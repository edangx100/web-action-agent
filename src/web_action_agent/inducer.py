"""Turning one successful run into a reusable skill.

This is the project's actual claim, in one module: the agent did a job once, and
the useful part of that afternoon becomes a recipe it can replay instead of
thinking the same thoughts again. Everything before it -- the loop, the trace,
the skill format, the durable selectors -- exists so that this can be attempted
on evidence rather than on a transcript somebody remembers.

**The model is asked only what needs judgment.** It would be easy to hand a
trace to a model and ask for a skill file. It would also be wrong, because most
of a skill is *known* and the model would be inventing it: which elements were
clicked, in which order, what they were called. Those are facts sitting in the
trace, and a model asked for them will occasionally produce a plausible
accessible name for a button that says something else -- a mistake nothing
downstream could catch, because the result looks exactly like a correct skill.

So the division is strict. The code reads the steps, resolves each element
against the observation that step was decided from, and builds the route. The
model answers four questions the recording genuinely cannot answer about
itself:

* which typed values were specific to this one task, and so should become
  blanks;
* what the skill should be called;
* what it does, in a line;
* and how somebody would check it works, with *different* values.

Judgment for the model, bookkeeping for the code, and no overlap.

**Only successful runs are eligible.** A failed run is a recording of something
that did not work, and a recipe made from one is a recipe for failing. The rule
is checked here rather than assumed, because the tempting exception -- "it only
failed at the last step, the earlier ones were fine" -- is exactly the case
where nobody can say which earlier step was the one that doomed it.

**Steps that failed are left out of the route.** A successful run may still
contain a click that went nowhere before the agent found the way. Recording
those would make a recipe that deliberately does things known not to work. What
it cannot do is tell a *wasted* step from one whose failure still changed the
page the later steps rely on, so the pruning is reported rather than silent:
the notes on the result say what was dropped and why.

**Nothing induced is trusted yet, and here is the reason it must not be.**
What comes out is a *candidate*: schema-valid, with a verification block
attached, and not stored anywhere. Replaying it with the block's different
values, and only then admitting it to the library, is the next component's job.

That is not caution for its own sake. Induction has a blind spot it cannot see
around: a step can name something the run only found *by chance*. A run that
searched for "membership" and clicked the result called "Membership application"
induces a skill that reads as though it searches for anything and opens whatever
it finds -- while actually looking for that one link by name. Every part of that
is a faithful record of what happened, the model was never shown a second search
to compare against, and the result passes every check this module can make.

Nothing here can catch it. Replaying with a different search term can, because
the link is not there, and that is precisely what the verification block is for.
Induction proposes; verification disposes.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .actions import Click, Fill, Navigate, Press, Scroll
from .allowlist import derive_domain, host_of
from .model_client import CallRecord, ModelClient, ModelResponseError, system_message, user_message
from .observation import Observation
from .outcomes import SUCCESS
from .reporting import describe_action
from .selectors import selector_for
from .skill_schema import (
    ClickStep,
    FillStep,
    NavigateStep,
    Parameter,
    PressStep,
    ScrollStep,
    Skill,
    SkillStep,
    Verification,
)
from .trace import utc_now
from .trace_reader import ReplayedRun

#: The system prompt, shipped inside the package so an installed copy induces
#: the same way a checkout does.
PROMPT_PATH = Path(__file__).resolve().parent / "prompts" / "inducer.md"

#: Names the schema-constrained reply under, for endpoints that want one.
SCHEMA_NAME = "induced_skill"

#: Actions that can become a skill step, mapped to the step that stores them.
#: Anything not in here -- ``done``, ``use_skill`` -- is not a skill step, and
#: the skill format refuses it; dropping them here is what keeps that refusal
#: from ever being reached with a valid-looking run behind it.
STEP_FOR_ACTION = {
    Click: ClickStep,
    Fill: FillStep,
    Press: PressStep,
    Scroll: ScrollStep,
    Navigate: NavigateStep,
}


class InductionError(Exception):
    """This run cannot become a skill, and the message says why.

    Every reason is about the *input*: a run that failed, a run with nothing
    worth recording, a step whose element is no longer in its own observation.
    None of them is a fault in the model's answer -- that is a different
    failure with a different fix, and it arrives as a validation error from the
    skill format instead.
    """


@dataclass(frozen=True)
class Induction:
    """One candidate skill, and how it was arrived at.

    Attributes:
        skill: The candidate. Schema-valid and carrying a verification block,
            but not stored anywhere and not yet checked against a live page.
        record: What the model call cost, for the run's accounting. Induction
            is a deliberate, separate command rather than something a run does
            quietly, and a cost that nobody asked for is exactly the kind of
            surprise that would make it one.
        notes: What was left out of the route, and why. Empty for the ordinary
            run where nothing was dropped. Worth carrying because a skill built
            from a run with four discarded steps deserves more suspicion than
            one built from a run that went straight through, and nothing else
            would ever say so.
    """

    skill: Skill
    record: CallRecord
    notes: tuple[str, ...] = ()


@functools.cache
def _system_prompt(path: Path) -> str:
    """Read the inducer's instructions, once per process.

    Args:
        path: The prompt file.

    Returns:
        str: Its contents.

    Raises:
        FileNotFoundError: If it is missing, which means the package was built
            without its data files rather than that anything went wrong here.
    """
    return path.read_text(encoding="utf-8")


def induction_schema() -> dict[str, Any]:
    """Describe the answer the model must give.

    Deliberately small, and deliberately not the skill format. The model is
    being asked four questions, so it answers four questions; handing it the
    whole skill schema would invite it to fill in the parts that are already
    known.

    Two shapes here are arrays where a mapping would read more naturally --
    the parameters, and the verification values. That is for the endpoints
    rather than for the reader: a JSON object with arbitrary keys cannot be
    pinned down by a schema, and schema-constrained decoding is what stops a
    reply being prose that has to be guessed at.

    Returns:
        dict[str, Any]: A JSON Schema for the reply.
    """
    parameter = {
        "type": "object",
        "properties": {
            "name": {"type": "string", "description": "lowercase_with_underscores"},
            "value": {"type": "string", "description": "the exact text from the run"},
            "description": {"type": "string", "description": "one line on what belongs here"},
            "required": {"type": "boolean"},
        },
        "required": ["name", "value", "description", "required"],
    }
    value = {
        "type": "object",
        "properties": {"name": {"type": "string"}, "value": {"type": "string"}},
        "required": ["name", "value"],
    }
    return {
        "type": "object",
        "properties": {
            "name": {"type": "string", "description": "lowercase_with_underscores"},
            "description": {"type": "string", "description": "one line on what it achieves"},
            "parameters": {"type": "array", "items": parameter},
            "verification": {
                "type": "object",
                "properties": {
                    "params": {"type": "array", "items": value},
                    "expected_outcome": {"type": "string"},
                },
                "required": ["params", "expected_outcome"],
            },
        },
        "required": ["name", "description", "parameters", "verification"],
    }


class Inducer:
    """Turns a successful run into a candidate skill, with one model call.

    One call, not one per step. The questions are about the run as a whole --
    which values were specific to it, what the whole thing achieves -- and
    asking them step by step would be asking a model to decide in the dark
    about a route it has only seen part of.
    """

    def __init__(self, client: ModelClient, *, prompt_path: Path | None = None) -> None:
        """Build an inducer.

        Args:
            client: The model transport. The same one the planner uses, so an
                induction pass is billed and recorded exactly like a run's own
                calls rather than through a second route nobody is watching.
            prompt_path: The instructions to use. Defaults to the shipped
                ones; a test overrides it to check the file is read rather
                than the text being built into the code.
        """
        self._client = client
        self._prompt_path = prompt_path or PROMPT_PATH

    @property
    def system_prompt(self) -> str:
        """str: The instructions the model is given."""
        return _system_prompt(self._prompt_path)

    def induce(self, run: ReplayedRun) -> Induction:
        """Turn one successful run into a candidate skill.

        Args:
            run: A run read back from its folder.

        Returns:
            Induction: The candidate, what the call cost, and what was left out
            of the route.

        Raises:
            InductionError: If the run is not eligible, has no steps worth
                recording, or names an element its own observation does not
                have.
            pydantic.ValidationError: If the model's answer does not make a
                valid skill -- a verification block missing a value, say.
                Raised from the skill format rather than caught here, because
                those are the format's rules and it says exactly which one was
                broken. (A proposed parameter no step uses is not one of
                these: it is dropped, with a note.)
            ModelClientError: If the call itself failed. A broken endpoint is
                not a bad run, and is not this component's to retry.
        """
        self._refuse_ineligible(run)
        usable, notes = self._route(run)
        if not usable:
            raise InductionError(
                f"{run.task!r} succeeded without doing anything a skill could repeat"
            )

        answer, record = self._ask(run, usable)
        skill, dropped = self._assemble(run, usable, answer)
        return Induction(skill=skill, record=record, notes=(*notes, *dropped))

    def _refuse_ineligible(self, run: ReplayedRun) -> None:
        """Refuse a run that has no business becoming a skill.

        Args:
            run: The run to check.

        Raises:
            InductionError: If the run was interrupted or did not succeed.
                Checked rather than assumed, because the tempting exception --
                "it only failed at the last step" -- is exactly the case where
                nobody can say which earlier step was the one that doomed it.
        """
        if not run.finished:
            raise InductionError(
                f"{run.folder} is an interrupted run: it never reached a verdict, "
                "so there is nothing to say it did the job"
            )
        assert run.outcome is not None, "a finished run has an outcome"
        if run.outcome.category != SUCCESS:
            raise InductionError(
                f"{run.folder} ended in {run.outcome.category!r}, and only successful runs "
                "may be learnt from: a recipe made from a failure is a recipe for failing"
            )

    def _route(self, run: ReplayedRun) -> tuple[list[tuple[Any, Observation]], tuple[str, ...]]:
        """Pick out the steps a skill can repeat, and say what was left behind.

        Args:
            run: The successful run.

        Returns:
            tuple: The usable steps, each with the observation it was decided
            against, and a note per dropped step.
        """
        usable: list[tuple[Any, Observation]] = []
        notes: list[str] = []

        for step in run.steps:
            action = step.action
            if action is None:
                # The step never chose anything: every attempt was refused, or
                # the endpoint broke. There is nothing to repeat.
                notes.append(f"step {step.number}: chose no action")
                continue
            if type(action) not in STEP_FOR_ACTION:
                # `done` and `use_skill`. Not a fault -- every run ends with a
                # `done` -- so it is not worth a note.
                continue
            if step.result is not None and not step.result.ok:
                notes.append(
                    f"step {step.number}: {describe_action(action, step.observation)} "
                    f"did not work ({step.result.status})"
                )
                continue
            usable.append((action, step.observation))

        return usable, tuple(notes)

    def _ask(
        self, run: ReplayedRun, usable: list[tuple[Any, Observation]]
    ) -> tuple[dict[str, Any], CallRecord]:
        """Ask the model the four questions the recording cannot answer.

        Args:
            run: The successful run.
            usable: The steps that will make up the route.

        Returns:
            tuple[dict[str, Any], CallRecord]: The model's answer and what the
            call cost.

        Raises:
            InductionError: If the reply was not a JSON object. The caller
                asked once and got prose; retrying is the caller's decision,
                and there is nothing partial worth keeping.
        """
        try:
            response = self._client.complete(
                [
                    system_message(self.system_prompt),
                    user_message(self._render(run, usable)),
                ],
                schema=induction_schema(),
                schema_name=SCHEMA_NAME,
            )
        except ModelResponseError as error:
            raise InductionError(f"the reply was not a skill description: {error}") from error

        if response.data is None:
            raise InductionError("the reply was not a JSON object")

        # The schema says these are required, but a route that cannot constrain
        # decoding is shown the schema in the prompt and no more -- so a reply
        # missing half of them is a real possibility rather than a theoretical
        # one. Named here so the caller gets "the reply had no verification
        # block" rather than a KeyError from three functions further in.
        missing = [field for field in induction_schema()["required"] if field not in response.data]
        if missing:
            raise InductionError(f"the reply left out {', '.join(missing)}")
        return response.data, response.record

    def _render(self, run: ReplayedRun, usable: list[tuple[Any, Observation]]) -> str:
        """Write out the run for the model to read.

        The route is rendered with elements *named* rather than numbered, the
        same way a person watching the run would have seen it -- the numbers
        mean nothing outside the observation that assigned them, and showing
        them would invite the model to refer to one.

        Args:
            run: The successful run.
            usable: The steps making up the route.

        Returns:
            str: The task, where it happened, what was done, and the answer.
        """
        steps = "\n".join(
            f"{number}. {describe_action(action, observation)}"
            for number, (action, observation) in enumerate(usable, start=1)
        )
        # The last page's own words, because the answer usually came from them
        # and they are the best evidence of what a *different* set of values
        # would plausibly find.
        final = usable[-1][1]
        answer = (
            run.outcome.answer if run.outcome and run.outcome.answer else "(no answer asked for)"
        )
        return (
            f"## The task\n\n{run.task}\n\n"
            f"## The site\n\n{run.start_url}\n\n"
            f"## What the agent did\n\n{steps}\n\n"
            f"## What the last page said\n\n{final.text[:1500]}\n\n"
            f"## The answer it gave\n\n{answer}\n"
        )

    def _assemble(
        self,
        run: ReplayedRun,
        usable: list[tuple[Any, Observation]],
        answer: dict[str, Any],
    ) -> Skill:
        """Build the candidate from the known route and the model's judgment.

        Args:
            run: The successful run.
            usable: The steps making up the route.
            answer: The model's reply.

        Returns:
            tuple[Skill, tuple[str, ...]]: The candidate, valid by construction,
            and a note for each parameter the model proposed that no step
            could use.

        Raises:
            InductionError: If a step names an element its own observation does
                not have, which would leave a step nothing could describe.
            pydantic.ValidationError: If the model's judgment does not make a
                valid skill.
        """
        proposed = answer.get("parameters", [])
        # Longest first, so that a parameter whose value contains another's --
        # "reading room" and "reading" -- does not get half-replaced by the
        # shorter one and leave a placeholder inside a literal.
        blanks = sorted(
            ((item["value"], item["name"]) for item in proposed),
            key=lambda pair: len(pair[0]),
            reverse=True,
        )
        steps = tuple(self._step_for(action, observation, blanks) for action, observation in usable)

        # Bookkeeping, not judgment: a parameter is a value the run typed or
        # visited, replaced by a blank. One whose value appears in no step left
        # no blank, so no step can use it -- typically a fact the model read
        # off the page, such as an age group, on a run that only clicked. Kept,
        # it would make the skill claim to need something it never uses, which
        # the skill format rightly refuses. So it is dropped, here and from the
        # verification values, and the notes say so.
        used = {name for step in steps for name in step.parameters_used()}
        kept = [item for item in proposed if item["name"] in used]
        dropped = tuple(
            f"proposed parameter {item['name']!r} ({item['value']!r}) was dropped: "
            "the run never typed or visited that value, so no step could use it"
            for item in proposed
            if item["name"] not in used
        )
        params = tuple(
            Parameter(
                name=item["name"],
                description=item.get("description", ""),
                required=item.get("required", True),
                example=item["value"],
            )
            for item in kept
        )
        verification = Verification(
            params={
                item["name"]: item["value"]
                for item in answer["verification"]["params"]
                if item["name"] in used
            },
            expected_outcome=answer["verification"]["expected_outcome"],
        )
        skill = Skill(
            name=answer["name"],
            description=answer["description"],
            domain=derive_domain(host_of(run.start_url)),
            params=params,
            steps=steps,
            verification=verification,
            induced_from=run.run_id,
            created_at=utc_now(),
        )
        return skill, dropped

    def _step_for(
        self, action: Any, observation: Observation, blanks: list[tuple[str, str]]
    ) -> SkillStep:
        """Turn one executed action into a stored step.

        Args:
            action: What was done.
            observation: The page it was decided against, which is where the
                element's description comes from.
            blanks: ``(value, parameter name)`` pairs to replace, longest
                first.

        Returns:
            SkillStep: The step, with the element described rather than
            numbered and the task-specific values replaced by placeholders.

        Raises:
            InductionError: If the element is not in the observation. That
                should be impossible for a step that executed, and is worth
                refusing over rather than working around: the alternative is a
                step with no way of ever being found again.
        """
        if isinstance(action, Press):
            return PressStep(action="press", key=action.key)
        if isinstance(action, Scroll):
            return ScrollStep(action="scroll", direction=action.direction)
        if isinstance(action, Navigate):
            return NavigateStep(action="navigate", url=_blank_out(action.url, blanks))

        element = observation.element_by_id(action.element_id)
        if element is None:
            raise InductionError(
                f"step {describe_action(action, observation)} names element "
                f"{action.element_id}, which is not in the page it was decided from"
            )
        selector = selector_for(element, observation.elements)
        if isinstance(action, Fill):
            return FillStep(
                action="fill",
                selector=selector,
                text=_blank_out(action.text, blanks),
                note=f"{element.role} on {observation.title}".strip(),
            )
        return ClickStep(action="click", selector=selector, note=element.name or element.text)


def _blank_out(text: str, blanks: list[tuple[str, str]]) -> str:
    """Replace the run's own values with the parameters that stand for them.

    Done here rather than by the model. The model says *which* values are
    task-specific; putting the placeholders in is a substitution, and a model
    asked to retype a URL with one blank filled in will occasionally retype the
    rest of it slightly differently.

    Args:
        text: The text or URL as the run used it.
        blanks: ``(value, parameter name)`` pairs, longest value first.

    Returns:
        str: The text with each value replaced by ``{name}``. An exact,
        whole-string match becomes the placeholder alone, which is the common
        case -- a search box holds the search term and nothing else. Otherwise
        the value is replaced where it appears, for the addresses and
        sentences that carry one inside something longer.
    """
    for value, name in blanks:
        if not value:
            continue
        if text == value:
            return f"{{{name}}}"
        text = text.replace(value, f"{{{name}}}")
    return text
