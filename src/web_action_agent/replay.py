"""Running a learned route, with no model involved at all.

This is where the project's claim is actually cashed. Everything before it built
the machinery to learn a route, prove it and offer it; replaying one is the part
that is *cheaper*, and the saving is not incidental -- it is the whole reason a
library exists. A four-step skill replayed costs **zero planner calls**, against
four for working the same route out again, and the difference is the headline
number the benchmark is built to measure.

**Zero is a promise, not an outcome.** Nothing in this module can call a model:
it is handed an executor, a validator and a page, and there is no model client
anywhere in reach. The guarantee is structural rather than a matter of care --
the cheapest way to keep a count at zero is to make the call impossible.

**A replayed action goes through the same gate and the same executor as a fresh
one.** Each stored step is turned into an ordinary numbered action against the
page as it is *now*, and handed to the validator and then the executor. That
costs an extra hop and buys the thing worth having: a stored route gets no
private path to the browser. Every rule the gate enforces, every timeout, every
browser-event handler applies to a replayed step exactly as it does to one the
planner proposed -- so a skill cannot become a way around any of them, and a
replay that hangs trips the same ceiling a fresh click would.

**Stopping at the first failure is the point, not a shortcut.** Once a step has
not happened, every later step is being attempted from a page the route did not
expect. Carrying on would produce a cascade of failures that say nothing except
that the run went off the rails several steps ago, which is already known -- and
it would act on pages nobody planned for. So the replay stops, reports how far
it got, and the controller hands back to the planner from there.

**Progress is kept.** A replay that completed two of four steps leaves those two
done. The planner resumes from the page in front of it rather than from the
beginning, which is the difference between a skill being a gamble and a skill
being a head start: even a failed replay is usually worth more than not having
tried.

**One implementation, two callers.** The agent replays a skill the planner
chose; the verifier replays a candidate with deliberately different values
before it may be stored. Those must be the same code, or the thing checked is
not the thing that will run.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .actions import Action, Click, Fill, Navigate, Press, Scroll, UseSkill
from .config import Config
from .executor import Executor
from .observation import Observation, capture
from .reporting import describe_action
from .selectors import Resolution, resolve
from .skill_schema import (
    ClickStep,
    FillStep,
    NavigateStep,
    PressStep,
    ScrollStep,
    Skill,
    SkillStep,
    placeholders_in,
)
from .validator import Validator

#: Where a skill library is looked up by name. Structural rather than an
#: import, for the same reason the validator declares one: the two components
#: are built separately and neither should have to change when the other does.
SkillSource = Any


class ReplayError(Exception):
    """A skill cannot be replayed at all, before any step is attempted.

    A value missing for one of the skill's own blanks, or a named skill the
    library does not hold. Both are faults in the *request* rather than in the
    route, which is why they are raised: a replay that failed halfway is
    ordinary and reported, but a replay that could never have started is a
    caller's mistake and worth stopping on.
    """


@dataclass(frozen=True)
class ReplayedStep:
    """One step of a replay, and how it went.

    Attributes:
        number: Its position in the skill, from 1.
        described: The step as a person reads it, with the values already
            filled in and the element named -- so a failure says what was
            actually attempted rather than what the file says.
        resolution: How the element was found, or why it was not. None for the
            steps that name no element.
        ok: Whether it resolved and executed.
        detail: What went wrong, in a sentence. Empty when nothing did.
    """

    number: int
    described: str
    resolution: Resolution | None = None
    ok: bool = False
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Render this step as plain data, for the run's trace.

        Returns:
            dict[str, Any]: JSON-safe values, with the resolution's own record
            nested where there was one -- which is what shows a route that
            succeeded only by falling back to a weak description, before that
            weakness becomes a failure.
        """
        return {
            "number": self.number,
            "step": self.described,
            "ok": self.ok,
            "detail": self.detail,
            "resolution": self.resolution.to_dict() if self.resolution else None,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ReplayedStep:
        """Rebuild one replayed step from the plain data of a trace.

        The resolution is deliberately not rebuilt into a
        :class:`~web_action_agent.selectors.Resolution`. Reading a run back
        needs to know *that* a step resolved and by which description, and the
        record says so; reconstructing the element it found would mean
        reconstructing a page that no longer exists.

        Args:
            data: One step record.

        Returns:
            ReplayedStep: The step, with its resolution left as the record.

        Raises:
            KeyError: If ``number`` or ``ok`` is missing.
        """
        return cls(
            number=data["number"],
            described=data.get("step", ""),
            resolution=None,
            ok=data["ok"],
            detail=data.get("detail", ""),
        )


@dataclass(frozen=True)
class Replay:
    """What came of replaying one skill.

    Carries ``ok`` and ``detail``, which is all the controller's replay seam
    asks for -- and rather more besides, because a trace that recorded only
    "the skill failed" could not answer the first question anybody has about a
    failed replay, which is how far it got.

    Attributes:
        skill: The name of the skill replayed.
        steps: Every step attempted, in order. Shorter than the skill when the
            replay stopped early.
        total: How many steps the skill has, so "2 of 4" is answerable.
        ok: Whether every step ran.
        detail: What happened, in a sentence, for the trace and for the
            planner's notes when control comes back.
    """

    skill: str
    steps: tuple[ReplayedStep, ...] = ()
    total: int = 0
    ok: bool = False
    detail: str = ""

    @property
    def completed(self) -> int:
        """int: How many steps actually ran."""
        return sum(1 for step in self.steps if step.ok)

    @property
    def handover(self) -> int | None:
        """int | None: The step the planner is taking over at, or None.

        The number of the step that did not run. What makes a fallback a
        *handover* rather than a restart: the planner picks up here, with
        everything before it already done.
        """
        return None if self.ok else self.completed + 1

    def summary(self) -> str:
        """Render the replay as the line a trace and the planner's notes record.

        Returns:
            str: e.g. ``replayed search_catalogue: 4 of 4 steps`` or
            ``search_catalogue stopped at step 3 of 4 (2 done): no element
            matched`` -- the counts first, because how far it got is the first
            thing anybody asks.
        """
        if self.ok:
            return f"replayed {self.skill}: {self.total} of {self.total} steps"
        return (
            f"{self.skill} stopped at step {self.handover} of {self.total} "
            f"({self.completed} done): {self.detail}"
        )

    def to_dict(self) -> dict[str, Any]:
        """Render the replay as plain data, for the run's trace.

        ``completed`` and ``handover`` are written out even though both are
        derived from the steps. A reader of a run folder should not have to
        work out where the planner took over by counting; it is the first
        question a failed replay raises.

        Returns:
            dict[str, Any]: What ran, what did not, and where control went
            back.
        """
        return {
            "skill": self.skill,
            "ok": self.ok,
            "completed": self.completed,
            "total": self.total,
            "handover": self.handover,
            "detail": self.detail,
            "steps": [step.to_dict() for step in self.steps],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Replay:
        """Rebuild a replay from the plain data of a trace.

        The inverse of :meth:`to_dict`, written beside it so a reader of a run
        cannot come to disagree with its writer. ``completed`` and ``handover``
        are not read back: both are derived from the steps, and reading them
        would let a trace assert a count that contradicts the steps it carries.

        Args:
            data: One replay record, as the action log stores it.

        Returns:
            Replay: The same value the trace was written from.

        Raises:
            KeyError: If the record is missing a field always written.
        """
        return cls(
            skill=data["skill"],
            steps=tuple(ReplayedStep.from_dict(step) for step in data.get("steps", ())),
            total=data.get("total", 0),
            ok=data["ok"],
            detail=data.get("detail", ""),
        )


def fill_in(step: SkillStep, params: dict[str, str]) -> SkillStep:
    """Put the parameter values into one stored step.

    Args:
        step: The stored step, with ``{name}`` where values go.
        params: The values to use.

    Returns:
        SkillStep: The same step with its placeholders replaced. Steps carrying
        no text come back unchanged, which is most of them -- only what gets
        typed and where the run goes are ever task-specific.

    Raises:
        ReplayError: If the step refers to a value that was not supplied. The
            skill format already refuses a placeholder the skill does not
            declare and the gate refuses a call missing a required value, so
            reaching this means something upstream let an incomplete set
            through. Refused rather than left alone, because leaving it would
            type the characters ``{query}`` into a search box and report the
            step as having worked.
    """
    if isinstance(step, FillStep):
        return step.model_copy(update={"text": _substitute(step.text, params)})
    if isinstance(step, NavigateStep):
        return step.model_copy(update={"url": _substitute(step.url, params)})
    return step


class SkillReplayer:
    """Runs a stored route against a live page, with no model in reach.

    Built per run and handed to the controller as its replay seam, and built
    again by the verifier for checking a candidate. Both get the same code,
    deliberately: a candidate verified through one path and replayed through
    another would be a candidate checked for something other than what happens.
    """

    def __init__(
        self,
        *,
        executor: Executor,
        validator: Validator,
        config: Config,
        library: SkillSource | None = None,
        screenshot_dir: Path | None = None,
    ) -> None:
        """Assemble a replayer.

        Args:
            executor: Performs each resolved action. The run's own executor, so
                a replayed step shares its timeouts, its structured errors and
                its browser-event handling.
            validator: The gate each resolved action passes through. Replaying
                is not a reason to suspend the rules a fresh action follows.
            config: The run's settings, for observing pages between steps.
            library: Where a skill named in a ``use_skill`` is looked up. Not
                needed by a caller that already holds the skill, which is how
                the verifier uses this.
            screenshot_dir: Where the replay's screenshots go, if anywhere.
        """
        self._executor = executor
        self._validator = validator
        self._config = config
        self._library = library
        self._screenshot_dir = screenshot_dir

    def __call__(self, action: UseSkill, observation: Observation) -> Replay:
        """Replay the skill a ``use_skill`` names.

        The shape the controller's replay seam expects, so an instance of this
        can be handed straight to a run.

        Args:
            action: The invocation the planner chose, already approved by the
                gate -- so the skill exists, is active, belongs to this site,
                and has every value it needs.
            observation: The page before the replay. Unused: each step takes
                its own fresh look, because the numbering of an observation is
                only good for the moment it was taken.

        Returns:
            Replay: What ran and what did not.

        Raises:
            ReplayError: If the library has no such skill. The gate checks that
                first, so reaching this means the library changed underneath
                the run.
        """
        del observation  # each step looks again; see `_look`.
        skill = self._library.find(action.skill) if self._library else None
        if skill is None:
            raise ReplayError(
                f"the library has no skill named {action.skill!r}, though the gate approved it"
            )
        return self.run(skill, dict(action.params))

    def run(self, skill: Skill, params: dict[str, str]) -> Replay:
        """Replay one skill with one set of values.

        Args:
            skill: The route to run.
            params: The values for its blanks.

        Returns:
            Replay: Every step attempted, how far it got, and why it stopped.
            Never raises for a step that simply did not work -- a route a site
            has changed under is ordinary, and it is what retirement exists for.

        Raises:
            ReplayError: If a value for one of the skill's own blanks is
                missing, which would make the route unrunnable from the start.
        """
        done: list[ReplayedStep] = []

        for number, stored in enumerate(skill.steps, start=1):
            step = fill_in(stored, params)
            observation = self.look()
            action, resolution = self._action_for(step, observation)

            if action is None:
                detail = resolution.summary() if resolution else "the step named no action"
                done.append(ReplayedStep(number, step.describe(), resolution, False, detail))
                # The reason carries the *why* as well as the which, because it
                # is what the planner is told when control comes back and what
                # a reader of the trace sees first. "Step 3 failed" sends
                # somebody looking; "step 3: no element matched, and here is
                # every description that was tried" answers the question.
                return self._stopped(skill, done, f"step {number}: {detail}")

            described = describe_action(action, observation)
            verdict = self._validator.check(action, observation)
            if not verdict.ok:
                # The gate refused a replayed action. Rare, and its own finding:
                # the route wants to do something this run is not allowed to,
                # which is a different problem from an element being missing.
                detail = verdict.summary()
                done.append(ReplayedStep(number, described, resolution, False, detail))
                return self._stopped(skill, done, f"step {number} was refused: {detail}")

            result = self._executor.execute(verdict.action, observation=observation)
            done.append(
                ReplayedStep(
                    number, described, resolution, result.ok, "" if result.ok else result.summary()
                )
            )
            if not result.ok:
                return self._stopped(skill, done, f"step {number} failed: {result.summary()}")

        return Replay(
            skill=skill.name,
            steps=tuple(done),
            total=len(skill.steps),
            ok=True,
            detail=f"all {len(skill.steps)} steps ran",
        )

    def _stopped(self, skill: Skill, done: list[ReplayedStep], why: str) -> Replay:
        """Build the result for a replay that could not finish.

        Args:
            skill: The route being replayed.
            done: Every step attempted, the last of which failed.
            why: What stopped it.

        Returns:
            Replay: The record, with the progress made kept rather than
            discarded -- the steps that worked really did happen, and the page
            is where they left it.
        """
        return Replay(
            skill=skill.name, steps=tuple(done), total=len(skill.steps), ok=False, detail=why
        )

    def _action_for(
        self, step: SkillStep, observation: Observation
    ) -> tuple[Action | None, Resolution | None]:
        """Turn one stored step into a concrete action against the page in front of it.

        The step says "the button called Reserve"; the executor acts on numbers.
        This is the join, and the place a route that no longer fits its site
        comes apart: the element it describes is not on the page, resolution
        finds nothing, and there is no action to perform.

        Args:
            step: The step, with its values already filled in.
            observation: The page as it is now.

        Returns:
            tuple: The action, or None when the element could not be found, and
            the resolution saying why. The resolution is None for steps that
            name no element, which cannot fail to resolve because they resolve
            nothing.
        """
        if isinstance(step, PressStep):
            return Press(action="press", key=step.key), None
        if isinstance(step, ScrollStep):
            return Scroll(action="scroll", direction=step.direction), None
        if isinstance(step, NavigateStep):
            return Navigate(action="navigate", url=step.url), None

        resolution = resolve(step.selector, observation.elements)
        if resolution.element is None:
            return None, resolution
        if isinstance(step, FillStep):
            return Fill(action="fill", element_id=resolution.element.id, text=step.text), resolution
        assert isinstance(step, ClickStep), "the five step kinds are covered above"
        return Click(action="click", element_id=resolution.element.id), resolution

    def look(self) -> Observation:
        """Take a fresh observation of whatever page the run is on now.

        Fresh for every step, not once at the start. Element numbers are
        assigned per look, so an observation taken before a click describes a
        page that no longer exists -- and resolving the next step against it
        would find the right description at a number that now belongs to
        something else.

        Returns:
            Observation: The current page, from the executor, because a click
            that opened a tab moved the run to a different one.
        """
        observation, _ = capture(
            self._executor.page,
            config=self._config,
            screenshot_dir=self._screenshot_dir,
            label="replay",
        )
        return observation


def _substitute(text: str, params: dict[str, str]) -> str:
    """Replace every ``{name}`` in one piece of stored text.

    Args:
        text: The stored text or URL.
        params: The values to put in.

    Returns:
        str: The text with the values in place.

    Raises:
        ReplayError: If a placeholder has no value.
    """
    for name in placeholders_in(text):
        if name not in params:
            raise ReplayError(f"no value was given for {{{name}}}, so {text!r} cannot be replayed")
        text = text.replace(f"{{{name}}}", params[name])
    return text
