"""The one decision per step: look at the page, choose one action.

This is the only component that talks to the language model during a run, and
it does exactly one thing with it -- turns "here is the task and here is the
page" into one typed action from the closed vocabulary in :mod:`actions`. It
does not carry the action out, does not decide whether it is allowed, and does
not decide whether the run should continue. Those belong to the executor, the
validator and the controller, and keeping them apart is what makes it possible
to say afterwards which of the four got a step wrong.

**A prompt, not a conversation.** The model is not shown a growing transcript.
Every step is assembled fresh from five pieces -- the task, a bounded window of
recent steps, the scratchpad, the current observation, and any skills retrieved
for this site -- so the prompt stays roughly the same size on step 2 and on step
20. A transcript that grows with the run costs more every step and, worse,
buries the current page under twenty stale copies of pages that have already
been left behind.

**Failure is a return value, not an exception.** A model that answers with
prose, or with an action that does not exist, is having an ordinary bad step.
The planner reports that as a :class:`PlannerDecision` carrying a
:class:`PlannerFailure` -- a stable code plus a sentence written to be quoted
back to the model -- so that the controller's retry loop reads it the same way
it reads a validator rejection. Two things follow from that choice. The failed
call's cost record comes back with it, because a wasted call is still billed
and a run's reported cost has to include it. And *no* partially parsed action
is ever returned: either the reply was one of the seven actions or the decision
carries nothing but the failure. There is no state in between, which is the
whole point of a closed schema.

What is still allowed to raise is the call itself failing -- no network, a bad
key, a rate limit. That is not a bad step but a broken endpoint, and the retry
policy for it is a different policy, owned elsewhere.

**Two inputs are strings on purpose.** History lines and retrieved skills
arrive already rendered. The components that own them -- memory and the skill
library -- decide what a step or a skill looks like when written down, and the
planner would only be guessing at their formats if it took their objects
instead. The one thing it does enforce is the *bound* on history, as a backstop
in case whatever supplies it forgets to.
"""

from __future__ import annotations

import functools
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .actions import Action, ActionError, action_schema, parse_action
from .model_client import (
    CallRecord,
    ModelClient,
    ModelResponseError,
    system_message,
)
from .observation import Observation
from .observer import describe_elements

#: The planner's system prompt. Packaged beside the module, like the injected
#: browser scripts, so it is found identically from a checkout and from an
#: installed wheel -- and so it stays a readable document rather than a string
#: literal wrapped at 100 columns.
PROMPT_PATH = Path(__file__).resolve().parent / "prompts" / "planner.md"

#: Name the schema is sent under. Some endpoints quote it back in errors, so it
#: is worth being a word rather than "response".
SCHEMA_NAME = "web_action"

#: Most recent steps included in the prompt, as a backstop. Memory owns the real
#: window; this is here so that a memory bug shows up as a slightly short prompt
#: rather than as a run whose cost climbs every step.
MAX_HISTORY_STEPS = 10

#: The reply could not be read as a JSON object at all -- prose, an apology, a
#: truncated object. Distinguished from the next code because the fixes differ:
#: this one is usually a route that cannot constrain decoding.
FAILURE_NOT_JSON = "not_json"

#: The reply was a JSON object, but not one of the seven actions. A model that
#: understands it must answer in JSON and still invents ``teleport``.
FAILURE_NOT_AN_ACTION = "not_an_action"


@functools.cache
def _system_prompt(path: Path) -> str:
    """Read a prompt file, once per path per process.

    Args:
        path: The file to read.

    Returns:
        str: Its contents.

    Raises:
        FileNotFoundError: If the prompt is missing, which means the package was
            built without its data files rather than that anything went wrong at
            run time.
    """
    return path.read_text(encoding="utf-8").strip()


@dataclass(frozen=True)
class PlannerFailure:
    """Why a step produced no action, in a form both a machine and a model read.

    Attributes:
        code: :data:`FAILURE_NOT_JSON` or :data:`FAILURE_NOT_AN_ACTION`. A short
            stable token, so a hundred runs can be counted by failure kind.
        detail: What was wrong, as a sentence. This is the text quoted back to
            the model on the retry, so it names the field rather than describing
            the problem in general terms.
        text: The reply as it arrived, untouched. Kept because the difference
            between "the model wrote an essay" and "the model wrote valid JSON
            with one field misspelled" is invisible in the code alone, and a
            person reading the trace needs to see it.
    """

    code: str
    detail: str
    text: str

    def summary(self) -> str:
        """Render this failure as the single line the step log records.

        Returns:
            str: For example ``planner failed (not_an_action): 'teleport' is
            not one of the 7 actions``.
        """
        return f"planner failed ({self.code}): {self.detail}"

    def to_dict(self) -> dict[str, Any]:
        """Render this failure as plain data, for the run's trace.

        Returns:
            dict[str, Any]: Every field, all JSON-safe already. The raw reply is
            written in full: it is at most a few hundred characters, and it is
            the evidence.
        """
        return {"code": self.code, "detail": self.detail, "text": self.text}


@dataclass(frozen=True)
class PlannerDecision:
    """One step's answer from the planner: an action, or the reason there isn't one.

    Exactly one of ``action`` and ``failure`` is set. That is enforced by
    construction rather than by a check, because both constructors are in this
    module and neither can produce a half-parsed action -- see the module
    docstring.

    Attributes:
        action: The chosen action, or None when the reply could not be read as
            one.
        failure: Why there is no action, or None on success.
        text: The model's raw reply.
        record: What the call cost. Present even when the step failed: the
            endpoint answered and billed for it.
    """

    action: Action | None
    failure: PlannerFailure | None
    text: str
    record: CallRecord | None

    @property
    def ok(self) -> bool:
        """bool: Whether this step produced an action to hand to the validator."""
        return self.action is not None

    def summary(self) -> str:
        """Render this decision as the single line the step log records.

        Returns:
            str: The action's own description on success -- ``click 4`` -- and
            the failure's summary otherwise.
        """
        if self.action is not None:
            return self.action.describe()
        return self.failure.summary() if self.failure else "planner returned nothing"

    def to_dict(self) -> dict[str, Any]:
        """Render this decision as plain data, for the run's trace.

        Returns:
            dict[str, Any]: The action or the failure, the raw reply, and the
            cost record. The raw reply is kept even on success, because it is
            the only way to tell later whether a well-formed action came from a
            model that answered cleanly or from one that had to be salvaged out
            of a fenced code block.
        """
        return {
            "action": self.action.to_dict() if self.action else None,
            "failure": self.failure.to_dict() if self.failure else None,
            "text": self.text,
            "record": asdict(self.record) if self.record else None,
        }


class Planner:
    """Makes one model call per step and returns one typed action.

    Construct one per run and reuse it: the system prompt is read once, and the
    model client underneath pools its connections.
    """

    def __init__(
        self,
        client: ModelClient,
        *,
        prompt_path: Path | None = None,
        max_history: int = MAX_HISTORY_STEPS,
    ) -> None:
        """Build a planner for one run.

        Args:
            client: The model client to call through. The planner never builds
                its own, so a run's profile, key and timeout are decided once,
                where they are configured.
            prompt_path: The system prompt. Defaults to the packaged one;
                overridden in tests and when trying a different prompt without
                editing the package.
            max_history: Most recent steps to include. See
                :data:`MAX_HISTORY_STEPS`.
        """
        self._client = client
        self._prompt_path = prompt_path or PROMPT_PATH
        self._max_history = max_history

    @property
    def system_prompt(self) -> str:
        """str: The instructions the model is given once per call."""
        return _system_prompt(self._prompt_path)

    def decide(
        self,
        *,
        task: str,
        observation: Observation,
        history: Sequence[str] = (),
        scratchpad: str = "",
        skills: Sequence[str] = (),
        feedback: Sequence[str] = (),
        screenshot_png: bytes | None = None,
    ) -> PlannerDecision:
        """Ask the model for one action, and return it typed or say why not.

        Args:
            task: The task in the user's own words, unchanged for the whole run.
            observation: The current page. The only page the model may name
                elements from.
            history: Recent steps, already rendered one per line by memory. The
                last :attr:`MAX_HISTORY_STEPS` are used.
            scratchpad: The running notes memory keeps across the window, or
                empty where there are none yet.
            skills: Retrieved skills, already rendered one per line by the skill
                library. Empty when none apply to this site, which is the normal
                case on a cold library.
            feedback: Complaints about this step's earlier attempts -- validator
                rejections, or a previous parse failure -- one per line. Empty on
                a first attempt.
            screenshot_png: The annotated screenshot, when the cost gate let one
                in. Passed through the client's own vision check, so a profile
                that cannot see images degrades to text rather than failing.

        Returns:
            PlannerDecision: The action, or the failure and the reply it came
            from. Either way with the call's cost record attached.

        Raises:
            ModelClientError: If the call itself failed -- network, auth, rate
                limit. A broken endpoint is not a bad step, and is not this
                component's to retry.
        """
        prompt = self.render_prompt(
            task=task,
            observation=observation,
            history=history,
            scratchpad=scratchpad,
            skills=skills,
            feedback=feedback,
        )
        messages = [
            system_message(self.system_prompt),
            self._client.observation_message(prompt, screenshot_png=screenshot_png),
        ]

        try:
            response = self._client.complete(
                messages, schema=action_schema(), schema_name=SCHEMA_NAME
            )
        except ModelResponseError as error:
            # The reply was not JSON. Nothing was parsed, so nothing is
            # returned but the failure -- there is no "it probably meant click".
            return PlannerDecision(
                action=None,
                failure=PlannerFailure(
                    code=FAILURE_NOT_JSON,
                    detail="reply was not a JSON object; answer with one action object and no prose",
                    text=error.text,
                ),
                text=error.text,
                record=error.record,
            )

        try:
            action = parse_action(response.data or {})
        except ActionError as error:
            return PlannerDecision(
                action=None,
                failure=PlannerFailure(
                    code=FAILURE_NOT_AN_ACTION,
                    detail="; ".join((str(error), *error.detail)),
                    text=response.text,
                ),
                text=response.text,
                record=response.record,
            )

        return PlannerDecision(
            action=action, failure=None, text=response.text, record=response.record
        )

    def render_prompt(
        self,
        *,
        task: str,
        observation: Observation,
        history: Sequence[str] = (),
        scratchpad: str = "",
        skills: Sequence[str] = (),
        feedback: Sequence[str] = (),
    ) -> str:
        """Assemble the per-step user message.

        Public because it is worth being able to print one without making a
        call: a prompt that is wrong is much easier to see than to infer from
        the actions it produced.

        Sections with nothing in them are left out rather than written as
        empty headings. On a first step that is four of the six, and a heading
        with nothing under it invites the model to wonder what it is missing.
        The current page is never left out, even when the page is blank -- that
        it is blank is the observation.

        Args:
            task: The task in the user's own words.
            observation: The current page.
            history: Recent steps, one per line; only the last
                :attr:`MAX_HISTORY_STEPS` are used.
            scratchpad: Running notes, or empty.
            skills: Retrieved skills, one per line.
            feedback: Complaints about this step's earlier attempts.

        Returns:
            str: The message text, with the task first and the page last -- the
            page being what the next decision is actually about, and the end of
            a prompt being where a model attends most reliably.
        """
        blocks = [f"# Task\n{task.strip()}"]

        if scratchpad.strip():
            blocks.append(f"# Notes so far\n{scratchpad.strip()}")
        if history:
            recent = list(history)[-self._max_history :]
            blocks.append("# Recent steps\n" + "\n".join(recent))
        if skills:
            # The one line of instruction is here rather than in the system
            # prompt because it only makes sense next to a list: a model told
            # in advance how to invoke skills, on a run where none were
            # retrieved, is a model that has been invited to invent one.
            blocks.append(
                "# Skills available for this site\n"
                "Routines already proved to work here. Invoke one with "
                "`use_skill`, giving every value it asks for; values in square "
                "brackets are optional. Prefer one over working the same route "
                "out step by step.\n" + "\n".join(skills)
            )

        blocks.append(self.render_observation(observation))

        if feedback:
            # Last, and phrased as an instruction rather than a report: this is
            # a retry, and the one thing that has to change is named here.
            blocks.append(
                "# Your last answer was rejected\n"
                + "\n".join(f"- {line}" for line in feedback)
                + "\nAnswer again, fixing that."
            )

        return "\n\n".join(blocks)

    @staticmethod
    def render_observation(observation: Observation) -> str:
        """Render one observation as the page section of the prompt.

        Args:
            observation: The page to describe.

        Returns:
            str: The URL and title, then the numbered element list, then the
            page text. Truncated text says so in the heading rather than
            trailing off silently, because a model that cannot tell a short page
            from a cut-off one will answer confidently from half of a page.
        """
        text_heading = "## Page text"
        if observation.text_truncated:
            text_heading += " (cut short to fit; scroll or narrow the page for more)"
        body = observation.text.strip() or "(this page has no main text)"
        return (
            f"# Current page\n"
            f"URL: {observation.url}\n"
            f"Title: {observation.title}\n\n"
            f"## Interactive elements\n{describe_elements(observation.elements)}\n\n"
            f"{text_heading}\n{body}"
        )


def render_history(
    entries: Sequence[Mapping[str, str]],
    *,
    start: int = 1,
) -> tuple[str, ...]:
    """Render step records as the history lines the planner prompt takes.

    A convenience for callers that hold step records rather than lines, and the
    one place the shape of a history line is decided, so that the trace and the
    prompt cannot end up describing the same step two ways.

    Args:
        entries: One mapping per step, with an ``action`` (what was chosen, as
            :meth:`Action.describe` renders it) and a ``result`` (what came of
            it, in a few words).
        start: The number of the first entry. One for a whole run; later for a
            window of recent steps, which would otherwise always appear to
            start at the beginning -- telling the model on step 14 that it has
            only just begun.

    Returns:
        tuple[str, ...]: One numbered line per step, e.g.
        ``1. click 4 -> navigated to /contact``.
    """
    return tuple(
        f"{number}. {entry.get('action', '?')} -> {entry.get('result', 'no result recorded')}"
        for number, entry in enumerate(entries, start=start)
    )
