"""Proving a candidate skill works before it is allowed to be used.

Induction produces a recipe from *one* run, which means every candidate arrives
with the same defect: it has been shown to repeat itself, and nothing else. The
dangerous version of that is not a recipe that obviously breaks -- it is one
that looks general and is secretly specific. A run that searched for
"membership" and clicked the result called "Membership application" induces a
skill whose every step is a faithful record of what happened, which is
schema-valid, and which will do the wrong thing for every other search. Nothing
in induction can see it.

Replaying with *different* values can. That is this module, and it is the gate
between "this worked once" and "this works".

**Two layers, cheapest first, and the order is the point.** The structural layer
replays every step with the verification block's alternate values and requires
each one to resolve and execute. It costs no model calls at all. Only if every
step worked does the semantic layer spend one judge call asking whether the
page it ended on is the right page.

That ordering is not a micro-optimisation. Most bad candidates fail
structurally -- the link named after the original search is not on a different
search's results -- and a verifier that asked a model first would pay for an
opinion about a page the recipe never reached. So a structural failure discards
immediately, with the judge never called, and a test asserts the call count is
zero.

**Both layers are needed, and neither would do alone.** Structural asks "did
every click land?"; a recipe can pass that and be on entirely the wrong page,
because clicking *something* successfully is not the same as arriving
somewhere. Semantic asks "is this the right page?"; it cannot be asked at all
until the steps have actually run. One is mechanical and free, the other is
judgment and costs a call, and putting them in the other order would buy
judgment about a mechanism that had not been checked.

**A discard records why.** Every failure carries the layer that refused it and
a sentence naming what went wrong -- the step that could not resolve, the
element that was gone, the page the judge did not accept. The project reports a
discard rate, and a rate with no reasons behind it says only that something is
wrong somewhere.

**Nothing here stores anything.** The verifier answers a question. Putting a
passing skill into the library is the caller's decision, made one line later,
and keeping it out of here is what stops "verified" and "stored" from becoming
the same step by accident.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .actions import Navigate
from .config import Config
from .executor import Executor
from .model_client import CallRecord, ModelClient, ModelClientError, system_message
from .observation import Observation
from .replay import SkillReplayer
from .selectors import Resolution
from .skill_schema import (
    Skill,
    placeholders_in,
)
from .validator import Validator

#: The judge's instructions, shipped inside the package so an installed copy
#: verifies the same way a checkout does.
PROMPT_PATH = Path(__file__).resolve().parent / "prompts" / "skill_verify.md"

#: Names the schema-constrained reply, for endpoints that want one.
SCHEMA_NAME = "skill_verdict"

#: The replay refused it: a step's element could not be found, or the browser
#: would not do what the step asked. Costs no model call, and catches most bad
#: candidates -- a recipe that memorised one search's results fails here.
STRUCTURAL = "structural"

#: The replay worked but the page it reached was not the right one. The layer
#: that costs a judge call, and the only one that can tell a routine that
#: *ran* from a routine that *worked*.
SEMANTIC = "semantic"

#: The candidate could not be replayed at all -- no browser, a judge that could
#: not be reached. Distinguished from the two above because it says nothing
#: about the skill: a candidate discarded for this reason should be tried
#: again, not written off.
UNCHECKED = "unchecked"

#: How much of the final page's own words the judge is shown. Generous, because
#: the evidence that a routine reached the right page is usually in the text,
#: and stingy enough that a long page does not become the whole prompt.
PAGE_TEXT_LIMIT = 2000


class VerificationError(Exception):
    """The check could not be carried out, which is not the same as failing it.

    Raised when the verifier cannot do its job -- a skill whose steps cannot be
    prepared at all, a judge endpoint that is unreachable. A candidate that is
    merely *wrong* does not raise: it comes back as a failed
    :class:`Verification`, because being wrong is the ordinary outcome this
    module exists to detect and report.
    """


@dataclass(frozen=True)
class StepReplay:
    """One step of a verification replay, and how it went.

    Attributes:
        number: Its position in the skill, from 1.
        described: The step as a person would read it, with the parameters
            already filled in -- so a failure report says what was actually
            attempted rather than what the file says.
        resolution: How the element was found, or why it was not. None for the
            steps that name no element.
        ok: Whether the step resolved and executed.
        detail: What went wrong, in a sentence. Empty when nothing did.
    """

    number: int
    described: str
    resolution: Resolution | None = None
    ok: bool = False
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Render this step as plain data, for a discard record.

        Returns:
            dict[str, Any]: JSON-safe values, with the resolution's own record
            nested where there was one.
        """
        return {
            "number": self.number,
            "step": self.described,
            "ok": self.ok,
            "detail": self.detail,
            "resolution": self.resolution.to_dict() if self.resolution else None,
        }


@dataclass(frozen=True)
class Verification:
    """The answer about one candidate skill.

    Attributes:
        passed: Whether the candidate may be stored. False is the ordinary
            outcome for a recipe that memorised its one example, which is what
            this module exists to catch.
        layer: Which layer decided, when one refused -- :data:`STRUCTURAL`,
            :data:`SEMANTIC` or :data:`UNCHECKED`. Empty on a pass.
        reason: Why, in a sentence, for the discard record. Empty on a pass
            where the judge said nothing worth keeping.
        replayed: Every step that was attempted, in order. A candidate that
            failed on step four is a different finding from one that failed on
            step one, and only this says which.
        judge_calls: How many model calls the check cost. Zero for a structural
            failure, which is the whole reason the layers are in this order.
        record: What the judge call cost, or None if there was none.
    """

    passed: bool
    layer: str = ""
    reason: str = ""
    replayed: tuple[StepReplay, ...] = ()
    judge_calls: int = 0
    record: CallRecord | None = None

    def summary(self) -> str:
        """Render the verdict as the line an induction pass prints.

        Returns:
            str: e.g. ``verified: the results page lists a privacy record`` or
            ``discarded (structural): step 3 could not find the element it
            needs``.
        """
        if self.passed:
            return f"verified: {self.reason}" if self.reason else "verified"
        return f"discarded ({self.layer}): {self.reason}"

    def to_dict(self) -> dict[str, Any]:
        """Render the verdict as plain data, so a discard is reportable.

        Returns:
            dict[str, Any]: JSON-safe values. The discard rate is one of the
            project's reported numbers, and a rate with no reasons behind it
            says only that something is wrong somewhere.
        """
        return {
            "passed": self.passed,
            "layer": self.layer,
            "reason": self.reason,
            "judge_calls": self.judge_calls,
            "steps": [step.to_dict() for step in self.replayed],
        }


def verdict_schema() -> dict[str, Any]:
    """Describe the answer the judge must give.

    Returns:
        dict[str, Any]: A JSON Schema for a pass/fail and a sentence. Small on
        purpose: the judge is being asked one question, and a reply with room
        for more would invite it to re-litigate the steps, which are already
        known to have worked.
    """
    return {
        "type": "object",
        "properties": {
            "passed": {"type": "boolean"},
            "reason": {"type": "string", "description": "one sentence"},
        },
        "required": ["passed", "reason"],
    }


@functools.cache
def _system_prompt(path: Path) -> str:
    """Read the judge's instructions, once per process.

    Args:
        path: The prompt file.

    Returns:
        str: Its contents.

    Raises:
        FileNotFoundError: If it is missing, which means the package was built
            without its data files.
    """
    return path.read_text(encoding="utf-8")


class SkillVerifier:
    """Replays a candidate with different values, and asks whether it worked.

    Built per verification pass rather than held open, because it needs a live
    browser and a judge, and both are things a caller should be deciding about
    deliberately.
    """

    def __init__(
        self,
        *,
        executor: Executor,
        validator: Validator,
        config: Config,
        judge: ModelClient | None = None,
        screenshot_dir: Path | None = None,
        prompt_path: Path | None = None,
    ) -> None:
        """Assemble a verifier.

        Args:
            executor: Performs the replayed steps. The *same* executor the
                agent uses, deliberately: a skill being checked has to be
                subject to the same timeouts, the same browser-event handling
                and the same failure vocabulary as a skill being used, or the
                check is of something other than what will run.
            validator: The gate every replayed action passes through. A stored
                skill is not a way around the rules a fresh action follows, and
                checking one is not a reason to suspend them.
            config: The run's settings, for observing pages.
            judge: The model that decides whether the final page is right.
                None runs the structural layer alone, which the spec allows as
                a deliberate fallback position when the semantic layer proves
                unreliable -- with the retirement threshold as the quality
                filter instead. Not a default anyone should reach by accident,
                which is why the verdict says which layers actually ran.
            screenshot_dir: Where the replay's screenshots go, if anywhere.
            prompt_path: The judge's instructions. Defaults to the shipped
                ones.
        """
        self._executor = executor
        self._judge = judge
        self._prompt_path = prompt_path or PROMPT_PATH
        # The same replayer the agent itself uses. A candidate checked through
        # one path and replayed through another would be a candidate checked
        # for something other than what will happen to it.
        self._replayer = SkillReplayer(
            executor=executor,
            validator=validator,
            config=config,
            screenshot_dir=screenshot_dir,
        )

    @property
    def system_prompt(self) -> str:
        """str: The instructions the judge is given."""
        return _system_prompt(self._prompt_path)

    def verify(self, skill: Skill, *, start_url: str) -> Verification:
        """Check one candidate, and say whether it may be stored.

        Args:
            skill: The candidate, fresh from induction and stored nowhere.
            start_url: The page to begin from. Not taken from the skill,
                because a skill has no starting page: it is an action the
                planner may take at any point in a run, from wherever the
                agent already is. For a check there is no run to already be in,
                so the caller supplies the page the skill was learnt on.

        Returns:
            Verification: The verdict, the layer that decided it, and every
            step that was attempted.

        Raises:
            VerificationError: If the check could not be carried out at all --
            which is not the same as the candidate failing, and is reported
            separately so that a skill is never written off for something that
            was wrong with the machinery.
        """
        params = dict(skill.verification.params)
        arrival = self._executor.execute(Navigate(action="navigate", url=start_url))
        if not arrival.ok:
            raise VerificationError(
                f"cannot check {skill.name!r}: the starting page {start_url} would not open "
                f"({arrival.detail or arrival.status})"
            )

        replay = self._replayer.run(skill, params)
        replayed = tuple(
            StepReplay(step.number, step.described, step.resolution, step.ok, step.detail)
            for step in replay.steps
        )
        if not replay.ok:
            # No judge call. A recipe whose steps did not run is not a recipe
            # whose destination is worth an opinion, and most bad candidates
            # end here.
            return Verification(
                passed=False, layer=STRUCTURAL, reason=replay.detail, replayed=replayed
            )

        if self._judge is None:
            return Verification(
                passed=True,
                reason="every step replayed with different values; no judge configured",
                replayed=replayed,
            )
        return self._ask_the_judge(skill, params, replayed)

    def _ask_the_judge(
        self, skill: Skill, params: dict[str, str], replayed: tuple[StepReplay, ...]
    ) -> Verification:
        """Ask whether the page the replay ended on is the right page.

        Args:
            skill: The candidate.
            params: The values it was replayed with.
            replayed: The steps that ran.

        Returns:
            Verification: The verdict, with the judge's own sentence as the
            reason either way -- a pass worth recording says what on the page
            showed it.

        Raises:
            VerificationError: If the judge could not be reached or did not
                answer with a verdict. Raised rather than treated as a failure:
                a candidate must not be written off because an endpoint was
                down.
        """
        assert self._judge is not None, "only called when a judge is configured"
        # The page the replay left the browser on. Taken through the same
        # replayer the steps ran through, so the judge is shown the page
        # the replay actually produced rather than a second opinion of it.
        final = self._replayer.look()
        try:
            response = self._judge.complete(
                [
                    system_message(self.system_prompt),
                    self._judge.observation_message(
                        self._render(skill, params, final),
                        screenshot_png=_screenshot_bytes(final),
                    ),
                ],
                schema=verdict_schema(),
                schema_name=SCHEMA_NAME,
            )
        except ModelClientError as error:
            raise VerificationError(f"the judge could not be reached: {error}") from error

        if response.data is None or "passed" not in response.data:
            raise VerificationError(f"the judge did not answer with a verdict: {response.text!r}")

        passed = bool(response.data["passed"])
        return Verification(
            passed=passed,
            layer="" if passed else SEMANTIC,
            reason=str(response.data.get("reason", "")),
            replayed=replayed,
            judge_calls=1,
            record=response.record,
        )

    def _render(self, skill: Skill, params: dict[str, str], final: Observation) -> str:
        """Write out the question for the judge.

        Args:
            skill: The candidate.
            params: The values it was replayed with.
            final: The page it ended on.

        Returns:
            str: What should have been true, what values were used, and what
            the page says. The values are named because the whole failure being
            looked for is a page that belongs to the recipe's *original*
            example, and a judge that did not know which values were used could
            not tell.
        """
        values = "\n".join(f"- {name}: {value}" for name, value in sorted(params.items()))
        return (
            f"## What should be true afterwards\n\n{skill.verification.expected_outcome}\n\n"
            f"## Replayed with these values\n\n{values or '(none)'}\n\n"
            f"## The final page\n\n{final.url}\n{final.title}\n\n"
            f"{final.text[:PAGE_TEXT_LIMIT]}\n"
        )


def _substitute(text: str, params: dict[str, str]) -> str:
    """Replace every ``{name}`` in one piece of stored text.

    Args:
        text: The stored text or URL.
        params: The values to put in.

    Returns:
        str: The text with the values in place.

    Raises:
        VerificationError: If a placeholder has no value. Refused rather than
            left alone, because leaving it would type the characters ``{query}``
            into a search box and report the step as having worked.
    """
    for name in placeholders_in(text):
        if name not in params:
            raise VerificationError(
                f"no value was given for {{{name}}}, so {text!r} cannot be replayed"
            )
        text = text.replace(f"{{{name}}}", params[name])
    return text


def _screenshot_bytes(observation: Observation) -> bytes | None:
    """Read the final page's picture, when there is one to read.

    Args:
        observation: The final observation.

    Returns:
        bytes | None: The PNG, or None when the step carried no image or the
        file has gone. None is ordinary: the judge is meant to decide from the
        page's words, and a run against an endpoint that cannot see images has
        no picture to offer anyway.
    """
    if observation.screenshot is None:
        return None
    try:
        return observation.screenshot.read_bytes()
    except OSError:
        return None
