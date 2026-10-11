"""The AI judge: a second, stronger model that marks what code cannot.

Some benchmark answers can be checked by plain code -- see
:mod:`~web_action_agent.checkers`. Others cannot. "What is Singapore's latest
total population?" has a right answer that changes every year, so no fixed
string can mark it. For those, a second model looks at how the run ended and
says pass or fail.

**What the judge is shown.** The task, exactly as the agent was given it; a
sentence from the task file describing what success looks like; the agent's
final answer; the last few screenshots of the run, the final one last; and a
short list of the steps the agent took. The steps are there for the second
half of the judge's job (below), not to decide pass or fail: the judge is told
that the agent's own claim of success is not evidence.

**What it answers.** Three things, in a fixed shape:

* ``passed`` -- true or false;
* ``justification`` -- one sentence saying what on the page or in the answer
  decided it;
* ``category`` -- for a failure, *which kind* of failure:

  * ``grounding``: the agent meant the right thing but acted on the wrong
    element -- typed into the wrong box, clicked the wrong link;
  * ``planning``: the agent chose the wrong thing to do -- searched for the
    wrong term, stopped too early, misread what was asked;
  * ``navigation``: the agent got lost between pages -- never reached the page
    with the answer, or wandered away from it;
  * ``site_change``: the website itself got in the way -- the page has moved,
    changed, or shows something the task could not have expected.

  The categories are what turns a success rate into something useful: "62%"
  says how often the agent works, and the categories say what to fix.

**Every failure gets exactly one category.** Not every failure is decided by
the judge: an agent that ran out of steps, or a wrong answer caught by a
machine check, has already failed before the judge is involved. For those,
:meth:`Judge.categorize` asks a narrower question -- "this run failed, for
this reason; which of the four kinds of failure was it?" -- whose only valid
answers are the four categories. The judge is told not to decide pass or
fail again, and cannot: there is no "passed" in that answer, and no "none".

**Nothing grades its own homework.** The judge must come from a different
*family* of models than the agent it marks. A model is much more likely to
accept an answer that looks like one it would have given. So
:func:`independence_problem` compares the two before anything is judged, and
the program refuses to start with a judge from the planner's own family. How a
model's family is worked out is described at :func:`model_family`.

**A judge that cannot answer has not passed anything.** Every way a judge call
can go wrong -- the service is down, the reply is not JSON, a field is missing
or the wrong type, a failure comes without a category -- is recorded as a
*judging error*: its own outcome, distinct from both pass and fail, with the
reason and the raw reply kept. It is never turned into a pass, and it is never
silently counted as a fail either: a fail would be blaming the agent for the
judge's problem.

**Temperature zero.** The judge is asked to answer the same way every time it
is shown the same thing, so a verdict can be recomputed and compared. A fixed
seed is sent too, for services that honour one.
"""

from __future__ import annotations

import functools
import re
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .benchmark import BenchmarkTask
from .benchmark import Judge as JudgeVerdictMethod
from .checkers import final_state
from .config import Config, ConfigError
from .model_client import (
    CallRecord,
    ModelClient,
    ModelClientError,
    ModelResponseError,
    Profile,
    ProfileError,
    load_profile,
    resolve_profile,
    system_message,
    user_message,
)
from .reporting import describe_action
from .trace_reader import ReplayedRun

#: The four kinds of failure a judge may name, in the order they are explained
#: in the module docstring.
FAILURE_CATEGORIES = ("grounding", "planning", "navigation", "site_change")

#: What the judge writes as the category of a run that passed. A word rather
#: than an empty value, so that "the judge forgot" and "there was nothing to
#: categorise" cannot look the same.
NO_FAILURE = "none"

#: The three outcomes of judging one run.
PASS = "pass"
FAIL = "fail"
JUDGING_ERROR = "judging_error"

#: The judge's instructions, shipped inside the package.
DEFAULT_PROMPT_PATH = Path(__file__).resolve().parent / "prompts" / "judge.md"

#: How many of the run's last screenshots the judge sees. The final page
#: decides the verdict; the one or two before it show how the agent got there,
#: which the failure category needs. More would cost more and add little.
SCREENSHOTS_SHOWN = 3

#: Name for the answer's schema in the request.
SCHEMA_NAME = "judge_verdict"

#: Name for the narrower answer's schema, when only the category is asked for.
CATEGORY_SCHEMA_NAME = "failure_category"


class JudgeConfigError(ConfigError):
    """The judge cannot be used as configured.

    A :class:`~web_action_agent.config.ConfigError`, because it is the same
    kind of problem -- a setting that cannot work -- and is reported the same
    way, before anything runs.
    """


# --- Model families ------------------------------------------------------------


@dataclass(frozen=True)
class ModelFamily:
    """Who made a model, and which line of models it belongs to.

    Attributes:
        vendor: The part of the model id before the ``/``, such as ``nvidia``
            or ``google``. Empty if the id has none.
        line: The first word of the model's own name, such as ``nemotron``,
            ``gemini``, ``claude``, ``gpt``, or ``qwen``.
    """

    vendor: str
    line: str

    def __str__(self) -> str:
        """Render as ``vendor/line``, or just the line when there is no vendor."""
        return f"{self.vendor}/{self.line}" if self.vendor else self.line


def model_family(model_id: str) -> ModelFamily:
    """Work out a model's family from its id.

    Model ids follow a common shape, ``vendor/name-version-size``:
    ``nvidia/nemotron-3-nano-30b-a3b``, ``google/gemini-3.8-flash``. The
    vendor is the part before the slash. The line is the leading letters of
    the name, before the first digit or punctuation: ``nemotron``, ``gemini``.

    Args:
        model_id: A model id as a profile names it.

    Returns:
        ModelFamily: The vendor and line, both lowercased.
    """
    vendor, _, name = model_id.strip().lower().rpartition("/")
    match = re.match(r"[a-z]+", name)
    return ModelFamily(vendor=vendor, line=match.group(0) if match else name)


def same_family(a: ModelFamily, b: ModelFamily) -> bool:
    """Report whether two models count as the same family.

    Deliberately cautious: sharing *either* the vendor or the line is enough.
    Sharing a vendor catches a vendor's model built on another's base, such as
    ``nvidia/llama-3.1-nemotron``. Sharing a line catches one model served
    under two vendors' names, such as ``meta-llama/llama-4`` and
    ``nvidia/llama-4``. Being wrongly refused costs choosing another judge;
    being wrongly allowed costs every published number.

    Args:
        a: One model's family.
        b: The other's.

    Returns:
        bool: True if they share a vendor or a line.
    """
    return bool(a.vendor and a.vendor == b.vendor) or a.line == b.line


def route_models(profile: Profile) -> tuple[str, ...]:
    """List every model a profile can send a call to.

    Args:
        profile: A resolved profile.

    Returns:
        tuple[str, ...]: The text model, and the vision model if there is one.
    """
    models = [profile.text.model]
    if profile.vision is not None:
        models.append(profile.vision.model)
    return tuple(models)


def family_clash(planner: Profile, judge_models: tuple[str, ...]) -> tuple[str, str] | None:
    """Find a judge model in the same family as a planner model, if any.

    Both of the planner's routes are checked, text and vision, because either
    may have produced the steps being judged.

    Args:
        planner: The profile the agent runs under.
        judge_models: Every model the judge may use.

    Returns:
        tuple[str, str] | None: The first clashing pair, planner model first,
        or None if every judge model is from another family.
    """
    for planner_model in route_models(planner):
        for judge_model in judge_models:
            if same_family(model_family(planner_model), model_family(judge_model)):
                return planner_model, judge_model
    return None


def check_independence(planner: Profile, judge: Profile) -> None:
    """Refuse a judge from the same family as any model the planner uses.

    Args:
        planner: The profile the agent runs under.
        judge: The profile chosen to judge it. Both its routes are checked,
            because either may do the judging.

    Raises:
        JudgeConfigError: If any judge model shares a family with any planner
            model. The message names both models and their families.
    """
    clash = family_clash(planner, route_models(judge))
    if clash is not None:
        planner_model, judge_model = clash
        raise JudgeConfigError(
            f"judge profile {judge.name!r} uses {judge_model!r}, which is in the same "
            f"family as the planner's {planner_model!r} ({model_family(planner_model)} and "
            f"{model_family(judge_model)}). The judge must come from a different family of "
            "models, so the agent is not marked by its own kind. Choose another judge profile"
        )


def independence_problem(config: Config, *, profiles_path: Path | None = None) -> str | None:
    """Check the configured judge against the configured planner, at startup.

    Only a judge that is named *and* loads is checked here. A judge profile
    that does not exist, or has no key, is already reported in the startup
    capabilities list, and running a task does not need a judge at all.

    Args:
        config: The resolved settings.
        profiles_path: Profiles file, or None for the packaged one.

    Returns:
        str | None: Why the judge may not be used, or None if it may (or none
        is configured).
    """
    if not config.judge_profile:
        return None
    try:
        planner = resolve_profile(config, path=profiles_path)
        judge = load_profile(config.judge_profile, path=profiles_path)
    except ProfileError:
        return None
    try:
        check_independence(planner, judge)
    except JudgeConfigError as error:
        return str(error)
    return None


# --- The verdict -------------------------------------------------------------------


@dataclass(frozen=True)
class JudgeVerdict:
    """What the judge decided about one run, or why it could not decide.

    Attributes:
        outcome: :data:`PASS`, :data:`FAIL`, or :data:`JUDGING_ERROR`.
        justification: The judge's one sentence. Empty for a judging error.
        category: For a failure, one of :data:`FAILURE_CATEGORIES`. None for a
            pass or a judging error.
        error: For a judging error, what went wrong. Empty otherwise.
        raw_reply: The judge's reply exactly as received, kept for a judging
            error so a person can see what the judge actually said.
        model: Which model judged.
        record: What the call cost, when a call was made and answered.
    """

    outcome: str
    justification: str = ""
    category: str | None = None
    error: str = ""
    raw_reply: str = ""
    model: str = ""
    record: CallRecord | None = None

    @property
    def passed(self) -> bool:
        """bool: True only for a pass. A judging error is not a pass."""
        return self.outcome == PASS

    def to_dict(self) -> dict[str, Any]:
        """Turn the verdict into plain data, for a results file.

        Returns:
            dict[str, Any]: Every field, with the cost record as plain data.
        """
        return {
            "outcome": self.outcome,
            "justification": self.justification,
            "category": self.category,
            "error": self.error,
            "raw_reply": self.raw_reply,
            "model": self.model,
            "record": asdict(self.record) if self.record is not None else None,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> JudgeVerdict:
        """Rebuild a verdict written by :meth:`to_dict`.

        Args:
            data: The plain data.

        Returns:
            JudgeVerdict: The verdict.
        """
        record = data.get("record")
        return cls(
            outcome=data["outcome"],
            justification=data.get("justification", ""),
            category=data.get("category"),
            error=data.get("error", ""),
            raw_reply=data.get("raw_reply", ""),
            model=data.get("model", ""),
            record=CallRecord.from_dict(record) if record else None,
        )


def verdict_schema() -> dict[str, Any]:
    """Describe the exact shape of answer the judge must give.

    Returns:
        dict[str, Any]: A JSON Schema. Sent to services that can enforce it,
        and checked by :func:`parse_verdict` either way.
    """
    return {
        "type": "object",
        "properties": {
            "passed": {"type": "boolean"},
            "justification": {"type": "string", "description": "one sentence"},
            "category": {"type": "string", "enum": [NO_FAILURE, *FAILURE_CATEGORIES]},
        },
        "required": ["passed", "justification", "category"],
        "additionalProperties": False,
    }


def parse_verdict(data: dict[str, Any], *, raw_reply: str, model: str = "") -> JudgeVerdict:
    """Turn the judge's reply into a verdict, refusing anything malformed.

    Every rule here exists to stop a broken reply becoming a pass:

    * ``passed`` must be an actual true or false. The *text* ``"false"`` is
      refused, because in Python any non-empty text counts as true.
    * ``justification`` must be a non-empty sentence.
    * ``category`` must be one of the allowed words.
    * A failure must name a failure category. ``none`` on a failure is refused:
      every failure gets a category, or the results cannot say what went wrong.
    * A pass that names a failure category is accepted as a pass, and the
      category is dropped. The verdict is the decision; a stray category
      beside a pass is noise.

    Args:
        data: The reply, already read as JSON.
        raw_reply: The reply text, kept on a judging error.
        model: Which model replied.

    Returns:
        JudgeVerdict: A pass, a fail, or a judging error saying which rule
        the reply broke.
    """

    def error(reason: str) -> JudgeVerdict:
        """Build the judging error for one broken rule."""
        return JudgeVerdict(JUDGING_ERROR, error=reason, raw_reply=raw_reply, model=model)

    passed = data.get("passed")
    if not isinstance(passed, bool):
        return error(f"'passed' must be true or false, got {passed!r}")

    justification = data.get("justification")
    if not isinstance(justification, str) or not justification.strip():
        return error("the reply has no justification")

    category = data.get("category")
    if category not in (NO_FAILURE, *FAILURE_CATEGORIES):
        return error(
            f"'category' must be one of {[NO_FAILURE, *FAILURE_CATEGORIES]}, got {category!r}"
        )

    if passed:
        return JudgeVerdict(PASS, justification=justification.strip(), model=model)
    if category == NO_FAILURE:
        return error("the run failed but the reply names no failure category")
    return JudgeVerdict(FAIL, justification=justification.strip(), category=category, model=model)


def category_schema() -> dict[str, Any]:
    """Describe the answer when only the kind of failure is asked for.

    Returns:
        dict[str, Any]: A JSON Schema with a category from the four kinds --
        not ``none``, since the run has already failed -- and one sentence.
    """
    return {
        "type": "object",
        "properties": {
            "category": {"type": "string", "enum": list(FAILURE_CATEGORIES)},
            "justification": {"type": "string", "description": "one sentence"},
        },
        "required": ["category", "justification"],
        "additionalProperties": False,
    }


def parse_category(data: dict[str, Any], *, raw_reply: str, model: str = "") -> JudgeVerdict:
    """Turn a category-only reply into a categorised fail, or a judging error.

    Args:
        data: The reply, already read as JSON.
        raw_reply: The reply text, kept on a judging error.
        model: Which model replied.

    Returns:
        JudgeVerdict: A fail with exactly one of the four categories, or a
        judging error naming what was wrong -- a missing or unknown category,
        ``none``, or no justification.
    """
    category = data.get("category")
    if category not in FAILURE_CATEGORIES:
        return JudgeVerdict(
            JUDGING_ERROR,
            error=f"'category' must be one of {list(FAILURE_CATEGORIES)}, got {category!r}",
            raw_reply=raw_reply,
            model=model,
        )
    justification = data.get("justification")
    if not isinstance(justification, str) or not justification.strip():
        return JudgeVerdict(
            JUDGING_ERROR, error="the reply has no justification", raw_reply=raw_reply, model=model
        )
    return JudgeVerdict(FAIL, justification=justification.strip(), category=category, model=model)


# --- The judge -----------------------------------------------------------------------


@functools.cache
def _instructions(path: Path) -> str:
    """Read the judge's instructions, once per process.

    Args:
        path: The prompt file.

    Returns:
        str: Its contents.
    """
    return path.read_text(encoding="utf-8")


class Judge:
    """Marks finished runs with a second model.

    Attributes:
        client: The model client the judge's calls go through.
    """

    def __init__(self, client: ModelClient, *, prompt_path: Path | None = None) -> None:
        """Set up a judge around a model client.

        Args:
            client: The client for the judge's model. Build it with
                :meth:`from_config` to get the family check.
            prompt_path: The judge's instructions. Defaults to the shipped
                ``prompts/judge.md``.
        """
        self.client = client
        self._prompt_path = prompt_path or DEFAULT_PROMPT_PATH

    @classmethod
    def from_config(
        cls,
        config: Config,
        *,
        env: Mapping[str, str] | None = None,
        profiles_path: Path | None = None,
        transport: Any | None = None,
    ) -> Judge:
        """Build the configured judge, after checking it is independent.

        Args:
            config: The resolved settings. ``judge_profile`` names the judge.
            env: Environment to read the judge's key from. Defaults to
                ``.env`` merged under the process environment.
            profiles_path: Profiles file, or None for the packaged one.
            transport: A stand-in for the model service, for tests.

        Returns:
            Judge: A judge ready to mark runs.

        Raises:
            JudgeConfigError: If no judge is configured, or it is from the
                planner's family.
            ModelClientError: If the judge's profile cannot be loaded or has
                no key.
        """
        if not config.judge_profile:
            raise JudgeConfigError(
                "no judge is configured: set judge_profile in config.yaml, or "
                "WAA_JUDGE_PROFILE in .env"
            )
        planner = resolve_profile(config, path=profiles_path)
        judge_profile = load_profile(config.judge_profile, path=profiles_path)
        check_independence(planner, judge_profile)
        # The judge gets the run's own timeouts and backoff policy by being
        # built from the same settings, with only the profile swapped.
        client = ModelClient.from_config(
            config.model_copy(
                update={"profile": config.judge_profile, "model": None, "base_url": None}
            ),
            env=env,
            profiles_path=profiles_path,
            transport=transport,
        )
        return cls(client)

    @property
    def instructions(self) -> str:
        """str: The system prompt the judge is given."""
        return _instructions(self._prompt_path)

    def judge(self, task: BenchmarkTask, run: ReplayedRun) -> JudgeVerdict:
        """Mark one finished run of one task.

        Args:
            task: The benchmark task the run attempted.
            run: The run, read back from its folder.

        Returns:
            JudgeVerdict: Pass, fail with a category, or a judging error.
            Never raises for a problem with the judge's reply or service:
            those are recorded as judging errors, so a sweep of a hundred runs
            is not stopped by one bad reply.
        """
        return self._ask(run, render_case(task, run), verdict_schema(), SCHEMA_NAME, parse_verdict)

    def categorize(self, task: BenchmarkTask, run: ReplayedRun, reason: str) -> JudgeVerdict:
        """Ask only which kind of failure a run that has already failed was.

        For a run failed by something other than the judge -- the agent ran
        out of steps, a machine check caught a wrong answer -- so that it still
        gets exactly one category. Also used when the judge's own pass/fail
        disagrees with the machine checks: the machine checks decide, and the
        judge is asked to explain the failure they found.

        Args:
            task: The task the run attempted.
            run: The run.
            reason: Why it was failed, shown to the judge as already decided.

        Returns:
            JudgeVerdict: A fail with one of the four categories, or a judging
            error. Never a pass: the answer has no field that could say so.
        """
        return self._ask(
            run,
            render_case(task, run, decided=reason),
            category_schema(),
            CATEGORY_SCHEMA_NAME,
            parse_category,
        )

    def _ask(
        self,
        run: ReplayedRun,
        case: str,
        schema: dict[str, Any],
        schema_name: str,
        parse: Any,
    ) -> JudgeVerdict:
        """Send one case to the judge and read its answer.

        Shared by :meth:`judge` and :meth:`categorize`, so that both get the
        same screenshots, temperature, error handling and cost record.

        Args:
            run: The run, whose last screenshots are attached.
            case: The case file, as text.
            schema: The shape the answer must have.
            schema_name: The schema's name in the request.
            parse: Turns the parsed answer into a verdict.

        Returns:
            JudgeVerdict: The parsed verdict, or a judging error.
        """
        model = self.client.profile.route_for(vision=True).model
        try:
            response = self.client.complete(
                [
                    system_message(self.instructions),
                    user_message(case, screenshots_png=_last_screenshots(run)),
                ],
                schema=schema,
                schema_name=schema_name,
                temperature=0.0,
            )
        except ModelResponseError as error:
            # The service answered, but not with JSON.
            return JudgeVerdict(
                JUDGING_ERROR,
                error=f"the reply was not a JSON object: {error}",
                raw_reply=error.text,
                model=model,
                record=error.record,
            )
        except ModelClientError as error:
            # The service could not be reached, refused, or timed out.
            return JudgeVerdict(
                JUDGING_ERROR, error=f"the judge could not be reached: {error}", model=model
            )

        assert response.data is not None, "a schema was asked for, so the reply was parsed"
        verdict = parse(response.data, raw_reply=response.text, model=response.record.model)
        # The cost record is attached whatever the outcome: a judging error
        # still cost a call, and a budget that missed it would run over.
        return JudgeVerdict(
            outcome=verdict.outcome,
            justification=verdict.justification,
            category=verdict.category,
            error=verdict.error,
            raw_reply=verdict.raw_reply,
            model=verdict.model,
            record=response.record,
        )


def render_case(task: BenchmarkTask, run: ReplayedRun, *, decided: str | None = None) -> str:
    """Write out what the judge is shown, as text.

    Args:
        task: The task the run attempted.
        run: The run.
        decided: For a category-only question, why the run was already
            failed. Shown at the end, with the instruction not to decide pass
            or fail again.

    Returns:
        str: The task, what success looks like, the answer, how the run
        ended, and the steps it took. The screenshots go alongside, as images.
    """
    final = final_state(run)
    expected = next(
        (v.expected_outcome for v in task.verdicts if isinstance(v, JudgeVerdictMethod)),
        None,
    )
    ended = (
        f"{run.outcome.category}" + (f" ({run.outcome.reason})" if run.outcome.reason else "")
        if run.outcome is not None
        else "interrupted before it finished"
    )
    steps = "\n".join(
        f"{step.number}. "
        f"{describe_action(step.action, step.observation) if step.action else '(no action)'}"
        f"  [on {step.observation.url}]"
        for step in run.steps
    )
    screenshot_note = (
        "The last screenshot shows the final page."
        if final.page_is_current
        else (
            "The run's last action moved to a new page that was never captured, so the "
            f"last screenshot shows the page before it. The run ended at {final.url}."
        )
    )
    return (
        f"## The task\n\n{task.instruction}\n\n"
        f"## What success looks like\n\n"
        f"{expected or '(not written down: judge from the task alone)'}\n\n"
        f"## The agent's final answer\n\n{final.answer or '(no answer given)'}\n\n"
        f"## How the run ended\n\n{ended}\n\n"
        f"## The steps it took\n\n{steps or '(none)'}\n\n"
        f"## The screenshots\n\n{screenshot_note} The numbers drawn on them are "
        "labels the agent used to refer to elements, not part of the website.\n"
        + (
            f"\n## Already decided: this run failed\n\n{decided}\n\n"
            "Do not decide pass or fail again. Answer only which kind of failure it was.\n"
            if decided
            else ""
        )
    )


def _last_screenshots(run: ReplayedRun) -> tuple[bytes, ...]:
    """Read the run's last few screenshots, oldest first.

    Args:
        run: The run.

    Returns:
        tuple[bytes, ...]: Up to :data:`SCREENSHOTS_SHOWN` PNG images, the
        final one last. Files that have gone missing are skipped rather than
        stopping the judging: the text still says what happened.
    """
    images: list[bytes] = []
    for path in run.screenshots[-SCREENSHOTS_SHOWN:]:
        try:
            images.append(path.read_bytes())
        except OSError:
            continue
    return tuple(images)
