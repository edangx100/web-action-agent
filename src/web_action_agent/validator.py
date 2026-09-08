"""The gate between what the model proposes and what the browser does.

Everything upstream of here is a suggestion. The planner reads a page and says
"click 12"; nothing about that sentence guarantees there is a twelfth element,
that the run is allowed on the site it wants to open, or that the skill it
wants to replay still exists. This module is where a suggestion becomes an
approved action, and it is the only place that transition happens -- an
executor that took actions from anywhere else would make the whole arrangement
decorative.

**Five checks, in one pass.**

1. *Schema.* The proposal is one of the seven actions, spelled correctly. This
   repeats what the planner already did, and repeats it on purpose: proposals
   also arrive from a stored skill, from a hand-written retry, from a trace
   being replayed. Only the ones that came through the planner have been parsed
   before, so the gate parses everything.
2. *The element exists.* Element numbers are handed out per observation and
   mean nothing outside it. A number from the previous page, or one the model
   invented, is caught here rather than becoming a click on whatever now
   happens to be twelfth.
3. *The action type is permitted.* A run can be configured with a narrower
   vocabulary than the full seven -- the evaluation's cold pass runs with no
   skill library, so ``use_skill`` is simply not on the table.
4. *Navigation is allowed, by us and by them.* Where the run may go was decided
   before its first step, and this is where that decision is applied. It is
   also where the *site's* decision is applied: an address its ``robots.txt``
   asks automated visitors to leave alone is refused here, before the browser
   is asked, because fetching a page and then discarding it is exactly what
   that file exists to prevent.
5. *The skill is usable.* Named, active, belongs to this site, and given the
   parameters it declares.

**A rejection is feedback, not an error.** Nothing here raises when a proposal
is bad. Each check produces a :class:`Rejection` carrying a stable code for
counting and a sentence written to be handed straight back to the model, and
the step is tried again. That is the whole recovery mechanism: the model is not
told "invalid action", it is told *which element number does not exist*, which
is a thing it can act on.

**The retry cycle is bounded, and running out is an outcome.**
:func:`plan_until_valid` runs the ask-check-complain cycle up to the run's
configured budget. When the budget is gone the step does not silently do
nothing -- it produces a ``done(failed, ...)`` naming validation exhaustion as
the reason, so a run that could not get a usable action out of the model ends
with a verdict rather than a shrug.

**Two things are deliberately not checked.** Whether the element can actually
be clicked -- whether it moved, went stale, or is covered by a banner -- is not
knowable from an observation, which is a snapshot taken some moments ago. That
is the executor's report to make, in categories, after trying. And whether the
action is a *good idea* is nobody's check: a validator that second-guessed the
plan would be a second planner with no view of the task.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from .actions import ACTION_NAMES, Action, ActionError, Done, Navigate, UseSkill, parse_action
from .allowlist import Allowlist, AllowlistError, derive_domain, host_of, normalize_host
from .model_client import CallRecord
from .observation import Observation
from .planner import PlannerDecision
from .politeness import ROBOTS_DISALLOWED, RobotsPolicy

#: The proposal was not one of the seven actions, correctly spelled. Also the
#: code a planner reply that could not be parsed at all is reported under: a
#: model that answered in prose failed this same check, one stage earlier.
INVALID_SCHEMA = "invalid_schema"

#: The proposal named an element number the current observation does not have.
UNKNOWN_ELEMENT = "unknown_element"

#: The action type is not one this run allows.
FORBIDDEN_ACTION = "forbidden_action"

#: The URL may not be navigated to -- off the allowlist, not a web URL, or a
#: path the site's own robots.txt excludes. Which of those it was is the
#: rejection's ``cause``.
BLOCKED_NAVIGATION = "blocked_navigation"

#: The named skill cannot be replayed here: unknown, retired, from another
#: site, or called with the wrong parameters.
UNUSABLE_SKILL = "unusable_skill"

#: The five codes, in the order the checks run. Tests assert the list has not
#: grown a sixth member quietly, and a run's summary counts rejections by it.
REJECTION_CODES = (
    INVALID_SCHEMA,
    FORBIDDEN_ACTION,
    UNKNOWN_ELEMENT,
    BLOCKED_NAVIGATION,
    UNUSABLE_SKILL,
)

#: Finer reasons behind an :data:`UNUSABLE_SKILL` rejection. The rejection code
#: stays one of the five, because there are five checks; these say which way
#: the fifth one failed, for whoever is counting later.
SKILL_NOT_FOUND = "skill_not_found"
SKILL_RETIRED = "skill_retired"
SKILL_WRONG_SITE = "skill_wrong_site"
SKILL_BAD_PARAMS = "skill_bad_params"
NO_SKILL_LIBRARY = "no_skill_library"

#: Reason prefix on the ``done(failed, ...)`` a step produces when its retry
#: budget runs out. A prefix rather than a sentence so that a hundred runs can
#: be sorted into "the model could not produce a usable action" without anyone
#: reading the sentences.
VALIDATION_EXHAUSTED = "validation_exhausted"


class SkillEntry(Protocol):
    """What the gate needs to know about a skill, and nothing more.

    A structural type rather than an import, because the skill library is a
    separate component with its own richer record -- steps, selectors, usage
    statistics, none of which are the gate's business. Anything carrying these
    five attributes can be checked, which keeps the two components from having
    to be built or changed together.

    Attributes:
        name: The skill's name, as ``use_skill`` refers to it.
        domain: The registrable domain the skill was learnt on. A skill is a
            recorded route through one particular site's pages, so replaying it
            somewhere else is not a long shot, it is meaningless.
        active: False for a skill retirement has taken out of service. Retired
            skills are kept rather than deleted -- their statistics are the
            evidence for why they were retired -- so being in the library is
            not the same as being usable.
        required_params: Parameter names the skill cannot replay without.
        optional_params: Parameter names it accepts but does not need.
    """

    name: str
    domain: str
    active: bool
    required_params: tuple[str, ...]
    optional_params: tuple[str, ...]


class SkillLibrary(Protocol):
    """The one question the gate asks the skill library."""

    def find(self, name: str) -> SkillEntry | None:
        """Look a skill up by name.

        Args:
            name: The name the planner used.

        Returns:
            SkillEntry | None: The skill, active or retired, or None if the
            library has never heard of it. Retired skills are returned rather
            than hidden so the gate can tell the model "that skill was retired"
            instead of "no such skill", which are different problems.
        """
        ...


@dataclass(frozen=True)
class Rejection:
    """Why one proposed action may not be carried out.

    Attributes:
        code: One of :data:`REJECTION_CODES`. Stable, and the thing to branch
            on or count; the wording of ``detail`` is free to change.
        detail: What was wrong, as a sentence addressed to the model. This is
            the text quoted back on the retry, so it names the element number
            or the missing parameter rather than describing the problem in
            general terms.
        cause: A finer reason within the same check, where the check has one --
            the allowlist's own refusal code, or which way a skill was
            unusable. None where the code already says everything.
    """

    code: str
    detail: str
    cause: str | None = None

    def summary(self) -> str:
        """Render this rejection as the single line the step log records.

        Returns:
            str: For example ``rejected (unknown_element): this page has no
            element 12; ...``, with the finer cause appended where there is
            one.
        """
        code = self.code if self.cause is None else f"{self.code}/{self.cause}"
        return f"rejected ({code}): {self.detail}"

    def to_dict(self) -> dict[str, Any]:
        """Render this rejection as plain data, for the run's trace.

        Returns:
            dict[str, Any]: Every field, all JSON-safe already.
        """
        return {"code": self.code, "detail": self.detail, "cause": self.cause}


@dataclass(frozen=True)
class Verdict:
    """The gate's answer about one proposed action.

    Exactly one of the two fields is set. There is no verdict carrying both an
    action and a complaint about it, because an action that is only partly
    approved is exactly the thing this module exists to make impossible.

    Attributes:
        action: The approved action, typed, or None if it was refused. This is
            the value an executor may act on -- and, when the proposal arrived
            as plain data, the parsed form of it.
        rejection: Why it was refused, or None if it was approved.
    """

    action: Action | None
    rejection: Rejection | None

    @property
    def ok(self) -> bool:
        """bool: Whether this proposal may be carried out."""
        return self.action is not None

    def summary(self) -> str:
        """Render this verdict as the single line the step log records.

        Returns:
            str: ``approved: click 4``, or the rejection's own summary.
        """
        if self.action is not None:
            return f"approved: {self.action.describe()}"
        return self.rejection.summary() if self.rejection else "rejected: no reason recorded"

    def to_dict(self) -> dict[str, Any]:
        """Render this verdict as plain data, for the run's trace.

        Returns:
            dict[str, Any]: The action or the rejection, whichever there is.
        """
        return {
            "action": self.action.to_dict() if self.action else None,
            "rejection": self.rejection.to_dict() if self.rejection else None,
        }


class Validator:
    """The five checks, held together with the run's policy.

    Built once per run, from things settled before the first step: where the
    run may navigate, which action types it may use, and which skills it has.
    None of them change while the run is going, so a trace can say what the run
    was allowed to do without listing it again at every step.
    """

    def __init__(
        self,
        allowlist: Allowlist,
        *,
        skills: SkillLibrary | None = None,
        permitted: Iterable[str] | None = None,
        robots: RobotsPolicy | None = None,
    ) -> None:
        """Build the gate for one run.

        Args:
            allowlist: Where this run may navigate. Required: a run with no
                allowlist is a run with no navigation policy, and defaulting to
                "anywhere" is not a default anyone should be able to reach by
                forgetting an argument.
            skills: The run's skill library, or None for a run without one --
                a cold evaluation pass, or any run before a library exists.
            robots: What each site's ``robots.txt`` allows, or None for a run
                configured not to read them. Consulted only for ``navigate``,
                and only after the allowlist has passed -- so a run never
                requests a file from a site it was never going to visit.
            permitted: Action names this run allows. Defaults to all seven,
                minus ``use_skill`` when there is no library to replay from.

        Raises:
            ValueError: If ``permitted`` names something that is not an action,
                or leaves out ``done``. Both are setup mistakes, and both are
                worth stopping for: a misspelt name would silently forbid
                nothing, and a run that cannot say ``done`` cannot report how
                it went.
        """
        self._allowlist = allowlist
        self._skills = skills
        self._robots = robots

        if permitted is None:
            # No library means no skill can be usable, so the honest answer to
            # a `use_skill` is "not in this run" rather than "no such skill" --
            # which is what the type check gives, one check earlier.
            allowed = [name for name in ACTION_NAMES if skills is not None or name != "use_skill"]
        else:
            allowed = list(dict.fromkeys(permitted))

        unknown = [name for name in allowed if name not in ACTION_NAMES]
        if unknown:
            raise ValueError(
                f"not action names: {', '.join(sorted(unknown))}; "
                f"the actions are {', '.join(ACTION_NAMES)}"
            )
        if "done" not in allowed:
            raise ValueError("'done' must be permitted: a run has to be able to say how it ended")

        self._permitted = tuple(allowed)

    @property
    def permitted(self) -> tuple[str, ...]:
        """tuple[str, ...]: The action names this run allows, in vocabulary order."""
        return self._permitted

    def check(self, proposal: Action | Mapping[str, Any], observation: Observation) -> Verdict:
        """Run all five checks against one proposed action.

        The single way in. It accepts plain data as readily as a typed action
        so that a caller holding a stored skill step cannot get to the browser
        by skipping the parse -- handing in a dict runs one more check, not one
        fewer.

        Args:
            proposal: The action, either already typed or as the plain data it
                was stored or decoded as.
            observation: The page the action will be applied to. This must be
                the *current* observation, not the one the planner was shown,
                on the rare step where those differ -- the question being asked
                is whether the action makes sense against the page as it is now.

        Returns:
            Verdict: The approved action, or the reason it was refused. Never
            raises for a bad proposal; a bad proposal is an ordinary event.
        """
        if isinstance(proposal, Mapping):
            try:
                action: Action = parse_action(proposal)
            except ActionError as error:
                return _refuse(INVALID_SCHEMA, "; ".join((str(error), *error.detail)))
        else:
            action = proposal

        # Type before element: if the run has no skills, complaining about the
        # parameters of a `use_skill` it was never going to run is noise.
        if action.action not in self._permitted:
            return _refuse(
                FORBIDDEN_ACTION,
                f"{action.action!r} is not available in this run; "
                f"use one of {', '.join(self._permitted)}",
            )

        # Read the field rather than listing the action types that have one, so
        # that an action added later is checked without anyone remembering to
        # add it here.
        element_id = getattr(action, "element_id", None)
        if element_id is not None and observation.element_by_id(element_id) is None:
            return _refuse(
                UNKNOWN_ELEMENT,
                f"there is no element {element_id} on this page; "
                f"the page offers {_element_range(observation)}. "
                "Use a number from the list above",
            )

        if isinstance(action, Navigate):
            refusal = self._allowlist.check(action.url)
            if refusal is not None:
                return _refuse(BLOCKED_NAVIGATION, refusal.message, cause=refusal.code)
            # Second, and only if the address survived the first: reading a
            # site's robots.txt means asking that site for a file, which is not
            # worth doing for an address this run was never going to visit.
            if self._robots is not None:
                complaint = self._robots.refusal(action.url)
                if complaint is not None:
                    return _refuse(BLOCKED_NAVIGATION, complaint, cause=ROBOTS_DISALLOWED)

        if isinstance(action, UseSkill):
            rejection = self._check_skill(action, observation)
            if rejection is not None:
                return Verdict(action=None, rejection=rejection)

        return Verdict(action=action, rejection=None)

    def _check_skill(self, action: UseSkill, observation: Observation) -> Rejection | None:
        """Check the fifth condition: that this skill can be replayed here.

        Args:
            action: The proposed ``use_skill``.
            observation: The current page, whose site the skill must match.

        Returns:
            Rejection | None: The reason the skill is unusable, or None if it
            is fine.
        """
        if self._skills is None:
            # Only reachable when a caller passed `permitted` by hand and put
            # use_skill in it without supplying a library. Refused rather than
            # crashed, because a setup mistake should not end a run mid-step.
            return _rejection(
                UNUSABLE_SKILL,
                "this run has no skill library; act on the page directly",
                cause=NO_SKILL_LIBRARY,
            )

        skill = self._skills.find(action.skill)
        if skill is None:
            return _rejection(
                UNUSABLE_SKILL,
                f"there is no skill named {action.skill!r}; act on the page directly",
                cause=SKILL_NOT_FOUND,
            )
        if not skill.active:
            return _rejection(
                UNUSABLE_SKILL,
                f"the skill {skill.name!r} was retired and can no longer be replayed; "
                "act on the page directly",
                cause=SKILL_RETIRED,
            )

        site = _site_of(observation.url)
        if site is None or site != normalize_host(skill.domain):
            here = site or observation.url
            return _rejection(
                UNUSABLE_SKILL,
                f"the skill {skill.name!r} belongs to {skill.domain}, not to {here}; "
                "act on the page directly",
                cause=SKILL_WRONG_SITE,
            )

        return _check_params(skill, action.params)


def _check_params(skill: SkillEntry, given: Mapping[str, str]) -> Rejection | None:
    """Check a skill call's parameters against the ones the skill declares.

    Both directions matter. A missing required parameter leaves a replay with a
    blank to fill and no value for it; an unexpected one means the model has
    misunderstood what the skill does, and quietly ignoring it would let that
    misunderstanding through.

    Args:
        skill: The skill being called.
        given: The parameters the action supplied.

    Returns:
        Rejection | None: The complaint, naming the parameters by name, or None
        if the call matches.
    """
    required = frozenset(skill.required_params)
    known = required | frozenset(skill.optional_params)
    missing = sorted(required - frozenset(given))
    unexpected = sorted(frozenset(given) - known)
    if not missing and not unexpected:
        return None

    problems = []
    if missing:
        problems.append(f"missing {', '.join(missing)}")
    if unexpected:
        problems.append(f"unexpected {', '.join(unexpected)}")
    accepted = ", ".join(sorted(known)) or "no parameters"
    return _rejection(
        UNUSABLE_SKILL,
        f"the skill {skill.name!r} was called with the wrong parameters "
        f"({'; '.join(problems)}); it takes {accepted}",
        cause=SKILL_BAD_PARAMS,
    )


def _site_of(url: str) -> str | None:
    """Reduce a page URL to the registrable domain a skill is compared against.

    Args:
        url: The current observation's URL.

    Returns:
        str | None: The registrable domain, or None for a URL with no usable
        host -- ``about:blank`` on a fresh tab, for instance. None is not an
        error here; it simply means no skill can match this page.
    """
    try:
        return derive_domain(host_of(url))
    except AllowlistError:
        return None


def _element_range(observation: Observation) -> str:
    """Say which element numbers the page does offer.

    Args:
        observation: The current page.

    Returns:
        str: ``elements 1 to 8``, or a sentence for a page with none. Written
        out rather than left implicit because "pick a valid number" is not
        actionable and "pick one between 1 and 8" is.
    """
    if not observation.elements:
        return "no interactive elements at all"
    return f"elements 1 to {len(observation.elements)}"


def _refuse(code: str, detail: str, *, cause: str | None = None) -> Verdict:
    """Build a rejecting verdict.

    Args:
        code: One of :data:`REJECTION_CODES`.
        detail: The sentence the model is shown.
        cause: The finer reason, where there is one.

    Returns:
        Verdict: Carrying the rejection and no action.
    """
    return Verdict(action=None, rejection=_rejection(code, detail, cause=cause))


def _rejection(code: str, detail: str, *, cause: str | None = None) -> Rejection:
    """Build a rejection.

    Args:
        code: One of :data:`REJECTION_CODES`.
        detail: The sentence the model is shown.
        cause: The finer reason, where there is one.

    Returns:
        Rejection: The complaint.
    """
    return Rejection(code=code, detail=detail, cause=cause)


@dataclass(frozen=True)
class StepAttempt:
    """One pass through "ask the planner, check the answer".

    Attributes:
        decision: What the planner returned, including what the call cost.
        rejection: Why the answer was refused, or None if it was approved. A
            planner failure -- prose instead of JSON, an invented action type --
            appears here as an :data:`INVALID_SCHEMA` rejection, because that is
            what it is: the same first check, failed one stage earlier.
    """

    decision: PlannerDecision
    rejection: Rejection | None

    @property
    def ok(self) -> bool:
        """bool: Whether this attempt produced an approved action."""
        return self.rejection is None and self.decision.action is not None

    def summary(self) -> str:
        """Render this attempt as the single line the step log records.

        Returns:
            str: What was proposed and what became of it, e.g.
            ``click 12 -> rejected (unknown_element): there is no element 12
            on this page; ...``.
        """
        proposed = self.decision.summary()
        if self.rejection is None:
            return f"{proposed} -> approved"
        return f"{proposed} -> {self.rejection.summary()}"

    def to_dict(self) -> dict[str, Any]:
        """Render this attempt as plain data, for the run's trace.

        Returns:
            dict[str, Any]: The whole decision, cost record included, and the
            rejection if there was one.
        """
        return {
            "decision": self.decision.to_dict(),
            "rejection": self.rejection.to_dict() if self.rejection else None,
        }


@dataclass(frozen=True)
class StepOutcome:
    """What one step of the loop produced, attempts and all.

    Attributes:
        action: The approved action, or None if no attempt produced one.
        giving_up: The ``done(failed, ...)`` built when the retry budget ran
            out, and None otherwise. Built here rather than left to the caller
            so that there is no path on which a step simply stops without
            saying why.
        attempts: Every attempt made, in order, including the successful one.
            Kept in full because the interesting thing about a recovered step
            is what it recovered *from*, and a trace that recorded only the
            action that worked would lose exactly that.
    """

    action: Action | None
    giving_up: Done | None
    attempts: tuple[StepAttempt, ...]

    @property
    def ok(self) -> bool:
        """bool: Whether this step has an action to execute."""
        return self.action is not None

    @property
    def records(self) -> tuple[CallRecord, ...]:
        """tuple[CallRecord, ...]: What every attempt cost, rejected ones included.

        A rejected attempt was still a model call, and still billed. A run that
        counted only the calls it liked would under-report its own cost.
        """
        return tuple(
            attempt.decision.record for attempt in self.attempts if attempt.decision.record
        )

    def summary(self) -> str:
        """Render this step as the single line the run log records.

        Returns:
            str: The approved action and how many attempts it took, or the
            reason the step gave up.
        """
        if self.action is not None:
            retries = len(self.attempts) - 1
            if retries == 0:
                return self.action.describe()
            plural = "retry" if retries == 1 else "retries"
            return f"{self.action.describe()} (after {retries} {plural})"
        return self.giving_up.describe() if self.giving_up else "no action and no reason recorded"

    def to_dict(self) -> dict[str, Any]:
        """Render this step as plain data, for the run's trace.

        Returns:
            dict[str, Any]: The action, the give-up action if there was one,
            and every attempt.
        """
        return {
            "action": self.action.to_dict() if self.action else None,
            "giving_up": self.giving_up.to_dict() if self.giving_up else None,
            "attempts": [attempt.to_dict() for attempt in self.attempts],
        }


def plan_until_valid(
    *,
    attempt: Callable[[tuple[str, ...]], PlannerDecision],
    validator: Validator,
    observation: Observation,
    retry_budget: int,
) -> StepOutcome:
    """Ask for an action until one passes the gate, or the budget runs out.

    The cycle is: ask, check, and on a refusal ask again with the complaint
    attached. Complaints accumulate rather than replace, so a model that fixes
    the element number by proposing an off-allowlist URL still sees that the
    old number was wrong and does not go back to it.

    The planner is taken as a callable rather than as an object, because the
    only thing this cycle needs from it is "given these complaints, produce a
    decision". The task, the history and the screenshot are the caller's to
    hold, and repeating that whole argument list here would mean changing this
    function every time the prompt gains a section.

    Args:
        attempt: Called once per try with the complaints so far, oldest first,
            and empty on the first try. Returns the planner's decision.
        validator: The gate to check each proposal against.
        observation: The page every proposal is checked against.
        retry_budget: Retries allowed *after* the first try, so 0 means one
            try and 2 means up to three. The run's configuration owns this
            number; it is required here rather than defaulted so that no caller
            can quietly run on a different budget from the one configured.

    Returns:
        StepOutcome: The approved action and the attempts it took, or -- when
        every attempt was refused -- a ``done(failed, ...)`` naming validation
        exhaustion, with the whole sequence of refusals kept.

    Raises:
        ModelClientError: If a call itself fails -- network, auth, rate limit.
            A broken endpoint is not a bad step and is not retried here; that
            is the model client's own policy, one layer down.
    """
    attempts: list[StepAttempt] = []
    complaints: list[str] = []

    # A negative budget is treated as none rather than as "no tries at all",
    # which would make a misconfigured run look like a model failure.
    for _ in range(max(0, retry_budget) + 1):
        decision = attempt(tuple(complaints))
        rejection = _reject_decision(decision, validator, observation)
        attempts.append(StepAttempt(decision=decision, rejection=rejection))
        if rejection is None:
            return StepOutcome(action=decision.action, giving_up=None, attempts=tuple(attempts))
        complaints.append(rejection.detail)

    return StepOutcome(
        action=None,
        giving_up=_give_up(tuple(attempts)),
        attempts=tuple(attempts),
    )


def _reject_decision(
    decision: PlannerDecision,
    validator: Validator,
    observation: Observation,
) -> Rejection | None:
    """Turn one planner decision into a rejection, or None if it passes.

    This is where the two kinds of bad step become one kind. A reply that could
    not be parsed and an action that names a nonexistent element are different
    events, but the response to both is identical -- tell the model what was
    wrong and ask again -- so the retry cycle sees a single shape.

    Args:
        decision: What the planner returned.
        validator: The gate.
        observation: The page to check against.

    Returns:
        Rejection | None: Why this attempt cannot be used, or None if it can.
    """
    if decision.action is None:
        failure = decision.failure
        return _rejection(
            INVALID_SCHEMA,
            failure.detail if failure else "the reply could not be read as an action",
            cause=failure.code if failure else None,
        )
    return validator.check(decision.action, observation).rejection


def _give_up(attempts: Sequence[StepAttempt]) -> Done:
    """Build the ``done(failed, ...)`` for a step whose retries ran out.

    Args:
        attempts: Every attempt made, each carrying its rejection.

    Returns:
        Done: A terminal action whose reason starts with
        :data:`VALIDATION_EXHAUSTED` and then lists what was wrong each time.
        The list is there because "it failed three times" and "it failed the
        same way three times" call for different fixes.
    """
    codes = ", ".join(
        attempt.rejection.code for attempt in attempts if attempt.rejection is not None
    )
    tried = f"{len(attempts)} attempt{'' if len(attempts) == 1 else 's'} rejected"
    return Done(
        action="done",
        status="failed",
        reason=f"{VALIDATION_EXHAUSTED}: {tried} ({codes})" if codes else VALIDATION_EXHAUSTED,
    )
