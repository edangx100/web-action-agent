"""The gate: nothing reaches the browser that has not been checked five ways.

Two properties carry this file. The first is that each of the five checks has
its own error code, because a rejection is not just a "no" -- it is the text
the model is shown before it tries again, and a run whose element-number
mistakes and whose allowlist refusals arrive under one code is a run nobody can
diagnose afterwards.

The second is that a rejected proposal yields *nothing*. Not a partial action,
not the action with the bad field blanked out. Every rejection test below
asserts the verdict carries no action at all, because the executor takes what
the verdict hands it, and the only safe thing to hand it on a refusal is
nothing.

The retry cycle gets the same treatment from the other end: it has to recover
when the model can be told what was wrong, and it has to stop -- with a stated
reason -- when it cannot. A step that ran out of retries and quietly did
nothing would be the one failure mode a trace could not explain.

Nothing here reaches a network or a model. The planner is a stub returning
scripted decisions, and the one test that opens a browser does so against the
local fixture site.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest
from playwright.sync_api import Page

from web_action_agent.actions import ACTION_NAMES, Click, Done, parse_action
from web_action_agent.allowlist import Allowlist
from web_action_agent.model_client import CallRecord
from web_action_agent.observation import Observation
from web_action_agent.observer import Box, Element, extract_elements
from web_action_agent.planner import (
    FAILURE_NOT_AN_ACTION,
    FAILURE_NOT_JSON,
    PlannerDecision,
    PlannerFailure,
)
from web_action_agent.trace import new_id
from web_action_agent.validator import (
    BLOCKED_NAVIGATION,
    FORBIDDEN_ACTION,
    INVALID_SCHEMA,
    NO_SKILL_LIBRARY,
    REJECTION_CODES,
    SKILL_BAD_PARAMS,
    SKILL_NOT_FOUND,
    SKILL_RETIRED,
    SKILL_WRONG_SITE,
    UNKNOWN_ELEMENT,
    UNUSABLE_SKILL,
    VALIDATION_EXHAUSTED,
    StepOutcome,
    Validator,
    plan_until_valid,
)

#: The site every observation below is on, and the only one the allowlist
#: admits. Short enough to read inside an assertion.
SITE = "example.com"

#: A URL on that site, used wherever the page's own address is beside the point.
PAGE_URL = f"https://www.{SITE}/search"

#: A well-formed proposal for each of the seven actions. Written out by hand
#: rather than generated from the models, so that a model whose fields changed
#: shape shows up as a failure here instead of being followed automatically.
VALID_PROPOSALS: dict[str, dict[str, Any]] = {
    "click": {"action": "click", "element_id": 1},
    "fill": {"action": "fill", "element_id": 2, "text": "rust"},
    "press": {"action": "press", "key": "Enter"},
    "scroll": {"action": "scroll", "direction": "down"},
    "navigate": {"action": "navigate", "url": f"https://{SITE}/help"},
    "use_skill": {"action": "use_skill", "skill": "search", "params": {"query": "rust"}},
    "done": {"action": "done", "status": "success", "answer": "1,024"},
}


@dataclass(frozen=True)
class StubSkill:
    """A skill library entry, cut down to what the gate looks at.

    Attributes:
        name: The skill's name.
        domain: The registrable domain it was learnt on.
        active: False once retirement has taken it out of service.
        required_params: Parameters it cannot replay without.
        optional_params: Parameters it accepts but does not need.
    """

    name: str = "search"
    domain: str = SITE
    active: bool = True
    required_params: tuple[str, ...] = ("query",)
    optional_params: tuple[str, ...] = ("sort",)


class StubLibrary:
    """A skill library holding a handful of skills in a dict."""

    def __init__(self, *skills: StubSkill) -> None:
        """Set up a library.

        Args:
            *skills: The skills it holds. One ordinary active skill by default.
        """
        self._skills = {skill.name: skill for skill in (skills or (StubSkill(),))}

    def find(self, name: str) -> StubSkill | None:
        """Look a skill up by name.

        Args:
            name: The name the planner used.

        Returns:
            StubSkill | None: The skill, or None if there is none by that name.
        """
        return self._skills.get(name)


def an_observation(*, elements: int = 3, url: str = PAGE_URL) -> Observation:
    """Build a page to check proposals against.

    Args:
        elements: How many interactive elements it offers, numbered from 1.
        url: The page's address, which decides which site a skill must match.

    Returns:
        Observation: A page with plausible contents.
    """
    box = Box(x=0, y=0, width=80, height=18)
    return Observation(
        url=url,
        title="Search",
        elements=tuple(
            Element(id=number, tag="a", role="link", name=f"Link {number}", text="", box=box)
            for number in range(1, elements + 1)
        ),
        text="Three results, priced from 4.50.",
    )


def a_validator(**kwargs: Any) -> Validator:
    """Build a gate for a run on :data:`SITE`.

    Args:
        **kwargs: Passed to :class:`Validator`. Without ``skills`` the run has
            no library, which is the cold-start case.

    Returns:
        Validator: The gate.
    """
    return Validator(Allowlist.derive(PAGE_URL), **kwargs)


def a_record() -> CallRecord:
    """Build a cost record, as every model call produces one.

    Returns:
        CallRecord: A plausible record. The numbers only have to be recognisable
        when a test asserts a rejected attempt was still counted.
    """
    return CallRecord(
        call_id=new_id("call"),
        started_at="2026-09-17T08:00:00.000Z",
        ended_at="2026-09-17T08:00:00.400Z",
        profile="openrouter",
        model="a-model",
        vision=False,
        prompt_tokens=900,
        completion_tokens=20,
        latency_s=0.4,
        finish_reason="stop",
    )


def a_decision(proposal: dict[str, Any]) -> PlannerDecision:
    """Build a planner decision carrying one well-formed action.

    Args:
        proposal: The action, as plain data.

    Returns:
        PlannerDecision: A successful decision, with a cost record attached.
    """
    return PlannerDecision(
        action=parse_action(proposal),
        failure=None,
        text=str(proposal),
        record=a_record(),
    )


def a_failed_decision(code: str = FAILURE_NOT_JSON, detail: str = "not JSON") -> PlannerDecision:
    """Build a planner decision that produced no action at all.

    Args:
        code: The planner's own failure code.
        detail: The sentence it would quote back to the model.

    Returns:
        PlannerDecision: A failed decision, still carrying its cost record --
        the call happened, so it was billed.
    """
    return PlannerDecision(
        action=None,
        failure=PlannerFailure(code=code, detail=detail, text="I think I should..."),
        text="I think I should...",
        record=a_record(),
    )


class StubPlanner:
    """A planner that returns scripted decisions and remembers what it was told.

    Attributes:
        complaints: The feedback passed to each call, in order. Empty tuple on
            a first attempt, which is what makes "the retry actually carried
            the complaint" checkable.
    """

    def __init__(self, *decisions: PlannerDecision) -> None:
        """Set up a planner answering with the given decisions in turn.

        Args:
            *decisions: One per call. The last repeats if asked again.
        """
        self._decisions = list(decisions)
        self.complaints: list[tuple[str, ...]] = []

    def __call__(self, complaints: tuple[str, ...]) -> PlannerDecision:
        """Return the next scripted decision.

        Args:
            complaints: What earlier attempts were told, oldest first.

        Returns:
            PlannerDecision: The next decision.
        """
        self.complaints.append(complaints)
        return self._decisions[min(len(self.complaints) - 1, len(self._decisions) - 1)]


def run_step(
    *decisions: PlannerDecision,
    validator: Validator | None = None,
    observation: Observation | None = None,
    retry_budget: int = 2,
) -> tuple[StepOutcome, StubPlanner]:
    """Run one bounded retry cycle against scripted planner decisions.

    Args:
        *decisions: What the planner answers, one per attempt.
        validator: The gate. A default cold-start one if not given.
        observation: The page. A three-element one if not given.
        retry_budget: Retries allowed after the first try.

    Returns:
        tuple[StepOutcome, StubPlanner]: The outcome, and the stub so a test can
        inspect what each attempt was told.
    """
    planner = StubPlanner(*decisions)
    outcome = plan_until_valid(
        attempt=planner,
        validator=validator or a_validator(),
        observation=observation or an_observation(),
        retry_budget=retry_budget,
    )
    return outcome, planner


# --- The five checks, five codes --------------------------------------------


@pytest.mark.unit
def test_there_are_exactly_five_codes_and_they_are_distinct() -> None:
    """Five checks, five codes. A sixth means a check was added silently."""
    assert len(REJECTION_CODES) == 5
    assert len(set(REJECTION_CODES)) == 5


@pytest.mark.unit
@pytest.mark.parametrize(
    ("proposal", "code"),
    [
        ({"action": "teleport", "to": "the moon"}, INVALID_SCHEMA),
        ({"action": "click", "element_id": 99}, UNKNOWN_ELEMENT),
        ({"action": "use_skill", "skill": "search"}, FORBIDDEN_ACTION),
        ({"action": "navigate", "url": "https://elsewhere.example/"}, BLOCKED_NAVIGATION),
        ({"action": "click"}, INVALID_SCHEMA),
    ],
    ids=["invented action", "no such element", "not in this run", "off the allowlist", "no field"],
)
def test_each_rejection_path_reports_its_own_code(proposal: dict[str, Any], code: str) -> None:
    """The code says which check refused, so failures can be counted by kind."""
    verdict = a_validator().check(proposal, an_observation())

    assert verdict.rejection is not None
    assert verdict.rejection.code == code


@pytest.mark.unit
def test_the_fifth_check_has_its_own_code_too() -> None:
    """Separated from the table above because it needs a library to reach."""
    validator = a_validator(skills=StubLibrary())

    verdict = validator.check({"action": "use_skill", "skill": "nope"}, an_observation())

    assert verdict.rejection is not None
    assert verdict.rejection.code == UNUSABLE_SKILL


@pytest.mark.unit
@pytest.mark.parametrize(
    "proposal",
    [
        {"action": "teleport"},
        {"action": "click", "element_id": 99},
        {"action": "use_skill", "skill": "search"},
        {"action": "navigate", "url": "https://elsewhere.example/"},
    ],
    ids=["invented action", "no such element", "not in this run", "off the allowlist"],
)
def test_a_rejection_hands_back_no_action_at_all(proposal: dict[str, Any]) -> None:
    """The executor acts on what the verdict carries, so a refusal carries nothing."""
    verdict = a_validator().check(proposal, an_observation())

    assert verdict.action is None
    assert verdict.ok is False


@pytest.mark.unit
def test_every_code_a_rejection_reports_is_one_of_the_five() -> None:
    """A code invented at a call site would defeat counting by code."""
    proposals = [
        {"action": "teleport"},
        {"action": "click", "element_id": 99},
        {"action": "use_skill", "skill": "search"},
        {"action": "navigate", "url": "https://elsewhere.example/"},
        {"action": "navigate", "url": "javascript:alert(1)"},
    ]

    for proposal in proposals:
        rejection = a_validator().check(proposal, an_observation()).rejection
        assert rejection is not None
        assert rejection.code in REJECTION_CODES


# --- What a rejection says --------------------------------------------------


@pytest.mark.unit
def test_the_element_complaint_names_the_number_and_the_range() -> None:
    """ "Pick a valid number" is not actionable; "pick one from 1 to 3" is."""
    verdict = a_validator().check({"action": "click", "element_id": 99}, an_observation())

    assert verdict.rejection is not None
    assert "99" in verdict.rejection.detail
    assert "1 to 3" in verdict.rejection.detail


@pytest.mark.unit
def test_a_page_with_no_elements_says_so_rather_than_naming_an_empty_range() -> None:
    """A range of "1 to 0" would read as a bug, and invite a retry that cannot work."""
    observation = an_observation(elements=0)

    verdict = a_validator().check({"action": "click", "element_id": 1}, observation)

    assert verdict.rejection is not None
    assert "no interactive elements" in verdict.rejection.detail


@pytest.mark.unit
def test_the_forbidden_action_complaint_lists_what_is_available() -> None:
    """A model told only "no" has nothing to change on the retry."""
    verdict = a_validator().check({"action": "use_skill", "skill": "search"}, an_observation())

    assert verdict.rejection is not None
    assert "use_skill" in verdict.rejection.detail
    for name in ("click", "fill", "press", "scroll", "navigate", "done"):
        assert name in verdict.rejection.detail


@pytest.mark.unit
def test_every_rejection_carries_a_sentence_worth_showing_a_model() -> None:
    """An empty detail would render as a blank bullet under "your answer was rejected"."""
    proposals = [
        {"action": "teleport"},
        {"action": "click", "element_id": 99},
        {"action": "use_skill", "skill": "search"},
        {"action": "navigate", "url": "https://elsewhere.example/"},
    ]

    for proposal in proposals:
        rejection = a_validator().check(proposal, an_observation()).rejection
        assert rejection is not None
        assert len(rejection.detail) > 20


# --- What gets through ------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize("name", ACTION_NAMES)
def test_every_action_in_the_vocabulary_is_approved_when_it_is_well_formed(name: str) -> None:
    """Parametrized over the vocabulary, so an eighth action without a case fails here."""
    validator = a_validator(skills=StubLibrary())

    verdict = validator.check(VALID_PROPOSALS[name], an_observation())

    assert verdict.rejection is None
    assert verdict.action is not None
    assert verdict.action.action == name


@pytest.mark.unit
def test_plain_data_comes_back_as_a_typed_action() -> None:
    """The gate is also the parse, so a caller holding a dict gets the action back."""
    verdict = a_validator().check({"action": "click", "element_id": 2}, an_observation())

    assert isinstance(verdict.action, Click)
    assert verdict.action.element_id == 2


@pytest.mark.unit
def test_an_already_typed_action_passes_through_unchanged() -> None:
    """A planner's action is not re-serialized on the way past."""
    action = parse_action({"action": "click", "element_id": 2})

    verdict = a_validator().check(action, an_observation())

    assert verdict.action is action


@pytest.mark.unit
def test_an_action_with_no_element_is_not_asked_for_one() -> None:
    """press and scroll name nothing on the page, so the element check must skip them."""
    validator = a_validator()
    observation = an_observation(elements=0)

    assert validator.check({"action": "press", "key": "Enter"}, observation).ok
    assert validator.check({"action": "scroll", "direction": "down"}, observation).ok


@pytest.mark.unit
def test_every_action_that_names_an_element_has_that_element_checked() -> None:
    """A drift guard: an action added later with an element_id is covered without edits.

    Built by asking the models themselves which of them carry the field, rather
    than by listing click and fill here -- a list would go stale in exactly the
    way the check must not.
    """
    observation = an_observation(elements=1)
    checked = []

    for name, proposal in VALID_PROPOSALS.items():
        if "element_id" not in proposal:
            continue
        checked.append(name)
        verdict = a_validator().check({**proposal, "element_id": 99}, observation)
        assert verdict.rejection is not None, f"{name} was not checked"
        assert verdict.rejection.code == UNKNOWN_ELEMENT

    assert checked, "no action in the table names an element; the guard is checking nothing"


# --- Navigation -------------------------------------------------------------


@pytest.mark.unit
def test_a_url_on_the_allowlist_is_approved() -> None:
    """Subdomains of the start URL's registrable domain are the ordinary case."""
    verdict = a_validator().check(
        {"action": "navigate", "url": f"https://help.{SITE}/faq"}, an_observation()
    )

    assert verdict.ok


@pytest.mark.unit
@pytest.mark.parametrize(
    ("url", "cause"),
    [
        ("https://elsewhere.example/", "off_allowlist"),
        ("javascript:alert(1)", "unsupported_scheme"),
        ("https:///nohost", "malformed_url"),
    ],
    ids=["another site", "not a web URL", "no host"],
)
def test_a_blocked_url_keeps_the_allowlists_own_finer_reason(url: str, cause: str) -> None:
    """One code for the check, and the cause underneath it.

    A ``javascript:`` URL and a wrong domain are both refused navigation, but
    they are not the same event, and a run where the first started happening is
    worth noticing separately.
    """
    verdict = a_validator().check({"action": "navigate", "url": url}, an_observation())

    assert verdict.rejection is not None
    assert verdict.rejection.code == BLOCKED_NAVIGATION
    assert verdict.rejection.cause == cause


# --- Skills -----------------------------------------------------------------


@pytest.mark.unit
def test_a_skill_call_that_matches_the_library_is_approved() -> None:
    """The baseline the rejections below are variations on."""
    validator = a_validator(skills=StubLibrary())

    verdict = validator.check(VALID_PROPOSALS["use_skill"], an_observation())

    assert verdict.ok


@pytest.mark.unit
def test_an_optional_parameter_may_be_left_out() -> None:
    """Optional has to mean optional, or every skill call needs every field."""
    validator = a_validator(skills=StubLibrary())

    verdict = validator.check(
        {"action": "use_skill", "skill": "search", "params": {"query": "rust"}}, an_observation()
    )

    assert verdict.ok


@pytest.mark.unit
@pytest.mark.parametrize(
    ("skill", "params", "cause"),
    [
        (StubSkill(name="other"), {"query": "rust"}, SKILL_NOT_FOUND),
        (StubSkill(active=False), {"query": "rust"}, SKILL_RETIRED),
        (StubSkill(domain="elsewhere.example"), {"query": "rust"}, SKILL_WRONG_SITE),
        (StubSkill(), {}, SKILL_BAD_PARAMS),
        (StubSkill(), {"query": "rust", "colour": "blue"}, SKILL_BAD_PARAMS),
    ],
    ids=["unknown", "retired", "another site", "missing parameter", "unexpected parameter"],
)
def test_an_unusable_skill_is_refused_and_says_which_way(
    skill: StubSkill, params: dict[str, str], cause: str
) -> None:
    """One code for the check, with the cause naming which condition failed."""
    validator = a_validator(skills=StubLibrary(skill))

    verdict = validator.check(
        {"action": "use_skill", "skill": "search", "params": params}, an_observation()
    )

    assert verdict.action is None
    assert verdict.rejection is not None
    assert verdict.rejection.code == UNUSABLE_SKILL
    assert verdict.rejection.cause == cause


@pytest.mark.unit
def test_a_retired_skill_is_not_reported_as_a_missing_one() -> None:
    """Different problems: one means the library never had it, one means it broke."""
    validator = a_validator(skills=StubLibrary(StubSkill(active=False)))

    verdict = validator.check(VALID_PROPOSALS["use_skill"], an_observation())

    assert verdict.rejection is not None
    assert verdict.rejection.cause == SKILL_RETIRED
    assert "retired" in verdict.rejection.detail


@pytest.mark.unit
def test_a_missing_parameter_is_named_in_the_complaint() -> None:
    """The model can only fix the call if it is told which value it left out."""
    validator = a_validator(skills=StubLibrary())

    verdict = validator.check(
        {"action": "use_skill", "skill": "search", "params": {}}, an_observation()
    )

    assert verdict.rejection is not None
    assert "query" in verdict.rejection.detail


@pytest.mark.unit
def test_a_skill_from_another_site_is_refused_on_this_one() -> None:
    """A skill is a route through one site's pages; elsewhere it means nothing."""
    validator = a_validator(skills=StubLibrary(StubSkill(domain="elsewhere.example")))

    verdict = validator.check(VALID_PROPOSALS["use_skill"], an_observation())

    assert verdict.rejection is not None
    assert verdict.rejection.cause == SKILL_WRONG_SITE
    assert "elsewhere.example" in verdict.rejection.detail


@pytest.mark.unit
def test_a_page_with_no_usable_host_refuses_skills_rather_than_raising() -> None:
    """A fresh tab is ``about:blank``; no skill matches it, and nothing crashes."""
    validator = a_validator(skills=StubLibrary())

    verdict = validator.check(VALID_PROPOSALS["use_skill"], an_observation(url="about:blank"))

    assert verdict.rejection is not None
    assert verdict.rejection.cause == SKILL_WRONG_SITE


@pytest.mark.unit
def test_a_run_with_a_library_but_use_skill_switched_off_refuses_it_as_forbidden() -> None:
    """The cold evaluation pass runs this way, and the reason should say so."""
    validator = a_validator(
        skills=StubLibrary(),
        permitted=[name for name in ACTION_NAMES if name != "use_skill"],
    )

    verdict = validator.check(VALID_PROPOSALS["use_skill"], an_observation())

    assert verdict.rejection is not None
    assert verdict.rejection.code == FORBIDDEN_ACTION


@pytest.mark.unit
def test_use_skill_without_a_library_is_refused_even_if_someone_permits_it() -> None:
    """A setup mistake, refused rather than crashed halfway through a run."""
    validator = a_validator(permitted=ACTION_NAMES)

    verdict = validator.check(VALID_PROPOSALS["use_skill"], an_observation())

    assert verdict.rejection is not None
    assert verdict.rejection.code == UNUSABLE_SKILL
    assert verdict.rejection.cause == NO_SKILL_LIBRARY


# --- Which actions a run allows ---------------------------------------------


@pytest.mark.unit
def test_a_run_with_a_library_allows_all_seven() -> None:
    """The warm case: everything in the vocabulary is on the table."""
    assert a_validator(skills=StubLibrary()).permitted == ACTION_NAMES


@pytest.mark.unit
def test_a_run_without_a_library_allows_the_other_six() -> None:
    """No library means nothing to replay, so the action is dropped rather than left to fail."""
    permitted = a_validator().permitted

    assert "use_skill" not in permitted
    assert set(permitted) == set(ACTION_NAMES) - {"use_skill"}


@pytest.mark.unit
def test_a_misspelt_action_name_in_the_configuration_is_refused_at_startup() -> None:
    """Left alone it would forbid nothing, silently, for the length of the run."""
    with pytest.raises(ValueError, match="not action names"):
        a_validator(permitted=["click", "clcik", "done"])


@pytest.mark.unit
def test_a_run_that_cannot_say_done_is_refused_at_startup() -> None:
    """Every run ends with a verdict; one that cannot say so has no way to end."""
    with pytest.raises(ValueError, match="done"):
        a_validator(permitted=["click", "scroll"])


@pytest.mark.unit
def test_a_repeated_action_name_is_listed_once() -> None:
    """Duplicates are a typo, not a policy; they should not appear twice in a message."""
    assert a_validator(permitted=["click", "click", "done"]).permitted == ("click", "done")


# --- The bounded retry cycle ------------------------------------------------


@pytest.mark.unit
def test_a_good_first_answer_costs_one_attempt_and_no_complaints() -> None:
    """The ordinary step: nothing about the retry machinery should show up in it."""
    outcome, planner = run_step(a_decision(VALID_PROPOSALS["click"]))

    assert outcome.ok
    assert len(outcome.attempts) == 1
    assert planner.complaints == [()]


@pytest.mark.unit
def test_a_rejected_answer_is_retried_with_the_complaint_attached() -> None:
    """This is the whole recovery mechanism: the model is told what was wrong."""
    outcome, planner = run_step(
        a_decision({"action": "click", "element_id": 99}),
        a_decision({"action": "click", "element_id": 2}),
    )

    assert outcome.ok
    assert outcome.action is not None
    assert outcome.action.describe() == "click 2"
    assert planner.complaints[0] == ()
    assert "99" in planner.complaints[1][0]


@pytest.mark.unit
def test_complaints_accumulate_rather_than_replace() -> None:
    """A model that fixes one mistake should not be free to make the first one again."""
    _, planner = run_step(
        a_decision({"action": "click", "element_id": 99}),
        a_decision({"action": "navigate", "url": "https://elsewhere.example/"}),
        a_decision({"action": "click", "element_id": 1}),
    )

    assert len(planner.complaints[2]) == 2
    assert "99" in planner.complaints[2][0]
    assert "elsewhere.example" in planner.complaints[2][1]


@pytest.mark.unit
def test_the_budget_caps_the_attempts() -> None:
    """Two retries means three tries, not three retries and not an unbounded loop."""
    outcome, planner = run_step(a_decision({"action": "click", "element_id": 99}), retry_budget=2)

    assert len(outcome.attempts) == 3
    assert len(planner.complaints) == 3


@pytest.mark.unit
@pytest.mark.parametrize("budget", [0, -1], ids=["no retries", "a nonsense budget"])
def test_a_budget_of_none_still_gets_one_try(budget: int) -> None:
    """Zero retries is one attempt. A negative number is a misconfiguration, not a ban."""
    outcome, _ = run_step(a_decision({"action": "click", "element_id": 99}), retry_budget=budget)

    assert len(outcome.attempts) == 1


@pytest.mark.unit
def test_exhaustion_produces_a_categorized_done_rather_than_nothing() -> None:
    """A step that ran out of tries still ends with a stated reason."""
    outcome, _ = run_step(a_decision({"action": "click", "element_id": 99}), retry_budget=2)

    assert outcome.ok is False
    assert outcome.action is None
    assert isinstance(outcome.giving_up, Done)
    assert outcome.giving_up.status == "failed"
    assert outcome.giving_up.reason is not None
    assert outcome.giving_up.reason.startswith(VALIDATION_EXHAUSTED)


@pytest.mark.unit
def test_the_give_up_reason_lists_what_was_wrong_each_time() -> None:
    """Failing the same way three times and failing three different ways differ."""
    outcome, _ = run_step(
        a_decision({"action": "click", "element_id": 99}),
        a_decision({"action": "navigate", "url": "https://elsewhere.example/"}),
        a_failed_decision(),
        retry_budget=2,
    )

    assert outcome.giving_up is not None
    assert outcome.giving_up.reason is not None
    assert UNKNOWN_ELEMENT in outcome.giving_up.reason
    assert BLOCKED_NAVIGATION in outcome.giving_up.reason
    assert INVALID_SCHEMA in outcome.giving_up.reason


@pytest.mark.unit
def test_a_reply_that_was_not_an_action_enters_the_same_cycle() -> None:
    """A model that answers in prose has failed the first check, one stage earlier."""
    outcome, planner = run_step(
        a_failed_decision(FAILURE_NOT_JSON, "reply was not a JSON object"),
        a_decision(VALID_PROPOSALS["click"]),
    )

    assert outcome.ok
    first = outcome.attempts[0]
    assert first.rejection is not None
    assert first.rejection.code == INVALID_SCHEMA
    assert first.rejection.cause == FAILURE_NOT_JSON
    assert "JSON" in planner.complaints[1][0]


@pytest.mark.unit
def test_an_invented_action_type_is_reported_under_the_schema_check() -> None:
    """Whether the planner or the gate caught it, it is the same check failing."""
    outcome, _ = run_step(
        a_failed_decision(FAILURE_NOT_AN_ACTION, "'teleport' is not one of the 7 actions"),
        retry_budget=0,
    )

    assert outcome.attempts[0].rejection is not None
    assert outcome.attempts[0].rejection.code == INVALID_SCHEMA
    assert outcome.attempts[0].rejection.cause == FAILURE_NOT_AN_ACTION


@pytest.mark.unit
def test_rejected_attempts_are_still_counted_as_cost() -> None:
    """A wasted call is billed like any other; a run that skipped them would flatter itself."""
    outcome, _ = run_step(a_decision({"action": "click", "element_id": 99}), retry_budget=2)

    assert len(outcome.records) == 3
    assert sum(record.prompt_tokens or 0 for record in outcome.records) == 2700


@pytest.mark.unit
def test_the_planner_is_not_called_again_once_an_answer_passes() -> None:
    """Retries are for recovery, not a fixed number of calls per step."""
    _, planner = run_step(a_decision(VALID_PROPOSALS["click"]), retry_budget=2)

    assert len(planner.complaints) == 1


# --- What the trace keeps ---------------------------------------------------


@pytest.mark.unit
def test_the_trace_record_keeps_the_rejection_and_the_recovery() -> None:
    """The interesting thing about a recovered step is what it recovered from."""
    outcome, _ = run_step(
        a_decision({"action": "click", "element_id": 99}),
        a_decision({"action": "click", "element_id": 2}),
    )

    record = outcome.to_dict()

    assert record["action"] == {"action": "click", "element_id": 2}
    assert record["giving_up"] is None
    assert len(record["attempts"]) == 2
    assert record["attempts"][0]["rejection"]["code"] == UNKNOWN_ELEMENT
    assert record["attempts"][0]["decision"]["action"] == {"action": "click", "element_id": 99}
    assert record["attempts"][1]["rejection"] is None


@pytest.mark.unit
def test_the_trace_record_of_an_exhausted_step_carries_its_verdict() -> None:
    """A run killed at this step should still show how the step ended."""
    outcome, _ = run_step(a_decision({"action": "click", "element_id": 99}), retry_budget=1)

    record = outcome.to_dict()

    assert record["action"] is None
    assert record["giving_up"]["status"] == "failed"
    assert VALIDATION_EXHAUSTED in record["giving_up"]["reason"]


@pytest.mark.unit
def test_a_step_summary_says_how_many_tries_it_took() -> None:
    """One line per step in the terminal, and a recovered step should look different."""
    clean, _ = run_step(a_decision(VALID_PROPOSALS["click"]))
    recovered, _ = run_step(
        a_decision({"action": "click", "element_id": 99}),
        a_decision({"action": "click", "element_id": 2}),
    )

    assert clean.summary() == "click 1"
    assert "click 2" in recovered.summary()
    assert "after 1 retry" in recovered.summary()


@pytest.mark.unit
def test_an_attempt_summary_names_the_proposal_and_its_fate() -> None:
    """Read on its own, a log line has to say what was asked for as well as what happened."""
    outcome, _ = run_step(a_decision({"action": "click", "element_id": 99}), retry_budget=0)

    line = outcome.attempts[0].summary()

    assert line.startswith("click 99")
    assert UNKNOWN_ELEMENT in line


@pytest.mark.unit
def test_a_verdict_summary_reads_the_same_way_for_both_answers() -> None:
    """The step log is scanned by eye; approvals and refusals should line up."""
    validator = a_validator()

    approved = validator.check(VALID_PROPOSALS["click"], an_observation())
    refused = validator.check({"action": "click", "element_id": 99}, an_observation())

    assert approved.summary() == "approved: click 1"
    assert refused.summary().startswith(f"rejected ({UNKNOWN_ELEMENT})")


# --- Against a real page ----------------------------------------------------


@pytest.mark.fixture
def test_a_made_up_element_number_is_caught_and_recovered_on_the_real_site(
    page: Page, fixture_site: Any
) -> None:
    """The whole cycle over a real observation, from a real browser.

    Everything above builds its element list by hand, which proves the checks
    agree with each other and nothing about whether they agree with a page. Here
    the numbering comes from the extractor, the planner names a number that page
    does not have, and the step has to recover -- with both halves left in the
    trace, which is what makes the recovery reviewable afterwards.
    """
    page.goto(fixture_site.url_for("search.html"))
    elements = extract_elements(page)
    observation = Observation(
        url=page.url,
        title=page.title(),
        elements=elements,
        text="Search the catalogue.",
    )
    beyond_the_page = len(elements) + 50
    real_element = elements[0].id

    outcome, planner = run_step(
        a_decision({"action": "click", "element_id": beyond_the_page}),
        a_decision({"action": "click", "element_id": real_element}),
        validator=Validator(Allowlist.derive(page.url)),
        observation=observation,
    )

    assert outcome.ok
    assert outcome.action is not None
    assert outcome.action.describe() == f"click {real_element}"
    assert str(beyond_the_page) in planner.complaints[1][0]

    record = outcome.to_dict()
    assert record["attempts"][0]["rejection"]["code"] == UNKNOWN_ELEMENT
    assert record["attempts"][1]["rejection"] is None
