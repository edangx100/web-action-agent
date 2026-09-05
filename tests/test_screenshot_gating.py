"""That the expensive channel stays shut unless something says to open it.

The screenshot is the one part of an observation whose cost is large enough to
show up in the project's headline cost-per-task number, so the rule that decides
whether to send one is worth more than a passing test. Two things can go wrong
with it, and neither announces itself at runtime.

**Gating in too readily** costs money quietly. Every run still works; the
numbers are simply several times worse than they should be, and nothing in the
output says which steps did it. That is what the trigger on every decision is
for, and why these tests assert on it rather than only on the yes/no.

**Gating out too readily** costs accuracy quietly. The agent goes on failing on
the pages that most needed an image, in a way that looks like a weak model
rather than a gate set too tight.

The third property here is not a heuristic at all: a profile with no vision
model must never be handed an image, whatever the cost rules would prefer. That
one is a guarantee, so it is tested as an override — asserted against inputs
chosen so that both cost rules would otherwise say yes.
"""

import pytest

from web_action_agent.config import Config
from web_action_agent.screenshot_gate import (
    TRIGGER_DOM_SUFFICIENT,
    TRIGGER_FEW_ELEMENTS,
    TRIGGER_NO_VISION_ROUTE,
    TRIGGER_PLANNER_FAILURES,
    GateDecision,
    ScreenshotGate,
)

MIN_ELEMENTS = 5
"""int: Element floor used throughout, matching the committed default."""

MAX_DOM_FAILURES = 2
"""int: Failure limit used throughout, matching the committed default."""

BUSY_PAGE = 30
"""int: Elements on a page the extractor clearly read well — well clear of the
floor, so only a failure streak should ever gate an image in for it."""

SPARSE_PAGE = 2
"""int: Elements on a page the extractor probably read badly. Below the floor,
which is the whole signal: a page a person would call busy, seen as nearly
empty, is more likely a bad extraction than an empty page."""


def gate(*, supports_vision: bool = True, **overrides) -> ScreenshotGate:
    """Build a gate at the committed defaults, with thresholds replaced as needed.

    Args:
        supports_vision: Whether the run's profile has a vision route.
        **overrides: ``min_elements`` or ``max_dom_failures``, replacing the
            default.

    Returns:
        ScreenshotGate: A fresh gate with no failures recorded.
    """
    defaults = {"min_elements": MIN_ELEMENTS, "max_dom_failures": MAX_DOM_FAILURES}
    return ScreenshotGate(supports_vision=supports_vision, **{**defaults, **overrides})


def fail_dom_only(instance: ScreenshotGate, times: int) -> None:
    """Record a run of planner failures on text-only steps.

    Args:
        instance: The gate to report to.
        times: How many consecutive failures to record.
    """
    for _ in range(times):
        instance.record_planner_outcome(succeeded=False, had_screenshot=False)


@pytest.mark.unit
def test_busy_page_gates_the_screenshot_out() -> None:
    """A page the extractor read well is answered from text alone.

    This is the ordinary case and by far the most common one, so it is the case
    that decides what a run costs.
    """
    decision = gate().decide(element_count=BUSY_PAGE)

    assert decision.send_screenshot is False
    assert decision.trigger == TRIGGER_DOM_SUFFICIENT


@pytest.mark.unit
def test_sparse_page_gates_the_screenshot_in() -> None:
    """Too few elements is read as a bad extraction, not an empty page."""
    decision = gate().decide(element_count=SPARSE_PAGE)

    assert decision.send_screenshot is True
    assert decision.trigger == TRIGGER_FEW_ELEMENTS


@pytest.mark.unit
def test_element_count_exactly_at_the_floor_is_enough() -> None:
    """The floor is a minimum, not a threshold to clear.

    Worth pinning because an off-by-one here is invisible: the agent keeps
    working and simply pays for an image on every page with exactly the
    boundary number of controls.
    """
    decision = gate().decide(element_count=MIN_ELEMENTS)

    assert decision.send_screenshot is False
    assert decision.trigger == TRIGGER_DOM_SUFFICIENT


@pytest.mark.unit
def test_repeated_dom_only_failures_gate_the_screenshot_in() -> None:
    """A failure streak overrules a healthy element count.

    The element count is a guess made before the model is asked; the streak is
    evidence from the run itself. When they disagree, the evidence wins — so
    this is asserted on the busy page, where the count alone would say no.
    """
    instance = gate()
    fail_dom_only(instance, MAX_DOM_FAILURES)

    decision = instance.decide(element_count=BUSY_PAGE)

    assert decision.send_screenshot is True
    assert decision.trigger == TRIGGER_PLANNER_FAILURES
    assert decision.dom_failures == MAX_DOM_FAILURES


@pytest.mark.unit
def test_one_failure_short_of_the_limit_still_runs_on_text() -> None:
    """The streak escalates at the limit, not on the way to it."""
    instance = gate()
    fail_dom_only(instance, MAX_DOM_FAILURES - 1)

    decision = instance.decide(element_count=BUSY_PAGE)

    assert decision.send_screenshot is False
    assert decision.trigger == TRIGGER_DOM_SUFFICIENT


@pytest.mark.unit
def test_a_success_clears_the_streak() -> None:
    """A streak is consecutive, so the run drops back to text once it is unstuck.

    Without this, one bad patch early in a run would put every remaining step
    on the expensive channel for the rest of the run.
    """
    instance = gate()
    fail_dom_only(instance, MAX_DOM_FAILURES)
    instance.record_planner_outcome(succeeded=True, had_screenshot=True)

    decision = instance.decide(element_count=BUSY_PAGE)

    assert instance.dom_failures == 0
    assert decision.send_screenshot is False


@pytest.mark.unit
def test_failures_on_a_step_that_already_had_an_image_do_not_count() -> None:
    """The streak counts evidence that *text* was insufficient, nothing else.

    A failure on a step that already carried a screenshot says the planner is
    struggling, not that the text channels were the thing missing — the run had
    already escalated. Counting it would let a model having a bad day pull the
    streak up on its own.
    """
    instance = gate()
    for _ in range(5):
        instance.record_planner_outcome(succeeded=False, had_screenshot=True)

    assert instance.dom_failures == 0
    assert instance.decide(element_count=BUSY_PAGE).send_screenshot is False


@pytest.mark.unit
def test_no_vision_route_overrules_both_cost_rules() -> None:
    """A profile with no vision model is never handed an image.

    Deliberately asserted under the conditions where both heuristics would say
    yes — a sparse page *and* a failure streak — because the guarantee is only
    worth anything in exactly the case where the cost rules disagree with it.
    """
    instance = gate(supports_vision=False)
    fail_dom_only(instance, MAX_DOM_FAILURES + 1)

    decision = instance.decide(element_count=SPARSE_PAGE)

    assert decision.send_screenshot is False
    assert decision.trigger == TRIGGER_NO_VISION_ROUTE


@pytest.mark.unit
def test_a_zero_element_floor_disables_the_element_rule() -> None:
    """No element count can fall below zero, so the rule stops firing.

    The bound in the configuration schema allows zero, so what zero means is a
    question someone will ask; this is the answer.
    """
    decision = gate(min_elements=0).decide(element_count=0)

    assert decision.send_screenshot is False
    assert decision.trigger == TRIGGER_DOM_SUFFICIENT


@pytest.mark.unit
def test_a_zero_failure_limit_puts_every_step_on_the_image_channel() -> None:
    """Tolerating no failures means escalating before any have happened.

    This is the setting a "does vision help?" comparison run would use, so it
    should behave predictably rather than being an unconsidered edge case.
    """
    decision = gate(max_dom_failures=0).decide(element_count=BUSY_PAGE)

    assert decision.send_screenshot is True
    assert decision.trigger == TRIGGER_PLANNER_FAILURES


@pytest.mark.unit
@pytest.mark.parametrize(
    ("element_count", "failures", "supports_vision", "expected_trigger"),
    [
        (BUSY_PAGE, 0, True, TRIGGER_DOM_SUFFICIENT),
        (SPARSE_PAGE, 0, True, TRIGGER_FEW_ELEMENTS),
        (BUSY_PAGE, MAX_DOM_FAILURES, True, TRIGGER_PLANNER_FAILURES),
        (BUSY_PAGE, 0, False, TRIGGER_NO_VISION_ROUTE),
    ],
)
def test_every_decision_carries_its_trigger_and_the_counts_behind_it(
    element_count: int,
    failures: int,
    supports_vision: bool,
    expected_trigger: str,
) -> None:
    """Each of the four outcomes is recorded, not just the expensive ones.

    A run whose cost came out high has to be answerable from its own trace, and
    that needs the steps that sent nothing recorded too — otherwise the trace
    shows the images without showing what the alternative looked like.

    Args:
        element_count: Elements extracted for the step.
        failures: DOM-only planner failures recorded before it.
        supports_vision: Whether the profile has a vision route.
        expected_trigger: The rule that should decide this combination.
    """
    instance = gate(supports_vision=supports_vision)
    fail_dom_only(instance, failures)

    decision = instance.decide(element_count=element_count)

    assert decision.trigger == expected_trigger
    assert decision.element_count == element_count
    assert decision.dom_failures == failures
    # The readable half of the record has to be filled in as well: a trigger
    # token alone tells a reader which rule fired, never on what numbers.
    assert decision.detail
    assert decision.trigger in decision.summary()


@pytest.mark.unit
def test_the_summary_line_says_which_way_it_went() -> None:
    """The step log's one line is legible without decoding the trigger token."""
    assert gate().decide(element_count=SPARSE_PAGE).summary().startswith("screenshot: yes")
    assert gate().decide(element_count=BUSY_PAGE).summary().startswith("screenshot: no")


@pytest.mark.unit
def test_a_decision_serializes_to_plain_data() -> None:
    """The trace is JSON, so every field of the record has to survive it."""
    decision = gate().decide(element_count=SPARSE_PAGE)

    record = decision.to_dict()

    assert record == {
        "send_screenshot": True,
        "trigger": TRIGGER_FEW_ELEMENTS,
        "detail": decision.detail,
        "element_count": SPARSE_PAGE,
        "dom_failures": 0,
    }


@pytest.mark.unit
def test_a_decision_cannot_be_edited_after_the_fact() -> None:
    """It is the record of a choice already made, like the observation it sits with."""
    decision = gate().decide(element_count=BUSY_PAGE)

    with pytest.raises(Exception, match="frozen|immutable|cannot assign"):
        decision.send_screenshot = True  # type: ignore[misc]


@pytest.mark.unit
def test_from_config_reads_both_thresholds() -> None:
    """The thresholds come from configuration, so an experiment can move them.

    Built from a Config with non-default values, so a gate that ignored one of
    them and used its own default would be caught rather than accidentally
    agreeing.
    """
    config = Config(screenshot_min_elements=12, screenshot_max_dom_failures=4)

    instance = ScreenshotGate.from_config(config, supports_vision=True)

    # The floor moved up: a page that clears the committed default of 5 no
    # longer clears 12.
    assert instance.decide(element_count=8).trigger == TRIGGER_FEW_ELEMENTS
    # The failure limit moved up too: the committed default of 2 is no longer
    # enough to escalate.
    fail_dom_only(instance, 2)
    assert instance.decide(element_count=BUSY_PAGE).trigger == TRIGGER_DOM_SUFFICIENT


@pytest.mark.unit
def test_the_gate_describes_its_own_state() -> None:
    """A gate in a debugger or a failure message should explain itself."""
    instance = gate()
    fail_dom_only(instance, 1)

    text = repr(instance)

    assert "min_elements=5" in text
    assert "max_dom_failures=2" in text
    assert "supports_vision=True" in text
    assert "dom_failures=1" in text


@pytest.mark.unit
def test_the_decision_record_is_constructible_directly() -> None:
    """Whoever reads a trace back builds these without a gate to hand.

    The trace reader has the plain data and no run state, so the record has to
    stand on its own rather than only ever coming out of :meth:`decide`.
    """
    decision = GateDecision(
        send_screenshot=False,
        trigger=TRIGGER_NO_VISION_ROUTE,
        detail="the profile serves no vision model; running on DOM text alone",
        element_count=7,
        dom_failures=0,
    )

    assert decision.summary().startswith("screenshot: no (no_vision_route)")
