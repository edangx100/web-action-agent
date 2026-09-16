"""The loop, and the promise that every run ends with a verdict.

One property matters more than everything else in this file: **a run always
produces an outcome, and the outcome always has a category**. Not a stack
trace, not a hang, not silence. The evaluation is a table of a few hundred
runs, and a run that ended in none of the known ways is a row nobody can
count -- so the tests here are mostly a list of ways to end a run, each
checked to end it in the right category.

The second property is that the loop cannot run forever. Three things bound
it: the step cap, the wall clock, and the agent's own ``done``. A fourth --
stuck detection -- is not a bound but the reason the first is rarely reached:
a model at temperature zero shown the same page twice answers the same way
twice, so the loop has to break the tie itself.

**How these tests drive the loop.** Every part the controller composes is
passed in, so most tests here hand it stubs: a planner that reads from a
script, an executor that records what it was asked and answers as told, and an
observer that returns a page without a browser. That is not a shortcut around
integration -- the last section runs the whole thing against a real Chromium
and the local fixture site, which is where the parts are proven to fit
together. The stubs are what make the *branches* testable: a dead end, an
exhausted retry budget, a busy endpoint.
"""

import ast
import inspect
import textwrap
from pathlib import Path
from types import SimpleNamespace

import pytest
from browser_steps import observe as observe_page

from web_action_agent import controller as controller_module
from web_action_agent.actions import parse_action
from web_action_agent.allowlist import Allowlist
from web_action_agent.config import Config
from web_action_agent.controller import (
    STUCK_NOTE,
    Controller,
    RunReport,
)
from web_action_agent.executor import (
    NAVIGATION_ERROR,
    OK,
    TIMEOUT,
    ExecutionResult,
    Executor,
)
from web_action_agent.memory import Memory
from web_action_agent.model_client import (
    ENDPOINT_UNAVAILABLE,
    RATE_LIMITED,
    CallRecord,
    ModelBusyError,
    ModelClientError,
    ModelTimeoutError,
)
from web_action_agent.observation import Observation
from web_action_agent.observer import Box, Element
from web_action_agent.outcomes import (
    ACTION_TIMEOUT,
    FAILURE_CATEGORIES,
    GAVE_UP,
    MODEL_ERROR,
    NAVIGATION_DEAD_END,
    RUN_TIMEOUT,
    STEP_CAP,
    SUCCESS,
    TERMINAL_CATEGORIES,
    VALIDATION_EXHAUSTED,
)
from web_action_agent.planner import PlannerDecision, PlannerFailure
from web_action_agent.screenshot_gate import ScreenshotGate
from web_action_agent.timeouts import RunBudget
from web_action_agent.validator import Validator

TASK = "Find the opening hours of the Hollow Lane branch."
"""str: The task every run here is given."""

START = "http://localhost:8000/search.html"
"""str: Where every run here opens."""

A_CLICK = {"action": "click", "element_id": 1}
"""dict: A well-formed action a stub planner can return forever."""

DONE = {"action": "done", "status": "success", "answer": "09:00 to 17:30"}
"""dict: The agent finishing the task."""


def a_page(url=START, *, elements=2):
    """Build an observation to decide against.

    Args:
        url: The page's address.
        elements: How many interactive elements it offers.

    Returns:
        Observation: The page.
    """
    return Observation(
        url=url,
        title="Search",
        elements=tuple(
            Element(
                id=number,
                tag="a",
                role="link",
                name=f"Link {number}",
                text=f"Link {number}",
                box=Box(x=0, y=20 * number, width=80, height=18),
            )
            for number in range(1, elements + 1)
        ),
        text="Opening hours are listed by branch.",
    )


class StubPlanner:
    """A planner that answers from a script instead of asking a model.

    Attributes:
        asked: Every call's arguments, so a test can check what the planner was
            told -- the history, the notepad, the complaints from a refusal.
    """

    def __init__(self, *replies, failures=()):
        """Set up the script.

        Args:
            *replies: Actions to return, as plain data. The last repeats for as
                long as the loop keeps asking, which is what lets one entry
                drive a run to the step cap.
            failures: Codes to fail with instead, one per call, before the
                replies are used. For the steps where the model answers with
                something that is not an action at all.
        """
        self._replies = list(replies) or [A_CLICK]
        self._failures = list(failures)
        self.asked = []

    def decide(self, **kwargs):
        """Answer one step.

        Args:
            **kwargs: What the controller passed, recorded for inspection.

        Returns:
            PlannerDecision: The next scripted answer.
        """
        self.asked.append(kwargs)
        record = CallRecord(
            profile="local",
            model="stub",
            vision=False,
            prompt_tokens=100,
            completion_tokens=10,
            latency_s=0.01,
            finish_reason="stop",
        )
        if self._failures:
            code = self._failures.pop(0)
            return PlannerDecision(
                action=None,
                failure=PlannerFailure(code=code, detail="not an action", text="sorry!"),
                text="sorry!",
                record=record,
            )
        index = min(len(self.asked) - 1, len(self._replies) - 1)
        reply = self._replies[index]
        return PlannerDecision(action=parse_action(reply), failure=None, text="{}", record=record)


class RaisingPlanner:
    """A planner whose endpoint is broken in a particular way."""

    def __init__(self, error):
        """Args: error: What to raise when asked."""
        self._error = error

    def decide(self, **kwargs):
        """Raise the configured failure.

        Args:
            **kwargs: Ignored.

        Raises:
            ModelClientError: Always.
        """
        raise self._error


class StubExecutor:
    """An executor that records actions and answers as instructed.

    Attributes:
        performed: Every action it was asked to perform, in order, as the
            planner's own description of it -- which is what the stuck-detection
            tests read.
        page: Stands in for the browser's page; nothing here touches it.
    """

    def __init__(self, *results, default=None):
        """Set up the answers.

        Args:
            *results: Statuses to return, one per action, in order.
            default: The status to return once the list runs out. Defaults to
                success, so a test only scripts the steps it cares about.
        """
        self.performed = []
        self._results = list(results)
        self._default = default or OK
        self.page = SimpleNamespace(url=START)

    def execute(self, action, *, observation=None):
        """Perform one action, or say why not.

        Args:
            action: The action to perform.
            observation: The page it was planned against. Unused here.

        Returns:
            ExecutionResult: The scripted result.
        """
        self.performed.append(action.describe())
        status = self._results.pop(0) if self._results else self._default
        return ExecutionResult(
            action=action,
            status=status,
            detail="" if status == OK else f"{status} happened",
            url=START,
            elapsed_s=0.01,
        )


def a_controller(
    *,
    planner=None,
    executor=None,
    observation=None,
    max_steps=5,
    retry_budget=1,
    budget=None,
    memory=None,
    validator=None,
    replay=None,
):
    """Assemble a controller over stubs.

    Args:
        planner: The planner. A stub clicking forever by default.
        executor: The executor. A stub that succeeds by default.
        observation: What every look at the page returns. One page by default,
            which is the right default for tests about the loop rather than
            about perception.
        max_steps: The step cap.
        retry_budget: Extra planning attempts per step.
        budget: The run's wall clock, or None for no clock.
        memory: The run's memory. A fresh one by default.
        validator: The gate. One permitting everything on this site by default.
        replay: A skill replay, for the fallback test.

    Returns:
        tuple[Controller, StubPlanner, StubExecutor]: The controller and the
        two stubs, since most tests assert on what those two saw.
    """
    planner = planner or StubPlanner()
    executor = executor or StubExecutor()
    page = observation or a_page()
    return (
        Controller(
            planner=planner,
            validator=validator or Validator(Allowlist.derive(START)),
            executor=executor,
            memory=memory or Memory(),
            max_steps=max_steps,
            retry_budget=retry_budget,
            budget=budget,
            replay=replay,
            observe=lambda label: (page, None),
        ),
        planner,
        executor,
    )


# --- Every run ends, and ends in a category ---------------------------------


@pytest.mark.mock_llm
def test_a_run_that_finishes_says_so_with_its_answer():
    """The ordinary ending: the agent decided it was done, and what the answer is."""
    controller, _, _ = a_controller(planner=StubPlanner(A_CLICK, DONE))

    report = controller.run(TASK, start_url=START)

    assert report.outcome.ok
    assert report.outcome.category == SUCCESS
    assert report.outcome.answer == "09:00 to 17:30"
    assert report.outcome.steps == 2


@pytest.mark.mock_llm
def test_a_run_that_never_finishes_stops_at_the_step_cap():
    """The loop's guarantee that it ends at all."""
    controller, _, _ = a_controller(max_steps=4)

    report = controller.run(TASK, start_url=START)

    assert report.outcome.category == STEP_CAP
    assert report.outcome.steps == 4
    assert len(report.steps) == 4
    assert "4" in report.outcome.reason


@pytest.mark.mock_llm
def test_the_agent_giving_up_is_its_own_category():
    """Stopping early with a reason costs less and says more than looping to the cap."""
    beaten = {"action": "done", "status": "failed", "reason": "no branch page exists"}
    controller, _, _ = a_controller(planner=StubPlanner(beaten))

    report = controller.run(TASK, start_url=START)

    assert report.outcome.category == GAVE_UP
    assert report.outcome.reason == "no branch page exists"
    assert report.outcome.answer is None, "a failed run has no answer"


@pytest.mark.mock_llm
def test_a_planner_that_never_produces_a_valid_action_exhausts_the_retries():
    """The step's own retries run out, and that ends the run rather than the step."""
    controller, _, _ = a_controller(
        planner=StubPlanner(failures=["not_json", "not_json", "not_json", "not_json"]),
        retry_budget=1,
    )

    report = controller.run(TASK, start_url=START)

    assert report.outcome.category == VALIDATION_EXHAUSTED
    assert report.steps[-1].action is None, "no action was ever approved"
    assert report.steps[-1].rejections, "and the refusals are on the record"


@pytest.mark.mock_llm
def test_a_start_page_that_cannot_be_reached_is_a_dead_end():
    """A run that never started has not failed at the task."""
    controller, _, _ = a_controller(executor=StubExecutor(NAVIGATION_ERROR))

    report = controller.run(TASK, start_url=START)

    assert report.outcome.category == NAVIGATION_DEAD_END
    assert report.outcome.steps == 0
    assert report.steps == (), "nothing was stepped"
    assert START in report.outcome.reason


@pytest.mark.mock_llm
def test_a_start_page_that_times_out_says_so_rather_than_unreachable():
    """Slow and absent are different findings, even on the first page."""
    controller, _, _ = a_controller(executor=StubExecutor(TIMEOUT))

    report = controller.run(TASK, start_url=START)

    assert report.outcome.category == ACTION_TIMEOUT


@pytest.mark.mock_llm
def test_navigations_that_keep_failing_are_a_dead_end():
    """One dead link is ordinary; two in a row means there is nowhere to go."""
    navigate = {"action": "navigate", "url": "http://localhost:8000/gone.html"}
    controller, _, _ = a_controller(
        planner=StubPlanner(navigate),
        executor=StubExecutor(OK, NAVIGATION_ERROR, NAVIGATION_ERROR),
        max_steps=6,
    )

    report = controller.run(TASK, start_url=START)

    assert report.outcome.category == NAVIGATION_DEAD_END
    assert report.outcome.steps == 2


@pytest.mark.mock_llm
def test_one_failed_navigation_does_not_end_the_run():
    """It is reported to the planner and the run carries on, which is the point."""
    navigate = {"action": "navigate", "url": "http://localhost:8000/gone.html"}
    controller, _, _ = a_controller(
        planner=StubPlanner(navigate, DONE),
        executor=StubExecutor(OK, NAVIGATION_ERROR),
        max_steps=6,
    )

    report = controller.run(TASK, start_url=START)

    assert report.outcome.category == SUCCESS


@pytest.mark.mock_llm
def test_a_page_that_cannot_be_read_ends_the_run_rather_than_raising():
    """Nothing to decide from is a verdict, not an exception."""
    unreadable = {"calls": 0}

    def observe(label):
        """Read the page once, then stop being able to.

        Args:
            label: The screenshot label, unused.

        Returns:
            tuple: The page and no gate decision.

        Raises:
            OSError: On every look after the first.
        """
        unreadable["calls"] += 1
        if unreadable["calls"] > 1:
            raise OSError("the page has gone")
        return a_page(), None

    controller = Controller(
        planner=StubPlanner(),
        validator=Validator(Allowlist.derive(START)),
        executor=StubExecutor(),
        memory=Memory(),
        max_steps=5,
        retry_budget=1,
        observe=observe,
    )

    report = controller.run(TASK, start_url=START)

    assert report.outcome.category == NAVIGATION_DEAD_END


@pytest.mark.mock_llm
def test_a_run_out_of_wall_clock_stops_between_steps():
    """The clock is checked where a step boundary is, so nothing is left half done."""
    clock = {"now": 0.0}
    budget = RunBudget(10.0, clock=lambda: clock["now"])

    class Slow(StubExecutor):
        """An executor whose every action takes four seconds of the run's clock."""

        def execute(self, action, *, observation=None):
            """Advance the clock, then answer normally.

            Args:
                action: The action to perform.
                observation: Unused.

            Returns:
                ExecutionResult: Success.
            """
            clock["now"] += 4.0
            return super().execute(action, observation=observation)

    controller, _, _ = a_controller(executor=Slow(), budget=budget, max_steps=50)

    report = controller.run(TASK, start_url=START)

    assert report.outcome.category == RUN_TIMEOUT
    # Four seconds each for step 0's navigation and two steps: twelve in all,
    # against a ten-second ceiling. The second step began while there was still
    # time and was allowed to finish, which is the behaviour being pinned --
    # the alternative is abandoning a half-performed action.
    assert clock["now"] == 12.0, "a step already underway ran past the ceiling"
    assert report.outcome.steps == 2
    assert len(report.steps) == 2, "and no third step was half-recorded"


@pytest.mark.mock_llm
@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (ModelTimeoutError("no answer within 90s"), "llm_timeout"),
        (ModelBusyError("refused 4 attempts", status=429), RATE_LIMITED),
        (ModelBusyError("refused 4 attempts", status=503), ENDPOINT_UNAVAILABLE),
        (ModelClientError("bad key"), MODEL_ERROR),
    ],
)
def test_a_broken_endpoint_ends_the_run_in_its_own_category(error, expected):
    """Slow, busy, down and misconfigured are four findings, not one."""
    controller, _, _ = a_controller(planner=RaisingPlanner(error))

    report = controller.run(TASK, start_url=START)

    assert report.outcome.category == expected
    assert not report.outcome.ok


@pytest.mark.unit
def test_every_category_a_run_can_end_in_is_one_the_table_knows():
    """A finding in a category nothing counts is a finding that will not appear."""
    assert SUCCESS in TERMINAL_CATEGORIES
    assert set(FAILURE_CATEGORIES) | {SUCCESS} == set(TERMINAL_CATEGORIES)
    assert len(TERMINAL_CATEGORIES) == len(set(TERMINAL_CATEGORIES))


@pytest.mark.unit
def test_no_way_out_of_the_loop_leaves_the_outcome_unset():
    """The task's own structural test, read off the loop's own source.

    Every ``return`` inside the loop has to carry a value, and the loop must not
    be able to fall off its end -- the statement after it is the step cap. A
    scenario test can only cover the paths somebody thought of; this covers the
    ones nobody did.
    """
    source = textwrap.dedent(inspect.getsource(controller_module.Controller._drive))
    tree = ast.parse(source)
    returns = [node for node in ast.walk(tree) if isinstance(node, ast.Return)]

    assert returns, "the loop returns something"
    assert all(node.value is not None for node in returns), (
        "a bare return would be a run with no verdict"
    )
    assert isinstance(tree.body[0].body[-1], ast.Return), "the loop cannot fall off its end"


# --- Going in circles -------------------------------------------------------


@pytest.mark.mock_llm
def test_a_third_identical_action_is_replaced():
    """The task's own test: two repeats are allowed, a third is a loop."""
    controller, _, executor = a_controller(max_steps=4)

    report = controller.run(TASK, start_url=START)

    assert executor.performed[:3] == [
        f"navigate {START}",
        "click 1",
        "click 1",
    ], "step 0's navigation, then the two allowed repeats"
    assert executor.performed[3] == "scroll down", "and then something else"
    assert report.steps[2].forced, "recorded as forced rather than passed off as the planner's idea"


@pytest.mark.mock_llm
def test_the_forced_alternative_is_written_into_the_notepad():
    """The planner reads the notepad, and being told it is repeating may change its mind."""
    memory = Memory()
    controller, _, _ = a_controller(memory=memory, max_steps=4)

    controller.run(TASK, start_url=START)

    assert STUCK_NOTE in memory.scratchpad()
    assert "click 1" in memory.scratchpad()


@pytest.mark.mock_llm
def test_an_agent_stuck_scrolling_down_is_sent_back_up():
    """Forcing another scroll down at the bottom of a page would achieve nothing."""
    scroll = {"action": "scroll", "direction": "down"}
    controller, _, executor = a_controller(planner=StubPlanner(scroll), max_steps=4)

    controller.run(TASK, start_url=START)

    assert executor.performed[1:4] == ["scroll down", "scroll down", "scroll up"]


@pytest.mark.mock_llm
def test_two_identical_actions_are_left_alone():
    """A click that missed while the page settled deserves a second go."""
    controller, _, executor = a_controller(planner=StubPlanner(A_CLICK, A_CLICK, DONE))

    report = controller.run(TASK, start_url=START)

    assert executor.performed[1:] == ["click 1", "click 1"]
    assert [step.forced for step in report.steps] == ["", "", ""], "nothing was forced"


@pytest.mark.mock_llm
def test_different_actions_never_look_stuck():
    """Alternating between two moves is progress as far as this rule is concerned."""
    controller, _, _ = a_controller(
        planner=StubPlanner(A_CLICK, {"action": "scroll", "direction": "down"}, A_CLICK, DONE),
        max_steps=6,
    )

    report = controller.run(TASK, start_url=START)

    assert report.outcome.ok
    assert all(step.forced == "" for step in report.steps)


# --- What the planner is told ----------------------------------------------


@pytest.mark.mock_llm
def test_the_planner_is_given_the_run_so_far():
    """Memory reaches the prompt: the loop is what wires them together."""
    controller, planner, _ = a_controller(planner=StubPlanner(A_CLICK, A_CLICK, DONE))

    controller.run(TASK, start_url=START)

    first, second = planner.asked[0], planner.asked[1]
    assert first["history"] == (), "nothing has happened yet on step 1"
    assert second["history"], "and by step 2 it has"
    assert "click 1" in second["history"][0]


@pytest.mark.mock_llm
def test_a_refused_action_sends_the_complaint_back_to_the_planner():
    """The retry is told what to fix, which is the whole point of refusing in words."""
    controller, planner, _ = a_controller(
        planner=StubPlanner({"action": "click", "element_id": 99}, DONE),
        retry_budget=2,
    )

    controller.run(TASK, start_url=START)

    assert planner.asked[1]["feedback"], "the second attempt carries the complaint"
    assert "99" in planner.asked[1]["feedback"][0]


@pytest.mark.mock_llm
def test_step_zero_is_not_written_into_the_history():
    """The planner is about to see that page in full; "you are where you started" is not news."""
    memory = Memory()
    controller, _, _ = a_controller(planner=StubPlanner(DONE), memory=memory)

    controller.run(TASK, start_url=START)

    assert memory.steps_recorded == 1, "one planned step, not two"
    assert "navigate" not in "\n".join(memory.history())


# --- Skill replay falls back here -------------------------------------------


@pytest.mark.mock_llm
def test_a_failed_skill_replay_hands_back_a_fresh_observation():
    """The browser is somewhere nobody planned for, so the page is read again."""
    looks = []

    def observe(label):
        """Count the looks and return a page.

        Args:
            label: The screenshot label, recorded.

        Returns:
            tuple: The page and no gate decision.
        """
        looks.append(label)
        return a_page(), None

    replayed = []

    def replay(action, observation):
        """Fail every replay, loudly enough to be recorded.

        Args:
            action: The skill invocation.
            observation: The page before the replay.

        Returns:
            SimpleNamespace: A failed replay result.
        """
        replayed.append(action.skill)
        return SimpleNamespace(ok=False, detail="step 2 of the skill found no search box")

    class Library:
        """A one-skill library, so the validator permits use_skill."""

        def find(self, name):
            """Return the one skill, whatever is asked for.

            Args:
                name: The skill's name.

            Returns:
                SimpleNamespace: The skill entry.
            """
            return SimpleNamespace(
                name=name,
                domain="localhost",
                active=True,
                required_params=(),
                optional_params=(),
            )

    controller = Controller(
        planner=StubPlanner({"action": "use_skill", "skill": "find_hours", "params": {}}, DONE),
        validator=Validator(Allowlist.derive(START), skills=Library()),
        executor=StubExecutor(),
        memory=Memory(),
        max_steps=4,
        retry_budget=1,
        replay=replay,
        observe=observe,
    )

    report = controller.run(TASK, start_url=START)

    assert replayed == ["find_hours"]
    # Step 0's look, then the fresh one the failed replay forces. The `done`
    # that follows ends the run without looking again, which is right: nothing
    # is decided from a page after the run is over.
    assert looks == ["step-00", "step-01"]
    assert report.steps[0].forced == "step 2 of the skill found no search box"
    assert report.outcome.ok, "the run carried on and finished"


# --- What the run leaves behind --------------------------------------------


@pytest.mark.mock_llm
def test_the_report_carries_every_step_and_what_it_cost():
    """The trace writer and the results table both read this, and neither re-derives it."""
    controller, _, _ = a_controller(planner=StubPlanner(A_CLICK, DONE))

    report = controller.run(TASK, start_url=START)

    assert isinstance(report, RunReport)
    assert [step.number for step in report.steps] == [1, 2]
    assert report.cost.calls == 2, "one model call per step"
    assert report.cost.prompt_tokens == 200


@pytest.mark.mock_llm
def test_a_refused_attempt_is_still_billed_to_the_run():
    """A wasted call is still a call, and a cost figure that hid them would flatter itself."""
    controller, _, _ = a_controller(
        planner=StubPlanner({"action": "click", "element_id": 99}, DONE),
        retry_budget=2,
    )

    report = controller.run(TASK, start_url=START)

    assert report.steps[0].attempts == 2, "one refused attempt and one accepted"
    assert report.cost.calls == 2, "and the run is billed for both"
    assert report.steps[0].records[0].prompt_tokens == 100, "including the wasted one"


@pytest.mark.mock_llm
def test_the_run_writes_itself_as_plain_data():
    """A trace is read after the run, so the report has to survive being written down."""
    controller, _, _ = a_controller(planner=StubPlanner(A_CLICK, DONE))

    record = controller.run(TASK, start_url=START).to_dict()

    assert record["task"] == TASK
    assert record["outcome"]["category"] == SUCCESS
    assert record["final_action"] == {
        "action": "done",
        "status": "success",
        "answer": "09:00 to 17:30",
        "reason": None,
    }
    assert [step["number"] for step in record["steps"]] == [1, 2]
    assert record["cost"]["calls"] == 2


@pytest.mark.mock_llm
@pytest.mark.parametrize(
    ("category", "controller_for"),
    [
        (STEP_CAP, lambda: a_controller(max_steps=2)[0]),
        (
            GAVE_UP,
            lambda: a_controller(
                planner=StubPlanner({"action": "done", "status": "failed", "reason": "beaten"})
            )[0],
        ),
        (
            VALIDATION_EXHAUSTED,
            lambda: a_controller(planner=StubPlanner(failures=["not_json"] * 4))[0],
        ),
        (
            NAVIGATION_DEAD_END,
            lambda: a_controller(executor=StubExecutor(NAVIGATION_ERROR))[0],
        ),
    ],
)
def test_each_failure_category_reaches_the_trace(category, controller_for):
    """The task's own test: all four are reachable, and all four are written down.

    A category that only exists in the code is a category no results table will
    ever show, so the assertion is on the *written* form rather than on the
    outcome object.
    """
    record = controller_for().run(TASK, start_url=START).to_dict()

    assert record["outcome"]["category"] == category
    assert record["outcome"]["ok"] is False
    assert record["final_action"]["reason"].startswith(f"{category}: ")


@pytest.mark.mock_llm
def test_a_run_that_ended_without_the_agent_choosing_still_ends_explicitly():
    """The action log always ends with an action, even one the agent never chose."""
    controller, _, _ = a_controller(max_steps=2)

    report = controller.run(TASK, start_url=START)

    assert report.outcome.as_done().status == "failed"
    assert report.outcome.as_done().reason.startswith(f"{STEP_CAP}: ")


@pytest.mark.unit
def test_a_controller_with_no_way_to_see_the_page_is_refused():
    """Better to fail here than on the first step of a run somebody was waiting for."""
    with pytest.raises(ValueError, match="observe"):
        Controller(
            planner=StubPlanner(),
            validator=Validator(Allowlist.derive(START)),
            executor=StubExecutor(),
            memory=Memory(),
            max_steps=5,
            retry_budget=1,
        )


# --- The whole thing, against a real browser -------------------------------


@pytest.mark.fixture
def test_a_whole_run_against_a_real_page(task_session, fixture_site, tmp_path):
    """Every part, fitted together, with a real Chromium and the fixture site.

    The stubs above prove the loop's branches. This proves the parts fit: a real
    navigation, a real observation with real element numbering, an action
    through the validator, a real click, and a real answer read off the page
    the click produced.
    """
    start = fixture_site.url_for("search.html")
    executor = Executor(task_session)
    planner = StubPlanner(
        {"action": "fill", "element_id": 7, "text": "membership"},
        {"action": "press", "key": "Enter"},
        {"action": "done", "status": "success", "answer": "the catalogue was searched"},
    )
    controller = Controller(
        planner=planner,
        validator=Validator(Allowlist.derive(start)),
        executor=executor,
        memory=Memory(),
        max_steps=6,
        retry_budget=1,
        config=Config(),
        screenshot_dir=tmp_path,
    )

    report = controller.run(TASK, start_url=start)

    assert report.outcome.ok, report.summary()
    assert report.outcome.answer == "the catalogue was searched"
    assert "q=membership" in executor.page.url, "the form really was submitted"
    assert report.steps[0].observation.elements, "a real page was really observed"
    assert planner.asked[1]["history"], "and the first step reached the second's prompt"


@pytest.mark.fixture
def test_the_loop_observes_the_page_it_is_actually_on(task_session, fixture_site, tmp_path):
    """A click that opened a tab moves the run, and the next step must see the tab.

    The executor follows a new tab; the controller asks it for the current page
    rather than remembering one. Without that, every step after a popup would
    describe the page the site had finished with.
    """
    start = fixture_site.url_for("popup.html")
    page = task_session.page
    page.goto(start)
    target = next(
        element.id
        for element in observe_page(page).elements
        if element.name == "Reading room refurbishment"
    )
    planner = StubPlanner(
        {"action": "click", "element_id": target},
        {"action": "done", "status": "success", "answer": "read the notice"},
    )
    controller = Controller(
        planner=planner,
        validator=Validator(Allowlist.derive(start)),
        executor=Executor(task_session),
        memory=Memory(),
        max_steps=4,
        retry_budget=1,
        config=Config(),
        screenshot_dir=tmp_path,
    )

    report = controller.run(TASK, start_url=start)

    assert report.outcome.ok, report.summary()
    assert report.steps[1].observation.url.endswith("article.html"), (
        "the second step saw the new tab"
    )


@pytest.mark.fixture
def test_a_screenshot_is_taken_when_the_gate_asks_for_one(task_session, fixture_site, tmp_path):
    """The perception pieces are composed in one place, and this is that place working."""
    start = fixture_site.url_for("search.html")
    controller = Controller(
        planner=StubPlanner({"action": "done", "status": "success", "answer": "x"}),
        validator=Validator(Allowlist.derive(start)),
        executor=Executor(task_session),
        memory=Memory(),
        max_steps=3,
        retry_budget=1,
        config=Config(),
        gate=ScreenshotGate(min_elements=99, max_dom_failures=2, supports_vision=True),
        screenshot_dir=tmp_path,
    )

    report = controller.run(TASK, start_url=start)

    observation = report.steps[0].observation
    assert observation.screenshot is not None, "the gate asked for one"
    assert observation.screenshot.exists()
    assert report.steps[0].gate.send_screenshot
    assert not list(Path(tmp_path).glob("*-plain.png")), "the unmarked copy is scaffolding"
