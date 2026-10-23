"""Reading a whole run back out of its folder, and getting the run.

The trace writer's claim is that a run folder holds enough to replay or audit
the run without re-executing it. That is a claim about a *reader*, and it can
only be checked one way: run something, write the folder, read the folder, and
compare what came back against what went in. Every test here is a version of
that, and the comparison is deliberately against the run's own report rather
than against a fixture — a fixture only proves the reader agrees with whoever
wrote the fixture.

**Why the round trip as well as the committed example.** The committed
example run in ``runs/example/`` is read back by this reader in
``tests/test_example_run.py``. But a committed example can only show the
reader agrees with one stored folder. The folders here are produced by the real
writer, from real runs through the real loop, and checked against the report
those runs returned: that shows the reader agrees with the thing the folder was
made from.

**Two failures are worth as much as the successes.** A folder from a killed run
has to read as a killed run rather than as an error, because flushing per step
exists precisely so that folder is worth something. And a folder with a record
that cannot be read has to *fail*, because a reader that skips what it cannot
parse produces a confident answer about a run that did not happen that way.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from web_action_agent.actions import parse_action
from web_action_agent.allowlist import Allowlist
from web_action_agent.config import Config
from web_action_agent.controller import Controller
from web_action_agent.executor import OK, ExecutionResult
from web_action_agent.memory import Memory
from web_action_agent.model_client import CallRecord
from web_action_agent.observation import Observation
from web_action_agent.observer import Box, Element
from web_action_agent.planner import PlannerDecision
from web_action_agent.trace import (
    ACTION_LOG,
    COST_LOG,
    OUTCOME_FILE,
    RUN_FILE,
    TraceWriter,
    new_id,
    utc_now,
)
from web_action_agent.trace_reader import TraceReadError, cost_log_agrees_with_steps, read_run
from web_action_agent.validator import Validator

TASK = "Find the opening hours of the Hollow Lane branch."
"""str: The task every run here is given."""

START = "http://localhost:8000/search.html"
"""str: Where the runs open."""

A_CLICK = {"action": "click", "element_id": 1}
"""dict: A well-formed action a stub planner can return forever."""

A_SCROLL = {"action": "scroll", "direction": "down"}
"""dict: A second action, so a run's sequence is worth comparing."""

FINISH = {"action": "done", "status": "success", "answer": "09:00 to 17:30"}
"""dict: The agent finishing the task."""


def a_page(url: str = START) -> Observation:
    """Build an observation to decide against.

    Args:
        url: The page's address.

    Returns:
        Observation: A page with one element and both text channels filled, so
        that a round trip has something in every field to lose.
    """
    return Observation(
        url=url,
        title="Search",
        elements=(
            Element(
                id=1,
                tag="a",
                role="link",
                name="Hollow Lane",
                text="Hollow Lane",
                box=Box(x=0, y=20, width=80, height=18),
            ),
        ),
        text="Opening hours are listed by branch.",
        text_truncated=True,
        screenshot_on_disk=Path("screenshots/step-00.png"),
    )


class StubPlanner:
    """A planner that answers from a script and bills every answer."""

    def __init__(self, *replies: dict[str, Any]) -> None:
        """Set up the script.

        Args:
            *replies: Actions to return, as plain data. The last repeats.
        """
        self._replies = list(replies) or [A_CLICK]
        self.asked = 0

    def decide(self, **kwargs: Any) -> PlannerDecision:
        """Answer one step, with a cost record attached.

        Args:
            **kwargs: What the controller passed, ignored by the script.

        Returns:
            PlannerDecision: The next scripted answer.
        """
        self.asked += 1
        record = CallRecord(
            call_id=new_id("call"),
            started_at=utc_now(),
            ended_at=utc_now(),
            profile="local",
            model="stub",
            vision=False,
            prompt_tokens=900,
            completion_tokens=20,
            latency_s=0.25,
            finish_reason="stop",
            cost_usd=0.0004,
        )
        reply = self._replies[min(self.asked - 1, len(self._replies) - 1)]
        return PlannerDecision(action=parse_action(reply), failure=None, text="{}", record=record)


class StubExecutor:
    """An executor that succeeds at everything and touches no browser."""

    def __init__(self) -> None:
        """Set up the stub."""
        self.page = SimpleNamespace(url=START)

    def execute(self, action: Any, *, observation: Any = None) -> ExecutionResult:
        """Perform one action, successfully.

        Args:
            action: The action to perform.
            observation: The page it was planned against. Unused here.

        Returns:
            ExecutionResult: A success.
        """
        return ExecutionResult(
            action=action, status=OK, detail="", url=START, elapsed_s=0.12, waited_s=0.01
        )


def a_run(folder: Path, *replies: dict[str, Any], planner: Any = None) -> Any:
    """Drive one scripted run into a real trace folder.

    The run is real in every way that matters to this file: the real loop, the
    real writer, the real records. Only the planner and the browser are stood
    in for, because what is being tested is whether a folder can be read back,
    not what the agent decides or what a page does.

    Args:
        folder: Where the run's files go.
        *replies: The actions the planner answers with, in order.
        planner: A planner to use instead of a scripted one, for the test that
            interrupts a run.

    Returns:
        RunReport: What the run produced, to compare the folder against.
    """
    trace = TraceWriter.open(
        folder, task=TASK, start_url=START, config=Config(start_url=START), text_only=True
    )
    controller = Controller(
        planner=planner or StubPlanner(*replies),
        validator=Validator(Allowlist.derive(START)),
        executor=StubExecutor(),
        memory=Memory(),
        max_steps=5,
        retry_budget=1,
        observe=lambda label: (a_page(), None),
        trace=trace,
    )
    return controller.run(TASK, start_url=START)


# --- The round trip ---------------------------------------------------------


@pytest.mark.unit
def test_a_folder_reads_back_as_the_run_that_wrote_it(tmp_path: Path) -> None:
    """The claim, checked the only way it can be: written, read, compared.

    Not "the reader produces something plausible" but "the reader produces the
    run". Every step, every action, every result, every call — compared against
    the report the loop returned, field for field. Anything a folder cannot
    carry across this comparison is a hole in the folder.
    """
    folder = tmp_path / "run"
    report = a_run(folder, A_CLICK, A_SCROLL, FINISH)

    rebuilt = read_run(folder)

    assert rebuilt.steps == report.steps, "the steps came back exactly as they went in"
    assert rebuilt.outcome == report.outcome
    assert rebuilt.cost == report.cost
    assert rebuilt.as_report().summary() == report.summary()


@pytest.mark.unit
def test_the_run_describes_itself_without_being_told(tmp_path: Path) -> None:
    """A folder handed to somebody else has to say what it is on its own."""
    folder = tmp_path / "run"
    a_run(folder, FINISH)

    rebuilt = read_run(folder)

    assert rebuilt.task == TASK
    assert rebuilt.start_url == START
    assert rebuilt.run_id.startswith("run-")
    assert rebuilt.text_only is True, "the mode the run was in, stated rather than inferred"
    assert rebuilt.config["max_steps"] == Config().max_steps
    assert rebuilt.folder == folder


@pytest.mark.unit
def test_the_step_sequence_and_the_cost_total_match_the_run(tmp_path: Path) -> None:
    """The three numbers an auditor checks first, from the folder alone.

    How many steps, what the agent did in them, and what it spent. A trace that
    could not answer these without the run being re-executed would not be
    evidence of anything.
    """
    folder = tmp_path / "run"
    report = a_run(folder, A_CLICK, A_SCROLL, FINISH)

    rebuilt = read_run(folder)

    assert len(rebuilt.steps) == len(report.steps) == 3
    assert [step.action.describe() for step in rebuilt.steps] == [
        "click 1",
        "scroll down",
        "done success: 09:00 to 17:30",
    ]
    assert rebuilt.cost.calls == report.cost.calls == 3
    assert rebuilt.cost.total_tokens == report.cost.total_tokens
    assert rebuilt.cost.cost_usd == report.cost.cost_usd


@pytest.mark.unit
def test_every_step_keeps_its_id_and_its_timestamps(tmp_path: Path) -> None:
    """The fields the trace schema exists to guarantee, checked after a read.

    Writing them is one thing; a reader that dropped them on the way back would
    make the guarantee worthless, since nothing downstream reads the file
    directly.
    """
    folder = tmp_path / "run"
    a_run(folder, A_CLICK, FINISH)

    for step in read_run(folder).steps:
        assert step.step_id.startswith("step-")
        assert step.started_at.endswith("Z") and step.ended_at.endswith("Z")
        for call in step.records:
            assert call.call_id.startswith("call-")
            assert call.started_at <= call.ended_at


@pytest.mark.unit
def test_the_page_each_step_saw_comes_back_whole(tmp_path: Path) -> None:
    """An audit's first question is what the agent was looking at.

    Both text channels and the element list, not a summary of them — the
    numbered elements are how an action like ``click 1`` is read at all, so a
    trace that lost them would record decisions nobody could check.
    """
    folder = tmp_path / "run"
    a_run(folder, A_CLICK, FINISH)

    observation = read_run(folder).steps[0].observation

    assert observation == a_page()
    assert observation.element_by_id(1) is not None
    assert observation.text_truncated is True, "including how the page was captured"


@pytest.mark.unit
def test_the_two_views_of_the_calls_agree(tmp_path: Path) -> None:
    """The cost log and the steps hold the same calls, and a reader can ask.

    They are written from one rendering, so they cannot drift when written. A
    folder that has since been copied, truncated or edited by hand is a
    different matter, and an auditor should be able to check rather than assume.
    """
    folder = tmp_path / "run"
    a_run(folder, A_CLICK, A_SCROLL, FINISH)

    rebuilt = read_run(folder)

    assert cost_log_agrees_with_steps(rebuilt) is True

    # Truncate the cost log by a line, as a partial copy would, and it no
    # longer agrees -- which is the point of being able to ask.
    lines = (folder / COST_LOG).read_text(encoding="utf-8").splitlines()
    (folder / COST_LOG).write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")
    assert cost_log_agrees_with_steps(read_run(folder)) is False


# --- A run that did not finish ----------------------------------------------


class InterruptedPlanner(StubPlanner):
    """A planner whose process is taken away from it on the third step."""

    def decide(self, **kwargs: Any) -> PlannerDecision:
        """Answer twice, then be interrupted.

        Args:
            **kwargs: What the controller passed, ignored by the script.

        Returns:
            PlannerDecision: The scripted answer, for the first two steps.

        Raises:
            KeyboardInterrupt: On the third step, which is what a Ctrl-C is.
        """
        if self.asked >= 2:
            raise KeyboardInterrupt
        return super().decide(**kwargs)


@pytest.mark.unit
def test_a_killed_run_reads_as_a_killed_run_rather_than_an_error(tmp_path: Path) -> None:
    """The whole reason steps are flushed as they happen.

    A folder with no verdict is the expected shape of an interrupted run, not a
    corrupt folder. A reader that refused it would throw the guarantee away at
    exactly the moment it was supposed to pay off.
    """
    folder = tmp_path / "run"
    with pytest.raises(KeyboardInterrupt):
        a_run(folder, A_CLICK, planner=InterruptedPlanner(A_CLICK))

    rebuilt = read_run(folder)

    assert rebuilt.finished is False
    assert rebuilt.outcome is None
    assert [step.number for step in rebuilt.steps] == [1, 2], "every step that completed"
    assert not (folder / OUTCOME_FILE).exists()
    assert "interrupted" in rebuilt.summary()
    assert rebuilt.cost.calls == 2, "and what it had spent by then"


# --- Folders that cannot be read --------------------------------------------


@pytest.mark.unit
def test_a_folder_that_is_not_a_run_says_so(tmp_path: Path) -> None:
    """Two ways of not being a run folder, and both are named rather than guessed."""
    with pytest.raises(TraceReadError, match="not a run folder"):
        read_run(tmp_path / "nowhere")

    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(TraceReadError, match=RUN_FILE):
        read_run(empty)


@pytest.mark.unit
def test_a_folder_missing_what_it_was_asked_to_do_is_refused(tmp_path: Path) -> None:
    """A run with no task is not a run with an empty task.

    Filling it in with a blank would produce a readable folder describing
    something nobody ever asked for, which is worse than refusing it.
    """
    folder = tmp_path / "run"
    a_run(folder, FINISH)
    described = json.loads((folder / RUN_FILE).read_text(encoding="utf-8"))
    del described["task"]
    (folder / RUN_FILE).write_text(json.dumps(described), encoding="utf-8")

    with pytest.raises(TraceReadError, match="task"):
        read_run(folder)


@pytest.mark.unit
def test_a_step_that_cannot_be_read_fails_and_names_itself(tmp_path: Path) -> None:
    """A reader that skipped a bad record would answer confidently and wrongly.

    The step number is in the message because "step 2 cannot be read" is
    something a person can act on and "a record is broken" is not.
    """
    folder = tmp_path / "run"
    a_run(folder, A_CLICK, A_SCROLL, FINISH)

    lines = (folder / ACTION_LOG).read_text(encoding="utf-8").splitlines()
    broken = json.loads(lines[1])
    # The timestamp the schema exists to guarantee, removed.
    del broken["started_at"]
    lines[1] = json.dumps(broken)
    (folder / ACTION_LOG).write_text("\n".join(lines) + "\n", encoding="utf-8")

    with pytest.raises(TraceReadError, match="step 2"):
        read_run(folder)


@pytest.mark.unit
def test_a_verdict_in_a_category_no_table_counts_is_refused(tmp_path: Path) -> None:
    """The same rule a fresh outcome is held to, applied on the way back in.

    A run recorded in a category nothing counts has, for reporting purposes,
    not ended — and a reader that accepted one would put it in a results table
    where it would silently never appear.
    """
    folder = tmp_path / "run"
    a_run(folder, FINISH)
    verdict = json.loads((folder / OUTCOME_FILE).read_text(encoding="utf-8"))
    verdict["outcome"]["category"] = "went_badly"
    (folder / OUTCOME_FILE).write_text(json.dumps(verdict), encoding="utf-8")

    with pytest.raises(TraceReadError, match="verdict"):
        read_run(folder)


@pytest.mark.unit
def test_the_verdict_is_read_from_the_category_not_from_the_ok_flag(tmp_path: Path) -> None:
    """One answer to "did this work", not two that could disagree.

    ``ok`` is written into the trace so a reader need not know which categories
    count as success. It is not read back, so a folder claiming a successful
    category and ``ok: false`` cannot produce an outcome that contradicts
    itself.
    """
    folder = tmp_path / "run"
    a_run(folder, FINISH)
    verdict = json.loads((folder / OUTCOME_FILE).read_text(encoding="utf-8"))
    verdict["outcome"]["ok"] = False
    (folder / OUTCOME_FILE).write_text(json.dumps(verdict), encoding="utf-8")

    outcome = read_run(folder).outcome

    assert outcome is not None
    assert outcome.ok is True, "derived from the category, which still says success"
