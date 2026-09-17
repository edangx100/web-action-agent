"""The run folder: written as the run happens, and readable if the run is killed.

Two promises are being checked here, and only one of them is about content.

The first is the **flush-per-step contract**. A trace that is assembled in
memory and written at the end is a trace that a Ctrl-C, an out-of-memory kill
or a benchmark worker hitting its wall clock erases entirely -- and those are
precisely the runs whose traces are worth the most. So two tests interrupt a
run rather than finishing it: one interrupts a real browser run mid-step, and
one kills the writing process outright with a signal it cannot catch. Both then
read the folder and expect every completed step to be in it.

The second is the **schema**. ``schemas/action_log.schema.json`` is the one
decision in this module that cannot be revised later: a field missing from it
is missing from every run already recorded under it, and no amount of later
code can put it back. So the timestamps, the ids and the parent links are
checked field by field, and a record without them is checked to *fail* -- a
schema that accepts everything would pass every other test here while
guaranteeing nothing.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from jsonschema import Draft202012Validator

from web_action_agent.actions import parse_action
from web_action_agent.allowlist import Allowlist
from web_action_agent.config import Config
from web_action_agent.controller import Controller, StepReport
from web_action_agent.executor import OK, ExecutionResult, Executor
from web_action_agent.memory import Memory
from web_action_agent.model_client import CallRecord
from web_action_agent.observation import Observation
from web_action_agent.observer import Box, Element
from web_action_agent.planner import PlannerDecision
from web_action_agent.screenshot_gate import ScreenshotGate
from web_action_agent.trace import (
    ACTION_LOG,
    CONFIG_FILE,
    COST_LOG,
    OUTCOME_FILE,
    RUN_FILE,
    TraceError,
    TraceWriter,
    action_log_schema,
    new_id,
    read_action_log,
    utc_now,
    validate_record,
)
from web_action_agent.validator import Validator

TASK = "Find the opening hours of the Hollow Lane branch."
"""str: The task every run here is given."""

START = "http://localhost:8000/search.html"
"""str: Where the stub-driven runs open."""

A_CLICK = {"action": "click", "element_id": 1}
"""dict: A well-formed action a stub planner can return forever."""

DONE = {"action": "done", "status": "success", "answer": "09:00 to 17:30"}
"""dict: The agent finishing the task."""

# --- Building a run without a browser ---------------------------------------


def a_page(url: str = START) -> Observation:
    """Build an observation to decide against.

    Args:
        url: The page's address.

    Returns:
        Observation: A page with one link on it, which is enough for a loop
        that is being tested for what it writes rather than what it sees.
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
    )


class StubPlanner:
    """A planner that answers from a script and bills every answer.

    The billing is the point: the cost records it returns are what end up in
    the cost log, so a planner here that reported no calls would let the
    parent-link tests pass without ever exercising a link.
    """

    def __init__(self, *replies: dict[str, Any]) -> None:
        """Set up the script.

        Args:
            *replies: Actions to return, as plain data. The last repeats for as
                long as the loop keeps asking.
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
            prompt_tokens=100,
            completion_tokens=10,
            latency_s=0.01,
            finish_reason="stop",
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
        return ExecutionResult(action=action, status=OK, detail="", url=START, elapsed_s=0.01)


def a_run(folder: Path, *replies: dict[str, Any], max_steps: int = 5) -> Any:
    """Drive one scripted run into a real trace folder.

    Args:
        folder: Where the run's files go.
        *replies: The actions the planner answers with, in order.
        max_steps: The step cap.

    Returns:
        RunReport: What the run produced, so a test can compare the report in
        memory against the folder on disk.
    """
    trace = TraceWriter.open(folder, task=TASK, start_url=START, config=Config(start_url=START))
    controller = Controller(
        planner=StubPlanner(*replies),
        validator=Validator(Allowlist.derive(START)),
        executor=StubExecutor(),
        memory=Memory(),
        max_steps=max_steps,
        retry_budget=1,
        observe=lambda label: (a_page(), None),
        trace=trace,
    )
    return controller.run(TASK, start_url=START)


# --- What the folder holds --------------------------------------------------


@pytest.mark.mock_llm
def test_a_finished_run_leaves_every_file_a_reader_needs(tmp_path: Path) -> None:
    """Five files, and each one answers a different question about the run."""
    folder = tmp_path / "run"

    a_run(folder, A_CLICK, DONE)

    assert (folder / RUN_FILE).is_file(), "what the run was asked to do"
    assert (folder / CONFIG_FILE).is_file(), "the settings it ran under"
    assert (folder / ACTION_LOG).is_file(), "what it saw and did, step by step"
    assert (folder / COST_LOG).is_file(), "what each model call cost"
    assert (folder / OUTCOME_FILE).is_file(), "the verdict"
    assert (folder / "screenshots").is_dir(), "where the annotated images go"


@pytest.mark.mock_llm
def test_the_run_is_described_before_its_first_step(tmp_path: Path) -> None:
    """A run that dies on its opening navigation still says what it was trying to do."""
    folder = tmp_path / "run"

    TraceWriter.open(folder, task=TASK, start_url=START, config=Config(start_url=START))

    described = json.loads((folder / RUN_FILE).read_text(encoding="utf-8"))
    assert described["task"] == TASK
    assert described["start_url"] == START
    assert described["run_id"].startswith("run-")
    assert described["opened_at"].endswith("Z")
    assert not (folder / ACTION_LOG).exists(), "nothing has happened yet"
    assert not (folder / OUTCOME_FILE).exists(), "and nothing has been decided"


@pytest.mark.mock_llm
def test_the_settings_are_recorded_and_carry_no_key(tmp_path: Path) -> None:
    """A published result has to be readable back against what produced it.

    The second half matters more than the first: run folders are meant to be
    shared, and a folder that carried an API key would be a folder nobody could
    safely attach to a paper.
    """
    folder = tmp_path / "run"

    a_run(folder, DONE)

    settings = json.loads((folder / CONFIG_FILE).read_text(encoding="utf-8"))
    assert settings["max_steps"] == Config().max_steps
    assert settings["start_url"] == START
    assert not [name for name in settings if "key" in name.lower()]


@pytest.mark.mock_llm
def test_the_verdict_names_the_answer_and_what_the_run_cost(tmp_path: Path) -> None:
    """The one file written at the end, holding the two things a table needs."""
    folder = tmp_path / "run"

    report = a_run(folder, A_CLICK, DONE)

    outcome = json.loads((folder / OUTCOME_FILE).read_text(encoding="utf-8"))
    assert outcome["outcome"]["category"] == "success"
    assert outcome["outcome"]["answer"] == DONE["answer"]
    assert outcome["steps"] == len(report.steps) == 2
    assert outcome["cost"]["calls"] == 2
    assert outcome["started_at"] <= outcome["ended_at"]


# --- Timestamps, ids, and the links between them ----------------------------


@pytest.mark.mock_llm
def test_every_step_carries_both_timestamps_and_a_stable_id(tmp_path: Path) -> None:
    """The schema's one irreversible decision, checked on a real run's output.

    Timestamps rather than durations, because a duration cannot be turned back
    into a position on the run's timeline -- and an id per step, because
    ``step 3`` is only unique inside one run and these records are meant to
    survive being collected with others.
    """
    folder = tmp_path / "run"

    a_run(folder, A_CLICK, A_CLICK, DONE)

    steps = read_action_log(folder)
    assert [step["number"] for step in steps] == [1, 2, 3]
    for step in steps:
        assert step["step_id"].startswith("step-")
        assert step["started_at"].endswith("Z") and step["ended_at"].endswith("Z")
        assert step["started_at"] <= step["ended_at"]
    assert len({step["step_id"] for step in steps}) == 3, "ids are per step, not per run"


@pytest.mark.mock_llm
def test_every_call_carries_its_own_span_and_names_its_step(tmp_path: Path) -> None:
    """A model call is placed on the run's timeline, and under the step it served.

    Both halves are needed to lay a run out: the timestamps say *when* the call
    happened, and the parent link says *which* step's work it was.
    """
    folder = tmp_path / "run"

    a_run(folder, A_CLICK, DONE)

    steps = read_action_log(folder)
    for step in steps:
        assert step["calls"], "a step that asked the planner made a call"
        for call in step["calls"]:
            assert call["call_id"].startswith("call-")
            assert call["parent_step_id"] == step["step_id"]
            assert call["started_at"] <= call["ended_at"]
            assert call["started_at"] >= step["started_at"]


@pytest.mark.mock_llm
def test_the_cost_log_is_the_calls_flattened_out_of_their_steps(tmp_path: Path) -> None:
    """One line per call, each naming its step, so the two files agree by construction."""
    folder = tmp_path / "run"

    a_run(folder, A_CLICK, A_CLICK, DONE)

    steps = read_action_log(folder)
    logged = [
        json.loads(line)
        for line in (folder / COST_LOG).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert logged == [call for step in steps for call in step["calls"]]
    assert {call["parent_step_id"] for call in logged} == {step["step_id"] for step in steps}


# --- The schema -------------------------------------------------------------


@pytest.mark.unit
def test_the_shipped_schema_is_a_valid_json_schema() -> None:
    """A typo in a keyword name makes a schema that validates nothing, silently."""
    Draft202012Validator.check_schema(action_log_schema())


@pytest.mark.mock_llm
def test_a_real_step_validates_against_the_shipped_schema(tmp_path: Path) -> None:
    """The record the controller writes and the schema readers trust are one shape."""
    folder = tmp_path / "run"

    a_run(folder, A_CLICK, DONE)

    for step in read_action_log(folder):
        validate_record(step)


@pytest.mark.mock_llm
@pytest.mark.parametrize("missing", ["step_id", "started_at", "ended_at"])
def test_a_step_without_its_id_or_its_times_is_refused(tmp_path: Path, missing: str) -> None:
    """The fields that cannot be backfilled are the fields that must be required.

    Args:
        tmp_path: A folder to run into.
        missing: The field to remove before validating.
    """
    folder = tmp_path / "run"
    a_run(folder, DONE)
    record = read_action_log(folder)[0]
    del record[missing]

    with pytest.raises(TraceError, match=missing):
        validate_record(record)


@pytest.mark.mock_llm
@pytest.mark.parametrize("missing", ["call_id", "parent_step_id", "started_at", "ended_at"])
def test_a_call_without_its_id_or_its_step_is_refused(tmp_path: Path, missing: str) -> None:
    """A call that cannot be placed or attributed is not worth recording.

    Args:
        tmp_path: A folder to run into.
        missing: The field to remove from the call before validating.
    """
    folder = tmp_path / "run"
    a_run(folder, DONE)
    record = read_action_log(folder)[0]
    del record["calls"][0][missing]

    with pytest.raises(TraceError, match=missing):
        validate_record(record)


@pytest.mark.unit
def test_a_timestamp_that_is_not_utc_is_refused() -> None:
    """Local time in one run and UTC in another cannot be put on one timeline."""
    record = a_minimal_record()
    record["started_at"] = "2026-09-17T08:00:00+08:00"

    with pytest.raises(TraceError, match="started_at"):
        validate_record(record)


@pytest.mark.unit
def test_a_record_with_an_unknown_field_is_refused() -> None:
    """A field nobody declared is a field no reader will read, written silently."""
    record = a_minimal_record()
    record["mystery"] = "where did this come from"

    with pytest.raises(TraceError):
        validate_record(record)


def a_minimal_record() -> dict[str, Any]:
    """Build the smallest action log record the schema accepts.

    Returns:
        dict[str, Any]: A conforming record, for the tests that break exactly
        one thing about it and expect the break to be caught.
    """
    step = StepReport(
        step_id=new_id("step"),
        started_at=utc_now(),
        ended_at=utc_now(),
        number=1,
        observation=a_page(),
    )
    record = step.to_dict()
    validate_record(record)
    return record


# --- A run that does not get to finish --------------------------------------


@pytest.mark.fixture
def test_a_run_interrupted_mid_step_keeps_every_completed_step(
    tmp_path: Path,
    task_session: Any,
    fixture_site: Any,
) -> None:
    """Ctrl-C on step three, and steps one and two are still on disk.

    Driven against a real browser and a real page on purpose: the step records
    that survive here carry genuine observations, so this also shows that what
    survives a kill is a *usable* record rather than a stub of one.
    """
    folder = tmp_path / "run"
    start = fixture_site.url_for("search.html")
    config = Config(start_url=start)
    trace = TraceWriter.open(folder, task=TASK, start_url=start, config=config)

    class InterruptedPlanner(StubPlanner):
        """A planner whose process is taken away from it on the third step."""

        def decide(self, **kwargs: Any) -> PlannerDecision:
            """Answer twice, then be interrupted.

            Args:
                **kwargs: What the controller passed, ignored by the script.

            Returns:
                PlannerDecision: The scripted answer, for the first two steps.

            Raises:
                KeyboardInterrupt: On the third step. This is what a Ctrl-C
                    really is, and the controller does not catch it -- so the
                    run ends with no verdict, exactly as a kill would.
            """
            if self.asked >= 2:
                raise KeyboardInterrupt
            return super().decide(**kwargs)

    controller = Controller(
        planner=InterruptedPlanner({"action": "scroll", "direction": "down"}),
        validator=Validator(Allowlist.derive(start)),
        executor=Executor(task_session, action_timeout_s=config.action_timeout_s),
        memory=Memory(),
        max_steps=10,
        retry_budget=1,
        screenshot_dir=trace.screenshot_dir,
        config=config,
        trace=trace,
    )

    with pytest.raises(KeyboardInterrupt):
        controller.run(TASK, start_url=start)

    steps = read_action_log(folder)
    assert [step["number"] for step in steps] == [1, 2], "the steps that finished, and no more"
    assert all(step["observation"]["url"].startswith(fixture_site.base_url) for step in steps)
    for step in steps:
        validate_record(step)
    assert not (folder / OUTCOME_FILE).exists(), "no verdict, because the run reached none"


@pytest.mark.fixture
def test_a_step_that_carried_an_image_leaves_it_beside_the_log(
    tmp_path: Path,
    task_session: Any,
    fixture_site: Any,
) -> None:
    """The picture the planner saw, in the folder, named by the record that used it.

    Written by the observer straight into the folder the writer owns, rather
    than copied there afterwards -- so the path in the action log is the path
    of the file on disk, and a reader can follow one to the other.

    The gate is forced open by asking for more elements than the page has: what
    is being tested is that an image *reaches the folder*, not the cost rule
    that decides when one is worth taking.
    """
    folder = tmp_path / "run"
    start = fixture_site.url_for("search.html")
    config = Config(start_url=start, screenshot_min_elements=1000)
    trace = TraceWriter.open(folder, task=TASK, start_url=start, config=config)

    controller = Controller(
        planner=StubPlanner(DONE),
        validator=Validator(Allowlist.derive(start)),
        executor=Executor(task_session, action_timeout_s=config.action_timeout_s),
        memory=Memory(),
        max_steps=3,
        retry_budget=1,
        gate=ScreenshotGate.from_config(config, supports_vision=True),
        screenshot_dir=trace.screenshot_dir,
        config=config,
        trace=trace,
    )
    controller.run(TASK, start_url=start)

    step = read_action_log(folder)[0]
    assert step["screenshot_decision"]["send_screenshot"] is True
    image = Path(step["observation"]["screenshot"])
    assert image.is_file(), "the log names an image that is really there"
    assert image.parent == trace.screenshot_dir
    assert image.read_bytes().startswith(b"\x89PNG"), "and it is a real PNG"


@pytest.mark.unit
def test_a_step_survives_the_process_being_killed_outright(tmp_path: Path) -> None:
    """SIGKILL gives no chance to flush, so the flush must already have happened.

    The interrupted-run test above still lets Python unwind and close its files.
    This one does not: the child is killed with a signal it cannot handle,
    catch, or clean up after. If ``write_step`` returned before the bytes
    reached the disk, the log read here would be empty.
    """
    folder = tmp_path / "run"
    script = tmp_path / "write_one_step_then_hang.py"
    script.write_text(WRITE_THEN_HANG, encoding="utf-8")

    child = subprocess.Popen(
        [sys.executable, str(script), str(folder)],
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        # The child prints only after `write_step` has returned, so reading this
        # line is what makes the kill land *after* the write and before any
        # tidy shutdown could have helped.
        assert child.stdout.readline().strip() == "written"
        os.kill(child.pid, signal.SIGKILL)
        child.wait(timeout=10)
    finally:
        child.kill()

    assert child.returncode == -signal.SIGKILL, "the child was killed, not asked to stop"
    steps = read_action_log(folder)
    assert [step["number"] for step in steps] == [1]
    validate_record(steps[0])


WRITE_THEN_HANG = '''\
"""Write one step to a run folder, say so, and then wait to be killed.

Run as a child process by the test above. It must not exit on its own: the
whole point is that the step is on disk before anything gets a chance to tidy
up after it.
"""

import sys
import time
from pathlib import Path

from web_action_agent.controller import StepReport
from web_action_agent.observation import Observation
from web_action_agent.observer import Box, Element
from web_action_agent.trace import TraceWriter, new_id, utc_now

folder = Path(sys.argv[1])
writer = TraceWriter.open(folder, task="killed", start_url="http://localhost:8000/")
writer.write_step(
    StepReport(
        step_id=new_id("step"),
        started_at=utc_now(),
        ended_at=utc_now(),
        number=1,
        observation=Observation(
            url="http://localhost:8000/",
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
        ),
    )
)
print("written", flush=True)
time.sleep(60)
'''
"""str: The child program the kill test runs. Kept beside the test that uses it."""


# --- Reading a folder back --------------------------------------------------


@pytest.mark.unit
def test_a_folder_with_no_log_reads_as_no_steps(tmp_path: Path) -> None:
    """A run killed before its first step finished, which is a true reading."""
    folder = tmp_path / "run"
    TraceWriter.open(folder, task=TASK, start_url=START)

    assert read_action_log(folder) == []


@pytest.mark.unit
def test_a_log_line_that_is_not_json_is_an_error_rather_than_a_skip(tmp_path: Path) -> None:
    """A half-written line would mean the flush-per-step contract had broken.

    Skipping it would hide exactly the failure this module exists to prevent,
    so the reader refuses the file instead.
    """
    folder = tmp_path / "run"
    TraceWriter.open(folder, task=TASK, start_url=START)
    (folder / ACTION_LOG).write_text('{"number": 1}\n{"number":\n', encoding="utf-8")

    with pytest.raises(TraceError, match="line 2"):
        read_action_log(folder)
