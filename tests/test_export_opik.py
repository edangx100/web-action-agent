"""Exporting finished runs to Opik: a copy for viewing, never the record.

``waa export --to opik`` reads run folders that runs have already written and
logs them to Opik: a trace per run, a span per step with its screenshot, a
span per model call and per browser action, and an experiment per arm. These
tests hold what makes that copy trustworthy and harmless:

- **On the run's timeline.** Every span carries the times the run recorded,
  not the time of the export, and its id is made from the run's own ids.
- **Shared names, no content.** Spans use the OpenTelemetry names for AI
  systems where they exist, and carry no prompt, page text or model reply.
- **One experiment per arm**, each linking its runs to the tasks they ran.
- **Optional.** Without the package or an Opik key the export says so, and
  nothing else in the project needs either.

The export is tested against a stand-in for the Opik client, so no account,
network or package is needed. The committed example run in ``runs/example/``
is exported as it is. The other cases start from a small made-up run, copied
and given pictures for some of its steps.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import types
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from web_action_agent import cli
from web_action_agent.eval_harness import SWEEP_FILE, VERDICT_FILE
from web_action_agent.export_opik import (
    API_KEY_ENV,
    CHAT,
    EXECUTE_TOOL,
    FINISH_REASONS,
    INPUT_TOKENS,
    INVOKE_AGENT,
    OPERATION,
    OUTPUT_TOKENS,
    PROVIDER,
    REQUEST_MODEL,
    TOOL_NAME,
    ExportError,
    OpikKit,
    connect,
    export,
    find_targets,
    item_id,
    parse_time,
    record_id,
)
from web_action_agent.trace import ACTION_LOG, SCREENSHOT_DIR

#: The committed example run: a real run, with a real screenshot per step.
EXAMPLE = Path(__file__).resolve().parent.parent / "runs" / "example"

#: A small made-up run, for what the example cannot show: a step that types
#: text, a failed verdict, and runs grouped into sweeps.
FIXTURE = Path(__file__).parent / "fixtures" / "traces" / "search-and-open"

#: Text from the fixture run that must never leave the run folder: the task's
#: wording, the page's words, what the agent typed, and its answer.
PRIVATE_TEXT = (
    "find the record about membership and open it",
    "Records are matched on title and subject tags",
    "membership",
    "Membership application",
)

#: The fixture's steps that are given a screenshot (three of its four).
WITH_SCREENSHOTS = (1, 2, 3)

#: A made-up commit, as a sweep records one.
COMMIT = "0123456789ab"


# --- A run folder to export, and a stand-in for Opik ------------------------------------


def a_run(folder: Path, *, screenshots: tuple[int, ...] = WITH_SCREENSHOTS) -> Path:
    """Copy the fixture run, giving some of its steps a screenshot.

    The steps record their screenshot's path as a run started from the
    project folder does -- relative to where the run was started -- so the
    exporter has to find it in the run's own screenshot folder.

    Args:
        folder: Where the run goes.
        screenshots: The step numbers that get a picture.

    Returns:
        Path: The run folder.
    """
    shutil.copytree(FIXTURE, folder)
    (folder / SCREENSHOT_DIR).mkdir()
    steps = [json.loads(line) for line in (folder / ACTION_LOG).read_text().splitlines()]
    for step in steps:
        if step["number"] in screenshots:
            name = f"step-{step['number']:02d}.png"
            (folder / SCREENSHOT_DIR / name).write_bytes(b"\x89PNG\r\n\x1a\n" + b"0" * 16)
            step["observation"]["screenshot_on_disk"] = f"runs/somewhere/{SCREENSHOT_DIR}/{name}"
    (folder / ACTION_LOG).write_text("".join(json.dumps(s) + "\n" for s in steps))
    return folder


def a_verdict(folder: Path, *, status: str = "fail", category: str | None = "navigation") -> None:
    """Give a run the benchmark's mark, with the judge's call that named it."""
    record = {
        "call_id": "call-judge-1",
        "started_at": "2026-09-28T14:19:45.000Z",
        "ended_at": "2026-09-28T14:19:47.500Z",
        "profile": "openrouter-judge",
        "model": "google/gemini-3.8-flash",
        "vision": True,
        "prompt_tokens": 2000,
        "completion_tokens": 40,
        "latency_s": 2.5,
        "finish_reason": "stop",
        "cost_usd": 0.0017,
    }
    verdict = {
        "status": status,
        "reason": "Membership application was not the right page",
        "agent_outcome": "success",
        "category": category,
        "categorized_by": {
            "outcome": status,
            "justification": "Records are matched on title and subject tags, it says",
            "category": category,
            "model": "google/gemini-3.8-flash",
            "record": record,
        },
        "marked_at": "2026-09-28T14:19:48+00:00",
    }
    (folder / VERDICT_FILE).write_text(json.dumps(verdict))


def a_sweep(root: Path, name: str, *, tasks: tuple[str, ...] = ("task-a", "task-b")) -> Path:
    """Write a sweep folder holding one marked run per task."""
    sweep = root / name
    sweep.mkdir(parents=True)
    spec = {
        "suite": "sg-gov-v1",
        "profile": name.removeprefix("modality-"),
        "model": "nvidia/nemotron-3-nano-30b-a3b",
        "library": "cold",
        "judge_profile": "openrouter-judge",
        "started_at": "2026-09-28T14:19:00+00:00",
        "vision_model": "",
        "code_versions": [COMMIT],
    }
    (sweep / SWEEP_FILE).write_text(json.dumps(spec))
    for task in tasks:
        run = a_run(sweep / task / "trial-1")
        a_verdict(run)
    return sweep


class Recorded:
    """Something the stand-in client returns, with the id it was given."""

    def __init__(self, kwargs: dict[str, Any]) -> None:
        """Keep the call's arguments."""
        self.kwargs = kwargs
        self.id = kwargs.get("id")


class StubDataset:
    """A stand-in dataset that keeps what is inserted."""

    def __init__(self) -> None:
        """Start empty."""
        self.items: dict[str, dict[str, Any]] = {}

    def insert(self, items: list[dict[str, Any]]) -> None:
        """Keep each item under its id, as Opik does: once."""
        for item in items:
            self.items[item["id"]] = item


class StubExperiment:
    """A stand-in experiment that keeps the links inserted into it."""

    def __init__(self, name: str, dataset_name: str, config: dict[str, Any]) -> None:
        """Keep how it was made."""
        self.name = name
        self.dataset_name = dataset_name
        self.config = config
        self.links: list[tuple[str, str]] = []

    def insert(self, references: list[tuple[str, str]]) -> None:
        """Keep the links."""
        self.links.extend(references)


class StubClient:
    """A stand-in for the Opik client that records everything sent to it."""

    def __init__(self) -> None:
        """Start with nothing sent."""
        self.traces: list[dict[str, Any]] = []
        self.spans: list[dict[str, Any]] = []
        self.datasets: dict[str, StubDataset] = {}
        self.experiments: list[StubExperiment] = []
        self.flushed = 0

    def trace(self, **kwargs: Any) -> Recorded:
        """Record a trace."""
        self.traces.append(kwargs)
        return Recorded(kwargs)

    def span(self, **kwargs: Any) -> Recorded:
        """Record a span."""
        self.spans.append(kwargs)
        return Recorded(kwargs)

    def get_or_create_dataset(self, name: str, description: str | None = None) -> StubDataset:
        """Find or make a dataset."""
        return self.datasets.setdefault(name, StubDataset())

    def create_experiment(
        self,
        dataset_name: str,
        name: str | None = None,
        experiment_config: dict[str, Any] | None = None,
    ) -> StubExperiment:
        """Make an experiment."""
        experiment = StubExperiment(name or "", dataset_name, experiment_config or {})
        self.experiments.append(experiment)
        return experiment

    def flush(self, timeout: int | None = None) -> None:
        """Count the flushes."""
        self.flushed += 1


def a_kit() -> tuple[OpikKit, StubClient]:
    """A kit around the stand-in client, with plain stand-ins for Opik's types."""
    client = StubClient()
    kit = OpikKit(
        client=client,
        attachment=lambda path: ("attachment", path.name),
        item_reference=lambda item, trace: (item, trace),
    )
    return kit, client


def step_spans(client: StubClient) -> list[dict[str, Any]]:
    """The spans that stand for steps, in the order they were sent."""
    return [s for s in client.spans if s["name"].startswith("step ")]


def export_one(tmp_path: Path) -> tuple[StubClient, Path]:
    """Export the fixture run, as a run outside any sweep."""
    run = a_run(tmp_path / "run")
    kit, client = a_kit()
    export(kit, find_targets([run]), log=lambda line: None)
    return client, run


def recorded_steps(run: Path) -> list[dict[str, Any]]:
    """The fixture's step records, as written."""
    return [json.loads(line) for line in (run / ACTION_LOG).read_text().splitlines()]


# --- The committed example run ----------------------------------------------------------
#
# The real run in `runs/example/`, exported as it is: five steps on a live
# site, each with its own screenshot. The made-up fixture run further down
# covers what the example has none of (typed text, a failed verdict, sweeps).


def export_example() -> StubClient:
    """Export the committed example run to the stand-in client.

    Returns:
        StubClient: What was sent.
    """
    kit, client = a_kit()
    export(kit, find_targets([EXAMPLE]), log=lambda line: None)
    return client


@pytest.mark.unit
def test_the_example_run_exports_a_span_per_step_with_its_screenshot() -> None:
    client = export_example()
    steps = step_spans(client)

    assert [s["name"] for s in steps] == [
        "step 1: click",
        "step 2: click",
        "step 3: click",
        "step 4: click",
        "step 5: done",
    ]
    # Each step's picture is the page it acted on, saved before the step, so
    # step 1 carries step-00.png.
    assert [s["attachments"] for s in steps] == [
        [("attachment", f"step-{n:02d}.png")] for n in range(5)
    ]


@pytest.mark.unit
def test_the_example_runs_spans_carry_its_recorded_times() -> None:
    client = export_example()

    for span, step in zip(step_spans(client), recorded_steps(EXAMPLE), strict=True):
        assert span["start_time"] == parse_time(step["started_at"])
        assert span["end_time"] == parse_time(step["ended_at"])


@pytest.mark.unit
def test_the_example_runs_pass_is_its_score() -> None:
    trace = export_example().traces[0]

    assert trace["feedback_scores"] == [{"name": "passed", "value": 1.0}]
    assert trace["tags"] == ["pass"], "no arm: the example is not part of a sweep"


@pytest.mark.unit
def test_no_text_from_the_example_run_leaves_it() -> None:
    # The task's wording, the start of every page's text, the names of the
    # links the agent clicked, and its answer: all of it stays in the folder.
    steps = recorded_steps(EXAMPLE)
    run = json.loads((EXAMPLE / "run.json").read_text(encoding="utf-8"))
    private = [run["task"], "$3,350"]
    private += [step["observation"]["text"][:60] for step in steps]
    private += [
        element["name"]
        for step in steps
        for element in step["observation"]["elements"]
        if element["id"] == step["action"].get("element_id")
    ]

    sent = json.dumps(export_example().__dict__, default=str)

    for text in private:
        assert text not in sent, f"{text!r} left the run folder"


# --- On the run's timeline ---------------------------------------------------------------


@pytest.mark.unit
def test_one_span_per_step(tmp_path: Path) -> None:
    client, _ = export_one(tmp_path)

    assert [s["name"] for s in step_spans(client)] == [
        "step 1: fill",
        "step 2: press",
        "step 3: click",
        "step 4: done",
    ]
    assert len(client.traces) == 1, "and one trace for the whole run"


@pytest.mark.unit
def test_one_attachment_per_screenshot(tmp_path: Path) -> None:
    client, _ = export_one(tmp_path)

    attached = [s["attachments"] for s in step_spans(client)]
    assert attached == [
        [("attachment", "step-01.png")],
        [("attachment", "step-02.png")],
        [("attachment", "step-03.png")],
        None,
    ], "each picture on its own step; a step with none gets none"


@pytest.mark.unit
def test_step_spans_carry_the_recorded_times_not_the_export_time(tmp_path: Path) -> None:
    client, run = export_one(tmp_path)

    for span, step in zip(step_spans(client), recorded_steps(run), strict=True):
        assert span["start_time"] == parse_time(step["started_at"])
        assert span["end_time"] == parse_time(step["ended_at"])
    # The fixture was recorded in 2026-09; an export-time clock would say now.
    assert all(s["start_time"].year == 2026 and s["start_time"].month == 9 for s in client.spans)


@pytest.mark.unit
def test_the_trace_spans_the_run(tmp_path: Path) -> None:
    client, run = export_one(tmp_path)
    steps = recorded_steps(run)

    trace = client.traces[0]
    assert trace["start_time"] == parse_time(steps[0]["started_at"])
    assert trace["end_time"] == parse_time(steps[-1]["ended_at"])


@pytest.mark.unit
def test_ids_are_uuid7_and_the_same_on_every_export(tmp_path: Path) -> None:
    run = a_run(tmp_path / "run")
    first, second = a_kit(), a_kit()
    export(first[0], find_targets([run]), log=lambda line: None)
    export(second[0], find_targets([run]), log=lambda line: None)

    ids = [first[1].traces[0]["id"], *(s["id"] for s in first[1].spans)]
    assert all(uuid.UUID(i).version == 7 for i in ids)
    assert len(set(ids)) == len(ids), "every span its own id"
    assert ids == [second[1].traces[0]["id"], *(s["id"] for s in second[1].spans)]


@pytest.mark.unit
def test_an_id_starts_with_the_time_it_stands_for() -> None:
    when = datetime(2026, 9, 28, 14, 19, 38, 449000, tzinfo=UTC)

    made = uuid.UUID(record_id("step-x", when))

    # A UUID v7 keeps the time in milliseconds in its first 48 bits.
    assert made.int >> 80 == int(when.timestamp() * 1000)
    assert made.version == 7 and made.variant == uuid.RFC_4122
    assert record_id("step-x", when) != record_id("step-y", when)


@pytest.mark.unit
def test_child_spans_sit_under_their_step(tmp_path: Path) -> None:
    client, _ = export_one(tmp_path)
    steps = step_spans(client)

    children = [s for s in client.spans if s.get("parent_span_id")]
    assert {c["parent_span_id"] for c in children} <= {s["id"] for s in steps}
    # Four steps, each with one model call; three browser actions (done is the
    # agent's own ending, not a browser action).
    assert sum(c["type"] == "llm" for c in children) == 4
    assert sum(c["type"] == "tool" for c in children) == 3


# --- Shared names, no content -------------------------------------------------------------


@pytest.mark.unit
def test_the_trace_is_an_agent_invocation(tmp_path: Path) -> None:
    client, _ = export_one(tmp_path)

    assert client.traces[0]["metadata"][OPERATION] == INVOKE_AGENT


@pytest.mark.unit
def test_model_call_spans_use_the_genai_names(tmp_path: Path) -> None:
    client, _ = export_one(tmp_path)

    call = next(s for s in client.spans if s["type"] == "llm")
    meta = call["metadata"]
    assert meta[OPERATION] == CHAT
    for name in (PROVIDER, REQUEST_MODEL, INPUT_TOKENS, OUTPUT_TOKENS, FINISH_REASONS):
        assert name in meta, name
    assert isinstance(meta[FINISH_REASONS], list)
    # This project's own facts keep their own names.
    for name in ("vision", "schema_enforced", "cost_usd", "backoffs", "waited_s"):
        assert name in meta, name


@pytest.mark.unit
def test_browser_action_spans_are_tool_executions(tmp_path: Path) -> None:
    client, _ = export_one(tmp_path)

    tools = [s for s in client.spans if s["type"] == "tool"]
    assert [t["metadata"][OPERATION] for t in tools] == [EXECUTE_TOOL] * 3
    assert [t["metadata"][TOOL_NAME] for t in tools] == ["fill", "press", "click"]


@pytest.mark.unit
def test_an_action_span_ends_with_its_step(tmp_path: Path) -> None:
    client, _ = export_one(tmp_path)
    by_id = {s["id"]: s for s in step_spans(client)}

    for tool in (s for s in client.spans if s["type"] == "tool"):
        step = by_id[tool["parent_span_id"]]
        assert tool["end_time"] == step["end_time"]
        assert step["start_time"] <= tool["start_time"] <= tool["end_time"]


@pytest.mark.unit
def test_no_span_carries_prompt_page_or_reply_text(tmp_path: Path) -> None:
    run = a_run(tmp_path / "sweep" / "task-a" / "trial-1")
    a_verdict(run)
    (tmp_path / "sweep" / SWEEP_FILE).write_text(json.dumps({"suite": "s", "code_versions": []}))
    kit, client = a_kit()

    export(kit, find_targets([tmp_path / "sweep"]), log=lambda line: None)

    sent = json.dumps([client.traces, client.spans], default=str)
    for text in PRIVATE_TEXT:
        assert text not in sent, f"{text!r} left the run folder"


@pytest.mark.unit
def test_what_an_action_may_say_is_its_type_and_position(tmp_path: Path) -> None:
    client, _ = export_one(tmp_path)

    fill = next(s for s in step_spans(client) if s["name"] == "step 1: fill")
    assert fill["metadata"]["action"] == "fill"
    assert fill["metadata"]["element_id"] == 7
    assert "text" not in fill["metadata"], "what was typed stays in the run folder"


# --- The verdict, the judge and the experiments ------------------------------------------


@pytest.mark.unit
def test_the_verdict_is_a_score_and_tags_on_the_trace(tmp_path: Path) -> None:
    sweep = a_sweep(tmp_path / "runs", "modality-openrouter", tasks=("task-a",))
    kit, client = a_kit()

    export(kit, find_targets([sweep]), log=lambda line: None)

    trace = client.traces[0]
    assert trace["feedback_scores"] == [
        {"name": "passed", "value": 0.0, "category_name": "navigation"}
    ]
    assert trace["tags"] == ["modality-openrouter", "fail", "navigation"]
    assert trace["metadata"]["code_versions"] == [COMMIT]


@pytest.mark.unit
def test_the_judges_call_is_a_chat_span_under_the_trace(tmp_path: Path) -> None:
    sweep = a_sweep(tmp_path / "runs", "modality-openrouter", tasks=("task-a",))
    kit, client = a_kit()

    export(kit, find_targets([sweep]), log=lambda line: None)

    judge = next(s for s in client.spans if s.get("model") == "google/gemini-3.8-flash")
    assert judge["parent_span_id"] is None, "after the run, not inside a step"
    assert judge["metadata"][OPERATION] == CHAT
    assert judge["start_time"] == parse_time("2026-09-28T14:19:45.000Z")
    # The trace ends when the judge did, not when the last step did.
    assert client.traces[0]["end_time"] == parse_time("2026-09-28T14:19:47.500Z")


@pytest.mark.unit
def test_one_experiment_per_arm_linking_each_run_to_its_task(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    a_sweep(runs, "modality-openrouter")
    a_sweep(runs, "modality-openrouter-text")
    kit, client = a_kit()

    export(kit, find_targets([runs]), log=lambda line: None)

    assert [e.name for e in client.experiments] == [
        "modality-openrouter",
        "modality-openrouter-text",
    ]
    assert {e.dataset_name for e in client.experiments} == {"sg-gov-v1"}
    # The tasks are one set of items, shared by both arms.
    items = client.datasets["sg-gov-v1"].items
    assert set(items) == {item_id("sg-gov-v1", "task-a"), item_id("sg-gov-v1", "task-b")}
    traces_by_task = {}
    for trace in client.traces:
        traces_by_task.setdefault(trace["metadata"]["task_id"], set()).add(trace["id"])
    for experiment in client.experiments:
        for item, trace in experiment.links:
            task = items[item]["task_id"]
            assert trace in traces_by_task[task]
        assert len(experiment.links) == 2


@pytest.mark.unit
def test_the_export_says_what_it_sent(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    a_sweep(runs, "modality-openrouter")
    a_sweep(runs, "modality-openrouter-text")
    kit, client = a_kit()

    counts = export(kit, find_targets([runs]), log=lambda line: None)

    # 4 runs of 4 steps, 3 of them with a screenshot; per run: 4 model calls,
    # 3 browser actions and 1 judge call besides the steps.
    assert (counts.runs, counts.steps, counts.attachments, counts.experiments) == (4, 16, 12, 2)
    assert counts.spans == len(client.spans) == 4 * (4 + 4 + 3 + 1)
    assert counts.summary() == (
        "exported 4 runs: 16 steps, 48 spans, 12 screenshots, 2 experiments"
    )


@pytest.mark.unit
def test_an_experiment_records_what_its_arm_was(tmp_path: Path) -> None:
    sweep = a_sweep(tmp_path / "runs", "modality-openrouter-text", tasks=("task-a",))
    kit, client = a_kit()

    export(kit, find_targets([sweep]), log=lambda line: None)

    config = client.experiments[0].config
    assert config["profile"] == "openrouter-text"
    assert config["vision_model"] == ""
    assert config["code_versions"] == [COMMIT]


@pytest.mark.unit
def test_a_run_outside_a_sweep_gets_no_experiment(tmp_path: Path) -> None:
    client, _ = export_one(tmp_path)

    assert client.experiments == []
    assert client.flushed == 1, "and what was queued is sent"


# --- Finding the runs --------------------------------------------------------------------


@pytest.mark.unit
def test_runs_are_found_from_a_run_a_sweep_or_a_folder_of_sweeps(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    sweep = a_sweep(runs, "modality-openrouter")
    one = sweep / "task-a" / "trial-1"

    assert [t.folder for t in find_targets([one])] == [one]
    assert len(find_targets([sweep])) == 2
    assert len(find_targets([runs])) == 2
    # The same run named twice is exported once.
    assert len(find_targets([one, sweep, runs])) == 2


@pytest.mark.unit
def test_a_run_named_inside_a_sweep_keeps_its_arm(tmp_path: Path) -> None:
    sweep = a_sweep(tmp_path / "runs", "modality-openrouter")

    target = find_targets([sweep / "task-b" / "trial-1"])[0]

    assert (target.arm, target.task_id, target.trial) == ("modality-openrouter", "task-b", 1)


@pytest.mark.unit
def test_runs_set_aside_are_never_exported(tmp_path: Path) -> None:
    sweep = a_sweep(tmp_path / "runs", "modality-openrouter", tasks=("task-a",))
    a_run(sweep / "_interrupted" / "task-a" / "trial-1")

    assert [t.task_id for t in find_targets([sweep])] == ["task-a"]


@pytest.mark.unit
def test_an_unreadable_run_is_skipped_and_named(tmp_path: Path) -> None:
    broken = a_run(tmp_path / "run")
    (broken / ACTION_LOG).write_text("{not json\n")
    kit, client = a_kit()

    counts = export(kit, find_targets([broken]), log=lambda line: None)

    assert client.traces == []
    assert counts.runs == 0 and len(counts.skipped) == 1
    assert str(broken) in counts.skipped[0]


# --- Optional ------------------------------------------------------------------------------


def fake_opik(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Put a fake ``opik`` package in place, recording how it was set up."""
    seen: dict[str, Any] = {}

    class Opik:
        def __init__(self, **kwargs: Any) -> None:
            seen.update(kwargs)

    package = types.ModuleType("opik")
    package.Opik = Opik  # type: ignore[attr-defined]
    package.Attachment = lambda **kwargs: kwargs  # type: ignore[attr-defined]
    item_module = types.ModuleType("opik.api_objects.experiment.experiment_item")
    item_module.ExperimentItemReferences = lambda **kwargs: kwargs  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "opik", package)
    monkeypatch.setitem(sys.modules, "opik.api_objects", types.ModuleType("opik.api_objects"))
    monkeypatch.setitem(
        sys.modules, "opik.api_objects.experiment", types.ModuleType("opik.api_objects.experiment")
    )
    monkeypatch.setitem(sys.modules, "opik.api_objects.experiment.experiment_item", item_module)
    return seen


@pytest.mark.unit
def test_no_key_and_no_address_means_nothing_to_connect_to() -> None:
    assert connect(env={}, url=None, workspace=None, project="p") is None


@pytest.mark.unit
def test_opik_cloud_is_reached_with_the_key_and_no_address(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen = fake_opik(monkeypatch)

    connect(env={API_KEY_ENV: "test-opik-key"}, url=None, workspace="ws", project="p")

    assert seen == {
        "project_name": "p",
        "workspace": "ws",
        "host": None,
        "api_key": "test-opik-key",
    }


@pytest.mark.unit
def test_a_self_hosted_opik_is_reached_by_its_address_alone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen = fake_opik(monkeypatch)

    connect(env={}, url="http://localhost:5173/api", workspace=None, project="p")

    assert seen["host"] == "http://localhost:5173/api"
    assert seen["api_key"] is None


@pytest.mark.unit
def test_a_missing_package_is_reported_with_how_to_install_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "opik", None)

    with pytest.raises(ExportError, match="uv sync --extra opik"):
        connect(env={API_KEY_ENV: "test-opik-key"}, url=None, workspace=None, project="p")


def invoke_export(monkeypatch: pytest.MonkeyPatch, env: dict[str, str], *args: str) -> Any:
    """Run ``waa export`` with a given environment and nothing from ``.env``."""
    monkeypatch.setattr(cli, "resolve_env", lambda: env)
    return CliRunner().invoke(cli.app, ["export", *args])


@pytest.mark.unit
def test_waa_export_without_a_key_skips_and_says_so(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    run = a_run(tmp_path / "run")

    result = invoke_export(monkeypatch, {}, str(run))

    assert result.exit_code == 0
    assert "Opik export skipped" in result.output and "Nothing was sent" in result.output


@pytest.mark.unit
def test_waa_export_without_the_package_says_how_to_install_it(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    run = a_run(tmp_path / "run")
    monkeypatch.setitem(sys.modules, "opik", None)

    result = invoke_export(monkeypatch, {API_KEY_ENV: "test-opik-key"}, str(run))

    assert result.exit_code == 2
    assert "uv sync --extra opik" in result.output


@pytest.mark.unit
def test_waa_export_refuses_another_destination(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    result = invoke_export(monkeypatch, {}, str(a_run(tmp_path / "run")), "--to", "langfuse")

    assert result.exit_code == 2 and "only destination is 'opik'" in result.output


@pytest.mark.unit
def test_waa_export_with_nothing_to_export_says_so(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    result = invoke_export(monkeypatch, {}, str(tmp_path))

    assert result.exit_code == 2 and "no finished run folders" in result.output


@pytest.mark.unit
def test_the_rest_of_the_project_never_imports_opik() -> None:
    # `waa run` and `waa eval` load the whole command line; none of it may
    # need the optional package. Checked in a fresh interpreter, so nothing
    # this test run imported earlier can hide it.
    code = "import sys, web_action_agent.cli; print('opik' in sys.modules)"
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)

    assert done.stdout.strip() == "False"


# --- Against the real package, where it is installed -------------------------------------


@pytest.mark.unit
def test_the_calls_fit_the_real_opik_client(tmp_path: Path) -> None:
    """Every argument the export passes is one the real client accepts.

    Skipped without the optional package. Binds each recorded call to the real
    method's signature, so a renamed argument upstream fails here rather than
    in someone's first real export.
    """
    opik = pytest.importorskip("opik")
    import inspect

    sweep = a_sweep(tmp_path / "runs", "modality-openrouter", tasks=("task-a",))
    kit, client = a_kit()
    export(kit, find_targets([sweep]), log=lambda line: None)

    for trace in client.traces:
        inspect.signature(opik.Opik.trace).bind(None, **trace)
    for span in client.spans:
        inspect.signature(opik.Opik.span).bind(None, **span)
    real_dataset = inspect.signature(opik.Opik.get_or_create_dataset)
    real_dataset.bind(None, name="sg-gov-v1", description="d")
    inspect.signature(opik.Opik.create_experiment).bind(
        None, dataset_name="sg-gov-v1", name="arm", experiment_config={}
    )
    attachment = opik.Attachment(
        data=str(tmp_path), file_name="step-01.png", content_type="image/png"
    )
    assert attachment.file_name == "step-01.png"
