"""Copy finished run folders to Opik, to review them side by side.

A run folder is complete evidence on its own, but it is read one folder at a
time. Opik (an open-source tool for looking at AI agent traces, run by Comet
as a free cloud service or self-hosted) shows many runs together: each step
on a timeline, its annotated screenshot inline, and one *experiment* per arm
of a comparison, so two arms can be compared row by row. This module copies
finished runs there. It changes nothing about the runs themselves.

**After the fact, and never the other way round.** The export reads folders
that runs have already written, through the same trace reader every other
consumer uses. It is a separate command, not instrumentation inside the agent
loop, so nothing here can change how a run behaves; and the run folders and
the results tables stay the record. Opik holds a copy, for viewing.

**Optional, all the way down.** The Opik package is an optional dependency,
imported only when the export actually runs. Without it, or without an Opik
key or address, every other command works exactly as before, and the export
says what is missing instead of failing.

**On the run's timeline, not the upload's.** Every trace and span is given the
start and end times the run recorded, so a run exported a week later still
shows when each step happened and how long it took. Their ids are made from
the run's own ids and times too, so exporting the same run twice writes the
same ids again instead of a second copy.

**Shared names for shared ideas.** Where the OpenTelemetry conventions for AI
systems have a name for a fact, it is used: ``gen_ai.operation.name``
(``invoke_agent`` for a whole run, ``chat`` for a model call,
``execute_tool`` for a browser action), ``gen_ai.request.model``,
``gen_ai.provider.name``, ``gen_ai.usage.input_tokens`` and
``gen_ai.usage.output_tokens``, ``gen_ai.response.finish_reasons``. Facts with
no such name -- ``vision``, ``schema_enforced``, ``cost_usd``, ``backoffs``,
``waited_s`` -- keep this project's own. Only the names are borrowed: the
export goes through the Opik client, because screenshots are attachments and
the OpenTelemetry protocol has no such thing.

**Numbers and labels only, never content.** No prompt, no page text, no model
reply, no typed text, no page address the agent chose: those stay in the run
folder, behind the redaction its writer applies. What is exported is what a
reviewer needs to find the run worth opening -- which step, which action
type, which element number, how it went, how long, how many tokens, what it
cost -- plus the screenshots, which are the reason to look at Opik at all.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import urlparse

from .eval_harness import SWEEP_FILE, VERDICT_FILE, TrialVerdict
from .judge import FAIL, PASS
from .model_client import CallRecord, ProfileError, load_profiles
from .trace import RUN_FILE, SCREENSHOT_DIR, TraceError
from .trace_reader import ReplayedRun, read_run

#: The Opik project runs are exported into, unless the settings name another.
DEFAULT_PROJECT = "web-action-agent"

#: The environment variable holding the Opik Cloud key. Like every key, it
#: lives in ``.env`` and is never printed.
API_KEY_ENV = "WAA_OPIK_API_KEY"

#: How to install the optional package, for the message when it is missing.
INSTALL_HINT = "uv sync --extra opik (or: pip install 'web-action-agent[opik]')"

# The OpenTelemetry names for AI systems, where one exists for a fact.
OPERATION = "gen_ai.operation.name"
AGENT_NAME = "gen_ai.agent.name"
PROVIDER = "gen_ai.provider.name"
REQUEST_MODEL = "gen_ai.request.model"
INPUT_TOKENS = "gen_ai.usage.input_tokens"
OUTPUT_TOKENS = "gen_ai.usage.output_tokens"
FINISH_REASONS = "gen_ai.response.finish_reasons"
TOOL_NAME = "gen_ai.tool.name"

#: ``gen_ai.operation.name`` for a whole run, a model call, a browser action.
INVOKE_AGENT = "invoke_agent"
CHAT = "chat"
EXECUTE_TOOL = "execute_tool"

#: The time put into every dataset item's id. Items are tasks, not events,
#: so they have no time of their own; a fixed one keeps their ids the same on
#: every export, which is what lets every arm share one item per task.
_ITEM_EPOCH = datetime(2026, 1, 1, tzinfo=UTC)

#: The fields of an action that may be exported: its type and the numbers
#: that say where it acted. Typed text, page addresses and answers are the
#: model's own words, and stay in the run folder.
_ACTION_FIELDS = ("action", "element_id", "direction")


class ExportError(Exception):
    """The export cannot start: a missing package, or nothing to export."""


# --- Talking to Opik --------------------------------------------------------------------


class OpikClient(Protocol):
    """The part of the Opik client the export uses.

    Written out so a test can pass a stand-in that records every call, and so
    it is plain which of the client's methods this module depends on.
    """

    def trace(self, **kwargs: Any) -> Any:
        """Log one trace (a whole run)."""

    def span(self, **kwargs: Any) -> Any:
        """Log one span (a step, a model call, a browser action)."""

    def get_or_create_dataset(self, name: str, description: str | None = None) -> Any:
        """Find or make the dataset an experiment's items refer to."""

    def create_experiment(
        self,
        dataset_name: str,
        name: str | None = None,
        experiment_config: dict[str, Any] | None = None,
    ) -> Any:
        """Make one experiment: one arm of a comparison."""

    def flush(self, timeout: int | None = None) -> Any:
        """Send whatever is still queued."""


@dataclass(frozen=True)
class OpikKit:
    """The Opik client, plus the two Opik types the export has to build.

    Kept together so the whole of Opik can be swapped for a stand-in in a
    test, without the real package installed.

    Attributes:
        client: The client to log to.
        attachment: Makes an attachment from a file path.
        item_reference: Makes the link between a dataset item and a trace that
            an experiment is made of, from their two ids.
    """

    client: OpikClient
    attachment: Callable[[Path], Any]
    item_reference: Callable[[str, str], Any]


def connect(
    *, env: Mapping[str, str], url: str | None, workspace: str | None, project: str
) -> OpikKit | None:
    """Connect to Opik Cloud or a self-hosted Opik, as the settings say.

    The package is imported here, not at the top of the module, so that
    nothing else in the project needs it to be installed.

    Args:
        env: The environment, for :data:`API_KEY_ENV`.
        url: A self-hosted Opik's address, or None for Opik Cloud.
        workspace: The Opik Cloud workspace, or None for the key's default.
        project: The Opik project to log into.

    Returns:
        OpikKit | None: A connected kit, or None when there is nowhere to send
        to: no key for Opik Cloud, and no self-hosted address. That is a reason
        to skip the export, not an error.

    Raises:
        ExportError: If the optional package is not installed.
    """
    key = env.get(API_KEY_ENV, "").strip() or None
    if key is None and not url:
        return None
    try:
        # Optional, so imported here, only when an export actually runs.
        import opik
        from opik.api_objects.experiment.experiment_item import (
            ExperimentItemReferences,
        )
    except ImportError as error:
        raise ExportError(
            f"the Opik export needs the optional 'opik' package: {INSTALL_HINT}"
        ) from error

    # Neither address is assumed: no `host` means the package's own default,
    # Opik Cloud; a configured one means that instance, wherever it runs.
    client = opik.Opik(project_name=project, workspace=workspace, host=url, api_key=key)
    return OpikKit(
        client=client,
        attachment=lambda path: opik.Attachment(
            data=str(path), file_name=path.name, content_type="image/png"
        ),
        item_reference=lambda item_id, trace_id: ExperimentItemReferences(
            dataset_item_id=item_id, trace_id=trace_id
        ),
    )


# --- Times and ids -----------------------------------------------------------------------


def parse_time(text: str) -> datetime:
    """Read a time the trace recorded, as an aware time in UTC.

    Args:
        text: An RFC 3339 time, such as ``2026-10-20T11:31:10.347Z``.

    Returns:
        datetime: The same moment, with its time zone.
    """
    moment = datetime.fromisoformat(text)
    return moment if moment.tzinfo else moment.replace(tzinfo=UTC)


def record_id(key: str, when: datetime) -> str:
    """Make an Opik id for a recorded thing, the same every time.

    Opik ids are UUID version 7: the first 48 bits are a time in
    milliseconds, the rest are free. Here the time is the moment the thing
    happened, and the rest comes from a hash of its own id. So the id sorts
    in the run's time order, and exporting a run twice gives every trace and
    span the same id again rather than a duplicate.

    Args:
        key: Something unique to the thing, such as a step's id.
        when: When it happened.

    Returns:
        str: The id, in the usual 8-4-4-4-12 form.
    """
    millis = int(when.timestamp() * 1000) & ((1 << 48) - 1)
    digest = int.from_bytes(hashlib.sha256(key.encode("utf-8")).digest()[:10], "big")
    rand_a = digest >> 68 & 0xFFF  # 12 bits
    rand_b = digest & ((1 << 62) - 1)  # 62 bits
    value = millis << 80 | 0x7 << 76 | rand_a << 64 | 0b10 << 62 | rand_b
    return str(uuid.UUID(int=value))


# --- What to export ----------------------------------------------------------------------


@dataclass(frozen=True)
class RunTarget:
    """One run folder to export, and the arm it belongs to, if any.

    Attributes:
        folder: The run folder.
        arm: The sweep folder's name, such as ``modality-openrouter``, or None
            for a run that was not part of a sweep.
        spec: The sweep's ``sweep.json``, or an empty mapping.
        task_id: The benchmark task, from the folder's place in the sweep, or
            None outside one.
        trial: The trial number, or None outside a sweep.
    """

    folder: Path
    arm: str | None = None
    spec: Mapping[str, Any] = field(default_factory=dict)
    task_id: str | None = None
    trial: int | None = None


def _sweep_targets(sweep: Path) -> list[RunTarget]:
    """List the run folders of one sweep, as export targets.

    Args:
        sweep: A sweep folder, holding a ``sweep.json``.

    Returns:
        list[RunTarget]: Its trials, in task then trial order. Two levels down
        only, so runs set aside after being killed or stopped by the AI
        service, which sit a level deeper, are never exported.
    """
    try:
        spec = json.loads((sweep / SWEEP_FILE).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        spec = {}
    targets = []
    for run_file in sorted(sweep.glob(f"*/trial-*/{RUN_FILE}")):
        trial_dir = run_file.parent
        number = trial_dir.name.removeprefix("trial-")
        targets.append(
            RunTarget(
                folder=trial_dir,
                arm=sweep.name,
                spec=spec,
                task_id=trial_dir.parent.name,
                trial=int(number) if number.isdigit() else None,
            )
        )
    return targets


def find_targets(paths: Iterable[Path]) -> list[RunTarget]:
    """Work out which run folders some paths mean.

    Each path may be a run folder, a sweep folder (its trials), or a folder of
    sweeps, such as ``runs`` (every sweep directly inside it).

    Args:
        paths: The paths given on the command line.

    Returns:
        list[RunTarget]: Every run folder found, each once.
    """
    found: dict[Path, RunTarget] = {}
    for path in paths:
        if (path / RUN_FILE).exists():
            sweep = path.parent.parent
            in_sweep = (sweep / SWEEP_FILE).exists()
            targets = (
                [t for t in _sweep_targets(sweep) if t.folder == path]
                if in_sweep
                else [RunTarget(folder=path)]
            )
        elif (path / SWEEP_FILE).exists():
            targets = _sweep_targets(path)
        else:
            targets = [
                target
                for sweep_file in sorted(path.glob(f"*/{SWEEP_FILE}"))
                for target in _sweep_targets(sweep_file.parent)
            ]
        for target in targets:
            found.setdefault(target.folder.resolve(), target)
    return list(found.values())


# --- Building the spans ------------------------------------------------------------------


def _provider_names() -> dict[str, str]:
    """Map each shipped profile to the service it reaches, for spans.

    Returns:
        dict[str, str]: Profile name to the service's host name, such as
        ``openrouter.ai``. Empty if the profiles cannot be read.
    """
    try:
        profiles = load_profiles()
    except ProfileError:
        return {}
    return {name: urlparse(p.base_url).hostname or name for name, p in profiles.items()}


def _screenshot_file(run: ReplayedRun, recorded: Path | None) -> Path | None:
    """Find a step's screenshot on disk.

    The path is recorded relative to wherever the run was started from, so it
    is tried as written and then in the run's own screenshot folder, which is
    where it is however the folder has been moved since.

    Args:
        run: The run.
        recorded: The path the step recorded, or None.

    Returns:
        Path | None: The file, or None if the step has none or it is gone.
    """
    if recorded is None:
        return None
    for candidate in (recorded, run.folder / SCREENSHOT_DIR / recorded.name):
        if candidate.is_file():
            return candidate
    return None


def read_verdict(folder: Path) -> TrialVerdict | None:
    """Read a run's mark from the benchmark, if it has one.

    Args:
        folder: The run folder.

    Returns:
        TrialVerdict | None: The mark, or None for a run the benchmark did not
        mark (such as one started with ``waa run``) or a file that cannot be
        read.
    """
    try:
        data = json.loads((folder / VERDICT_FILE).read_text(encoding="utf-8"))
        return TrialVerdict.from_dict(data)
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
        return None


def item_id(suite: str, task_id: str) -> str:
    """Make the id of a task's dataset item, the same for every arm and export.

    Args:
        suite: The benchmark suite's name.
        task_id: The task.

    Returns:
        str: A UUID version 7, as Opik requires.
    """
    return record_id(f"{suite}/{task_id}", _ITEM_EPOCH)


def _call_span(
    call: CallRecord, *, trace_id: str, parent: str | None, providers: Mapping[str, str]
) -> dict[str, Any]:
    """Describe one model call as a span: who, what model, how long, how much.

    Args:
        call: The call's record.
        trace_id: The run's trace.
        parent: The span it belongs under (its step), or None for the trace.
        providers: Profile name to service, from :func:`_provider_names`.

    Returns:
        dict[str, Any]: Keyword arguments for the client's ``span``.
    """
    provider = providers.get(call.profile, call.profile)
    usage = {}
    if call.prompt_tokens is not None and call.completion_tokens is not None:
        usage = {
            "prompt_tokens": call.prompt_tokens,
            "completion_tokens": call.completion_tokens,
            "total_tokens": call.prompt_tokens + call.completion_tokens,
        }
    started = parse_time(call.started_at)
    return {
        "trace_id": trace_id,
        "id": record_id(call.call_id, started),
        "parent_span_id": parent,
        "name": f"{CHAT} {call.model}",
        "type": "llm",
        "start_time": started,
        "end_time": parse_time(call.ended_at),
        "model": call.model,
        "provider": provider,
        "usage": usage or None,
        "total_cost": call.cost_usd,
        "metadata": {
            OPERATION: CHAT,
            PROVIDER: provider,
            REQUEST_MODEL: call.model,
            INPUT_TOKENS: call.prompt_tokens,
            OUTPUT_TOKENS: call.completion_tokens,
            FINISH_REASONS: [call.finish_reason] if call.finish_reason else [],
            # This project's own facts, under its own names.
            "profile": call.profile,
            "vision": call.vision,
            "schema_enforced": call.schema_enforced,
            "cost_usd": call.cost_usd,
            "latency_s": call.latency_s,
            "backoffs": [
                {"attempt": b.attempt, "status": b.status, "waited_s": b.waited_s}
                for b in call.backoffs
            ],
            "waited_s": sum(b.waited_s for b in call.backoffs),
        },
    }


def _action_fields(action: Any) -> dict[str, Any]:
    """Keep the parts of an action that may be exported: type and position.

    Args:
        action: The step's action.

    Returns:
        dict[str, Any]: Its ``action`` type, and ``element_id`` or
        ``direction`` where it has one. Never its text, address or answer.
    """
    data = action.model_dump() if hasattr(action, "model_dump") else dict(action)
    return {name: data[name] for name in _ACTION_FIELDS if data.get(name) is not None}


@dataclass
class ExportCounts:
    """What an export sent, for the closing summary.

    Attributes:
        runs: Traces logged, one per run.
        steps: Step spans logged.
        spans: Every span logged, steps included.
        attachments: Screenshots attached.
        experiments: Experiments made, one per arm.
        skipped: Run folders that could not be read, with the reason.
    """

    runs: int = 0
    steps: int = 0
    spans: int = 0
    attachments: int = 0
    experiments: int = 0
    skipped: list[str] = field(default_factory=list)

    def summary(self) -> str:
        """Say what was sent, in one line."""
        line = (
            f"exported {self.runs} runs: {self.steps} steps, {self.spans} spans, "
            f"{self.attachments} screenshots, {self.experiments} experiments"
        )
        return line + (f"; skipped {len(self.skipped)}" if self.skipped else "")


def export_run(
    kit: OpikKit,
    target: RunTarget,
    *,
    project: str,
    counts: ExportCounts,
    providers: Mapping[str, str],
) -> str | None:
    """Log one run: a trace, a span per step, and the spans inside each step.

    Args:
        kit: The Opik connection.
        target: The run folder, and its arm.
        project: The Opik project.
        counts: Added to as things are sent.
        providers: Profile name to service.

    Returns:
        str | None: The run's trace id, or None if the folder could not be
        read or holds no steps, which is noted in ``counts.skipped``.
    """
    try:
        run = read_run(target.folder)
    except TraceError as error:
        # Any fault in reading the folder -- a missing file, a broken line --
        # skips this run only; the others are still exported.
        counts.skipped.append(f"{target.folder}: {error}")
        return None
    if not run.steps:
        counts.skipped.append(f"{target.folder}: no steps recorded")
        return None

    verdict = read_verdict(target.folder)
    # The judge's calls: one to mark the run, and, for a failed run, one to
    # name its kind of failure. Each is a model call like the planner's.
    judge_calls: list[CallRecord] = []
    if verdict is not None:
        for answer in (verdict.judge, verdict.categorized_by):
            if answer is not None and answer.record is not None:
                judge_calls.append(answer.record)

    started = parse_time(run.steps[0].started_at)
    ended = max(
        [parse_time(run.steps[-1].ended_at), *(parse_time(c.ended_at) for c in judge_calls)]
    )
    trace_id = record_id(run.run_id, started)
    cost = run.cost
    outcome = run.outcome
    status = verdict.status if verdict is not None else None
    category = verdict.category if verdict is not None else None
    planner_model = run.config.get("model") or (run.calls[0].model if run.calls else None)

    scores = []
    if status in (PASS, FAIL):
        score: dict[str, Any] = {"name": "passed", "value": 1.0 if status == PASS else 0.0}
        if category:
            score["category_name"] = category
        scores.append(score)

    kit.client.trace(
        id=trace_id,
        name=target.task_id or run.run_id,
        start_time=started,
        end_time=ended,
        project_name=project,
        # Labels only: which task and where it started, never its wording.
        input={"task_id": target.task_id, "start_url": run.start_url},
        output={"verdict": status, "category": category},
        tags=[tag for tag in (target.arm, status, category) if tag],
        feedback_scores=scores or None,
        metadata={
            OPERATION: INVOKE_AGENT,
            AGENT_NAME: "web-action-agent",
            REQUEST_MODEL: planner_model,
            "run_id": run.run_id,
            "arm": target.arm,
            "task_id": target.task_id,
            "trial": target.trial,
            "profile": target.spec.get("profile") or run.config.get("profile"),
            "code_versions": list(target.spec.get("code_versions", [])),
            "text_only": run.text_only,
            "verdict": status,
            "failure_category": category,
            "outcome": outcome.category if outcome is not None else "interrupted",
            "steps": len(run.steps),
            "calls": cost.calls,
            INPUT_TOKENS: cost.prompt_tokens,
            OUTPUT_TOKENS: cost.completion_tokens,
            "cost_usd": cost.cost_usd,
            "unpriced_calls": cost.unpriced_calls,
        },
    )
    counts.runs += 1

    for step in run.steps:
        step_started = parse_time(step.started_at)
        step_ended = parse_time(step.ended_at)
        step_id = record_id(step.step_id, step_started)
        action = _action_fields(step.action) if step.action is not None else {}
        screenshot = _screenshot_file(run, step.observation.screenshot_on_disk)
        gate = step.gate
        kit.client.span(
            trace_id=trace_id,
            id=step_id,
            name=f"step {step.number}: {action.get('action', 'no action')}",
            type="general",
            start_time=step_started,
            end_time=step_ended,
            project_name=project,
            attachments=[kit.attachment(screenshot)] if screenshot else None,
            metadata={
                "step": step.number,
                **action,
                "result": step.result.status if step.result is not None else None,
                "attempts": step.attempts,
                "rejections": len(step.rejections),
                "forced": bool(step.forced),
                "replayed_skill": step.replay is not None,
                "elements": gate.element_count if gate is not None else None,
                "screenshot_sent": gate.send_screenshot if gate is not None else None,
                "screenshot_trigger": gate.trigger if gate is not None else None,
            },
        )
        counts.steps += 1
        counts.spans += 1
        counts.attachments += 1 if screenshot else 0

        for call in step.records:
            kit.client.span(
                **_call_span(call, trace_id=trace_id, parent=step_id, providers=providers),
                project_name=project,
            )
            counts.spans += 1

        if step.action is not None and step.result is not None:
            # The action ran at the end of the step, after the planning:
            # the step was recorded as soon as the action returned, so the
            # action's span is the last `elapsed_s` of the step.
            elapsed = max(0.0, float(step.result.elapsed_s))
            action_started = max(
                step_started, datetime.fromtimestamp(step_ended.timestamp() - elapsed, UTC)
            )
            kit.client.span(
                trace_id=trace_id,
                id=record_id(f"{step.step_id}/action", action_started),
                parent_span_id=step_id,
                name=f"{EXECUTE_TOOL} {action.get('action', '')}".strip(),
                type="tool",
                start_time=action_started,
                end_time=step_ended,
                project_name=project,
                metadata={
                    OPERATION: EXECUTE_TOOL,
                    TOOL_NAME: action.get("action"),
                    **action,
                    "status": step.result.status,
                    "elapsed_s": step.result.elapsed_s,
                    "waited_s": step.result.waited_s,
                },
            )
            counts.spans += 1

    # The judge marked the run after it ended: its calls sit under the trace.
    for call in judge_calls:
        kit.client.span(
            **_call_span(call, trace_id=trace_id, parent=None, providers=providers),
            project_name=project,
        )
        counts.spans += 1
    return trace_id


def _link_experiments(
    kit: OpikKit, exported: list[tuple[RunTarget, str]], *, counts: ExportCounts
) -> None:
    """Make one experiment per arm, linking each of its runs to its task.

    An Opik experiment is a set of traces, each tied to an item of a dataset:
    here the dataset is the benchmark suite, its items are the tasks, and an
    arm's experiment ties each trial's trace to its task. Two arms on the same
    suite can then be compared task by task. Each task's item has a fixed id
    (:func:`item_id`), so every arm points at the same one.

    Args:
        kit: The Opik connection.
        exported: Each exported run with its trace id.
        counts: Added to as experiments are made.
    """
    by_arm: dict[str, list[tuple[RunTarget, str]]] = {}
    for target, trace_id in exported:
        if target.arm and target.task_id:
            by_arm.setdefault(target.arm, []).append((target, trace_id))
    for arm, runs in sorted(by_arm.items()):
        spec = runs[0][0].spec
        suite = str(spec.get("suite") or "benchmark")
        dataset = kit.client.get_or_create_dataset(
            name=suite, description=f"Benchmark tasks of suite {suite}"
        )
        # Inserting an item that is already there, with the same id, changes
        # nothing, so every arm can insert the tasks it ran.
        tasks = sorted({t.task_id for t, _ in runs if t.task_id})
        dataset.insert([{"id": item_id(suite, task), "task_id": task} for task in tasks])
        experiment = kit.client.create_experiment(
            dataset_name=suite,
            name=arm,
            experiment_config={
                key: spec.get(key)
                for key in (
                    "profile",
                    "model",
                    "vision_model",
                    "base_url",
                    "library",
                    "judge_profile",
                    "code_versions",
                    "started_at",
                )
            },
        )
        experiment.insert(
            [
                kit.item_reference(item_id(suite, target.task_id), trace_id)
                for target, trace_id in runs
                if target.task_id
            ]
        )
        counts.experiments += 1


def export(
    kit: OpikKit,
    targets: Iterable[RunTarget],
    *,
    project: str = DEFAULT_PROJECT,
    log: Callable[[str], None] = print,
) -> ExportCounts:
    """Export run folders to Opik: a trace per run, an experiment per arm.

    Args:
        kit: The Opik connection.
        targets: The run folders, from :func:`find_targets`.
        project: The Opik project.
        log: Where progress lines go.

    Returns:
        ExportCounts: What was sent, and what was skipped and why.
    """
    counts = ExportCounts()
    providers = _provider_names()
    exported: list[tuple[RunTarget, str]] = []
    for target in targets:
        trace_id = export_run(kit, target, project=project, counts=counts, providers=providers)
        if trace_id is not None:
            exported.append((target, trace_id))
            log(f"exported {target.folder}")
    _link_experiments(kit, exported, counts=counts)
    kit.client.flush()
    return counts
