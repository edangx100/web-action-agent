"""The folder a run leaves behind, written as the run happens.

A run that reports "failed: step cap" and nothing else is unarguable and
useless. This module is what makes a run *examinable afterwards*: the page the
agent was looking at, what it decided, why the validator refused the first two
attempts, what the endpoint charged, and which screenshot goes with which
decision -- all on disk, in a folder named after the moment the run started.

**Written as it goes, not at the end.** The obvious implementation collects
everything in memory and serialises it once the verdict is known. That
implementation loses the entire run to a Ctrl-C, an OOM kill, a benchmark
worker that hits its wall clock, or a browser that takes the process down with
it -- which is to say, it loses exactly the runs whose traces are most worth
having. So every completed step is appended and flushed to disk before the next
one begins, and the run's verdict is the only thing written at exit. A trace
folder with no ``outcome.json`` is therefore a readable record of a run that
was killed, rather than a corrupt file or an empty directory.

**Append-only, one JSON object per line.** JSON Lines rather than one big JSON
array, because an array has to be closed to parse and a killed writer never
closes it. A line is written whole, flushed, and forced to disk; a reader can
take the file at any moment and get every step that finished.

**Timestamps, never durations.** Every step and every model call records when
it started and when it ended, in RFC 3339 UTC. A duration cannot be turned back
into a position on a timeline, so a trace that recorded only ``latency_s``
could never be laid out against the run it belongs to -- and this is the one
decision here that cannot be repaired later, because a field missing from the
schema is a field missing from every run already recorded under it. The same
reasoning gives every step and call a stable id, and every call the id of the
step it was made during: without those, two calls in one step are
indistinguishable and a flattened cost log cannot be put back together.

**The schema is a file, not a docstring.** ``schemas/action_log.schema.json``
describes one action-log record, every record is validated against it before
it is written, and a record that does not conform stops the writer rather than
being written anyway. Downstream -- the replayer, the benchmark, the span
exporter -- reads runs by that file, and a trace that quietly disagrees with it
is worse than no trace, because it is believed.
"""

from __future__ import annotations

import functools
import json
import os
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from jsonschema import Draft202012Validator
from jsonschema import ValidationError as SchemaValidationError

if TYPE_CHECKING:
    # Annotations only. The controller builds these and calls this module to
    # write them; importing it at runtime would be a cycle, and what is needed
    # here is the shape rather than the class.
    from .config import Config
    from .controller import RunReport, StepReport

#: The record schema, shipped inside the package so an installed copy validates
#: exactly the way a checkout does.
SCHEMA_PATH = Path(__file__).resolve().parent / "schemas" / "action_log.schema.json"

#: One JSON object per line, one line per completed step. The file a killed run
#: still leaves readable.
ACTION_LOG = "actions.jsonl"

#: The cost log: one line per model call, each naming the step it belongs to.
#: The same records also sit inside their step in the action log -- both are
#: written from one rendering of the step, in one call, so they cannot drift.
COST_LOG = "calls.jsonl"

#: What the run is: its id, its task, its start URL, and when its folder was
#: opened. Written before the first step, so a folder identifies itself even if
#: the run dies on its opening navigation.
RUN_FILE = "run.json"

#: Every setting the run resolved, so a result can be read back knowing what
#: produced it. Never contains an API key: keys are not configuration fields.
CONFIG_FILE = "config.json"

#: The verdict, the answer, and what the whole run cost. The only file written
#: at exit, and therefore the one whose absence means "this run was killed".
OUTCOME_FILE = "outcome.json"

#: Where the observer puts each step's annotated screenshot.
SCREENSHOT_DIR = "screenshots"


class TraceError(Exception):
    """The run folder cannot be written, or a record does not fit the schema.

    Both are programming or environment faults rather than anything the agent
    did: a full disk, a read-only path, or a record shape that has drifted from
    the schema every reader relies on. Raised rather than swallowed, because a
    trace that is quietly incomplete is trusted exactly as much as a complete
    one.
    """


def utc_now() -> str:
    """Take the current wall-clock time, in the form every trace record uses.

    Returns:
        str: RFC 3339, UTC, Z-suffixed, to the millisecond -- e.g.
        ``2026-09-17T08:31:04.218Z``. Milliseconds because two model calls in
        one step can start in the same second and their order matters;
        anything finer would be recording noise from the clock.
    """
    stamp = datetime.now(UTC).isoformat(timespec="milliseconds")
    # `isoformat` writes the UTC offset as `+00:00`; RFC 3339 allows `Z` for it
    # and every reader of these files expects the shorter form.
    return stamp.replace("+00:00", "Z")


def new_id(kind: str) -> str:
    """Mint a stable id for one run, step, or call.

    Random rather than sequential on purpose. A sequential id is only unique
    inside the thing that counted it, and these ids are meant to survive being
    lifted out of their folder -- into a benchmark's table, or into exported
    spans alongside other runs -- where ``step-3`` from two runs would collide
    and quietly merge.

    Args:
        kind: What the id points at: ``"run"``, ``"step"`` or ``"call"``. It
            is part of the id, so a bare id in a log line says what it refers
            to without needing the field name beside it.

    Returns:
        str: e.g. ``step-9f2c1d4a7b6e``. Twelve hex digits is 48 bits of
        randomness, which will not collide within a run or a benchmark sweep
        and stays short enough to read in a terminal.
    """
    return f"{kind}-{uuid.uuid4().hex[:12]}"


def action_log_schema() -> dict[str, Any]:
    """Read the action-log record schema from the shipped file.

    Returns:
        dict[str, Any]: The schema, as JSON.

    Raises:
        TraceError: If the file is missing or is not valid JSON. That is a
            broken build rather than a user mistake, and it should stop
            everything: without the schema nothing can be validated, and an
            unvalidated trace is the thing this module exists to prevent.
    """
    try:
        return json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    except OSError as error:
        raise TraceError(f"cannot read the action log schema at {SCHEMA_PATH}: {error}") from error
    except json.JSONDecodeError as error:
        raise TraceError(f"{SCHEMA_PATH} is not valid JSON: {error}") from error


@functools.cache
def _validator() -> Draft202012Validator:
    """Build the validator every record is checked against.

    Cached, because every record in every run goes through it and the schema is
    a file that does not change while the process runs. Compiling it per record
    would re-read and re-check the same file a few dozen times a run.

    Returns:
        Draft202012Validator: A validator for one action-log record.

    Raises:
        TraceError: If the shipped schema is not itself a valid JSON Schema.
            Checked here rather than trusted, so a typo in a keyword name is
            caught as a broken schema instead of silently validating nothing.
    """
    schema = action_log_schema()
    try:
        Draft202012Validator.check_schema(schema)
    except SchemaValidationError as error:
        raise TraceError(f"{SCHEMA_PATH} is not a valid JSON Schema: {error.message}") from error
    return Draft202012Validator(schema)


def validate_record(record: dict[str, Any]) -> None:
    """Check one action-log record against the schema.

    Args:
        record: One step, already rendered as plain data.

    Raises:
        TraceError: If the record does not conform, naming the field and what
            is wrong with it. Every path into the log goes through here, so a
            record shape that has drifted from the schema fails at the seam
            rather than three tools downstream.
    """
    try:
        _validator().validate(record)
    except SchemaValidationError as error:
        # `json_path` reads like `$.calls[0].parent_step_id`, which is the
        # single most useful thing to put in the message: it says which record
        # and which field, and it is how the schema file itself is navigated.
        raise TraceError(f"action log record at {error.json_path}: {error.message}") from error


class TraceWriter:
    """One run's folder, written step by step as the run happens.

    Built by whoever starts the run, handed to the controller, and asked to
    record each step as it completes. It owns the folder's layout and the
    format of everything in it; the controller owns *when* a step is finished
    and hands over a record rather than fields.

    Attributes:
        folder: The run's folder. Named by the caller -- in practice after the
            moment the run started -- and created here if it does not exist.
        run_id: The stable id for this run, written into ``run.json`` and into
            the outcome, so a run's files can be tied together after being
            copied somewhere else.
        opened_at: When the folder was opened, in the trace's timestamp form.
    """

    def __init__(self, folder: Path, *, run_id: str | None = None) -> None:
        """Prepare a run folder.

        Prefer :meth:`open`, which also writes the two files that describe the
        run before any step has happened. This constructor exists for the test
        that wants a writer and nothing else in the folder.

        Args:
            folder: Where the run's files go. Created, parents included, if it
                is not there.
            run_id: The run's id. A fresh one is minted when not given, which
                is the ordinary case.

        Raises:
            TraceError: If the folder cannot be created.
        """
        self.folder = folder
        self.run_id = run_id or new_id("run")
        self.opened_at = utc_now()
        try:
            # The screenshot directory is made now rather than on the first
            # screenshot: the observer writes into it directly, and a gate that
            # first fires on step 9 should not be the thing that discovers the
            # path is unwritable.
            self.screenshot_dir.mkdir(parents=True, exist_ok=True)
        except OSError as error:
            raise TraceError(f"cannot create the run folder at {folder}: {error}") from error

    @classmethod
    def open(
        cls,
        folder: Path,
        *,
        task: str,
        start_url: str,
        config: Config | None = None,
        text_only: bool = False,
    ) -> TraceWriter:
        """Open a run folder and write what is already known about the run.

        Two files land before the first step: what the run was asked to do, and
        the settings it resolved. Both are known before anything is attempted,
        and writing them now means a run that dies on its opening navigation
        still leaves a folder that says what it was trying to do.

        Args:
            folder: Where the run's files go.
            task: The task in the user's own words.
            start_url: The page the run opens on.
            config: The run's resolved settings. None for a test or a caller
                with no config, which leaves ``config.json`` unwritten rather
                than writing an empty one that would read as "no settings".
            text_only: Whether the run's endpoint serves no vision model, so no
                step could carry an image to it. Stated once here rather than
                left to be inferred from every step's screenshot trigger --
                and worth stating at all because the folder still holds a
                picture of every step, so its screenshots are no evidence
                either way about what the model was shown.

        Returns:
            TraceWriter: A writer ready to take steps.

        Raises:
            TraceError: If the folder or either file cannot be written.
        """
        writer = cls(folder)
        writer._write_json(
            RUN_FILE,
            {
                "run_id": writer.run_id,
                "task": task,
                "start_url": start_url,
                # `opened_at` rather than `started_at`: this is when the folder
                # was made, which is a little before the loop starts and is a
                # different fact from the run's own span in `outcome.json`.
                # Two files calling two different moments "started_at" would be
                # a small lie that a timeline would inherit.
                "opened_at": writer.opened_at,
                # Not a setting, which is why it is here and not in
                # `config.json`: it is a fact about the endpoint the run was
                # pointed at.
                "text_only": text_only,
                "schema": SCHEMA_PATH.name,
            },
        )
        if config is not None:
            # `mode="json"` rather than a plain dump: the config holds tuples
            # and other types json refuses, and pydantic already knows how each
            # one is meant to be written.
            writer._write_json(CONFIG_FILE, config.model_dump(mode="json"))
        return writer

    @property
    def screenshot_dir(self) -> Path:
        """Path: Where this run's annotated screenshots go.

        Handed to the observer, which writes each step's image here itself. The
        trace does not copy images around: the observation records the path it
        wrote, and that path is what lands in the action log.
        """
        return self.folder / SCREENSHOT_DIR

    def write_step(self, step: StepReport) -> None:
        """Record one completed step, and put it on disk before returning.

        This is the flush-per-step contract, and the reason it is a contract
        rather than an optimisation: when this method returns, the step is on
        disk, so whatever happens to the process during the *next* step cannot
        take this one with it.

        Args:
            step: The step, as the controller finished it.

        Raises:
            TraceError: If the record does not fit the schema, or cannot be
                written.
        """
        record = step.to_dict()
        validate_record(record)
        self._append(ACTION_LOG, [record])
        # The cost log is the same call records, lifted out of the step and
        # written flat. Taken from `record` rather than from `step.records` so
        # that the two files are two views of one rendering -- there is no
        # second code path here that could come to disagree with the first.
        self._append(COST_LOG, record["calls"])

    def finish(self, report: RunReport) -> None:
        """Write the run's verdict, once there is one.

        The only thing written at exit, and deliberately the only thing: every
        step is already on disk by now. A folder without this file is a run
        that was killed, which a reader can tell at a glance.

        Args:
            report: The finished run.

        Raises:
            TraceError: If the file cannot be written.
        """
        outcome = report.outcome
        self._write_json(
            OUTCOME_FILE,
            {
                "run_id": self.run_id,
                "task": report.task,
                "start_url": report.start_url,
                "started_at": report.started_at,
                "ended_at": report.ended_at,
                "steps": len(report.steps),
                "outcome": outcome.to_dict(),
                # Written out even when the agent never chose a `done`, so an
                # action log always ends with an explicit ending rather than
                # simply stopping.
                "final_action": outcome.as_done().to_dict(),
                "cost": report.cost.to_dict(),
            },
        )

    def _write_json(self, name: str, payload: dict[str, Any]) -> None:
        """Write one whole-file JSON document into the run folder.

        Args:
            name: The file's name inside the folder.
            payload: What to write.

        Raises:
            TraceError: If it cannot be written.
        """
        self._flush_to_disk(name, json.dumps(payload, indent=2, ensure_ascii=False) + "\n")

    def _append(self, name: str, records: list[dict[str, Any]]) -> None:
        """Append JSON Lines to one of the run's logs.

        Args:
            name: The log's name inside the folder.
            records: The objects to append, each on its own line. An empty list
                writes nothing, which is what a step that made no model calls
                should add to the cost log.

        Raises:
            TraceError: If they cannot be written.
        """
        if not records:
            return
        lines = "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records)
        self._flush_to_disk(name, lines, append=True)

    def _flush_to_disk(self, name: str, text: str, *, append: bool = False) -> None:
        """Put text in the run folder and make the operating system commit it.

        Opened and closed per write rather than held open for the run. A handle
        held across a run is faster and loses buffered lines to a process that
        never gets to close it -- and a run writes a few dozen of these, so the
        speed is worth nothing and the loss is the whole point of the module.

        Args:
            name: The file's name inside the folder.
            text: What to write. Already ends in a newline, so an append lands
                on its own line.
            append: True to add to the file, False to replace it.

        Raises:
            TraceError: If the write fails.
        """
        path = self.folder / name
        try:
            with path.open("a" if append else "w", encoding="utf-8") as handle:
                handle.write(text)
                # Two steps, and both are needed: `flush` moves the bytes out
                # of Python's buffer into the kernel, and `fsync` makes the
                # kernel commit them to the disk. Without the second, a power
                # loss or a hard kill of the machine can still lose a line that
                # Python believed it had written.
                handle.flush()
                os.fsync(handle.fileno())
        except OSError as error:
            raise TraceError(f"cannot write {path}: {error}") from error


def read_action_log(folder: Path) -> list[dict[str, Any]]:
    """Read back every completed step in a run folder.

    The reader half of the flush-per-step contract, and the one a test uses to
    check it: a run killed mid-step leaves a log whose last line is the last
    step that *finished*, and this parses it without complaint.

    Args:
        folder: The run folder.

    Returns:
        list[dict[str, Any]]: The step records, in the order they happened. An
        empty list for a run that was killed before its first step finished,
        which is a true reading of that folder rather than an error.

    Raises:
        TraceError: If the log exists but a line is not a JSON object. A
            half-written line would mean the flush-per-step contract had been
            broken, which is worth failing loudly over rather than skipping.
    """
    path = folder / ACTION_LOG
    if not path.is_file():
        return []
    records: list[dict[str, Any]] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise TraceError(f"{path} line {number} is not valid JSON: {error}") from error
        if not isinstance(record, dict):
            raise TraceError(f"{path} line {number} is not a JSON object")
        records.append(record)
    return records
