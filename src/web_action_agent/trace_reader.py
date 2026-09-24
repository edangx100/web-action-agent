"""Rebuilding a whole run from its folder, without re-running anything.

The trace writer's claim is that a run folder holds *enough information to
replay or audit the run without re-executing it*. That is a claim about a
reader, and until a reader exists it is an assertion nobody has checked. This
is the reader, and its existence is what turns the folders from decoration into
evidence.

**The test of a trace is whether it can be read back.** A folder can look
complete, validate against its schema, and still be missing the one field that
would let somebody reconstruct what happened. The only way to find that out is
to reconstruct it -- so this module rebuilds the *same objects the run was made
of*: the observation each step decided against, the action it chose, what came
of it, what every model call cost. Not a summary of them, and not a parallel set
of types that merely resemble them. If the round trip cannot produce the
original, the trace was not evidence.

**Nothing here knows the shape of a record.** Every part is rebuilt by
``from_dict`` on whichever class wrote it, exactly as the writer built each part
through ``to_dict`` on the same class. That is the whole reason this module is
short. A reader that parsed the records itself would be a second description of
every record in the project, kept correct by hand, and wrong in some corner
within a month -- which is the failure it is most important to avoid here,
because a trace reader that is quietly wrong produces confident answers about
runs that did not happen that way.

**A killed run is an ordinary input, not an error.** Flushing per step exists so
that a run cut short still leaves a readable folder, and a reader that refused
such a folder would throw away the guarantee at the moment it mattered. So a run
with no verdict comes back as a run with no verdict -- :attr:`ReplayedRun.
finished` says which, and every step that completed is there either way.

**Reading is strict about what a record claims.** A missing timestamp, an id
that is not there, an action that does not fit the action schema: all of them
raise rather than being filled in with a plausible default. A folder that cannot
be read is a finding; a folder that is read into something subtly different from
what happened is a trap, because everything downstream believes it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .controller import RunReport, StepReport
from .cost import RunCost, total_cost
from .model_client import CallRecord
from .outcomes import STEP_CAP, RunOutcome
from .trace import (
    ACTION_LOG,
    CONFIG_FILE,
    COST_LOG,
    OUTCOME_FILE,
    RUN_FILE,
    TraceError,
    read_action_log,
)


class TraceReadError(TraceError):
    """A run folder is missing, unreadable, or does not describe a run.

    A subclass of :class:`~web_action_agent.trace.TraceError` so that a caller
    working with run folders can catch one exception for both halves of the
    job. Reading and writing fail for the same kinds of reason -- a path that is
    not there, a file that is not JSON, a record whose shape has drifted -- and
    splitting them into unrelated exception types would make every caller
    handle the same situation twice.
    """


@dataclass(frozen=True)
class ReplayedRun:
    """One whole run, rebuilt from its folder.

    Frozen, like every record it is made of. This is what a run *was*, read back
    from the evidence, and a reconstruction that could be edited afterwards
    would be worth no more than a fresh guess.

    Attributes:
        folder: Where it was read from, so a finding can name its own source.
        run_id: The run's stable id, as the folder recorded it.
        task: The task in the user's own words.
        start_url: The page the run opened on.
        text_only: Whether the run's endpoint served no vision model, so no step
            could have carried an image to it. Recorded rather than inferred
            from the steps: a run folder keeps a screenshot of every step
            whatever the endpoint could see, so its pictures are no evidence
            either way.
        steps: Every step that completed, in order.
        outcome: The verdict, or None for a run that was killed before it
            reached one. The one field whose absence is a fact about the run
            rather than a fault in the folder.
        config: The settings the run resolved, or an empty mapping when the
            folder holds none.
        calls: Every model call the run made, in step order. Taken from the
            steps rather than from the cost log, because the steps are the
            authority on which call belonged where; see
            :func:`cost_log_agrees_with_steps` for holding the two level.
    """

    folder: Path
    run_id: str
    task: str
    start_url: str
    text_only: bool = False
    steps: tuple[StepReport, ...] = ()
    outcome: RunOutcome | None = None
    config: dict[str, Any] = field(default_factory=dict)
    calls: tuple[CallRecord, ...] = ()

    @property
    def finished(self) -> bool:
        """bool: Whether the run reached a verdict.

        False for a run that was interrupted -- a Ctrl-C, a worker that hit its
        wall clock, a machine that stopped. Such a folder is a legitimate and
        readable record of what happened up to that point, and this is how a
        caller tells it apart from a run that ended.
        """
        return self.outcome is not None

    @property
    def cost(self) -> RunCost:
        """RunCost: What the run's model calls came to, rolled up.

        Recomputed from the call records rather than read from the verdict
        file, on purpose. The verdict's own total is what the run *said* it
        spent; this is what its calls actually add up to, and a reader that
        simply repeated the claim could not notice the two disagreeing. The
        test that they agree is worth more than either number alone.
        """
        return total_cost(self.calls)

    @property
    def screenshots(self) -> tuple[Path, ...]:
        """tuple[Path, ...]: Every step's picture, in step order.

        Read from what each step recorded rather than by listing the folder, so
        the images come back tied to the steps they belong to. Steps that left
        no picture are skipped rather than represented by a gap, since a caller
        that wants the pairing has the steps.
        """
        return tuple(
            step.observation.screenshot_on_disk
            for step in self.steps
            if step.observation.screenshot_on_disk is not None
        )

    def as_report(self) -> RunReport:
        """Render this run in the same shape the controller produced it in.

        What makes the reconstruction checkable: the object that comes out of a
        folder can be compared, field for field, against the object that went
        into it. Anything a run folder cannot carry across that comparison is a
        gap in the folder rather than in the comparison.

        Returns:
            RunReport: The run, as the loop would have reported it. A run with
            no verdict is reported as a step-cap failure carrying the reason it
            has no verdict -- the report type has no way to say "unfinished",
            and inventing a success or a silent stop would be worse than saying
            plainly that the run was cut short.
        """
        outcome = self.outcome or RunOutcome.failure(
            STEP_CAP,
            "the run was interrupted before it reached a verdict",
            steps=len(self.steps),
        )
        return RunReport(
            task=self.task,
            start_url=self.start_url,
            outcome=outcome,
            started_at=self.steps[0].started_at if self.steps else "",
            ended_at=self.steps[-1].ended_at if self.steps else "",
            steps=self.steps,
            cost=self.cost,
        )

    def summary(self) -> str:
        """Render the run as the few lines somebody auditing it would read.

        Returns:
            str: One line per step, then the verdict, then the cost -- the same
            shape the run printed while it happened, rebuilt from the folder
            alone. That it reads identically is the point: if a trace is
            evidence, it should be able to say what the run said.
        """
        lines = [step.summary() for step in self.steps]
        if self.outcome is not None:
            lines.append(self.outcome.summary())
        else:
            lines.append(f"no verdict: this run was interrupted after {len(self.steps)} steps")
        lines.append(self.cost.summary())
        return "\n".join(lines)


def read_run(folder: Path) -> ReplayedRun:
    """Rebuild a whole run from its folder.

    Args:
        folder: A run folder, as the trace writer left it.

    Returns:
        ReplayedRun: Everything the folder records, with each part rebuilt by
        the class that wrote it.

    Raises:
        TraceReadError: If the folder is not there, is not a run folder, or
            holds a record that cannot be read. Raised rather than worked
            around, because the alternative -- a run reconstructed with the
            unreadable parts quietly left out -- is a confident answer about
            something that did not happen.
    """
    if not folder.is_dir():
        raise TraceReadError(f"{folder} is not a run folder")

    described = _read_json(folder / RUN_FILE, required=True)
    steps = _read_steps(folder)
    outcome_file = _read_json(folder / OUTCOME_FILE, required=False)

    outcome = None
    if outcome_file is not None:
        try:
            outcome = RunOutcome.from_dict(outcome_file["outcome"])
        except (KeyError, TypeError, ValueError) as error:
            raise TraceReadError(
                f"{folder / OUTCOME_FILE} does not hold a verdict: {error}"
            ) from error

    return ReplayedRun(
        folder=folder,
        run_id=_required(described, "run_id", folder / RUN_FILE),
        task=_required(described, "task", folder / RUN_FILE),
        start_url=_required(described, "start_url", folder / RUN_FILE),
        text_only=described.get("text_only", False),
        steps=steps,
        outcome=outcome,
        config=_read_json(folder / CONFIG_FILE, required=False) or {},
        calls=_calls_of(steps),
    )


def _read_steps(folder: Path) -> tuple[StepReport, ...]:
    """Rebuild every completed step from the action log.

    Args:
        folder: The run folder.

    Returns:
        tuple[StepReport, ...]: The steps, in the order they happened.

    Raises:
        TraceReadError: If a record cannot be read, naming the step it was.
            The step number is in the message because "record 14 is broken" is
            actionable and "a record is broken" is not.
    """
    steps = []
    for position, record in enumerate(read_action_log(folder), start=1):
        try:
            steps.append(StepReport.from_dict(record))
        except Exception as error:
            # Deliberately broad: every `from_dict` below this raises its own
            # kind -- a missing key, a bad action, an unknown outcome category
            # -- and a caller wants one exception naming the record, not five
            # types they have to know about to catch.
            raise TraceReadError(
                f"{folder / ACTION_LOG}: step {record.get('number', position)} "
                f"cannot be read: {error}"
            ) from error
    return tuple(steps)


def _calls_of(steps: tuple[StepReport, ...]) -> tuple[CallRecord, ...]:
    """Collect every model call the run made.

    Taken from the steps rather than from the cost log, even though the cost log
    holds the same records. The steps are the authority: they say which call
    belonged where, and a total built from them cannot include a call that no
    step claims. The cost log is the flattened view for a reader with a
    spreadsheet, and :func:`cost_log_agrees_with_steps` is how the two are held
    level.

    Args:
        steps: The steps already rebuilt.

    Returns:
        tuple[CallRecord, ...]: Every call record, in step order.
    """
    return tuple(record for step in steps for record in step.records)


def cost_log_agrees_with_steps(run: ReplayedRun) -> bool:
    """Check the run's two views of its model calls say the same thing.

    The action log keeps each call inside its step; the cost log keeps the same
    calls flattened, each naming its parent. They are written from one rendering
    in one call, so they cannot drift at *write* time -- but a folder that has
    been copied, edited, or truncated by hand is a different matter, and an
    auditor should be able to ask.

    Args:
        run: A rebuilt run.

    Returns:
        bool: True when the cost log holds exactly the calls the steps do, in
        the same order. True as well for a run whose folder has no cost log,
        since a run that made no calls writes none.
    """
    logged = _read_jsonl(run.folder / COST_LOG)
    if not logged:
        return not run.calls
    return [call["call_id"] for call in logged] == [call.call_id for call in run.calls]


def _read_json(path: Path, *, required: bool) -> dict[str, Any] | None:
    """Read one whole-file JSON document from a run folder.

    Args:
        path: The file to read.
        required: True when its absence is a fault in the folder, False when
            its absence is a fact about the run -- which is the case for the
            verdict of a run that was killed.

    Returns:
        dict[str, Any] | None: What the file holds, or None when it is not
        there and was not required.

    Raises:
        TraceReadError: If a required file is missing, or any file is present
            and not a JSON object.
    """
    if not path.is_file():
        if required:
            raise TraceReadError(f"{path} is missing; this is not a run folder")
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise TraceReadError(f"cannot read {path}: {error}") from error
    except json.JSONDecodeError as error:
        raise TraceReadError(f"{path} is not valid JSON: {error}") from error
    if not isinstance(data, dict):
        raise TraceReadError(f"{path} should hold a JSON object, not {type(data).__name__}")
    return data


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a JSON Lines file, a record per line.

    Args:
        path: The file to read. A missing file reads as no records, which is
            what a run that wrote none leaves behind.

    Returns:
        list[dict[str, Any]]: The records, in file order.

    Raises:
        TraceReadError: If a line is not a JSON object.
    """
    if not path.is_file():
        return []
    records = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise TraceReadError(f"{path} line {number} is not valid JSON: {error}") from error
        if not isinstance(record, dict):
            raise TraceReadError(f"{path} line {number} is not a JSON object")
        records.append(record)
    return records


def _required(data: dict[str, Any], key: str, path: Path) -> Any:
    """Read a field a run folder must have, or say which one it lacks.

    Args:
        data: The document read from ``path``.
        key: The field to read.
        path: The file it came from, for the message.

    Returns:
        Any: The value.

    Raises:
        TraceReadError: If the field is absent. Named individually rather than
            reported as a generic parse failure, because "this folder has no
            task" tells somebody what is wrong with it.
    """
    if key not in data:
        raise TraceReadError(f"{path} has no {key!r}; this is not a run folder")
    return data[key]
