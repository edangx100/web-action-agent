"""Turning finished benchmark sweeps into the results tables.

A sweep folder (see :mod:`~web_action_agent.eval_harness`) holds one run
folder and one mark per trial. This module reads them and writes the tables
that get published: one Markdown file per sweep, and one comparing the sweeps.

**The headline number: success rate, as a mean with a range.** Each task is
tried several times, because live websites are unpredictable and a single try
is mostly luck. So the success rate is worked out separately for each *round*
of trials -- all the tasks' first trials, then all their second trials, and so
on -- and the table shows the average of those rounds and the lowest and
highest. For example, rounds of 10/12, 8/12 and 9/12 give::

    75.0% (range 66.7%–83.3% over 3 rounds)

The range is the honest part. "75%" alone hides that one round was 67%.

**The other numbers**, averaged per trial: steps taken, tokens used (what the
AI read and wrote), wall-clock time, and dollars.

**Dollars only when known.** Tokens and time are always known, so they are
always shown. A dollar figure is shown only when *every* call behind it had a
price. Where any call had none, the table says ``unknown`` and how many calls
were unpriced, rather than inventing a total by treating them as free. The
judge's own cost is shown separately: it is the price of *marking* the agent,
not of the agent doing the task.

**Every verdict links to its evidence.** The last table lists every trial, its
mark, and a link to the run folder it came from, so any number in the results
can be traced back to the screenshots and steps that produced it.

**Stamped with when.** Websites change, so a result means little without a
date. Each table says when its sweep started and when its last trial was
marked.

**Every failure says what kind it was, or there is no table.** A failure
with no category -- grounding, planning, navigation, or site change -- would
be a blank cell nobody chases, and the failure analysis the project publishes
would quietly cover only some of the failures. So :func:`check_categories`
runs before anything is written, and stops with a list of every such failure
and how to fix it. A pass that carries a category is stopped the same way:
it means the mark was made inconsistently.

**Only marked trials count.** A trial with no pass or fail yet -- a judging
error, or waiting for a judge -- is left out of the rates and counted on its
own line, so it can neither help nor hurt the score without anyone noticing.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime
from os.path import relpath
from pathlib import Path
from typing import Any

from .budget import JUDGE_LOG, read_calls
from .eval_harness import SWEEP_FILE, VERDICT_FILE
from .judge import FAIL, FAILURE_CATEGORIES, PASS
from .trace import OUTCOME_FILE

#: The file comparing every sweep, in the results folder.
COMPARISON_FILE = "README.md"

#: Where human verdicts are kept, one JSON object per line. Written by the
#: review tool; each record names a trial folder and the human's verdict.
HUMAN_LABELS_FILE = "human_labels.jsonl"


class MetricsError(Exception):
    """A sweep folder could not be read as results."""


# --- One trial -------------------------------------------------------------------


@dataclass(frozen=True)
class TrialRow:
    """One trial's result, as the tables need it.

    Attributes:
        task_id: The task.
        trial: Which attempt, from 1.
        folder: The trial's run folder.
        status: Its mark: ``pass``, ``fail``, or something unfinished such as
            ``judging_error``.
        category: For a fail, the kind of mistake, when the judge said.
        reason: The one-sentence reason for the mark.
        steps: Steps the agent took.
        tokens: Tokens the agent's AI calls used, or None if unreported.
        wall_s: Seconds from the run's start to its end.
        cost_usd: The agent's dollars, or None if any call had no price.
        unpriced_calls: How many of the agent's calls had no price.
        marked_at: When the mark was made.
    """

    task_id: str
    trial: int
    folder: Path
    status: str
    category: str | None
    reason: str
    steps: int
    tokens: int | None
    wall_s: float
    cost_usd: float | None
    unpriced_calls: int
    marked_at: str

    @property
    def marked(self) -> bool:
        """bool: True for a pass or a fail -- a result that counts."""
        return self.status in (PASS, FAIL)


def _seconds_between(start: str, end: str) -> float:
    """Return the seconds between two RFC 3339 timestamps.

    Args:
        start: The earlier time.
        end: The later time.

    Returns:
        float: Seconds, or 0.0 if either cannot be read.
    """
    try:
        return (datetime.fromisoformat(end) - datetime.fromisoformat(start)).total_seconds()
    except (TypeError, ValueError):
        return 0.0


def read_trial(folder: Path, task_id: str, trial: int) -> TrialRow:
    """Read one trial's result from its run folder.

    Args:
        folder: The trial's run folder.
        task_id: The task.
        trial: Which attempt.

    Returns:
        TrialRow: The result.

    Raises:
        MetricsError: If the folder has no readable outcome or mark.
    """
    try:
        outcome = json.loads((folder / OUTCOME_FILE).read_text(encoding="utf-8"))
        verdict = json.loads((folder / VERDICT_FILE).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise MetricsError(f"{folder}: cannot read its outcome or mark: {error}") from error

    cost = outcome.get("cost") or {}
    return TrialRow(
        task_id=task_id,
        trial=trial,
        folder=folder,
        status=verdict.get("status", ""),
        category=verdict.get("category"),
        reason=verdict.get("reason", ""),
        steps=int(outcome.get("steps", 0)),
        tokens=cost.get("total_tokens"),
        wall_s=_seconds_between(outcome.get("started_at", ""), outcome.get("ended_at", "")),
        cost_usd=cost.get("cost_usd"),
        unpriced_calls=int(cost.get("unpriced_calls", 0)),
        marked_at=verdict.get("marked_at", ""),
    )


# --- One sweep -------------------------------------------------------------------


@dataclass(frozen=True)
class Rate:
    """A success rate given as a mean over rounds, with its lowest and highest.

    Attributes:
        mean: The average of the rounds' rates, from 0 to 1.
        low: The lowest round's rate.
        high: The highest round's rate.
        rounds: How many rounds had at least one marked trial.
        passed: Passes across all rounds.
        marked: Marked trials across all rounds.
    """

    mean: float
    low: float
    high: float
    rounds: int
    passed: int
    marked: int

    def render(self) -> str:
        """Render as the table shows it.

        Returns:
            str: e.g. ``75.0% (range 66.7%–83.3% over 3 rounds)``. With one
            round there is no spread to show, and the text says that a single
            round is not enough to rely on.
        """
        if self.rounds == 0:
            return "no marked trials"
        if self.rounds == 1:
            return f"{self.mean:.1%} (1 round only: no range, treat as a rough guide)"
        return f"{self.mean:.1%} (range {self.low:.1%}–{self.high:.1%} over {self.rounds} rounds)"


def success_rate(rows: Iterable[TrialRow]) -> Rate:
    """Work out the success rate as a mean over rounds, with a range.

    Round ``n`` is every task's ``n``-th trial. Each round's rate is its
    passes divided by its marked trials; unmarked trials are left out.

    Args:
        rows: Trial results.

    Returns:
        Rate: The mean of the round rates, the lowest and highest, and the
        counts behind them.
    """
    by_round: dict[int, list[bool]] = {}
    for row in rows:
        if row.marked:
            by_round.setdefault(row.trial, []).append(row.status == PASS)
    rates = [sum(results) / len(results) for _, results in sorted(by_round.items())]
    if not rates:
        return Rate(0.0, 0.0, 0.0, 0, 0, 0)
    passed = sum(sum(results) for results in by_round.values())
    marked = sum(len(results) for results in by_round.values())
    return Rate(sum(rates) / len(rates), min(rates), max(rates), len(rates), passed, marked)


@dataclass(frozen=True)
class Averages:
    """Per-trial averages over a set of trials.

    Attributes:
        trials: How many trials were averaged.
        steps: Average steps.
        tokens: Average tokens, or None if any trial's count is unknown.
        wall_s: Average seconds.
        cost_usd: Average dollars, or None if any trial had an unpriced call.
        unpriced_calls: Unpriced calls across the trials.
    """

    trials: int
    steps: float
    tokens: float | None
    wall_s: float
    cost_usd: float | None
    unpriced_calls: int


def averages(rows: Sequence[TrialRow]) -> Averages:
    """Average steps, tokens, time and dollars over some trials.

    Args:
        rows: The trials.

    Returns:
        Averages: The averages. Dollars are None if *any* trial had an
        unpriced call: an average that silently leaves some calls out is not
        an average of the cost.
    """
    n = len(rows)
    if n == 0:
        return Averages(0, 0.0, None, 0.0, None, 0)
    tokens = [r.tokens for r in rows]
    costs = [r.cost_usd for r in rows]
    return Averages(
        trials=n,
        steps=sum(r.steps for r in rows) / n,
        tokens=None if any(t is None for t in tokens) else sum(tokens) / n,  # type: ignore[arg-type]
        wall_s=sum(r.wall_s for r in rows) / n,
        cost_usd=None if any(c is None for c in costs) else sum(costs) / n,  # type: ignore[arg-type]
        unpriced_calls=sum(r.unpriced_calls for r in rows),
    )


def render_tokens(value: float | None) -> str:
    """Render an average token count, or ``unknown``."""
    return "unknown" if value is None else f"{value:,.0f}"


def render_cost(value: float | None, unpriced_calls: int) -> str:
    """Render a dollar figure, or say why there is none.

    Args:
        value: Dollars, or None.
        unpriced_calls: Calls with no price behind the figure.

    Returns:
        str: e.g. ``$0.0021``, or ``unknown (12 calls unpriced)``. Never a
        figure built by counting unpriced calls as free.
    """
    if value is None:
        return f"unknown ({unpriced_calls} calls unpriced)" if unpriced_calls else "unknown"
    return f"${value:.4f}"


@dataclass(frozen=True)
class SweepResults:
    """Everything one sweep produced, ready to tabulate.

    Attributes:
        name: The sweep folder's name, such as ``eval-openrouter-cold``.
        folder: The sweep folder.
        spec: The contents of its ``sweep.json``.
        rows: Every trial with a finished, marked or unmarked, run.
        judge_cost_usd: What the judge's calls cost, or None if any had no
            price.
        judge_calls: How many judge calls there were.
    """

    name: str
    folder: Path
    spec: dict[str, Any]
    rows: tuple[TrialRow, ...]
    judge_cost_usd: float | None
    judge_calls: int

    @property
    def marked(self) -> tuple[TrialRow, ...]:
        """tuple[TrialRow, ...]: The trials with a pass or a fail."""
        return tuple(r for r in self.rows if r.marked)

    @property
    def trials_per_task(self) -> int:
        """int: The highest trial number reached."""
        return max((r.trial for r in self.rows), default=0)

    @property
    def last_marked(self) -> str:
        """str: When the latest mark was made, or empty."""
        return max((r.marked_at for r in self.rows if r.marked_at), default="")


def load_sweep(folder: Path) -> SweepResults:
    """Read every trial of one sweep.

    Args:
        folder: The sweep folder.

    Returns:
        SweepResults: The sweep's trials, in task then trial order.

    Raises:
        MetricsError: If the folder holds no ``sweep.json``.
    """
    try:
        spec = json.loads((folder / SWEEP_FILE).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise MetricsError(f"{folder} is not a sweep folder: {error}") from error

    rows: list[TrialRow] = []
    # Two levels down: <task>/trial-<n>/. Killed runs, three levels down under
    # _interrupted/, never produced a result and are never read.
    for verdict in sorted(folder.glob(f"*/trial-*/{VERDICT_FILE}")):
        trial_dir = verdict.parent
        number = int(trial_dir.name.removeprefix("trial-"))
        rows.append(read_trial(trial_dir, trial_dir.parent.name, number))
    rows.sort(key=lambda r: (r.task_id, r.trial))

    judge_records = read_calls(folder / JUDGE_LOG)
    costs = [r.cost_usd for r in judge_records]
    judge_cost = None if any(c is None for c in costs) else sum(costs)  # type: ignore[arg-type]
    return SweepResults(folder.name, folder, spec, tuple(rows), judge_cost, len(judge_records))


# --- Agreement with humans ----------------------------------------------------------


def read_human_labels(path: Path) -> dict[str, bool]:
    """Read human verdicts, keyed by trial folder.

    Args:
        path: A ``human_labels.jsonl`` file. Each line is an object with
            ``run``, the trial folder's path, and ``passed``, the human's
            verdict. A relative ``run`` is relative to the file's own folder,
            so it means the same thing whatever folder this is called from.
            Later lines for the same run replace earlier ones.

    Returns:
        dict[str, bool]: The human's verdict for each trial folder, keyed by
        its resolved path. Empty if there is no file yet.
    """
    labels: dict[str, bool] = {}
    if not path.exists():
        return labels
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            record = json.loads(line)
            # Joining an absolute path leaves it unchanged, so both kinds work.
            run = (path.parent / record["run"]).resolve()
            labels[str(run)] = bool(record["passed"])
        except (json.JSONDecodeError, KeyError, TypeError):
            continue
    return labels


def raw_agreement(rows: Iterable[TrialRow], labels: dict[str, bool]) -> tuple[int, int]:
    """Count how often the recorded verdict and a human agreed.

    Args:
        rows: Trials.
        labels: Human verdicts, from :func:`read_human_labels`.

    Returns:
        tuple[int, int]: Agreements, and trials a human labelled. Only marked
        trials count: there is nothing to agree with on an unfinished one.
    """
    agree = total = 0
    for row in rows:
        human = labels.get(str(row.folder.resolve()))
        if human is None or not row.marked:
            continue
        total += 1
        agree += int(human == (row.status == PASS))
    return agree, total


# --- Every failure categorised -------------------------------------------------------


def category_problems(rows: Iterable[TrialRow]) -> list[str]:
    """List every trial whose category does not fit its verdict.

    Args:
        rows: Trial results.

    Returns:
        list[str]: One line per problem, naming the trial's folder: a fail
        whose category is missing or not one of the four, or a pass that
        carries a category. Empty when every trial is consistent.
    """
    problems: list[str] = []
    for row in rows:
        if row.status == FAIL and row.category not in FAILURE_CATEGORIES:
            found = "no category" if row.category is None else f"category {row.category!r}"
            problems.append(f"{row.folder}: a fail with {found}")
        elif row.status == PASS and row.category is not None:
            problems.append(f"{row.folder}: a pass that carries category {row.category!r}")
    return problems


def check_categories(sweeps: Iterable[SweepResults]) -> None:
    """Refuse to tabulate sweeps with a failure that does not say what kind.

    Args:
        sweeps: The sweeps about to be written.

    Raises:
        MetricsError: Listing every problem, and how to fix the usual one.
            Raised before any file is written, so a refusal never leaves half
            the tables updated and half stale.
    """
    problems = [line for sweep in sweeps for line in category_problems(sweep.rows)]
    if problems:
        raise MetricsError(
            f"{len(problems)} trial(s) cannot go in a results table, because every failure "
            "must say what kind it was (grounding, planning, navigation, or site_change):\n  "
            + "\n  ".join(problems)
            + "\nA fail with no category usually means no judge was configured, or the "
            "judge could not be reached. Set WAA_JUDGE_PROFILE (for example "
            "openrouter-judge) and run the same `waa eval` command again: it fills in "
            "the missing categories without running the trials again."
        )


# --- The tables ---------------------------------------------------------------------


def _cell(text: str) -> str:
    """Make text safe inside a Markdown table cell.

    A ``|`` would end the cell early and a line break would end the row, so
    both are replaced.
    """
    return text.replace("|", "\\|").replace("\n", " ")


def _link(target: Path, from_dir: Path) -> str:
    """Write a relative link from a results file to a folder.

    Args:
        target: The folder linked to.
        from_dir: The folder the results file is in.

    Returns:
        str: A path relative to ``from_dir``, with forward slashes, so the
        link works wherever the repository is checked out.
    """
    return Path(relpath(target.resolve(), from_dir.resolve())).as_posix()


def sweep_table(results: SweepResults, results_dir: Path, labels: dict[str, bool]) -> str:
    """Write the Markdown results for one sweep.

    Args:
        results: The sweep.
        results_dir: The folder the file will be written to, so that links to
            run folders can be made relative to it.
        labels: Human verdicts, for the agreement line.

    Returns:
        str: The whole Markdown document.

    Raises:
        MetricsError: If any failure has no category; see
            :func:`check_categories`.
    """
    check_categories([results])
    spec = results.spec
    rate = success_rate(results.rows)
    overall = averages(results.marked)
    unmarked = len(results.rows) - len(results.marked)
    agree, labelled = raw_agreement(results.rows, labels)
    judge = spec.get("judge_profile") or "none"

    lines = [
        f"# Results: {results.name}",
        "",
        (
            f"Suite **{spec.get('suite', '?')}** · planner **{spec.get('profile', '?')}** "
            f"(`{spec.get('model', '?')}`) · library **{spec.get('library', '?')}** · "
            f"judge **{judge}**"
        ),
        "",
        (
            f"Evaluated: sweep started {spec.get('started_at', 'unknown')}; "
            f"last trial marked {results.last_marked or 'not yet'}. Live websites change, "
            "so these numbers describe the sites as they were then."
        ),
        "",
        "## Summary",
        "",
        "| Measure | Value |",
        "|---|---|",
        f"| Success rate | {rate.render()} |",
        f"| Trials passed | {rate.passed} of {rate.marked} marked |",
        f"| Trials not yet marked | {unmarked} (left out of every figure) |",
        f"| Average steps per trial | {overall.steps:.1f} |",
        f"| Average tokens per trial | {render_tokens(overall.tokens)} |",
        f"| Average time per trial | {overall.wall_s:.1f} s |",
        (
            "| Average cost per trial (agent) | "
            f"{render_cost(overall.cost_usd, overall.unpriced_calls)} |"
        ),
        (
            "| Cost of judging (not the agent) | "
            + (
                f"{render_cost(results.judge_cost_usd, 0)} over {results.judge_calls} calls"
                if results.judge_calls
                # "$0.0000" would read as a measured cost of nothing.
                else "none: no judge calls"
            )
            + " |"
        ),
        "| Judge–human agreement | "
        + (
            f"{agree / labelled:.1%} ({agree} of {labelled} labelled trials)"
            if labelled
            else "not yet measured (no human labels)"
        )
        + " |",
        "",
        "## Per task",
        "",
        "| Task | Passed | Avg steps | Avg tokens | Avg time | Avg cost |",
        "|---|---|---|---|---|---|",
    ]
    for task_id in sorted({r.task_id for r in results.rows}):
        task_rows = [r for r in results.marked if r.task_id == task_id]
        avg = averages(task_rows)
        passed = sum(r.status == PASS for r in task_rows)
        lines.append(
            f"| {task_id} | {passed}/{len(task_rows)} | {avg.steps:.1f} | "
            f"{render_tokens(avg.tokens)} | {avg.wall_s:.1f} s | "
            f"{render_cost(avg.cost_usd, avg.unpriced_calls)} |"
        )

    failures = [r for r in results.marked if r.status == FAIL]
    lines += [
        "",
        "## Failures by kind",
        "",
        "| " + " | ".join(FAILURE_CATEGORIES) + " |",
        "|" + "---|" * len(FAILURE_CATEGORIES),
        "| "
        + " | ".join(str(sum(r.category == c for r in failures)) for c in FAILURE_CATEGORIES)
        + " |",
        "",
        "## Every verdict",
        "",
        "Each row links to the run folder the verdict was made from.",
        "",
        "| Task | Trial | Verdict | Kind of failure | Steps | Reason | Run |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in results.rows:
        kind = row.category or ""
        lines.append(
            f"| {row.task_id} | {row.trial} | {row.status} | {kind} | {row.steps} | "
            f"{_cell(row.reason)} | [run]({_link(row.folder, results_dir)}) |"
        )
    return "\n".join(lines) + "\n"


def comparison_table(sweeps: Sequence[SweepResults], results_dir: Path) -> str:
    """Write the Markdown table comparing every sweep.

    This is the table the cold-versus-warm and multi-model comparisons are
    read from: one row per sweep, same measures in every column.

    Args:
        sweeps: The sweeps to compare.
        results_dir: The folder the file will be written to.

    Returns:
        str: The whole Markdown document.
    """
    lines = [
        "# Benchmark results",
        "",
        (
            "One row per sweep. A sweep is one pass over the benchmark with one model "
            "and one skill library (cold: empty; warm: learned). Success rate is the "
            "mean over rounds of trials, with the lowest and highest round. Tokens and "
            "time are always shown; dollars only where every call had a price."
        ),
        "",
        (
            "| Sweep | Planner model | Library | Trials per task | Success rate | "
            "Avg steps | Avg tokens | Avg time | Avg cost | Started |"
        ),
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for sweep in sweeps:
        rate = success_rate(sweep.rows)
        avg = averages(sweep.marked)
        link = _link(results_dir / f"{sweep.name}.md", results_dir)
        lines.append(
            f"| [{sweep.name}]({link}) | `{sweep.spec.get('model', '?')}` | "
            f"{sweep.spec.get('library', '?')} | {sweep.trials_per_task} | {rate.render()} | "
            f"{avg.steps:.1f} | {render_tokens(avg.tokens)} | {avg.wall_s:.1f} s | "
            f"{render_cost(avg.cost_usd, avg.unpriced_calls)} | "
            f"{sweep.spec.get('started_at', '?')} |"
        )
    return "\n".join(lines) + "\n"


def find_sweeps(runs_root: Path) -> list[Path]:
    """Find every sweep folder directly under a runs folder.

    Args:
        runs_root: Usually ``runs``.

    Returns:
        list[Path]: Folders holding a ``sweep.json``, sorted by name.
    """
    return sorted(p.parent for p in runs_root.glob(f"*/{SWEEP_FILE}"))


def write_results(sweep_dirs: Iterable[Path], results_dir: Path) -> list[Path]:
    """Write the tables for some sweeps, and the comparison of them all.

    Args:
        sweep_dirs: The sweep folders.
        results_dir: Where the Markdown files go. Created if needed.

    Returns:
        list[Path]: The files written: one per sweep, then the comparison.

    Raises:
        MetricsError: If a folder is not a sweep folder, or any failure in
            any of the sweeps has no category. Checked for every sweep before
            any file is written.
    """
    sweeps = [load_sweep(folder) for folder in sweep_dirs]
    check_categories(sweeps)
    results_dir.mkdir(parents=True, exist_ok=True)
    labels = read_human_labels(results_dir / HUMAN_LABELS_FILE)
    written: list[Path] = []
    for sweep in sweeps:
        path = results_dir / f"{sweep.name}.md"
        path.write_text(sweep_table(sweep, results_dir, labels), encoding="utf-8")
        written.append(path)
    comparison = results_dir / COMPARISON_FILE
    comparison.write_text(comparison_table(sweeps, results_dir), encoding="utf-8")
    written.append(comparison)
    return written
