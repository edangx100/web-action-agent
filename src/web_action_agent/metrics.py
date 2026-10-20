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

**Checked by a person.** Some trials are labelled by hand (see
:mod:`~web_action_agent.review`), and the tables say how often the machine's
marks matched: on pass or fail, and on the kind of failure. Each is given as a
plain percentage and as Cohen's kappa, which discounts agreement by luck (see
:mod:`~web_action_agent.agreement`). Each sweep's table shows its own share
of the labels; the comparison shows every sweep's together, which is the
figure that rests on all of them.

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

from .agreement import DEFAULT_MIN_LABELS, Agreement, measure
from .budget import JUDGE_LOG, read_calls
from .eval_harness import (
    DIRTY,
    LIBRARY_DIR,
    NOT_RECORDED,
    SERVICE_ERROR,
    SERVICE_ERROR_OUTCOMES,
    SWEEP_FILE,
    VERDICT_FILE,
)
from .judge import FAIL, FAILURE_CATEGORIES, PASS
from .skill_store import SkillStore, SkillStoreError
from .trace import ACTION_LOG, COST_LOG, OUTCOME_FILE

#: The file comparing every sweep, in the results folder.
COMPARISON_FILE = "README.md"

#: The file comparing each cold sweep with its warm partner.
COLD_VS_WARM_FILE = "cold_vs_warm.md"

#: The file comparing the same benchmark on different models.
MULTI_MODEL_FILE = "multi_model.md"

#: How the multi-model comparison's sweeps are told apart from the others:
#: each arm runs in a folder named ``models-<profile>``. The comparison is a
#: separate experiment from cold-versus-warm, run on one day on one version
#: of the code, so its arms are kept in folders of their own.
MULTI_MODEL_PREFIX = "models-"

#: The file comparing the ways of reaching a model: an aggregator, the
#: maker's own service, and a server run by the user.
ENDPOINT_MODES_FILE = "endpoint_modes.md"

#: The folder prefix of the endpoint-modes check's sweeps, one per profile:
#: ``modes-<profile>``. Kept apart for the same reason as the multi-model ones.
ENDPOINT_MODES_PREFIX = "modes-"

#: The file comparing reading pages as text alone with text plus screenshots.
DOM_VS_DUAL_FILE = "dom_vs_dual.md"

#: The folder prefix of that comparison's sweeps, one per profile:
#: ``modality-<profile>``.
MODALITY_PREFIX = "modality-"

#: Every experiment kept in folders of its own. None of their sweeps is a
#: cold-versus-warm baseline, even when cold on the primary model.
EXPERIMENT_PREFIXES = (MULTI_MODEL_PREFIX, ENDPOINT_MODES_PREFIX, MODALITY_PREFIX)

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
            ``judging_error``, or ``service_error`` for a run the AI service
            stopped, which is never a result.
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
    status = verdict.get("status", "")
    # A run the AI service stopped is not a result, even if it was marked as
    # a fail before that rule existed: it counts as unmarked until run again.
    if (outcome.get("outcome") or {}).get("category") in SERVICE_ERROR_OUTCOMES:
        status = SERVICE_ERROR
    return TrialRow(
        task_id=task_id,
        trial=trial,
        folder=folder,
        status=status,
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
        str: e.g. ``$0.0021``, or ``unknown (12 calls unpriced)``, or
        ``unknown (1 call unpriced)``. Never a figure built by counting
        unpriced calls as free.
    """
    if value is None:
        if not unpriced_calls:
            return "unknown"
        calls = "call" if unpriced_calls == 1 else "calls"
        return f"unknown ({unpriced_calls} {calls} unpriced)"
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


@dataclass(frozen=True)
class HumanVerdict:
    """A person's verdict on one trial, as read from the labels file.

    Attributes:
        passed: Pass (True) or fail (False).
        category: For a fail, the person's kind of failure. None for a pass,
            or when the line does not say.
    """

    passed: bool
    category: str | None = None


def read_human_labels(path: Path) -> dict[str, HumanVerdict]:
    """Read human verdicts, keyed by trial folder.

    Args:
        path: A ``human_labels.jsonl`` file. Each line is an object with
            ``run``, the trial folder's path, ``passed``, the human's
            verdict, and for a fail ``category``, the human's kind of
            failure. A relative ``run`` is relative to the file's own folder,
            so it means the same thing whatever folder this is called from.
            Later lines for the same run replace earlier ones.

    Returns:
        dict[str, HumanVerdict]: The human's verdict for each trial folder,
        keyed by its resolved path. Empty if there is no file yet. Lines that
        cannot be read -- such as half a line left by a crash -- are skipped.
    """
    labels: dict[str, HumanVerdict] = {}
    if not path.exists():
        return labels
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            record = json.loads(line)
            passed = record["passed"]
            # Joining an absolute path leaves it unchanged, so both kinds work.
            run = (path.parent / record["run"]).resolve()
        except (json.JSONDecodeError, KeyError, TypeError):
            continue
        # Only a real true or false counts: the text "false" is not a verdict.
        if not isinstance(passed, bool):
            continue
        category = None if passed else record.get("category")
        labels[str(run)] = HumanVerdict(passed, category)
    return labels


def label_pairs(
    rows: Iterable[TrialRow], labels: dict[str, HumanVerdict]
) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """Pair the machine's marks with the human's, trial by trial.

    Args:
        rows: Trials.
        labels: Human verdicts, from :func:`read_human_labels`.

    Returns:
        tuple: Two lists of (machine, human) pairs.

        - **Verdicts:** ``pass`` or ``fail`` from each side, for every marked
          trial a human labelled.
        - **Kinds of failure:** the two kinds, for the trials *both* sides
          called a fail. A kind only exists for a fail, so a trial one side
          passed has no kind to compare; that disagreement is already counted
          in the verdicts.

        Unmarked trials are left out of both: there is nothing to agree with.
    """
    verdicts: list[tuple[str, str]] = []
    kinds: list[tuple[str, str]] = []
    for row in rows:
        human = labels.get(str(row.folder.resolve()))
        if human is None or not row.marked:
            continue
        verdicts.append((row.status, PASS if human.passed else FAIL))
        if row.status == FAIL and not human.passed and row.category and human.category:
            kinds.append((row.category, human.category))
    return verdicts, kinds


def judge_human_agreement(
    rows: Iterable[TrialRow],
    labels: dict[str, HumanVerdict],
    minimum: int = DEFAULT_MIN_LABELS,
) -> tuple[Agreement, Agreement]:
    """Measure how well the machine's marks and a person's agree.

    Args:
        rows: Trials.
        labels: Human verdicts, from :func:`read_human_labels`.
        minimum: The fewest labelled trials a figure should rest on.

    Returns:
        tuple[Agreement, Agreement]: Agreement on pass or fail, then on the
        kind of failure (over the trials both called a fail).
    """
    verdicts, kinds = label_pairs(rows, labels)
    return measure(verdicts, minimum), measure(kinds, minimum)


def _agreement_rows(
    rows: Iterable[TrialRow], labels: dict[str, HumanVerdict], minimum: int
) -> list[str]:
    """Write the two agreement rows of a two-column Markdown table.

    Args:
        rows: Trials.
        labels: Human verdicts.
        minimum: The fewest labelled trials a figure should rest on.

    Returns:
        list[str]: The pass-or-fail row, then the kind-of-failure row.
    """
    verdict, kind = judge_human_agreement(rows, labels, minimum)
    return [
        f"| Judge–human agreement: pass or fail | {verdict.render()} |",
        (
            "| Judge–human agreement: kind of failure (trials both called a fail) | "
            # With labels but no trial both sides failed, there is no kind to
            # compare yet -- which is different from having no labels at all.
            + (
                kind.render("no trial was called a fail by both sides yet")
                if verdict.n
                else verdict.render()
            )
            + " |"
        ),
    ]


#: Said under every agreement figure, so a reader knows what kappa is.
KAPPA_EXPLAINED = (
    "Agreement is the share of hand-labelled trials where the machine's mark and "
    "a person's matched. Cohen's κ discounts the agreement the two would reach by "
    "chance alone: 1 is perfect, 0 is no better than chance."
)


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


def sweep_table(
    results: SweepResults,
    results_dir: Path,
    labels: dict[str, HumanVerdict],
    minimum: int = DEFAULT_MIN_LABELS,
) -> str:
    """Write the Markdown results for one sweep.

    Args:
        results: The sweep.
        results_dir: The folder the file will be written to, so that links to
            run folders can be made relative to it.
        labels: Human verdicts, for the agreement lines.
        minimum: The fewest labelled trials an agreement figure should rest
            on before it is relied on.

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
        *_agreement_rows(results.rows, labels, minimum),
        "",
        KAPPA_EXPLAINED,
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


def comparison_table(
    sweeps: Sequence[SweepResults],
    results_dir: Path,
    labels: dict[str, HumanVerdict] | None = None,
    minimum: int = DEFAULT_MIN_LABELS,
) -> str:
    """Write the Markdown table comparing every sweep.

    This is the table the cold-versus-warm and multi-model comparisons are
    read from: one row per sweep, same measures in every column. Below it,
    the judge–human agreement over every sweep together: the hand-labelled
    trials are spread across the sweeps, so only the pooled figure rests on
    all of them.

    Args:
        sweeps: The sweeps to compare.
        results_dir: The folder the file will be written to.
        labels: Human verdicts, for the agreement section. None for none.
        minimum: The fewest labelled trials an agreement figure should rest
            on before it is relied on.

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
    every_row = [row for sweep in sweeps for row in sweep.rows]
    lines += [
        "",
        "## Judge–human agreement",
        "",
        "Over every sweep above together. " + KAPPA_EXPLAINED,
        "",
        "| Measure | Value |",
        "|---|---|",
        *_agreement_rows(every_row, labels or {}, minimum),
    ]
    return "\n".join(lines) + "\n"


# --- Cold versus warm ---------------------------------------------------------------


def _change(cold: float | None, warm: float | None, unit: str = "") -> str:
    """Write the change from cold to warm, with its relative size.

    Args:
        cold: The cold arm's figure, or None if unknown.
        warm: The warm arm's figure, or None if unknown.
        unit: Written after the number, such as `` s``.

    Returns:
        str: For example ``-4.1 (-44.6%)``. ``unknown`` if either side is
        unknown, since a change against an unknown is not a change.
    """
    if cold is None or warm is None:
        return "unknown"
    difference = warm - cold
    text = f"{difference:+,.1f}{unit}"
    if cold:
        text += f" ({difference / cold:+.1%})"
    return text


def _cost_change(cold: Averages, warm: Averages) -> str:
    """Write the change in dollars per trial, or why it cannot be given.

    Args:
        cold: The cold arm's averages.
        warm: The warm arm's averages.

    Returns:
        str: For example ``-$0.0021 (-40.0%)``, or ``unknown`` when either
        arm had a call with no price.
    """
    if cold.cost_usd is None or warm.cost_usd is None:
        return "unknown (some calls had no price)"
    difference = warm.cost_usd - cold.cost_usd
    # The sign goes before the dollar sign: "-$0.0010", not "$-0.0010".
    text = f"{'-' if difference < 0 else '+'}${abs(difference):.4f}"
    if cold.cost_usd:
        text += f" ({difference / cold.cost_usd:+.1%})"
    return text


def _library_line(sweep: SweepResults) -> str:
    """Say what a sweep's skill library held when it started.

    Args:
        sweep: A sweep.

    Returns:
        str: For example ``9 skills (8 active) on 4 sites``, or ``empty``.
    """
    library = sweep.folder / LIBRARY_DIR
    try:
        skills = SkillStore(library).load_all() if library.is_dir() else ()
    except SkillStoreError:
        return "unreadable"
    if not skills:
        return "empty"
    active = sum(s.active for s in skills)
    sites = len({s.domain for s in skills})
    return f"{len(skills)} skills ({active} active) on {sites} site{'s' if sites != 1 else ''}"


def used_a_skill(folder: Path) -> bool:
    """Say whether a trial replayed a learnt skill, from its own record of its steps.

    What happened, not what was available: a skill in the library that the
    agent never chose to use did nothing for that trial.

    Args:
        folder: A trial's run folder.

    Returns:
        bool: True if any step of the run was a skill replay.
    """
    try:
        lines = (folder / ACTION_LOG).read_text(encoding="utf-8").splitlines()
    except OSError:
        return False
    for line in lines:
        try:
            if json.loads(line).get("replay"):
                return True
        except (json.JSONDecodeError, AttributeError):
            continue
    return False


def _group_line(label: str, cold: Sequence[TrialRow], warm: Sequence[TrialRow]) -> str:
    """Write one row of the with-skill / without-skill split.

    Args:
        label: What the group is, for the first column.
        cold: The group's marked cold trials.
        warm: The group's marked warm trials.

    Returns:
        str: The passes, the share, and the average steps on each side, and
        the change in percentage points and in steps.
    """
    if not cold or not warm:
        return f"| {label} | — | — | no such tasks |"

    def side(rows: Sequence[TrialRow]) -> tuple[str, float, float]:
        passed = sum(r.status == PASS for r in rows)
        share = passed / len(rows)
        steps = averages(rows).steps
        return f"{passed}/{len(rows)} ({share:.1%}), {steps:.1f} steps", share, steps

    cold_text, cold_share, cold_steps = side(cold)
    warm_text, warm_share, warm_steps = side(warm)
    return (
        f"| {label} | {cold_text} | {warm_text} | "
        f"{(warm_share - cold_share) * 100:+.1f} points; steps {_change(cold_steps, warm_steps)} |"
    )


def cold_warm_pairs(sweeps: Iterable[SweepResults]) -> list[tuple[SweepResults, SweepResults]]:
    """Pair each cold sweep with the warm sweep of the same suite and model.

    Args:
        sweeps: Every sweep.

    Returns:
        list[tuple[SweepResults, SweepResults]]: (cold, warm) pairs that ran
        the same suite on the same profile and model, in name order. A sweep
        with no partner is left out: there is nothing to compare it with. So
        is a sweep of another experiment (see :data:`EXPERIMENT_PREFIXES`):
        it is cold, but no warm library was learnt from it, and on the
        primary model it would otherwise compete with the cold sweep that
        one was learnt from.
    """
    by_arm: dict[tuple[Any, ...], dict[str, SweepResults]] = {}
    for sweep in sweeps:
        if sweep.name.startswith(EXPERIMENT_PREFIXES):
            continue
        spec = sweep.spec
        key = (spec.get("suite"), spec.get("profile"), spec.get("model"))
        by_arm.setdefault(key, {})[spec.get("library", "")] = sweep
    pairs = [
        (arms["cold"], arms["warm"]) for arms in by_arm.values() if {"cold", "warm"} <= arms.keys()
    ]
    return sorted(pairs, key=lambda pair: pair[0].name)


def cold_vs_warm_table(
    pairs: Sequence[tuple[SweepResults, SweepResults]], results_dir: Path
) -> str:
    """Write the Markdown comparing each cold sweep with its warm one.

    This is the table behind the project's main claim: that learning skills
    from what worked makes the agent better at the same kind of task. Each
    measure is given for both arms, with the change from cold to warm, and
    every figure states how many trials it rests on.

    Args:
        pairs: (cold, warm) sweeps, from :func:`cold_warm_pairs`.
        results_dir: The folder the file will be written to.

    Returns:
        str: The whole Markdown document.
    """
    lines = [
        "# Cold vs. warm skill library",
        "",
        (
            "The same tasks on the same model, twice. **Cold**: an empty skill library. "
            "**Warm**: the library learnt from the cold pass's passing trials, each skill "
            "checked on a live page before it was kept. Success rate is the mean over "
            "rounds of trials, with the lowest and highest round. A change in success "
            "rate is in percentage points; other changes also give the relative size."
        ),
    ]
    for cold, warm in pairs:
        cold_rate, warm_rate = success_rate(cold.rows), success_rate(warm.rows)
        cold_avg, warm_avg = averages(cold.marked), averages(warm.marked)
        cold_tasks = len({r.task_id for r in cold.marked})
        warm_tasks = len({r.task_id for r in warm.marked})
        points = (
            f"{(warm_rate.mean - cold_rate.mean) * 100:+.1f} points"
            if cold_rate.rounds and warm_rate.rounds
            else "unknown"
        )
        cold_link = _link(results_dir / f"{cold.name}.md", results_dir)
        warm_link = _link(results_dir / f"{warm.name}.md", results_dir)
        # Split by whether the warm pass actually used a skill on the task.
        # The tasks it did not are a control: their change is luck alone.
        helped = {r.task_id for r in warm.marked if used_a_skill(r.folder)}
        n_helped = len(helped)
        n_other = len({r.task_id for r in (*cold.marked, *warm.marked)} - helped)
        lines += [
            "",
            f"## {cold.spec.get('profile', '?')} (`{cold.spec.get('model', '?')}`)",
            "",
            (
                f"Suite **{cold.spec.get('suite', '?')}** · cold [{cold.name}]({cold_link}), "
                f"started {cold.spec.get('started_at', '?')} · warm [{warm.name}]({warm_link}), "
                f"started {warm.spec.get('started_at', '?')}"
            ),
            "",
            "| Measure | Cold | Warm | Change |",
            "|---|---|---|---|",
            f"| Success rate (all tasks) | {cold_rate.render()} | {warm_rate.render()} | {points} |",
            _group_line(
                f"…on tasks where a skill was used ({n_helped})",
                [r for r in cold.marked if r.task_id in helped],
                [r for r in warm.marked if r.task_id in helped],
            ),
            _group_line(
                f"…on tasks where none was ({n_other})",
                [r for r in cold.marked if r.task_id not in helped],
                [r for r in warm.marked if r.task_id not in helped],
            ),
            (
                f"| Trials marked (N) | {cold_rate.marked} ({cold_tasks} tasks × "
                f"{cold.trials_per_task}) | {warm_rate.marked} ({warm_tasks} tasks × "
                f"{warm.trials_per_task}) | |"
            ),
            (
                f"| Trials passed | {cold_rate.passed} | {warm_rate.passed} | "
                f"{warm_rate.passed - cold_rate.passed:+d} |"
            ),
            (
                f"| Average steps per trial | {cold_avg.steps:.1f} | {warm_avg.steps:.1f} | "
                f"{_change(cold_avg.steps, warm_avg.steps)} |"
            ),
            (
                f"| Average tokens per trial | {render_tokens(cold_avg.tokens)} | "
                f"{render_tokens(warm_avg.tokens)} | {_change(cold_avg.tokens, warm_avg.tokens)} |"
            ),
            (
                f"| Average time per trial | {cold_avg.wall_s:.1f} s | {warm_avg.wall_s:.1f} s | "
                f"{_change(cold_avg.wall_s, warm_avg.wall_s, ' s')} |"
            ),
            (
                "| Average cost per trial (agent) | "
                f"{render_cost(cold_avg.cost_usd, cold_avg.unpriced_calls)} | "
                f"{render_cost(warm_avg.cost_usd, warm_avg.unpriced_calls)} | "
                f"{_cost_change(cold_avg, warm_avg)} |"
            ),
            f"| Skill library at the start | {_library_line(cold)} | {_library_line(warm)} | |",
            "",
            (
                "**Read the split before the headline.** A task counts as *a skill was "
                "used* when any of its warm trials replayed a learnt skill. Only those "
                "tasks can show the effect of learning. On the others the warm pass ran "
                "exactly as the cold one did, so their change is how much the results "
                "move between passes by chance alone. The split rows count passes over "
                "all marked trials, rather than averaging rounds."
            ),
            "",
            "### Per task",
            "",
            "| Task | Skill used (warm) | Cold passed | Warm passed | Cold avg steps | Warm avg steps |",
            "|---|---|---|---|---|---|",
        ]
        for task_id in sorted({r.task_id for r in (*cold.marked, *warm.marked)}):
            cold_rows = [r for r in cold.marked if r.task_id == task_id]
            warm_rows = [r for r in warm.marked if r.task_id == task_id]
            cold_passed = sum(r.status == PASS for r in cold_rows)
            warm_passed = sum(r.status == PASS for r in warm_rows)
            used = "yes" if task_id in helped else "no"
            lines.append(
                f"| {task_id} | {used} | {cold_passed}/{len(cold_rows)} | "
                f"{warm_passed}/{len(warm_rows)} | "
                f"{averages(cold_rows).steps:.1f} | {averages(warm_rows).steps:.1f} |"
            )
    return "\n".join(lines) + "\n"


# --- Multi-model comparison -----------------------------------------------------------


def multi_model_arms(sweeps: Iterable[SweepResults]) -> list[SweepResults]:
    """Pick out the arms of the multi-model comparison.

    Args:
        sweeps: Every sweep.

    Returns:
        list[SweepResults]: The sweeps in folders named
        ``models-<profile>``, in name order.
    """
    return sorted(
        (s for s in sweeps if s.name.startswith(MULTI_MODEL_PREFIX)), key=lambda s: s.name
    )


def code_versions(spec: dict[str, Any]) -> list[str]:
    """Read the code versions a sweep ran under, from its ``sweep.json``.

    Args:
        spec: The contents of the sweep's ``sweep.json``.

    Returns:
        list[str]: The versions, oldest first. ``[NOT_RECORDED]`` for a
        sweep started before versions were kept.
    """
    return list(spec.get("code_versions", [NOT_RECORDED]))


def same_code_check(arms: Sequence[SweepResults]) -> str:
    """Say whether every arm provably ran the same code, and if not, why not.

    "Same code" is only claimed when it can be shown: each arm ran under one
    committed version, with no uncommitted changes, and it is the same one
    for every arm. Anything less is reported as it is.

    Args:
        arms: The comparison's arms.

    Returns:
        str: One sentence, starting ``yes:`` or ``no:``.
    """
    for arm in arms:
        versions = code_versions(arm.spec)
        if not versions:
            return f"no: {arm.name} has not run any trial yet"
        if NOT_RECORDED in versions:
            return f"no: {arm.name} has trials with no recorded code version"
        if len(versions) > 1:
            listed = ", ".join(f"`{v}`" for v in versions)
            return f"no: {arm.name} was restarted on different code partway ({listed})"
        if versions[0].endswith(DIRTY):
            return (
                f"no: {arm.name} ran with changes not yet committed (`{versions[0]}`), "
                "which the commit alone does not pin down"
            )
    commits = sorted({code_versions(arm.spec)[0] for arm in arms})
    if len(commits) > 1:
        return "no: the arms ran different commits: " + ", ".join(f"`{c}`" for c in commits)
    return f"yes: every trial of every arm ran commit `{commits[0]}`, with nothing uncommitted"


def _same_setting(arms: Sequence[SweepResults], name: str) -> str:
    """Say whether every arm has the same value for one ``sweep.json`` field.

    Args:
        arms: The comparison's arms.
        name: The field, such as ``suite``.

    Returns:
        str: ``yes`` with the shared value, or ``no`` with each arm's.
    """
    values = {arm.name: arm.spec.get(name) or "none" for arm in arms}
    if len(set(values.values())) == 1:
        return f"yes: {next(iter(values.values()))}"
    return "no: " + ", ".join(f"{arm} {value}" for arm, value in values.items())


def _vision_model(spec: dict[str, Any]) -> str:
    """Write the model a sweep sent screenshots to, for a table cell.

    Args:
        spec: The contents of the sweep's ``sweep.json``.

    Returns:
        str: The model in backticks; ``none`` for a profile with no vision
        route; ``not recorded`` for a sweep from before it was kept.
    """
    model = spec.get("vision_model")
    if model is None:
        return NOT_RECORDED
    return f"`{model}`" if model else "none"


def multi_model_table(arms: Sequence[SweepResults], results_dir: Path) -> str:
    """Write the Markdown comparing the same benchmark on different models.

    The claim behind the comparison is that the agent works the same way
    whichever model it is given, with nothing changed but ``--profile``. So
    beside the figures, the document checks that claim against what each
    sweep recorded: the same code, the same tasks, the same judge, and an
    empty skill library in every arm.

    Args:
        arms: The comparison's sweeps, from :func:`multi_model_arms`.
        results_dir: The folder the file will be written to.

    Returns:
        str: The whole Markdown document.
    """
    lines = [
        "# Multi-model comparison",
        "",
        (
            "The same benchmark on different AI models. Each arm is one sweep with an "
            "empty skill library (cold), and the arms are meant to differ only in "
            "`--profile`, which picks the models -- one for steps that read the page as "
            "text, one for steps that also send a screenshot. The checks below test that "
            "against what each sweep recorded. Success rate is the mean over rounds of "
            "trials, with the lowest and highest round. N is how many marked trials each "
            "figure rests on: with fewer trials per task than the headline cold-vs-warm "
            "sweeps, these figures are rougher, and the ranges show by how much."
        ),
        "",
        "## Results",
        "",
        (
            "| Arm | Planner model | Screenshot model | Trials marked (N) | Success rate | "
            "Avg steps | Avg tokens | Avg time | Avg cost (agent) | Code | Started |"
        ),
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for arm in arms:
        rate = success_rate(arm.rows)
        avg = averages(arm.marked)
        tasks = len({r.task_id for r in arm.marked})
        link = _link(results_dir / f"{arm.name}.md", results_dir)
        code = ", ".join(f"`{v}`" for v in code_versions(arm.spec)) or "none yet"
        lines.append(
            f"| [{arm.spec.get('profile', arm.name)}]({link}) | `{arm.spec.get('model', '?')}` | "
            f"{_vision_model(arm.spec)} | {rate.marked} ({tasks} tasks × "
            f"{arm.trials_per_task}) | {rate.render()} | {avg.steps:.1f} | "
            f"{render_tokens(avg.tokens)} | {avg.wall_s:.1f} s | "
            f"{render_cost(avg.cost_usd, avg.unpriced_calls)} | {code} | "
            f"{arm.spec.get('started_at', '?')} |"
        )
    lines += [
        "",
        "## Did only the model change?",
        "",
        "| Check | Result |",
        "|---|---|",
        f"| Same code in every arm | {same_code_check(arms)} |",
        f"| Same benchmark suite | {_same_setting(arms, 'suite')} |",
        f"| Same judge | {_same_setting(arms, 'judge_profile')} |",
        f"| Empty skill library in every arm | {_same_setting(arms, 'library')} |",
        "",
        (
            "A code version is the git commit the sweep ran, with `-dirty` added when "
            "the agent's code, settings, tasks or dependencies had changes not yet "
            "committed. A sweep restarted on newer code lists both versions."
        ),
        "",
        "## Per task",
        "",
        "Trials passed out of trials marked, for each arm.",
        "",
        "| Task | " + " | ".join(str(arm.spec.get("profile", arm.name)) for arm in arms) + " |",
        "|---|" + "---|" * len(arms),
    ]
    for task_id in sorted({r.task_id for arm in arms for r in arm.marked}):
        cells = []
        for arm in arms:
            rows = [r for r in arm.marked if r.task_id == task_id]
            cells.append(f"{sum(r.status == PASS for r in rows)}/{len(rows)}")
        lines.append(f"| {task_id} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


# --- Endpoint modes ---------------------------------------------------------------------


def endpoint_mode_arms(sweeps: Iterable[SweepResults]) -> list[SweepResults]:
    """Pick out the sweeps of the endpoint-modes check.

    Args:
        sweeps: Every sweep.

    Returns:
        list[SweepResults]: The sweeps in folders named ``modes-<profile>``,
        in name order.
    """
    return sorted(
        (s for s in sweeps if s.name.startswith(ENDPOINT_MODES_PREFIX)), key=lambda s: s.name
    )


def _service(spec: dict[str, Any]) -> str:
    """Write the service a sweep's calls went to, for a table cell.

    Args:
        spec: The contents of the sweep's ``sweep.json``.

    Returns:
        str: The base URL in backticks, or ``not recorded`` for a sweep
        from before it was kept.
    """
    url = spec.get("base_url")
    return f"`{url}`" if url else NOT_RECORDED


def _different_services(arms: Sequence[SweepResults]) -> str:
    """Say whether every arm reached a service of its own.

    Args:
        arms: The check's arms.

    Returns:
        str: ``yes`` with the number of services, or ``no`` naming the arms
        that shared one -- two arms on one service would show nothing about
        reaching different ones.
    """
    by_url: dict[str, list[str]] = {}
    for arm in arms:
        by_url.setdefault(str(arm.spec.get("base_url") or NOT_RECORDED), []).append(arm.name)
    shared = [names for names in by_url.values() if len(names) > 1]
    if shared:
        return "no: " + "; ".join(" and ".join(names) + " used one service" for names in shared)
    return f"yes: {len(arms)} services"


def endpoint_modes_table(arms: Sequence[SweepResults], results_dir: Path) -> str:
    """Write the Markdown comparing the ways of reaching a model.

    The claim behind it: the agent reaches a model through configuration
    alone, so the same code runs against an aggregator, the model maker's own
    service, and a server the user runs, with only ``--profile`` changed. As
    with the multi-model table, the claim is checked against what each sweep
    recorded rather than asserted.

    Args:
        arms: The check's sweeps, from :func:`endpoint_mode_arms`.
        results_dir: The folder the file will be written to.

    Returns:
        str: The whole Markdown document.
    """
    lines = [
        "# Endpoint modes",
        "",
        (
            "The same task and the same code, reaching a model three ways: through an "
            "aggregator, through the model maker's own service, and through a server run "
            "by the user. Each arm is one sweep with an empty skill library, and the arms "
            "are meant to differ only in `--profile`; the checks below test that against "
            "what each sweep recorded. Tokens and time are always shown. Dollars are shown "
            "only where every call had a price: a self-hosted server has no per-token "
            "price, and free trial access is not one, so those arms read unknown by design."
        ),
        "",
        "## Results",
        "",
        (
            "| Arm | Service | Planner model | Screenshot model | Tasks | Passed | Avg steps | "
            "Avg tokens | Avg time | Avg cost (agent) | Code | Started |"
        ),
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for arm in arms:
        rate = success_rate(arm.rows)
        avg = averages(arm.marked)
        tasks = ", ".join(sorted({r.task_id for r in arm.rows})) or "none yet"
        link = _link(results_dir / f"{arm.name}.md", results_dir)
        code = ", ".join(f"`{v}`" for v in code_versions(arm.spec)) or "none yet"
        lines.append(
            f"| [{arm.spec.get('profile', arm.name)}]({link}) | {_service(arm.spec)} | "
            f"`{arm.spec.get('model', '?')}` | {_vision_model(arm.spec)} | {tasks} | "
            f"{rate.passed} of {rate.marked} | {avg.steps:.1f} | {render_tokens(avg.tokens)} | "
            f"{avg.wall_s:.1f} s | {render_cost(avg.cost_usd, avg.unpriced_calls)} | {code} | "
            f"{arm.spec.get('started_at', '?')} |"
        )
    lines += [
        "",
        "## Did only the endpoint change?",
        "",
        "| Check | Result |",
        "|---|---|",
        f"| Same code in every arm | {same_code_check(arms)} |",
        f"| Same tasks | {_same_setting(arms, 'suite')} |",
        f"| Same judge | {_same_setting(arms, 'judge_profile')} |",
        f"| Empty skill library in every arm | {_same_setting(arms, 'library')} |",
        f"| A different service in every arm | {_different_services(arms)} |",
        "",
        (
            "A code version is the git commit the sweep ran, with `-dirty` added when "
            "the agent's code, settings, tasks or dependencies had changes not yet "
            "committed. One commit, clean, in every arm is what shows that only the "
            "configuration changed: keys live in `.env`, which is not part of the code."
        ),
    ]
    return "\n".join(lines) + "\n"


# --- DOM-only versus dual-modality -----------------------------------------------------


def modality_arms(sweeps: Iterable[SweepResults]) -> list[SweepResults]:
    """Pick out the sweeps of the DOM-only versus dual-modality comparison.

    Args:
        sweeps: Every sweep.

    Returns:
        list[SweepResults]: The sweeps in folders named
        ``modality-<profile>``, in name order.
    """
    return sorted((s for s in sweeps if s.name.startswith(MODALITY_PREFIX)), key=lambda s: s.name)


def is_dom_only(spec: dict[str, Any]) -> bool:
    """Say whether a sweep ran on text alone, by its record.

    Args:
        spec: The contents of the sweep's ``sweep.json``.

    Returns:
        bool: True when its profile had no screenshot model -- recorded as
        an empty one -- so no screenshot could be sent. False for a sweep
        with a screenshot model, or one from before it was recorded.
    """
    return spec.get("vision_model") == ""


@dataclass(frozen=True)
class ImageCount:
    """How many model calls carried an image, over some trials.

    Read from each call's own record, whose ``vision`` flag is worked out from
    the request that was sent -- whether any message had an image part -- so
    it counts what went to the model, not what the agent meant to send.

    Attributes:
        with_image: Calls whose request carried an image.
        calls: Every call.
        trials: Trials with at least one call that carried an image.
    """

    with_image: int
    calls: int
    trials: int


def count_images(rows: Iterable[TrialRow]) -> ImageCount:
    """Count the model calls that carried an image, over some trials.

    Args:
        rows: The trials, each with its run folder.

    Returns:
        ImageCount: The counts. A trial whose call log cannot be read counts
        as having made no calls.
    """
    with_image = calls = trials = 0
    for row in rows:
        records = read_calls(row.folder / COST_LOG)
        images = sum(1 for record in records if record.vision)
        with_image += images
        calls += len(records)
        trials += images > 0
    return ImageCount(with_image, calls, trials)


def _arm_kind(spec: dict[str, Any]) -> str:
    """Name an arm by how it sees the page, for the table.

    Args:
        spec: The contents of the sweep's ``sweep.json``.

    Returns:
        str: ``DOM only`` or ``DOM + screenshots when gated in``.
    """
    return "DOM only" if is_dom_only(spec) else "DOM + screenshots when gated in"


def _no_images_check(arms: Sequence[SweepResults]) -> str:
    """Check that no request in a DOM-only arm carried an image.

    Args:
        arms: The comparison's arms.

    Returns:
        str: ``yes`` with the number of calls checked, or ``no`` with the
        count that carried one. Says so when there is no DOM-only arm.
    """
    dom_only = [arm for arm in arms if is_dom_only(arm.spec)]
    if not dom_only:
        return "no DOM-only arm"
    count = count_images(row for arm in dom_only for row in arm.rows)
    if count.with_image:
        return f"no: {count.with_image} of {count.calls} calls carried an image"
    return f"yes: 0 of {count.calls} calls carried an image"


def _tradeoff_lines(dual: SweepResults, dom: SweepResults) -> list[str]:
    """Write the change from text alone to text plus screenshots.

    Args:
        dual: The arm that could send screenshots.
        dom: The arm on text alone.

    Returns:
        list[str]: Markdown lines: what the screenshots bought, and what
        they cost.
    """
    dual_rate, dom_rate = success_rate(dual.rows), success_rate(dom.rows)
    dual_avg, dom_avg = averages(dual.marked), averages(dom.marked)
    points = (
        f"{(dual_rate.mean - dom_rate.mean) * 100:+.1f} points"
        if dual_rate.rounds and dom_rate.rounds
        else "unknown"
    )
    return [
        "## What the screenshots bought, and what they cost",
        "",
        "The change from text alone to text plus screenshots, per trial.",
        "",
        "| Measure | DOM only | DOM + screenshots | Change |",
        "|---|---|---|---|",
        f"| Success rate | {dom_rate.render()} | {dual_rate.render()} | {points} |",
        (
            f"| Trials passed | {dom_rate.passed} of {dom_rate.marked} | "
            f"{dual_rate.passed} of {dual_rate.marked} | |"
        ),
        (
            f"| Average steps | {dom_avg.steps:.1f} | {dual_avg.steps:.1f} | "
            f"{_change(dom_avg.steps, dual_avg.steps)} |"
        ),
        (
            f"| Average tokens | {render_tokens(dom_avg.tokens)} | {render_tokens(dual_avg.tokens)} | "
            f"{_change(dom_avg.tokens, dual_avg.tokens)} |"
        ),
        (
            f"| Average time | {dom_avg.wall_s:.1f} s | {dual_avg.wall_s:.1f} s | "
            f"{_change(dom_avg.wall_s, dual_avg.wall_s, ' s')} |"
        ),
        (
            f"| Average cost (agent) | {render_cost(dom_avg.cost_usd, dom_avg.unpriced_calls)} | "
            f"{render_cost(dual_avg.cost_usd, dual_avg.unpriced_calls)} | "
            f"{_cost_change(dom_avg, dual_avg)} |"
        ),
    ]


def dom_vs_dual_table(arms: Sequence[SweepResults], results_dir: Path) -> str:
    """Write the Markdown comparing text alone with text plus screenshots.

    The agent reads each page as text -- the numbered list of things to
    click, and the page's words -- and adds a screenshot only when a cheap
    rule says the text may be missing something. This table measures what
    that buys: the same suite, on the same planner model, once with the
    screenshot channel removed and once with it gated as usual.

    Args:
        arms: The comparison's sweeps, from :func:`modality_arms`.
        results_dir: The folder the file will be written to.

    Returns:
        str: The whole Markdown document.
    """
    lines = [
        "# DOM only vs. DOM + screenshots",
        "",
        (
            "The same benchmark on the same planner model, twice. **DOM only**: the page "
            "as text alone -- the numbered list of things to click, and the page's words "
            "-- with the screenshot channel removed (a profile with no screenshot model). "
            "**DOM + screenshots**: the usual setting, where a screenshot is added only "
            "when few elements were found or the planner keeps failing on text. An image "
            "counts as sent only when the request that went to the model carried one."
        ),
        "",
        "## Results",
        "",
        (
            "| Arm | Sees the page as | Planner model | Screenshot model | Trials marked (N) | "
            "Success rate | Calls with an image | Avg steps | Avg tokens | Avg time | "
            "Avg cost (agent) | Code |"
        ),
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for arm in arms:
        rate = success_rate(arm.rows)
        avg = averages(arm.marked)
        images = count_images(arm.marked)
        tasks = len({r.task_id for r in arm.marked})
        link = _link(results_dir / f"{arm.name}.md", results_dir)
        code = ", ".join(f"`{v}`" for v in code_versions(arm.spec)) or "none yet"
        lines.append(
            f"| [{arm.spec.get('profile', arm.name)}]({link}) | {_arm_kind(arm.spec)} | "
            f"`{arm.spec.get('model', '?')}` | {_vision_model(arm.spec)} | "
            f"{rate.marked} ({tasks} tasks × {arm.trials_per_task}) | {rate.render()} | "
            f"{images.with_image} of {images.calls} (in {images.trials} trials) | "
            f"{avg.steps:.1f} | {render_tokens(avg.tokens)} | {avg.wall_s:.1f} s | "
            f"{render_cost(avg.cost_usd, avg.unpriced_calls)} | {code} |"
        )
    dom = [arm for arm in arms if is_dom_only(arm.spec)]
    dual = [arm for arm in arms if not is_dom_only(arm.spec)]
    if len(dom) == 1 and len(dual) == 1:
        lines += ["", *_tradeoff_lines(dual[0], dom[0])]
    lines += [
        "",
        "## Did only the screenshot channel change?",
        "",
        "| Check | Result |",
        "|---|---|",
        f"| No image in any request of the DOM-only arm | {_no_images_check(arms)} |",
        f"| Same code in every arm | {same_code_check(arms)} |",
        f"| Same planner model | {_same_setting(arms, 'model')} |",
        f"| Same benchmark suite | {_same_setting(arms, 'suite')} |",
        f"| Same judge | {_same_setting(arms, 'judge_profile')} |",
        f"| Empty skill library in every arm | {_same_setting(arms, 'library')} |",
        "",
        "## Per task",
        "",
        "Trials passed out of trials marked, and in brackets the calls that carried an image.",
        "",
        "| Task | " + " | ".join(str(arm.spec.get("profile", arm.name)) for arm in arms) + " |",
        "|---|" + "---|" * len(arms),
    ]
    for task_id in sorted({r.task_id for arm in arms for r in arm.marked}):
        cells = []
        for arm in arms:
            rows = [r for r in arm.marked if r.task_id == task_id]
            images = count_images(rows)
            passed = sum(r.status == PASS for r in rows)
            cells.append(f"{passed}/{len(rows)} ({images.with_image})")
        lines.append(f"| {task_id} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def find_sweeps(runs_root: Path) -> list[Path]:
    """Find every sweep folder directly under a runs folder.

    Args:
        runs_root: Usually ``runs``.

    Returns:
        list[Path]: Folders holding a ``sweep.json``, sorted by name.
    """
    return sorted(p.parent for p in runs_root.glob(f"*/{SWEEP_FILE}"))


def write_results(
    sweep_dirs: Iterable[Path], results_dir: Path, *, min_labels: int = DEFAULT_MIN_LABELS
) -> list[Path]:
    """Write the tables for some sweeps, and the comparison of them all.

    Args:
        sweep_dirs: The sweep folders.
        results_dir: Where the Markdown files go. Created if needed.
        min_labels: The fewest hand-labelled trials an agreement figure
            should rest on before it is relied on. Fewer are still shown,
            with their count and a warning.

    Returns:
        list[Path]: The files written: one per sweep, then the cold-versus-
        warm table when a cold sweep and its warm partner are both there,
        then the multi-model table when any of its arms is there, then the
        endpoint-modes and DOM-versus-dual tables likewise, then the
        comparison.

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
        path.write_text(sweep_table(sweep, results_dir, labels, min_labels), encoding="utf-8")
        written.append(path)
    # The headline comparison, whenever a cold sweep and its warm partner
    # are both there.
    pairs = cold_warm_pairs(sweeps)
    if pairs:
        cold_vs_warm = results_dir / COLD_VS_WARM_FILE
        cold_vs_warm.write_text(cold_vs_warm_table(pairs, results_dir), encoding="utf-8")
        written.append(cold_vs_warm)
    arms = multi_model_arms(sweeps)
    if arms:
        multi_model = results_dir / MULTI_MODEL_FILE
        multi_model.write_text(multi_model_table(arms, results_dir), encoding="utf-8")
        written.append(multi_model)
    modes = endpoint_mode_arms(sweeps)
    if modes:
        endpoint_modes = results_dir / ENDPOINT_MODES_FILE
        endpoint_modes.write_text(endpoint_modes_table(modes, results_dir), encoding="utf-8")
        written.append(endpoint_modes)
    modality = modality_arms(sweeps)
    if modality:
        dom_vs_dual = results_dir / DOM_VS_DUAL_FILE
        dom_vs_dual.write_text(dom_vs_dual_table(modality, results_dir), encoding="utf-8")
        written.append(dom_vs_dual)
    comparison = results_dir / COMPARISON_FILE
    comparison.write_text(
        comparison_table(sweeps, results_dir, labels, min_labels), encoding="utf-8"
    )
    written.append(comparison)
    return written
