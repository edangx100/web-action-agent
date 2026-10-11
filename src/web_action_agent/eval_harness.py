"""Running the whole benchmark: many trials, several at once, safe to restart.

The benchmark is twelve tasks, each tried several times (*trials*), on each
configuration being compared. Done one after another at a few minutes each,
that is most of a day. So this module runs several trials at once, keeps a
careful record of each, and can be stopped and started again without losing or
repeating work.

**One sweep, one folder.** A *sweep* is one pass over the suite with one
configuration -- one model, and a library that is either empty (*cold*) or
the learned one (*warm*). Everything it produces goes in one folder::

    runs/eval-openrouter-cold/
        sweep.json                     what this sweep is, so a restart can check
        library/                       the skill library the sweep started with
        mom-annual-leave/
            trial-1/                   one complete run folder, plus verdict.json
            trial-2/
        moe-bishan-primary-count/
            trial-1/
        ...
        _interrupted/                  folders of runs that were killed midway

**Several at once, each kept apart.** A pool of *workers* (four by default)
takes trials from a shared list. Each worker has its own browser, and every
trial gets a fresh browser context in it -- new cookies, new storage, no
history -- so nothing one trial does can change how the next one starts.
Two rules keep the parallelism honest:

* one agent at a time per run: a trial is only ever run by one worker;
* one trial at a time per task: two trials of the *same* task never run side
  by side, so they never compete on the same pages.

Visits to each website stay polite across all workers, because the per-site
gap is kept in a shared file (see :mod:`~web_action_agent.politeness`), not
in each worker's memory.

**Safe to restart.** Each trial's state is read from its folder:

* no folder -- not started: run it;
* a folder with no verdict from the agent -- the run was killed partway: the
  folder is moved to ``_interrupted/`` (kept as evidence, never counted) and
  the trial is run again from the start;
* a finished run with no mark yet -- mark it, without running it again;
* a finished, marked run -- skip it.

So a sweep that dies halfway is restarted with the same command, and it does
exactly the trials that are left. A trial that *failed* is not run again: a
failure is a result, and retrying failures until they pass would be cheating.

**A fixed library underneath.** The skill library is copied into the sweep
folder once, when the sweep starts (an empty one for a cold sweep), and each
trial works on its *own* copy of that. A trial that replays a skill writes
its tally into its own copy, so no trial can change what the next one is
offered, and the real library is never touched by a benchmark.

**Marking.** When a trial's run finishes, it is marked straight away: the
machine checks from :mod:`~web_action_agent.checkers`, then the judge from
:mod:`~web_action_agent.judge` where the task needs one. The result goes in
``verdict.json`` beside the run. See :func:`mark_trial` for how they combine.

**Within budget.** Before each trial is handed out, the sweep's money and time
ceilings are checked (see :mod:`~web_action_agent.budget`). Once either is
reached, no new trial starts; the ones in progress finish and are marked, and
the sweep stops in the same restartable state as after a Ctrl-C.
"""

from __future__ import annotations

import json
import shutil
import threading
from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

from .agent_run import RunSetupError, run_agent
from .benchmark import BenchmarkTask, Suite
from .benchmark import Judge as JudgeMethod
from .browser import BrowserProcess, BrowserSettings
from .budget import Budget, record_judge_call
from .checkers import CheckResult, final_state, run_checks
from .config import Config
from .judge import FAIL, FAILURE_CATEGORIES, JUDGING_ERROR, PASS, Judge, JudgeVerdict
from .model_client import ModelClient, ModelClientError, resolve_profile
from .skill_store import SkillStore
from .trace import OUTCOME_FILE, TraceError
from .trace_reader import TraceReadError, read_run

#: The file describing a sweep, written when it starts and checked on restart.
SWEEP_FILE = "sweep.json"

#: The file holding a trial's mark, written beside its run folder's files.
VERDICT_FILE = "verdict.json"

#: Where the folders of killed runs are moved, inside the sweep folder.
INTERRUPTED_DIR = "_interrupted"

#: Where the sweep's starting copy of the skill library lives.
LIBRARY_DIR = "library"

#: Where each trial's own copy of the library lives, inside its run folder.
TRIAL_LIBRARY_DIR = "skills"

#: How many trials run at once when the configuration does not say.
DEFAULT_WORKERS = 4

#: A trial's state, read from its folder. See the module docstring.
NEW = "new"
INTERRUPTED = "interrupted"
UNMARKED = "unmarked"
DONE = "done"

#: What happened to a trial in this invocation of the harness.
RAN = "ran"
SKIPPED = "skipped"
REMARKED = "remarked"
SETUP_ERROR = "setup_error"
CRASHED = "crashed"

#: How a trial was marked. ``needs_judge`` means the task can only be marked
#: by the judge and none is configured: the run is kept, and a later restart
#: with a judge marks it without running it again.
NEEDS_JUDGE = "needs_judge"


class HarnessError(Exception):
    """The sweep cannot start: a mistake in how it was asked for.

    Raised before any trial runs -- an unusable model setting, or a sweep
    folder that holds a *different* sweep -- so nothing is spent.
    """


# --- Trials --------------------------------------------------------------------


@dataclass(frozen=True)
class Trial:
    """One attempt at one task: the unit the harness schedules.

    Attributes:
        task: The benchmark task.
        number: Which attempt, from 1.
    """

    task: BenchmarkTask
    number: int

    @property
    def name(self) -> str:
        """str: ``<task id>/trial-<n>``, the trial's folder below the sweep."""
        return f"{self.task.id}/trial-{self.number}"

    def folder(self, sweep_dir: Path) -> Path:
        """Return where this trial's run folder goes.

        Args:
            sweep_dir: The sweep's folder.

        Returns:
            Path: ``<sweep>/<task id>/trial-<n>``. Fixed by the trial alone,
            which is what lets a restart find the work already done.
        """
        return sweep_dir / self.task.id / f"trial-{self.number}"


def plan_trials(suite: Suite, trials: int) -> tuple[Trial, ...]:
    """List every trial a sweep must do, in the order to start them.

    All tasks' first trials, then all their second trials, and so on. That
    order spreads each task's trials out over the sweep, so a sweep stopped
    early has at least one trial of most tasks rather than three of a few.

    Args:
        suite: The benchmark suite.
        trials: How many times to try each task.

    Returns:
        tuple[Trial, ...]: Every trial, in starting order.
    """
    return tuple(Trial(task, n) for n in range(1, trials + 1) for task in suite.tasks)


def trial_state(folder: Path) -> str:
    """Read a trial's state from its folder.

    Args:
        folder: The trial's run folder.

    Returns:
        str: :data:`NEW`, :data:`INTERRUPTED`, :data:`UNMARKED`, or
        :data:`DONE`. A folder whose mark is a judging error, which still
        needs a judge, or which is a fail with no category yet, counts as
        unmarked, so a restart tries to finish the mark without running the
        trial again.
    """
    if not folder.exists():
        return NEW
    if not (folder / OUTCOME_FILE).exists():
        return INTERRUPTED
    verdict_path = folder / VERDICT_FILE
    if not verdict_path.exists():
        return UNMARKED
    verdict = _read_verdict(folder)
    return DONE if verdict is not None and verdict.complete else UNMARKED


# --- Marking -------------------------------------------------------------------


@dataclass(frozen=True)
class TrialVerdict:
    """A trial's mark: pass or fail, and how that was decided.

    Attributes:
        status: :data:`~web_action_agent.judge.PASS`,
            :data:`~web_action_agent.judge.FAIL`,
            :data:`~web_action_agent.judge.JUDGING_ERROR`, or
            :data:`NEEDS_JUDGE`.
        reason: One sentence on what decided it.
        agent_outcome: How the agent's own run ended, such as ``success`` or
            ``step_cap``.
        checks: The machine checks' results, in task-file order.
        judge: The judge's pass-or-fail verdict, when the task asked for one.
        category: For a fail, which kind of mistake: one of the four. None for
            a pass, and None for a fail that could not be categorised yet --
            which leaves the mark unfinished (see :attr:`complete`).
        categorized_by: The judge's answer when it was asked only for the
            category, for a fail it did not decide itself.
        marked_at: When the mark was made, RFC 3339 in UTC.
    """

    status: str
    reason: str
    agent_outcome: str
    checks: tuple[CheckResult, ...] = ()
    judge: JudgeVerdict | None = None
    category: str | None = None
    categorized_by: JudgeVerdict | None = None
    marked_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    @property
    def passed(self) -> bool:
        """bool: True only for a pass."""
        return self.status == PASS

    @property
    def complete(self) -> bool:
        """bool: True for a finished mark: a pass, or a fail with its category.

        A fail without a category is a decided result with a missing reason,
        and every failure must say what kind it was. Counting it as finished
        would let it reach the results tables as a blank.
        """
        if self.status == PASS:
            return True
        return self.status == FAIL and self.category in FAILURE_CATEGORIES

    def to_dict(self) -> dict[str, Any]:
        """Turn the mark into plain data for ``verdict.json``.

        Returns:
            dict[str, Any]: Every field.
        """
        return {
            "status": self.status,
            "reason": self.reason,
            "agent_outcome": self.agent_outcome,
            "checks": [
                {"method": c.method, "passed": c.passed, "detail": c.detail} for c in self.checks
            ],
            "judge": self.judge.to_dict() if self.judge is not None else None,
            "category": self.category,
            "categorized_by": (
                self.categorized_by.to_dict() if self.categorized_by is not None else None
            ),
            "marked_at": self.marked_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TrialVerdict:
        """Rebuild a mark written by :meth:`to_dict`.

        Args:
            data: The plain data.

        Returns:
            TrialVerdict: The mark.
        """
        judge = data.get("judge")
        categorized_by = data.get("categorized_by")
        return cls(
            status=data["status"],
            reason=data.get("reason", ""),
            agent_outcome=data.get("agent_outcome", ""),
            checks=tuple(CheckResult(**c) for c in data.get("checks", ())),
            judge=JudgeVerdict.from_dict(judge) if judge else None,
            category=data.get("category"),
            categorized_by=JudgeVerdict.from_dict(categorized_by) if categorized_by else None,
            marked_at=data.get("marked_at", ""),
        )


def mark_trial(task: BenchmarkTask, folder: Path, judge: Judge | None) -> TrialVerdict:
    """Mark one finished run, from its folder.

    A run passes only if **all** of these hold:

    1. the agent itself finished and reported success -- an agent that ran
       out of steps or gave up has not completed the task, whatever page it
       happened to be on;
    2. every machine check in the task file passes;
    3. if the task asks for the judge, the judge passes it.

    Every failure must get exactly one category -- grounding, planning,
    navigation, or site change. If the judge failed it, the judge's category
    is used. If something else failed it (the agent did not finish, or a
    machine check caught it), the judge is asked for the category alone, told
    that the fail is already decided -- including when the judge itself had
    said pass, since the machine checks are the stricter authority where they
    apply. A fail that cannot be categorised yet (no judge, or the judge
    could not answer) is kept as a fail with no category: an unfinished mark,
    which a restart with a judge completes without running the trial again.

    Args:
        task: The task the run attempted.
        folder: The run's folder.
        judge: The judge, or None if none is configured.

    Returns:
        TrialVerdict: The mark. :data:`NEEDS_JUDGE` when the task needs the
        judge and there is none; a judging error when the judge could not
        answer.
    """
    run = read_run(folder)
    agent_ok = run.outcome is not None and run.outcome.ok
    agent_outcome = run.outcome.category if run.outcome is not None else "interrupted"
    checks = run_checks(task.verdicts, final_state(run))
    needs_judge = any(isinstance(v, JudgeMethod) for v in task.verdicts)

    judged: JudgeVerdict | None = None
    if needs_judge:
        if judge is None:
            return TrialVerdict(
                NEEDS_JUDGE,
                "this task is marked by the judge, and no judge is configured",
                agent_outcome,
                checks,
            )
        judged = judge.judge(task, run)
        if judged.outcome == JUDGING_ERROR:
            return TrialVerdict(
                JUDGING_ERROR,
                f"the judge could not mark it: {judged.error}",
                agent_outcome,
                checks,
                judged,
            )

    # The first thing that failed, in the order the rules are listed above.
    failed_check = next((c for c in checks if not c.passed), None)
    if not agent_ok:
        reason = f"the agent did not finish the task ({agent_outcome})"
    elif failed_check is not None:
        reason = f"{failed_check.method} failed: {failed_check.detail}"
    elif judged is not None and not judged.passed:
        reason = f"the judge failed it: {judged.justification}"
    else:
        reason = judged.justification if judged is not None else "every check passed"
        return TrialVerdict(PASS, reason, agent_outcome, checks, judged)

    # A failure. Its category comes from the judge's own fail, if it gave one.
    if judged is not None and judged.outcome == FAIL:
        return TrialVerdict(FAIL, reason, agent_outcome, checks, judged, judged.category)
    if judge is None:
        return TrialVerdict(
            FAIL,
            f"{reason} (no judge is configured to say what kind of failure it was)",
            agent_outcome,
            checks,
            judged,
        )
    # Otherwise ask the judge which kind it was, telling it the fail is settled.
    categorized = judge.categorize(task, run, reason)
    category = categorized.category if categorized.outcome == FAIL else None
    return TrialVerdict(FAIL, reason, agent_outcome, checks, judged, category, categorized)


def write_verdict(folder: Path, verdict: TrialVerdict) -> None:
    """Save a trial's mark beside its run.

    Written to a temporary name and then renamed, so a harness killed
    mid-write never leaves a half-written mark that a restart would misread.

    Args:
        folder: The trial's run folder.
        verdict: Its mark.
    """
    final = folder / VERDICT_FILE
    partial = folder / f"{VERDICT_FILE}.partial"
    partial.write_text(json.dumps(verdict.to_dict(), indent=2) + "\n", encoding="utf-8")
    partial.replace(final)


# --- The sweep's identity ---------------------------------------------------------


@dataclass(frozen=True)
class SweepSpec:
    """What a sweep is, written to ``sweep.json`` when it starts.

    Restarting a sweep in its folder must not quietly change what it measures.
    So a restart compares these fields with the ones already written and
    refuses to go on if any differ: a folder with half its trials on one model
    and half on another would produce a number about neither.

    The number of trials is deliberately *not* compared: asking for more
    trials later simply adds them.

    Attributes:
        suite: The suite's name.
        profile: The planner's profile.
        model: The planner's text model.
        library: ``cold`` (empty library) or ``warm`` (the learned one).
        judge_profile: The judge's profile, or empty for none.
        started_at: When the sweep was first started, RFC 3339 in UTC. Kept
            from the first start, because published results must say when the
            live sites were visited.
    """

    suite: str
    profile: str
    model: str
    library: str
    judge_profile: str
    started_at: str = ""

    #: The fields a restart must agree on.
    IDENTITY = ("suite", "profile", "model", "library", "judge_profile")

    def differences(self, other: SweepSpec) -> list[str]:
        """List how another sweep differs from this one.

        Args:
            other: The sweep already recorded in the folder.

        Returns:
            list[str]: One line per differing field, empty if they agree.
        """
        return [
            f"{name}: {getattr(other, name)!r} in the folder, {getattr(self, name)!r} now"
            for name in self.IDENTITY
            if getattr(self, name) != getattr(other, name)
        ]


def default_sweep_dir(runs_root: Path, profile: str, cold: bool) -> Path:
    """Name a sweep's folder from what it measures.

    Fixed by the configuration alone, so running the same command again finds
    the same folder and carries on where it stopped.

    Args:
        runs_root: Where runs go, usually ``runs``.
        profile: The planner's profile.
        cold: Whether the library is empty.

    Returns:
        Path: ``<runs_root>/eval-<profile>-<cold|warm>``.
    """
    return runs_root / f"eval-{profile}-{'cold' if cold else 'warm'}"


# --- Running it -------------------------------------------------------------------


@dataclass(frozen=True)
class TrialResult:
    """What happened to one trial in this invocation of the harness.

    Attributes:
        trial: The trial.
        action: :data:`RAN`, :data:`SKIPPED`, :data:`REMARKED`,
            :data:`SETUP_ERROR`, or :data:`CRASHED`.
        verdict: Its mark, when it has one.
        error: What went wrong, for a setup error or a crash.
    """

    trial: Trial
    action: str
    verdict: TrialVerdict | None = None
    error: str = ""


@dataclass(frozen=True)
class SweepReport:
    """What one invocation of the harness did.

    Attributes:
        folder: The sweep's folder.
        results: One entry per trial that was looked at, in plan order.
            Trials never started -- because a ceiling was reached first --
            have no entry.
        moved_aside: Folders of killed runs moved to ``_interrupted/``.
        not_started: Trials left for a later run, because the sweep stopped
            before reaching them.
        stopped_by: Why the sweep stopped early, or empty if it did not.
    """

    folder: Path
    results: tuple[TrialResult, ...]
    moved_aside: tuple[Path, ...] = ()
    not_started: int = 0
    stopped_by: str = ""

    def count(self, action: str) -> int:
        """Count the trials that had one thing happen to them.

        Args:
            action: Such as :data:`RAN`.

        Returns:
            int: How many.
        """
        return sum(1 for r in self.results if r.action == action)

    def summary(self) -> str:
        """Render one line saying what this invocation did.

        Returns:
            str: e.g. ``36 trials: 12 ran, 24 already done; 10 passed, 2 failed``.
        """
        marks = [r.verdict.status for r in self.results if r.verdict is not None]
        parts = [f"{self.count(RAN)} ran", f"{self.count(SKIPPED)} already done"]
        for action, label in (
            (REMARKED, "marked again"),
            (SETUP_ERROR, "could not start"),
            (CRASHED, "crashed"),
        ):
            if self.count(action):
                parts.append(f"{self.count(action)} {label}")
        if self.not_started:
            parts.append(f"{self.not_started} not started")
        tally = [f"{marks.count(PASS)} passed", f"{marks.count(FAIL)} failed"]
        uncategorized = sum(
            1
            for r in self.results
            if r.verdict is not None and r.verdict.status == FAIL and not r.verdict.complete
        )
        if uncategorized:
            tally.append(f"{uncategorized} of the fails not yet categorised")
        for status, label in (
            (JUDGING_ERROR, "judging errors"),
            (NEEDS_JUDGE, "waiting for a judge"),
        ):
            if marks.count(status):
                tally.append(f"{marks.count(status)} {label}")
        # Every trial in the plan, including the ones never started.
        total = len(self.results) + self.not_started
        return f"{total} trials: {', '.join(parts)}; {', '.join(tally)}"


class TrialRunner(Protocol):
    """Runs one trial's agent into its folder. Swappable, for tests.

    The default, :meth:`Harness.run_one`, assembles the real agent. A test
    can supply one that writes a folder without a browser or a model.
    """

    def __call__(self, trial: Trial, config: Config, folder: Path, browser: Any) -> None:
        """Run the trial, leaving a complete run folder at ``folder``."""


class Harness:
    """Runs a sweep: every trial of a suite, several at once, restartable.

    Attributes:
        config: The settings every trial runs under.
        suite: The tasks.
        folder: The sweep's folder.
        trials: How many times to try each task.
        cold: True for an empty skill library, False for the learned one.
        workers: How many trials run at once.
    """

    def __init__(
        self,
        config: Config,
        suite: Suite,
        folder: Path,
        *,
        trials: int,
        cold: bool,
        workers: int | None = None,
        skills_root: Path = Path("skills"),
        judge: Judge | None = None,
        client_factory: Callable[[Config], ModelClient] | None = None,
        browser_factory: Callable[[Config], AbstractContextManager[Any]] | None = None,
        runner: TrialRunner | None = None,
        budget: Budget | None = None,
        log: Callable[[str], None] = print,
    ) -> None:
        """Set up a sweep, without starting anything.

        Args:
            config: The settings every trial runs under.
            suite: The tasks.
            folder: The sweep's folder. Created if it is not there.
            trials: How many times to try each task.
            cold: True for an empty skill library, False for the learned one.
            workers: How many trials at once. Defaults to ``config.workers``.
            skills_root: The real skill library, copied for a warm sweep.
            judge: The judge, or None to mark with machine checks only.
            client_factory: Builds the model client for each trial. Defaults
                to the configured endpoint; a test passes a stand-in.
            browser_factory: Starts one worker's browser. Defaults to a real
                browser process per worker.
            runner: Runs one trial. Defaults to :meth:`run_one`, the real
                agent.
            budget: The money and time ceilings. Defaults to the ones in
                ``config``.
            log: Where progress lines go.
        """
        self.config = config
        self.suite = suite
        self.folder = folder
        self.trials = trials
        self.cold = cold
        self.workers = workers if workers is not None else config.workers
        self._skills_root = skills_root
        self._judge = judge
        self._client_factory = client_factory or ModelClient.from_config
        self._browser_factory = browser_factory or (
            lambda cfg: BrowserProcess(BrowserSettings.from_config(cfg))
        )
        self._runner = runner or self.run_one
        self._budget = budget or Budget.from_config(folder, config)
        self._log = log

        # Shared between worker threads, always read and written under this
        # condition's lock.
        self._condition = threading.Condition()
        self._pending: list[Trial] = []
        self._running_tasks: set[str] = set()
        self._stop = threading.Event()
        self._results: dict[str, TrialResult] = {}
        self._stopped_by = ""

    # --- Preparing ---

    def spec(self) -> SweepSpec:
        """Describe this sweep, for ``sweep.json``.

        Returns:
            SweepSpec: The sweep's identity.

        Raises:
            HarnessError: If the planner's profile cannot be resolved.
        """
        try:
            profile = resolve_profile(self.config)
        except ModelClientError as error:
            raise HarnessError(str(error)) from error
        return SweepSpec(
            suite=self.suite.name,
            profile=profile.name,
            model=profile.model,
            library="cold" if self.cold else "warm",
            judge_profile=self.config.judge_profile or "",
            started_at=datetime.now(UTC).isoformat(),
        )

    def prepare(self) -> None:
        """Create the sweep folder, or check it holds this same sweep.

        Also takes the sweep's copy of the skill library, the first time only:
        a restart keeps using the copy the sweep started with, even if the
        real library has changed since.

        Raises:
            HarnessError: If the folder holds a different sweep, or the model
                settings cannot be used.
        """
        spec = self.spec()
        # Built once before any worker starts, so a missing key stops the
        # sweep here instead of failing every trial in turn.
        try:
            self._client_factory(self.config)
        except ModelClientError as error:
            raise HarnessError(str(error)) from error

        self.folder.mkdir(parents=True, exist_ok=True)
        spec_path = self.folder / SWEEP_FILE
        if spec_path.exists():
            recorded = SweepSpec(**json.loads(spec_path.read_text(encoding="utf-8")))
            differences = spec.differences(recorded)
            if differences:
                raise HarnessError(
                    f"{self.folder} holds a different sweep, so carrying on would mix two "
                    "configurations in one set of results: "
                    + "; ".join(differences)
                    + ". Use another folder with --out, or remove this one"
                )
        else:
            spec_path.write_text(json.dumps(asdict(spec), indent=2) + "\n", encoding="utf-8")

        library = self.folder / LIBRARY_DIR
        if not library.exists():
            if self.cold or not self._skills_root.exists():
                library.mkdir()
            else:
                shutil.copytree(self._skills_root, library)

    # --- Running ---

    def run(self) -> SweepReport:
        """Run every trial that is not already done, and report.

        Returns:
            SweepReport: What happened to each trial.

        Raises:
            HarnessError: If the sweep cannot start. Nothing has run.
        """
        self.prepare()
        self._budget.start()
        plan = plan_trials(self.suite, self.trials)
        moved: list[Path] = []
        to_mark: list[Trial] = []
        to_run: list[Trial] = []

        for trial in plan:
            folder = trial.folder(self.folder)
            state = trial_state(folder)
            if state == DONE:
                self._results[trial.name] = TrialResult(trial, SKIPPED, _read_verdict(folder))
            elif state == UNMARKED:
                to_mark.append(trial)
            elif state == INTERRUPTED:
                moved.append(self._move_aside(trial, folder))
                to_run.append(trial)
            else:
                to_run.append(trial)

        done = len(plan) - len(to_run) - len(to_mark)
        self._log(
            f"sweep: {len(plan)} trials, {done} already done, {len(to_run)} to run"
            + (f", {len(to_mark)} to mark" if to_mark else "")
        )
        if moved:
            self._log(f"sweep: {len(moved)} interrupted run(s) moved to {INTERRUPTED_DIR}/")
        self._log(f"sweep: spent so far {self._budget.spend().summary()}")

        # Marking may call the judge, which costs money too, so it waits on
        # the same ceilings as running.
        for trial in to_mark:
            if self._over_budget():
                break
            verdict = self._mark(trial, trial.folder(self.folder))
            self._results[trial.name] = TrialResult(trial, REMARKED, verdict)

        self._pending = [] if self._stopped_by else list(to_run)
        threads = [
            threading.Thread(target=self._work, name=f"worker-{n + 1}", daemon=True)
            for n in range(min(self.workers, len(self._pending)))
        ]
        for thread in threads:
            thread.start()
        try:
            for thread in threads:
                # A timeout on join keeps the main thread able to receive
                # Ctrl-C; a bare join would block it until the thread ended.
                while thread.is_alive():
                    thread.join(timeout=0.2)
        except KeyboardInterrupt:
            # Stop handing out new trials. The runs already going finish and
            # are marked; anything not started stays for the restart. A second
            # Ctrl-C ends the process at once, and the restart cleans up after
            # it the same way.
            self._stop.set()
            self._log("sweep: stopping after the runs in progress (Ctrl-C again to quit now)")
            for thread in threads:
                while thread.is_alive():
                    thread.join(timeout=0.2)
            raise
        finally:
            self._budget.save()

        results = tuple(self._results[t.name] for t in plan if t.name in self._results)
        self._log(f"sweep: spent in total {self._budget.spend().summary()}")
        return SweepReport(
            self.folder,
            results,
            tuple(moved),
            not_started=len(plan) - len(results),
            stopped_by=self._stopped_by,
        )

    def _over_budget(self) -> bool:
        """Check the ceilings, and stop the sweep the first time one is reached.

        Returns:
            bool: True if no more work may start.
        """
        if self._stopped_by:
            return True
        reason = self._budget.check()
        if reason is None:
            return False
        self._stopped_by = reason
        self._stop.set()
        self._log(f"sweep: stopping: {reason}")
        return True

    def _work(self) -> None:
        """One worker: start a browser, then run trials until none are left."""
        with self._browser_factory(self.config) as browser:
            while (trial := self._claim()) is not None:
                try:
                    result = self._attempt(trial, browser)
                finally:
                    self._release(trial)
                with self._condition:
                    self._results[trial.name] = result
                    # Saved after every trial, so a sweep killed without
                    # warning loses at most one trial's time from its total.
                    self._budget.save()
                mark = result.verdict.status if result.verdict else result.action
                self._log(f"{threading.current_thread().name}: {trial.name} -> {mark}")

    def _claim(self) -> Trial | None:
        """Take the next trial whose task is not already being run.

        Returns:
            Trial | None: A trial to run, or None when there is nothing left
            (or the sweep is stopping). Waits while the only trials left
            belong to tasks another worker is running.
        """
        with self._condition:
            while True:
                if self._stop.is_set() or not self._pending or self._over_budget():
                    return None
                for index, trial in enumerate(self._pending):
                    if trial.task.id not in self._running_tasks:
                        self._running_tasks.add(trial.task.id)
                        return self._pending.pop(index)
                self._condition.wait(timeout=0.5)

    def _release(self, trial: Trial) -> None:
        """Mark a trial's task as free for its next trial.

        Args:
            trial: The trial that finished.
        """
        with self._condition:
            self._running_tasks.discard(trial.task.id)
            self._condition.notify_all()

    def _attempt(self, trial: Trial, browser: Any) -> TrialResult:
        """Run one trial and mark it, turning any failure into a result.

        Args:
            trial: The trial.
            browser: This worker's browser.

        Returns:
            TrialResult: What happened. Never raises for a problem with one
            trial: one bad trial must not stop the other workers.
        """
        folder = trial.folder(self.folder)
        config = self.config.model_copy(update={"start_url": trial.task.start_url})
        try:
            self._runner(trial, config, folder, browser)
        except RunSetupError as error:
            return TrialResult(trial, SETUP_ERROR, error=str(error))
        except Exception as error:  # noqa: BLE001 -- one trial's crash is reported, not fatal
            return TrialResult(trial, CRASHED, error=f"{type(error).__name__}: {error}")
        if not (folder / OUTCOME_FILE).exists():
            # The run returned without a verdict: treat like a killed run.
            return TrialResult(trial, CRASHED, error="the run left no outcome")
        return TrialResult(trial, RAN, self._mark(trial, folder))

    def run_one(self, trial: Trial, config: Config, folder: Path, browser: Any) -> None:
        """Run the real agent for one trial.

        Args:
            trial: The trial.
            config: Its settings, with the task's start URL.
            folder: Where its run folder goes.
            browser: This worker's browser.

        Raises:
            RunSetupError: If the run could not be set up.
        """
        library_copy = folder / TRIAL_LIBRARY_DIR
        shutil.copytree(self.folder / LIBRARY_DIR, library_copy)
        try:
            client = self._client_factory(config)
        except ModelClientError as error:
            raise RunSetupError(str(error)) from error
        run_agent(
            config,
            trial.task.instruction,
            client=client,
            folder=folder,
            browser=browser,
            library=SkillStore.from_config(config, library_copy),
            # Workers run side by side, so their step-by-step lines would
            # interleave into nonsense on one screen. The run folder holds
            # every step; the sweep prints one line per finished trial.
            write=lambda line: None,
            task_extra=trial.task.extra_allowed_domains,
        )

    def _mark(self, trial: Trial, folder: Path) -> TrialVerdict | None:
        """Mark a finished run and save the mark.

        Args:
            trial: The trial.
            folder: Its run folder.

        Returns:
            TrialVerdict | None: The mark, or None if the folder could not be
            read back -- which leaves the trial unmarked for a restart.
        """
        try:
            verdict = mark_trial(trial.task, folder, self._judge)
        except (TraceReadError, TraceError) as error:
            self._log(f"sweep: could not mark {trial.name}: {error}")
            return None
        # Every judge call is paid for, and both kinds are counted.
        for asked in (verdict.judge, verdict.categorized_by):
            if asked is not None and asked.record is not None:
                record_judge_call(self.folder, asked.record)
        write_verdict(folder, verdict)
        return verdict

    def _move_aside(self, trial: Trial, folder: Path) -> Path:
        """Move a killed run's folder out of the way, keeping it.

        Args:
            trial: The trial.
            folder: Its partial run folder.

        Returns:
            Path: Where it went: ``_interrupted/<task>/trial-<n>-<k>``, with
            ``k`` counting earlier interruptions of the same trial.
        """
        parent = self.folder / INTERRUPTED_DIR / trial.task.id
        parent.mkdir(parents=True, exist_ok=True)
        k = 1
        while (target := parent / f"trial-{trial.number}-{k}").exists():
            k += 1
        folder.rename(target)
        return target


def _read_verdict(folder: Path) -> TrialVerdict | None:
    """Read a trial's saved mark.

    Args:
        folder: The trial's run folder.

    Returns:
        TrialVerdict | None: The mark, or None if it cannot be read.
    """
    try:
        return TrialVerdict.from_dict(
            json.loads((folder / VERDICT_FILE).read_text(encoding="utf-8"))
        )
    except (OSError, json.JSONDecodeError, KeyError, TypeError):
        return None


def iter_verdicts(sweep_dir: Path) -> Iterator[tuple[Path, TrialVerdict]]:
    """Walk a sweep's marked trials, skipping the interrupted folder.

    For whatever reads a sweep's results afterwards: the tables, the review.

    Args:
        sweep_dir: The sweep's folder.

    Yields:
        tuple[Path, TrialVerdict]: Each trial folder with its mark.
    """
    # Two levels down only: <task>/trial-<n>/. Killed runs sit three levels
    # down, under _interrupted/, and so are never picked up.
    for path in sorted(sweep_dir.glob(f"*/trial-*/{VERDICT_FILE}")):
        verdict = _read_verdict(path.parent)
        if verdict is not None:
            yield path.parent, verdict
