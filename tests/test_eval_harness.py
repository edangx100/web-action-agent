"""The benchmark harness: many trials at once, kept apart, and restartable.

Four groups of tests, one per promise the harness makes.

**It runs a whole suite in parallel.** A real suite of tasks on the local
practice site, real browsers, four workers. Every trial must leave its own
folder and a mark, and a counter wrapped around each run checks the two rules
of parallelism: never more runs at once than workers, and never two trials of
one task at the same time.

**Politeness holds across workers.** Four workers each waiting politely between
their *own* visits would still visit one site four times as often as allowed.
The per-site gap has to be shared, and a timed test checks it is.

**It is safe to restart.** The heart of the harness. Rerunning a finished sweep
runs nothing. A sweep killed with ``kill -9`` -- no warning, no clean-up, the
same as a crash or a power cut -- and then restarted does exactly the trials
that were left: none twice, none lost. That one runs the harness in a separate
process, because killing the test process itself would end the test.

**It marks fairly.** The rules that turn a run into pass or fail: an agent that
did not finish has not passed, a task needing the judge waits for one rather
than being guessed, and a failure gets the judge's category.
"""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from contextlib import nullcontext
from itertools import pairwise
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from test_skills_cli import a_skill
from typer.testing import CliRunner

from web_action_agent import cli
from web_action_agent.benchmark import Judge as JudgeMethod
from web_action_agent.benchmark import Suite, load_suite
from web_action_agent.budget import PREFLIGHT_LOG, read_calls
from web_action_agent.config import Config
from web_action_agent.eval_harness import (
    CRASHED,
    DONE,
    INTERRUPTED,
    INTERRUPTED_DIR,
    LIBRARY_DIR,
    NEEDS_JUDGE,
    NEW,
    RAN,
    REMARKED,
    SKIPPED,
    SWEEP_FILE,
    TRIAL_LIBRARY_DIR,
    UNMARKED,
    VERDICT_FILE,
    Harness,
    HarnessError,
    Trial,
    default_sweep_dir,
    iter_verdicts,
    mark_trial,
    plan_trials,
    trial_state,
)
from web_action_agent.judge import FAIL, JUDGING_ERROR, PASS, JudgeVerdict
from web_action_agent.model_client import ModelClient, load_profile
from web_action_agent.politeness import Throttle
from web_action_agent.skill_store import SkillStore
from web_action_agent.trace import OUTCOME_FILE, RUN_FILE

FIXTURES = Path(__file__).resolve().parent / "fixtures"
SUITE = FIXTURES / "benchmark" / "suite.yaml"
TRACE = FIXTURES / "traces" / "search-and-open"

#: The model's one reply in the real-browser tests: finish at once, with the
#: right answer. The tests are about running trials, not about the agent.
DONE_REPLY = '{"action": "done", "status": "success", "answer": "Northgate Public Library"}'


class Endpoint:
    """A stand-in model service, safe to call from several threads at once.

    Attributes:
        calls: How many times it was asked.
    """

    def __init__(self, reply: str) -> None:
        """Set up the reply."""
        self._reply = reply
        self._lock = threading.Lock()
        self.calls = 0
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **request: Any) -> SimpleNamespace:
        """Answer one call."""
        with self._lock:
            self.calls += 1
        return SimpleNamespace(
            choices=[
                SimpleNamespace(message=SimpleNamespace(content=self._reply), finish_reason="stop")
            ],
            usage=SimpleNamespace(prompt_tokens=500, completion_tokens=20),
        )


def fixture_suite(site: Any) -> Suite:
    """Load the fixture suite, pointed at the locally served practice site.

    Args:
        site: The running fixture site.

    Returns:
        Suite: The five fixture tasks, each starting on the real local page.
    """
    suite = load_suite(SUITE)
    tasks = tuple(
        task.model_copy(update={"start_url": site.url_for(task.start_url.rsplit("/", 1)[-1])})
        for task in suite.tasks
    )
    return Suite(name=suite.name, path=suite.path, sites=suite.sites, tasks=tasks)


def quick_config(**overrides: Any) -> Config:
    """Settings for a fast test sweep.

    Args:
        **overrides: Settings to change.

    Returns:
        Config: No politeness gap and no robots.txt (the practice site asks for
        a 2-second gap, which would make each test take minutes), and a small
        step cap.
    """
    values: dict[str, Any] = {"domain_delay_s": 0.0, "respect_robots": False, "max_steps": 3}
    values.update(overrides)
    return Config(**values)


class Concurrency:
    """Counts runs in progress, overall and per task, and remembers the peaks.

    Attributes:
        peak: Most runs at once.
        peak_per_task: Most runs of one task at once.
        peak_per_trial: Most runs of one trial at once.
    """

    def __init__(self) -> None:
        """Start with nothing running."""
        self._lock = threading.Lock()
        self._now = 0
        self._tasks: dict[str, int] = {}
        self._trials: dict[str, int] = {}
        self.peak = self.peak_per_task = self.peak_per_trial = 0

    def wrap(self, runner: Any) -> Any:
        """Return ``runner``, counting each call while it runs."""

        def counted(trial: Trial, config: Config, folder: Path, browser: Any) -> None:
            with self._lock:
                self._now += 1
                self._tasks[trial.task.id] = self._tasks.get(trial.task.id, 0) + 1
                self._trials[trial.name] = self._trials.get(trial.name, 0) + 1
                self.peak = max(self.peak, self._now)
                self.peak_per_task = max(self.peak_per_task, self._tasks[trial.task.id])
                self.peak_per_trial = max(self.peak_per_trial, self._trials[trial.name])
            try:
                runner(trial, config, folder, browser)
            finally:
                with self._lock:
                    self._now -= 1
                    self._tasks[trial.task.id] -= 1
                    self._trials[trial.name] -= 1

        return counted


def real_harness(
    tmp_path: Path, site: Any, endpoint: Endpoint, **kwargs: Any
) -> tuple[Harness, Concurrency]:
    """Build a harness that runs the real agent in real browsers.

    Args:
        tmp_path: Where the sweep folder goes.
        site: The running fixture site.
        endpoint: The stand-in model service.
        **kwargs: Passed to :class:`Harness`.

    Returns:
        tuple[Harness, Concurrency]: The harness, and the counter wrapped
        around its runs.
    """
    options: dict[str, Any] = {"trials": 2, "cold": True, "workers": 4}
    options.update(kwargs)
    config = options.pop("config", quick_config())
    harness = Harness(
        config,
        fixture_suite(site),
        tmp_path / "sweep",
        skills_root=tmp_path / "no-skills",
        client_factory=lambda cfg: ModelClient(load_profile("local"), transport=endpoint, env={}),
        log=lambda line: None,
        **options,
    )
    counter = Concurrency()
    harness._runner = counter.wrap(harness.run_one)
    return harness, counter


# --- Running a suite in parallel -------------------------------------------------


@pytest.mark.fixture
def test_a_suite_completes_under_four_workers_with_a_folder_per_trial(
    tmp_path: Path, fixture_site: Any
) -> None:
    endpoint = Endpoint(DONE_REPLY)
    harness, _ = real_harness(tmp_path, fixture_site, endpoint)

    report = harness.run()

    assert report.count(RAN) == 10, report.summary()
    for result in report.results:
        folder = result.trial.folder(harness.folder)
        assert (folder / OUTCOME_FILE).exists(), f"{result.trial.name} left a complete run"
        assert (folder / VERDICT_FILE).exists(), f"{result.trial.name} was marked"
        assert result.verdict is not None and result.verdict.status == PASS, result
    # One connection test before the sweep (the local profile has a text
    # route only), then one call per trial: each run did its task once.
    assert endpoint.calls == 1 + 10
    assert len(read_calls(harness.folder / PREFLIGHT_LOG)) == 1, "and the test call is counted"


@pytest.mark.fixture
def test_concurrency_never_exceeds_one_agent_per_task(tmp_path: Path, fixture_site: Any) -> None:
    harness, counter = real_harness(tmp_path, fixture_site, Endpoint(DONE_REPLY), trials=3)

    harness.run()

    assert counter.peak_per_trial == 1, "a trial is run by one worker only"
    assert counter.peak_per_task == 1, "two trials of one task never run side by side"
    assert 1 < counter.peak <= 4, "several at once, but never more than the workers"


@pytest.mark.fixture
def test_rerunning_a_finished_sweep_runs_nothing(tmp_path: Path, fixture_site: Any) -> None:
    endpoint = Endpoint(DONE_REPLY)
    harness, _ = real_harness(tmp_path, fixture_site, endpoint)
    harness.run()
    calls_after_first = endpoint.calls

    again, counter = real_harness(tmp_path, fixture_site, endpoint)
    report = again.run()

    assert report.count(RAN) == 0
    assert report.count(SKIPPED) == 10
    assert counter.peak == 0, "no agent was started"
    assert endpoint.calls == calls_after_first, "and nothing was spent"
    assert "0 ran, 10 already done" in report.summary()


@pytest.mark.fixture
def test_politeness_holds_across_workers(
    tmp_path: Path, fixture_site: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Every trial opens a page on the same site. With a 0.5-second gap, four
    # workers must still take turns, half a second apart, not go at once.
    gap = 0.5
    visits: list[float] = []
    lock = threading.Lock()
    real_wait = Throttle.wait

    def timed_wait(self: Throttle, url: str) -> float:
        waited = real_wait(self, url)
        with lock:
            visits.append(time.time())
        return waited

    monkeypatch.setattr(Throttle, "wait", timed_wait)
    harness, counter = real_harness(
        tmp_path,
        fixture_site,
        Endpoint(DONE_REPLY),
        trials=1,
        config=quick_config(domain_delay_s=gap),
    )

    harness.run()

    assert len(visits) == 5
    assert counter.peak > 1, "the workers really did overlap"
    times = sorted(visits)
    gaps = [later - earlier for earlier, later in pairwise(times)]
    assert min(gaps) >= gap - 0.05, f"visits too close together: {gaps}"


# --- Restarting, with stand-in runs ------------------------------------------------
#
# These use a runner that writes a finished run folder by copying the example
# run saved in the fixtures, instead of starting a browser and a model. What
# is being tested is the bookkeeping around runs, which is the same either way.


def finished_run(folder: Path, *, agent_ok: bool = True) -> None:
    """Write a complete run folder, as if the agent had just finished.

    Args:
        folder: Where the run goes.
        agent_ok: False to make the agent's own run end in failure.
    """
    folder.mkdir(parents=True, exist_ok=True)
    for path in TRACE.iterdir():
        if path.is_file():
            shutil.copy(path, folder / path.name)
    if not agent_ok:
        outcome_path = folder / OUTCOME_FILE
        outcome = json.loads(outcome_path.read_text())
        outcome["outcome"].update(
            ok=False, category="step_cap", reason="ran out of steps", answer=None
        )
        outcome_path.write_text(json.dumps(outcome))


class FakeRunner:
    """Writes finished runs without a browser or a model, and counts them.

    Attributes:
        ran: The trials it ran, in order.
    """

    def __init__(self, *, crash: str = "", agent_ok: bool = True, delay: float = 0.0) -> None:
        """Set up the runner.

        Args:
            crash: The name of a trial to crash on, or empty for none.
            agent_ok: False to make every run end in the agent failing.
            delay: Seconds each run takes.
        """
        self._crash = crash
        self._agent_ok = agent_ok
        self._delay = delay
        self._lock = threading.Lock()
        self.ran: list[str] = []

    def __call__(self, trial: Trial, config: Config, folder: Path, browser: Any) -> None:
        """Run one trial."""
        with self._lock:
            self.ran.append(trial.name)
        folder.mkdir(parents=True)
        if trial.name == self._crash:
            raise RuntimeError("the browser fell over")
        time.sleep(self._delay)
        finished_run(folder, agent_ok=self._agent_ok)


def fake_harness(folder: Path, runner: Any, **kwargs: Any) -> Harness:
    """Build a harness over the fixture suite with a stand-in runner.

    Args:
        folder: The sweep folder.
        runner: The stand-in runner.
        **kwargs: Passed to :class:`Harness`.

    Returns:
        Harness: A harness that needs no browser, model, or key.
    """
    options: dict[str, Any] = {"trials": 2, "cold": True, "workers": 3}
    options.update(kwargs)
    config = options.pop("config", Config())
    return Harness(
        config,
        stand_in_suite(),
        folder,
        skills_root=options.pop("skills_root", folder.parent / "no-skills"),
        client_factory=lambda cfg: None,  # type: ignore[arg-type,return-value]
        browser_factory=lambda cfg: nullcontext(),
        runner=runner,
        log=lambda line: None,
        **options,
    )


def stand_in_suite() -> Suite:
    """The fixture suite, with tasks whose checks the example run passes.

    The example run ended on ``form-step-1.html`` with the answer "Membership
    application", so these checks pass for it.

    Returns:
        Suite: Five tasks.
    """
    suite = load_suite(SUITE)
    tasks = tuple(
        task.model_copy(
            update={
                "verdicts": (
                    task.verdicts[0].model_copy(update={"accepted": ("Membership application",)}),
                    task.verdicts[1].model_copy(update={"value": "form-step-1"}),
                )
            }
        )
        for task in suite.tasks
    )
    return Suite(name=suite.name, path=suite.path, sites=suite.sites, tasks=tasks)


@pytest.mark.unit
def test_trials_are_planned_round_by_round() -> None:
    plan = plan_trials(stand_in_suite(), 2)
    names = [trial.name for trial in plan]
    assert names[:5] == [f"{t.id}/trial-1" for t in stand_in_suite().tasks]
    assert names[5:] == [f"{t.id}/trial-2" for t in stand_in_suite().tasks]


@pytest.mark.unit
def test_trial_state_is_read_from_the_folder(tmp_path: Path) -> None:
    folder = tmp_path / "t"
    assert trial_state(folder) == NEW
    folder.mkdir()
    (folder / RUN_FILE).write_text("{}")
    assert trial_state(folder) == INTERRUPTED, "started, never finished"
    finished_run(folder)
    assert trial_state(folder) == UNMARKED, "finished, not yet marked"
    (folder / VERDICT_FILE).write_text(json.dumps({"status": JUDGING_ERROR}))
    assert trial_state(folder) == UNMARKED, "a judging error is marked again later"
    (folder / VERDICT_FILE).write_text(json.dumps({"status": FAIL}))
    assert trial_state(folder) == UNMARKED, "a fail with no category is not finished"
    (folder / VERDICT_FILE).write_text(json.dumps({"status": FAIL, "category": "planning"}))
    assert trial_state(folder) == DONE, "a categorised fail is a result, not unfinished work"


@pytest.mark.unit
def test_a_killed_run_is_moved_aside_and_run_again(tmp_path: Path) -> None:
    sweep = tmp_path / "sweep"
    runner = FakeRunner()
    fake_harness(sweep, runner, trials=1).run()
    # Pretend the second task's run was killed partway: a folder, no outcome.
    victim = sweep / "fixture-search" / "trial-1"
    (victim / OUTCOME_FILE).unlink()
    (victim / VERDICT_FILE).unlink()

    again = FakeRunner()
    report = fake_harness(sweep, again, trials=1).run()

    assert again.ran == ["fixture-search/trial-1"], "only the killed trial ran again"
    assert report.count(SKIPPED) == 4
    assert len(report.moved_aside) == 1
    kept = sweep / INTERRUPTED_DIR / "fixture-search" / "trial-1-1"
    assert (kept / RUN_FILE).exists(), "the partial run is kept as evidence"
    assert trial_state(victim) == DONE
    assert all(f.parent.name != INTERRUPTED_DIR for f, _ in iter_verdicts(sweep)), (
        "and is never counted in the results"
    )


@pytest.mark.unit
def test_a_finished_but_unmarked_run_is_marked_without_running_again(tmp_path: Path) -> None:
    sweep = tmp_path / "sweep"
    fake_harness(sweep, FakeRunner(), trials=1).run()
    (sweep / "fixture-home" / "trial-1" / VERDICT_FILE).unlink()

    again = FakeRunner()
    report = fake_harness(sweep, again, trials=1).run()

    assert again.ran == []
    assert report.count(REMARKED) == 1
    assert trial_state(sweep / "fixture-home" / "trial-1") == DONE


@pytest.mark.unit
def test_a_failed_trial_is_not_run_again(tmp_path: Path) -> None:
    sweep = tmp_path / "sweep"
    first = fake_harness(sweep, FakeRunner(agent_ok=False), trials=1).run()
    assert all(r.verdict is not None and r.verdict.status == FAIL for r in first.results)

    again = FakeRunner()
    fake_harness(sweep, again, trials=1).run()

    assert again.ran == [], "retrying failures until they pass would be cheating"


@pytest.mark.unit
def test_asking_for_more_trials_later_adds_only_the_new_ones(tmp_path: Path) -> None:
    sweep = tmp_path / "sweep"
    fake_harness(sweep, FakeRunner(), trials=1).run()

    again = FakeRunner()
    fake_harness(sweep, again, trials=2).run()

    assert sorted(again.ran) == sorted(f"{t.id}/trial-2" for t in stand_in_suite().tasks)


@pytest.mark.unit
def test_one_crashing_trial_does_not_stop_the_others(tmp_path: Path) -> None:
    sweep = tmp_path / "sweep"
    report = fake_harness(sweep, FakeRunner(crash="fixture-browse/trial-1"), trials=1).run()

    assert report.count(CRASHED) == 1
    assert report.count(RAN) == 4

    again = FakeRunner()
    fake_harness(sweep, again, trials=1).run()
    assert again.ran == ["fixture-browse/trial-1"], "the crashed trial is run on restart"


@pytest.mark.unit
def test_a_crash_says_what_crashed(tmp_path: Path) -> None:
    lines: list[str] = []
    harness = fake_harness(tmp_path / "sweep", FakeRunner(crash="fixture-browse/trial-1"), trials=1)
    harness._log = lines.append

    harness.run()

    crashed = next(line for line in lines if "fixture-browse/trial-1" in line)
    assert "crashed (RuntimeError: the browser fell over)" in crashed


@pytest.mark.unit
@pytest.mark.parametrize(
    ("change", "named"),
    [
        ({"cold": False}, "library"),
        ({"config": Config(profile="nvidia")}, "profile"),
        ({"config": Config(judge_profile="openrouter-judge")}, "judge_profile"),
    ],
)
def test_a_folder_holding_a_different_sweep_is_refused(
    tmp_path: Path, change: dict[str, Any], named: str
) -> None:
    sweep = tmp_path / "sweep"
    fake_harness(sweep, FakeRunner(), trials=1).run()

    with pytest.raises(HarnessError, match=named):
        fake_harness(sweep, FakeRunner(), trials=1, **change).run()


@pytest.mark.unit
def test_the_sweep_records_what_it_is_and_when_it_started(tmp_path: Path) -> None:
    sweep = tmp_path / "sweep"
    fake_harness(sweep, FakeRunner(), trials=1).run()
    first = json.loads((sweep / SWEEP_FILE).read_text())

    fake_harness(sweep, FakeRunner(), trials=2).run()
    second = json.loads((sweep / SWEEP_FILE).read_text())

    assert first["library"] == "cold" and first["suite"] == "fixture-suite"
    assert second["started_at"] == first["started_at"], "a restart keeps the first start time"


@pytest.mark.unit
def test_a_warm_sweep_runs_on_a_fixed_copy_of_the_library(tmp_path: Path) -> None:
    skills = tmp_path / "skills"
    original = SkillStore(skills).save(a_skill("search", domain="fixture.test")).read_text()
    sweep = tmp_path / "sweep"

    def runner(trial: Trial, config: Config, folder: Path, browser: Any) -> None:
        # The real runner copies the library into each trial; check that the
        # copy is what it starts from, and that writing to it stays local.
        shutil.copytree(sweep / LIBRARY_DIR, folder / TRIAL_LIBRARY_DIR)
        (folder / TRIAL_LIBRARY_DIR / "search.json").write_text('{"name": "search", "uses": 1}')
        finished_run(folder)

    fake_harness(sweep, runner, trials=1, cold=False, skills_root=skills).run()
    (skills / "added-later.json").write_text("{}")
    fake_harness(sweep, runner, trials=2, cold=False, skills_root=skills).run()

    assert (sweep / LIBRARY_DIR / "search.json").read_text() == original, (
        "the sweep's copy is untouched by the trials"
    )
    assert not (sweep / LIBRARY_DIR / "added-later.json").exists(), (
        "and a restart keeps the copy the sweep started with"
    )
    assert (skills / "search.json").read_text() == original, "the real library too"


@pytest.mark.unit
def test_a_cold_sweep_starts_with_an_empty_library(tmp_path: Path) -> None:
    skills = tmp_path / "skills"
    skills.mkdir()
    (skills / "search.json").write_text("{}")
    sweep = tmp_path / "sweep"
    fake_harness(sweep, FakeRunner(), trials=1, cold=True, skills_root=skills).run()
    assert list((sweep / LIBRARY_DIR).iterdir()) == []


@pytest.mark.unit
def test_a_missing_key_stops_the_sweep_before_anything_runs(tmp_path: Path) -> None:
    runner = FakeRunner()
    harness = Harness(
        Config(),
        stand_in_suite(),
        tmp_path / "sweep",
        trials=1,
        cold=True,
        client_factory=lambda cfg: ModelClient.from_config(cfg, env={}),
        browser_factory=lambda cfg: nullcontext(),
        runner=runner,
        log=lambda line: None,
    )
    with pytest.raises(HarnessError, match="API key"):
        harness.run()
    assert runner.ran == []


# --- Killed for real ---------------------------------------------------------------

#: The script run in a separate process for the kill test. It runs the
#: harness with stand-in runs that each take a moment, and each trial writes
#: a line to a log when it starts and when it finishes.
KILL_SCRIPT = """
import sys, time, json, shutil
from contextlib import nullcontext
from pathlib import Path
sys.path.insert(0, {tests!r})
from test_eval_harness import stand_in_suite, finished_run
from web_action_agent.config import Config
from web_action_agent.eval_harness import Harness

sweep, log = Path(sys.argv[1]), Path(sys.argv[2])

def runner(trial, config, folder, browser):
    folder.mkdir(parents=True)
    (folder / "run.json").write_text("{{}}")
    with log.open("a") as f:
        f.write(f"start {{trial.name}}\\n")
    time.sleep(0.4)
    finished_run(folder)
    with log.open("a") as f:
        f.write(f"done {{trial.name}}\\n")

Harness(Config(), stand_in_suite(), sweep, trials=3, cold=True, workers=2,
        skills_root=sweep.parent / "none", client_factory=lambda c: None,
        browser_factory=lambda c: nullcontext(), runner=runner,
        log=lambda line: None).run()
"""


@pytest.mark.unit
def test_killing_the_harness_and_restarting_does_exactly_the_remaining_trials(
    tmp_path: Path,
) -> None:
    sweep, log = tmp_path / "sweep", tmp_path / "log.txt"
    script = tmp_path / "harness_child.py"
    script.write_text(KILL_SCRIPT.format(tests=str(Path(__file__).parent)))

    # Start the sweep in its own process, and kill it with SIGKILL -- no
    # clean-up of any kind -- once some trials are done and others are half way.
    child = subprocess.Popen([sys.executable, str(script), str(sweep), str(log)])
    deadline = time.time() + 30
    while time.time() < deadline:
        lines = log.read_text().splitlines() if log.exists() else []
        # Kill when at least three trials are done AND the newest line is a
        # trial starting: that run has its whole 0.4 seconds still to go, so
        # it is certainly caught half way. ("Some run has not finished yet" is
        # not enough: the two workers run in step, and the other one may be
        # milliseconds from done.)
        if sum(line.startswith("done") for line in lines) >= 3 and lines[-1].startswith("start"):
            break
        time.sleep(0.005)
    os.kill(child.pid, signal.SIGKILL)
    child.wait()

    # What the kill left on disk. A run "finished" if its folder holds the
    # agent's outcome; it may or may not have been marked before the kill.
    all_trials = {trial.name for trial in plan_trials(stand_in_suite(), 3)}
    finished = {name for name in all_trials if (sweep / name / OUTCOME_FILE).exists()}
    half_done = {name for name in all_trials if (sweep / name).exists()} - finished
    assert 3 <= len(finished) < len(all_trials), "the kill landed mid-sweep"
    assert half_done, "with at least one run caught half way"

    # Restart in this process, and count what it runs.
    again = FakeRunner()
    report = fake_harness(sweep, again, trials=3, workers=2).run()

    assert set(again.ran) == all_trials - finished, (
        "exactly the trials that had not finished: none twice, none lost"
    )
    assert len(again.ran) == len(set(again.ran)), "and each of those once"
    assert len(report.moved_aside) == len(half_done), "each half-done run was set aside"
    assert all(trial_state(sweep / name) == DONE for name in all_trials), "the sweep is complete"


# --- Marking -------------------------------------------------------------------------


class FakeJudge:
    """A judge that returns a fixed verdict and counts how often it was asked.

    Attributes:
        asked: How many times it was asked.
    """

    def __init__(self, verdict: JudgeVerdict) -> None:
        """Set up the verdict."""
        self._verdict = verdict
        self.asked = 0

    def judge(self, task: Any, run: Any) -> JudgeVerdict:
        """Return the fixed verdict."""
        self.asked += 1
        return self._verdict

    def categorize(self, task: Any, run: Any, reason: str) -> JudgeVerdict:
        """Return the fixed verdict, as the answer to "which kind of failure"."""
        self.asked += 1
        return self._verdict


def judged_task() -> Any:
    """A stand-in task that also asks for the judge."""
    task = stand_in_suite().tasks[0]
    return task.model_copy(
        update={
            "verdicts": (*task.verdicts, JudgeMethod(method="judge", expected_outcome="A form."))
        }
    )


@pytest.mark.unit
def test_a_run_passes_when_the_agent_finished_and_every_check_passes(tmp_path: Path) -> None:
    finished_run(tmp_path / "r")
    verdict = mark_trial(stand_in_suite().tasks[0], tmp_path / "r", None)
    assert verdict.status == PASS
    assert [c.passed for c in verdict.checks] == [True, True]


@pytest.mark.unit
def test_an_agent_that_did_not_finish_has_not_passed(tmp_path: Path) -> None:
    # The page and the address might be right, but the agent ran out of steps.
    finished_run(tmp_path / "r", agent_ok=False)
    verdict = mark_trial(stand_in_suite().tasks[0], tmp_path / "r", None)
    assert verdict.status == FAIL
    assert "did not finish" in verdict.reason


@pytest.mark.unit
def test_a_task_needing_the_judge_waits_for_one(tmp_path: Path) -> None:
    finished_run(tmp_path / "r")
    verdict = mark_trial(judged_task(), tmp_path / "r", None)
    assert verdict.status == NEEDS_JUDGE


@pytest.mark.unit
def test_a_judging_error_leaves_the_trial_to_be_marked_again(tmp_path: Path) -> None:
    finished_run(tmp_path / "r")
    judge = FakeJudge(JudgeVerdict(JUDGING_ERROR, error="service down"))
    verdict = mark_trial(judged_task(), tmp_path / "r", judge)  # type: ignore[arg-type]
    assert verdict.status == JUDGING_ERROR


@pytest.mark.unit
def test_the_judge_decides_a_task_that_asks_for_it(tmp_path: Path) -> None:
    finished_run(tmp_path / "r")
    judge = FakeJudge(JudgeVerdict(FAIL, justification="Wrong form.", category="planning"))
    verdict = mark_trial(judged_task(), tmp_path / "r", judge)  # type: ignore[arg-type]
    assert verdict.status == FAIL
    assert verdict.category == "planning"
    assert judge.asked == 1


@pytest.mark.unit
def test_a_failure_found_by_the_checks_gets_its_category_from_the_judge(tmp_path: Path) -> None:
    finished_run(tmp_path / "r", agent_ok=False)
    judge = FakeJudge(JudgeVerdict(FAIL, justification="Ran out.", category="navigation"))
    verdict = mark_trial(stand_in_suite().tasks[0], tmp_path / "r", judge)  # type: ignore[arg-type]
    assert verdict.status == FAIL
    assert verdict.category == "navigation"
    assert verdict.categorized_by is not None, "asked for the category alone"
    assert verdict.complete


@pytest.mark.unit
def test_a_passing_run_does_not_spend_a_judge_call_it_does_not_need(tmp_path: Path) -> None:
    finished_run(tmp_path / "r")
    judge = FakeJudge(JudgeVerdict(PASS, justification="x"))
    mark_trial(stand_in_suite().tasks[0], tmp_path / "r", judge)  # type: ignore[arg-type]
    assert judge.asked == 0


# --- The command -------------------------------------------------------------------


@pytest.mark.unit
def test_waa_eval_runs_the_harness_and_names_the_sweep_folder(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    seen: dict[str, Any] = {}

    class Recording:
        """Stands in for the harness, recording how it was built."""

        def __init__(self, config: Config, suite: Suite, folder: Path, **kwargs: Any) -> None:
            seen.update(folder=folder, suite=suite.name, **kwargs)

        def run(self) -> Any:
            return SimpleNamespace(
                results=(),
                summary=lambda: "0 trials",
                folder=seen["folder"],
                stopped_by="",
                not_started=0,
            )

    monkeypatch.setattr(cli, "Harness", Recording)
    monkeypatch.setattr(cli, "resolve_env", dict)
    monkeypatch.setattr(cli, "RUNS_ROOT", tmp_path / "runs")

    result = CliRunner().invoke(
        cli.app, ["eval", "--suite", str(SUITE), "--trials", "2", "--warm", "--workers", "3"]
    )

    assert result.exit_code == cli.EXIT_SUCCESS, result.output
    assert seen["folder"] == default_sweep_dir(tmp_path / "runs", "openrouter", cold=False)
    assert seen["trials"] == 2 and seen["cold"] is False and seen["workers"] == 3
    assert "no judge configured" in result.output


@pytest.mark.unit
def test_waa_eval_with_an_unreadable_suite_exits_two(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli, "resolve_env", dict)
    result = CliRunner().invoke(cli.app, ["eval", "--suite", "no-such-suite.yaml"])
    assert result.exit_code == cli.EXIT_USAGE
