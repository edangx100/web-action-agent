"""The money and time ceilings on a benchmark sweep.

The test that matters most is the one TASKS-style plain words would put first:
a sweep given a one-cent ceiling stops after the trial that crosses it, keeps
everything it finished, and -- with the ceiling raised -- carries on with
exactly the trials that were left. The money in it is real arithmetic: each
stand-in trial makes a real model-client call to a stand-in service, priced by
the real price table, so the cost the ceiling sees is computed the same way a
live sweep's is.

The rest pin the details that would otherwise quietly weaken the ceiling:
spend from killed runs and from the judge counts, an unpriced call is reported
rather than guessed, the time ceiling adds up across restarts but not the gaps
between them, and the command warns before a sweep whose models have no price.
"""

from __future__ import annotations

import json
import threading
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from test_eval_harness import Endpoint, finished_run, stand_in_suite
from typer.testing import CliRunner

from web_action_agent import cli
from web_action_agent.budget import (
    JUDGE_LOG,
    LEDGER_FILE,
    Budget,
    Spend,
    record_judge_call,
    sweep_spend,
    unpriced_routes,
)
from web_action_agent.config import Config
from web_action_agent.cost import Price, PriceList, shipped_prices
from web_action_agent.eval_harness import RAN, SKIPPED, Harness, Trial, plan_trials, trial_state
from web_action_agent.model_client import (
    CallRecord,
    ModelClient,
    load_profile,
    system_message,
    user_message,
)
from web_action_agent.trace import COST_LOG

#: The stand-in service reports 500 + 20 tokens per call. At $10 per million
#: tokens each way, one call costs exactly 520 / 1,000,000 * $10 = $0.0052.
PRICE = Price(input_usd_per_mtok=10.0, output_usd_per_mtok=10.0, checked="test")
COST_PER_CALL = 0.0052


def priced_client() -> ModelClient:
    """A real model client over a stand-in service, with a test price."""
    profile = load_profile("local")
    prices = PriceList({(profile.name, profile.text.model): PRICE}, source="test")
    return ModelClient(profile, transport=Endpoint('{"ok": true}'), env={}, prices=prices)


class PayingRunner:
    """A stand-in trial that makes one real, priced model call.

    The call goes through the real model client, which prices it from the
    price table and returns its cost record; the record is written to the
    trial's ``calls.jsonl`` exactly where a real run writes it. Then a
    finished run folder is filled in around it.

    Attributes:
        ran: The trials it ran, in order.
        on_run: Called after each trial, for a test that moves a clock.
    """

    def __init__(self, on_run: Any = None) -> None:
        """Set up the runner."""
        self.ran: list[str] = []
        self.on_run = on_run
        self._client = priced_client()
        self._lock = threading.Lock()

    def __call__(self, trial: Trial, config: Config, folder: Path, browser: Any) -> None:
        """Run one trial: one paid call, then a finished run folder."""
        with self._lock:
            self.ran.append(trial.name)
        folder.mkdir(parents=True)
        response = self._client.complete([system_message("x"), user_message("y")])
        finished_run(folder)
        (folder / COST_LOG).write_text(json.dumps(asdict(response.record)) + "\n")
        if self.on_run is not None:
            self.on_run()


def a_sweep(folder: Path, runner: Any, budget: Budget, **kwargs: Any) -> Harness:
    """Build a sweep of the stand-in suite with a given budget.

    Args:
        folder: The sweep folder.
        runner: The stand-in runner.
        budget: The ceilings.
        **kwargs: Passed to :class:`Harness`.

    Returns:
        Harness: One trial of each of the five tasks, one worker at a time
        unless told otherwise, so the order the money is spent in is fixed.
    """
    options: dict[str, Any] = {"trials": 1, "cold": True, "workers": 1}
    options.update(kwargs)
    return Harness(
        Config(),
        stand_in_suite(),
        folder,
        skills_root=folder.parent / "no-skills",
        client_factory=lambda cfg: None,  # type: ignore[arg-type,return-value]
        browser_factory=lambda cfg: nullcontext(),
        runner=runner,
        budget=budget,
        log=lambda line: None,
        **options,
    )


# --- The money ceiling -----------------------------------------------------------


@pytest.mark.mock_llm
def test_a_one_cent_ceiling_stops_after_the_run_that_crosses_it(tmp_path: Path) -> None:
    sweep = tmp_path / "sweep"
    runner = PayingRunner()

    report = a_sweep(sweep, runner, Budget(sweep, max_usd=0.01, max_hours=6)).run()

    # Trial 1 brings the spend to $0.0052, under the ceiling, so trial 2
    # starts; trial 2 brings it to $0.0104, over, so nothing else starts.
    assert len(runner.ran) == 2
    assert report.count(RAN) == 2
    assert report.not_started == 3
    assert "money ceiling of $0.01" in report.stopped_by
    assert report.summary().startswith("5 trials: 2 ran, 0 already done, 3 not started")
    assert sweep_spend(sweep).known_usd == pytest.approx(2 * COST_PER_CALL)


@pytest.mark.mock_llm
def test_stopping_leaves_finished_trials_intact(tmp_path: Path) -> None:
    sweep = tmp_path / "sweep"
    runner = PayingRunner()
    a_sweep(sweep, runner, Budget(sweep, max_usd=0.01, max_hours=6)).run()

    for name in runner.ran:
        assert trial_state(sweep / name) == "done", f"{name} kept, finished and marked"
    untouched = {t.name for t in plan_trials(stand_in_suite(), 1)} - set(runner.ran)
    for name in untouched:
        assert not (sweep / name).exists(), "and nothing was started that was not finished"


@pytest.mark.mock_llm
def test_raising_the_ceiling_resumes_with_exactly_the_trials_left(tmp_path: Path) -> None:
    sweep = tmp_path / "sweep"
    first = PayingRunner()
    a_sweep(sweep, first, Budget(sweep, max_usd=0.01, max_hours=6)).run()

    second = PayingRunner()
    report = a_sweep(sweep, second, Budget(sweep, max_usd=1.00, max_hours=6)).run()

    every = {t.name for t in plan_trials(stand_in_suite(), 1)}
    assert set(second.ran) == every - set(first.ran), "the rest, none twice"
    assert report.count(SKIPPED) == 2
    assert report.stopped_by == "" and report.not_started == 0
    assert sweep_spend(sweep).known_usd == pytest.approx(5 * COST_PER_CALL)


@pytest.mark.mock_llm
def test_a_sweep_already_over_its_ceiling_starts_nothing(tmp_path: Path) -> None:
    sweep = tmp_path / "sweep"
    a_sweep(sweep, PayingRunner(), Budget(sweep, max_usd=0.01, max_hours=6)).run()

    again = PayingRunner()
    report = a_sweep(sweep, again, Budget(sweep, max_usd=0.01, max_hours=6)).run()

    assert again.ran == [], "restarting with the same ceiling spends nothing more"
    assert "money ceiling" in report.stopped_by


@pytest.mark.mock_llm
def test_with_several_workers_the_overshoot_is_at_most_the_trials_in_progress(
    tmp_path: Path,
) -> None:
    # Three at once: the ceiling is checked before each trial starts, so at
    # most the three already running can carry the spend past it.
    sweep = tmp_path / "sweep"
    runner = PayingRunner()
    a_sweep(sweep, runner, Budget(sweep, max_usd=0.006, max_hours=6), workers=3, trials=3).run()
    assert 2 <= len(runner.ran) <= 2 + 3 - 1


# --- What counts as spent ----------------------------------------------------------


def a_record(cost: float | None) -> CallRecord:
    """A call record with a given cost (None for unpriced)."""
    return CallRecord(
        call_id="call-x",
        started_at="2026-10-11T00:00:00Z",
        ended_at="2026-10-11T00:00:01Z",
        profile="p",
        model="m",
        vision=False,
        prompt_tokens=1,
        completion_tokens=1,
        latency_s=0.1,
        finish_reason="stop",
        cost_usd=cost,
    )


def write_calls(folder: Path, *costs: float | None) -> None:
    """Write a ``calls.jsonl`` with one record per cost."""
    folder.mkdir(parents=True, exist_ok=True)
    (folder / COST_LOG).write_text("".join(json.dumps(asdict(a_record(c))) + "\n" for c in costs))


@pytest.mark.unit
def test_spend_counts_killed_runs_and_the_judge(tmp_path: Path) -> None:
    write_calls(tmp_path / "task-a" / "trial-1", 0.10, 0.20)
    write_calls(tmp_path / "_interrupted" / "task-a" / "trial-2-1", 0.05)  # killed: still paid
    record_judge_call(tmp_path, a_record(0.01))
    record_judge_call(tmp_path, a_record(0.01))  # marked again later: both were paid

    spend = sweep_spend(tmp_path)

    assert spend.known_usd == pytest.approx(0.37)
    assert spend.calls == 5
    assert (tmp_path / JUDGE_LOG).read_text().count("\n") == 2


@pytest.mark.unit
def test_an_unpriced_call_is_reported_not_guessed(tmp_path: Path) -> None:
    write_calls(tmp_path / "t" / "trial-1", 0.10, None, None)
    spend = sweep_spend(tmp_path)
    assert spend.known_usd == pytest.approx(0.10), "not counted as zero, not counted at all"
    assert spend.unpriced_calls == 2
    assert "2 with no price" in spend.summary()


@pytest.mark.unit
def test_a_half_written_last_line_is_skipped_not_fatal(tmp_path: Path) -> None:
    folder = tmp_path / "t" / "trial-1"
    write_calls(folder, 0.10)
    with (folder / COST_LOG).open("a") as log:
        log.write('{"call_id": "call-y", "cost_')  # a trial still writing
    assert sweep_spend(tmp_path).known_usd == pytest.approx(0.10)


@pytest.mark.unit
def test_the_harness_records_every_judge_call_it_makes(tmp_path: Path) -> None:
    from web_action_agent.benchmark import Judge as JudgeMethod
    from web_action_agent.judge import FAIL, JudgeVerdict

    class PricedJudge:
        def judge(self, task: Any, run: Any) -> JudgeVerdict:
            return JudgeVerdict(FAIL, "x", "planning", record=a_record(0.02))

    suite = stand_in_suite()
    tasks = tuple(
        t.model_copy(
            update={"verdicts": (*t.verdicts, JudgeMethod(method="judge", expected_outcome="x"))}
        )
        for t in suite.tasks
    )
    judged = type(suite)(name=suite.name, path=suite.path, sites=suite.sites, tasks=tasks)
    sweep = tmp_path / "sweep"

    def runner(trial: Trial, config: Config, folder: Path, browser: Any) -> None:
        finished_run(folder)

    Harness(
        Config(),
        judged,
        sweep,
        trials=1,
        cold=True,
        workers=1,
        skills_root=tmp_path / "none",
        judge=PricedJudge(),  # type: ignore[arg-type]
        client_factory=lambda cfg: None,  # type: ignore[arg-type,return-value]
        browser_factory=lambda cfg: nullcontext(),
        runner=runner,
        log=lambda line: None,
    ).run()

    assert sweep_spend(sweep).known_usd == pytest.approx(5 * 0.02)


# --- The time ceiling ------------------------------------------------------------


class Clock:
    """A clock a test moves forward by hand, in hours."""

    def __init__(self) -> None:
        """Start at zero."""
        self.now = 0.0

    def __call__(self) -> float:
        """Return the time, in seconds."""
        return self.now

    def advance_hours(self, hours: float) -> None:
        """Move forward."""
        self.now += hours * 3600


@pytest.mark.mock_llm
def test_the_time_ceiling_stops_new_trials(tmp_path: Path) -> None:
    sweep = tmp_path / "sweep"
    clock = Clock()
    runner = PayingRunner(on_run=lambda: clock.advance_hours(1))  # each trial takes "an hour"

    report = a_sweep(sweep, runner, Budget(sweep, max_usd=15, max_hours=2.5, clock=clock)).run()

    # Checked before each trial: 0 h, 1 h, 2 h are under 2.5 h; 3 h is not.
    assert len(runner.ran) == 3
    assert "time ceiling of 2.5 hours" in report.stopped_by


@pytest.mark.mock_llm
def test_the_time_total_is_saved_after_every_trial(tmp_path: Path) -> None:
    # Not only at the end: a sweep killed with no warning never reaches the
    # end, and its time would be lost from the total. Each trial reads the
    # saved total as it starts; by the third, two trials' time must be there.
    sweep = tmp_path / "sweep"
    clock = Clock()
    saved: list[float] = []

    def read_then_tick() -> None:
        ledger = sweep / LEDGER_FILE
        saved.append(json.loads(ledger.read_text())["elapsed_s"] if ledger.exists() else 0.0)
        clock.advance_hours(1)

    runner = PayingRunner(on_run=read_then_tick)
    a_sweep(sweep, runner, Budget(sweep, max_usd=15, max_hours=10, clock=clock)).run()

    assert saved[2] == pytest.approx(2 * 3600)


@pytest.mark.unit
def test_elapsed_time_adds_up_across_restarts_but_not_the_gap_between(tmp_path: Path) -> None:
    tmp_path.mkdir(exist_ok=True)
    clock = Clock()
    first = Budget(tmp_path, max_usd=15, max_hours=6, clock=clock)
    first.start()
    clock.advance_hours(2)
    first.save()

    clock.advance_hours(48)  # two days off: not running, not counted

    second = Budget(tmp_path, max_usd=15, max_hours=6, clock=clock)
    second.start()
    clock.advance_hours(1)
    assert second.elapsed_s() == pytest.approx(3 * 3600)
    assert json.loads((tmp_path / LEDGER_FILE).read_text())["elapsed_s"] == pytest.approx(2 * 3600)


# --- Prices -------------------------------------------------------------------------


@pytest.mark.unit
def test_the_default_planner_route_has_a_price() -> None:
    # Without it, the money ceiling would see only the judge, and a $15 limit
    # on a sweep whose main cost is invisible would never trigger.
    profile = load_profile("openrouter")
    price = shipped_prices().price_for(profile.name, profile.text.model)
    assert price is not None and price.known


@pytest.mark.unit
def test_unpriced_routes_are_listed() -> None:
    missing = unpriced_routes([load_profile("openrouter"), load_profile("openrouter-judge")])
    # The free vision tier is deliberately left unpriced; everything else has one.
    assert missing == ["openrouter -> nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free"]
    assert len(unpriced_routes([load_profile("nvidia")])) == 2


# --- The command -----------------------------------------------------------------


@pytest.mark.unit
def test_waa_eval_passes_the_ceilings_and_reports_an_early_stop(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    seen: dict[str, Any] = {}

    class Recording:
        """Stands in for the harness: records the settings, reports a stop."""

        def __init__(self, config: Config, suite: Any, folder: Path, **kwargs: Any) -> None:
            seen["config"] = config

        def run(self) -> Any:
            return SimpleNamespace(
                results=(),
                summary=lambda: "12 trials: 4 ran, 8 not started",
                folder=tmp_path,
                stopped_by="the money ceiling of $0.50 has been reached",
                not_started=8,
            )

    monkeypatch.setattr(cli, "Harness", Recording)
    monkeypatch.setattr(cli, "resolve_env", dict)
    result = CliRunner().invoke(
        cli.app,
        ["eval", "--budget-usd", "0.5", "--budget-hours", "1.5", "--profile", "nvidia"],
    )

    assert seen["config"].budget_usd == 0.5 and seen["config"].budget_hours == 1.5
    assert "budget: $0.50 and 1.5 hours" in result.output
    assert "no price in pricing.yaml" in result.output, "warned before anything was spent"
    assert "stopped early: the money ceiling" in result.output
    assert result.exit_code == cli.EXIT_TASK_FAILURE, "unfinished: rerun to carry on"


@pytest.mark.unit
def test_spend_summary_is_plain() -> None:
    assert Spend(known_usd=1.5, calls=10).summary() == "$1.5000 over 10 calls"
