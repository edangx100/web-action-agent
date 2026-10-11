"""The results tables: the numbers in them, and the links out of them.

Most of these tests run over one small, made-up sweep whose figures can be
worked out by hand, and are worked out by hand in the comments, so that a
reader can check the arithmetic without trusting the code:

    task      round 1   round 2   round 3
    alpha     pass      pass      pass
    bravo     pass      fail      pass
    charlie   fail      fail      pass
    delta     pass      pass      (judging error: not marked)

    round 1: 3 of 4 = 75%    round 2: 2 of 4 = 50%    round 3: 3 of 3 = 100%
    mean = (75 + 50 + 100) / 3 = 75%, range 50%–100%

Delta's unmarked third trial is left out of round 3, not counted as a fail:
an unfinished mark must neither help nor hurt the score.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from typer.testing import CliRunner

from web_action_agent import cli
from web_action_agent.budget import JUDGE_LOG
from web_action_agent.eval_harness import SWEEP_FILE, VERDICT_FILE
from web_action_agent.metrics import (
    COMPARISON_FILE,
    HUMAN_LABELS_FILE,
    Rate,
    TrialRow,
    averages,
    find_sweeps,
    load_sweep,
    raw_agreement,
    read_human_labels,
    render_cost,
    success_rate,
    write_results,
)
from web_action_agent.trace import OUTCOME_FILE

pytestmark = pytest.mark.unit

#: The made-up sweep: (task, round) -> (verdict, category, steps).
GRID: dict[tuple[str, int], tuple[str, str | None, int]] = {
    ("alpha", 1): ("pass", None, 4),
    ("alpha", 2): ("pass", None, 4),
    ("alpha", 3): ("pass", None, 4),
    ("bravo", 1): ("pass", None, 6),
    ("bravo", 2): ("fail", "grounding", 10),
    ("bravo", 3): ("pass", None, 6),
    ("charlie", 1): ("fail", "planning", 8),
    ("charlie", 2): ("fail", "navigation", 12),
    ("charlie", 3): ("pass", None, 7),
    ("delta", 1): ("pass", None, 5),
    ("delta", 2): ("pass", None, 5),
    ("delta", 3): ("judging_error", None, 5),
}


def make_sweep(
    root: Path,
    name: str = "eval-openrouter-cold",
    *,
    priced: bool = True,
    library: str = "cold",
) -> Path:
    """Write the made-up sweep to disk, in the harness's own layout.

    Every trial took 20 seconds and used 1,000 tokens. A priced sweep's
    trials each cost $0.002; an unpriced one's have no dollar figure and 3
    unpriced calls each.

    Args:
        root: Where the sweep folder goes.
        name: The sweep folder's name.
        priced: False for an arm whose model has no price.
        library: cold or warm.

    Returns:
        Path: The sweep folder.
    """
    sweep = root / name
    sweep.mkdir(parents=True)
    (sweep / SWEEP_FILE).write_text(
        json.dumps(
            {
                "suite": "sg-gov-v1",
                "profile": "openrouter",
                "model": "nvidia/nemotron-3-nano-30b-a3b",
                "library": library,
                "judge_profile": "openrouter-judge",
                "started_at": "2026-10-12T01:00:00+00:00",
            }
        )
    )
    for (task, number), (status, category, steps) in GRID.items():
        folder = sweep / task / f"trial-{number}"
        folder.mkdir(parents=True)
        (folder / OUTCOME_FILE).write_text(
            json.dumps(
                {
                    "started_at": "2026-10-12T01:00:00Z",
                    "ended_at": "2026-10-12T01:00:20Z",
                    "steps": steps,
                    "cost": {
                        "total_tokens": 1000,
                        "cost_usd": 0.002 if priced else None,
                        "unpriced_calls": 0 if priced else 3,
                    },
                }
            )
        )
        (folder / VERDICT_FILE).write_text(
            json.dumps(
                {
                    "status": status,
                    "reason": f"{task} round {number}",
                    "category": category,
                    "marked_at": f"2026-10-12T02:00:{number:02d}+00:00",
                }
            )
        )
    return sweep


# --- The arithmetic ---------------------------------------------------------------


def test_success_rate_is_the_hand_computed_mean_and_range(tmp_path: Path) -> None:
    rate = success_rate(load_sweep(make_sweep(tmp_path)).rows)
    assert rate.mean == pytest.approx(0.75)  # (0.75 + 0.50 + 1.00) / 3
    assert (rate.low, rate.high) == (pytest.approx(0.50), pytest.approx(1.00))
    assert rate.rounds == 3
    assert (rate.passed, rate.marked) == (8, 11)  # delta's third is not marked
    assert rate.render() == "75.0% (range 50.0%–100.0% over 3 rounds)"


def test_an_unmarked_trial_neither_helps_nor_hurts(tmp_path: Path) -> None:
    # Counting delta's unmarked third trial as a fail would give round 3 a
    # rate of 3/4 and the mean 66.7%; as a pass, 4/4 and 75%. It must be
    # left out entirely: 3/3.
    rows = load_sweep(make_sweep(tmp_path)).rows
    round_three = [r for r in rows if r.trial == 3 and r.marked]
    assert len(round_three) == 3


def test_a_single_round_says_it_has_no_range() -> None:
    rate = Rate(mean=0.5, low=0.5, high=0.5, rounds=1, passed=1, marked=2)
    assert "1 round only" in rate.render()


def test_averages_are_the_hand_computed_ones(tmp_path: Path) -> None:
    marked = load_sweep(make_sweep(tmp_path)).marked
    avg = averages(marked)
    # Steps of the 11 marked trials: 4+4+4 + 6+10+6 + 8+12+7 + 5+5 = 71.
    assert avg.steps == pytest.approx(71 / 11)
    assert avg.tokens == pytest.approx(1000)
    assert avg.wall_s == pytest.approx(20.0)
    assert avg.cost_usd == pytest.approx(0.002)


def test_one_unpriced_trial_makes_the_average_cost_unknown() -> None:
    def row(cost: float | None) -> TrialRow:
        return TrialRow(
            "t", 1, Path("x"), "pass", None, "", 3, 100, 1.0, cost, 0 if cost else 2, ""
        )

    avg = averages([row(0.01), row(None)])
    assert avg.cost_usd is None, "an average that skips some calls is not the cost"
    assert avg.tokens == pytest.approx(100), "tokens are still known"


def test_cost_renders_as_unknown_rather_than_inventing_a_figure() -> None:
    assert render_cost(None, 6) == "unknown (6 calls unpriced)"
    assert render_cost(0.00213, 0) == "$0.0021"


# --- The tables ---------------------------------------------------------------------


def test_a_null_priced_arm_shows_tokens_and_time_without_a_dollar_figure(tmp_path: Path) -> None:
    sweep = make_sweep(tmp_path / "runs", "eval-nvidia-cold", priced=False)
    [table, _] = write_results([sweep], tmp_path / "results")
    text = table.read_text()
    summary = text.split("## Per task")[0]
    assert "| Average tokens per trial | 1,000 |" in summary
    assert "| Average time per trial | 20.0 s |" in summary
    assert "| Average cost per trial (agent) | unknown (33 calls unpriced) |" in summary
    per_task = text.split("## Per task")[1].split("## Failures")[0]
    assert "$" not in per_task, "no dollar figure anywhere for an unpriced arm"


def test_every_verdict_row_links_to_an_existing_run_folder(tmp_path: Path) -> None:
    sweep = make_sweep(tmp_path / "runs")
    results = tmp_path / "results"
    [table, _] = write_results([sweep], results)

    verdicts = table.read_text().split("## Every verdict")[1]
    links = re.findall(r"\[run\]\(([^)]+)\)", verdicts)
    assert len(links) == len(GRID), "one link per trial"
    for link in links:
        target = (results / link).resolve()
        assert target.is_dir(), f"{link} does not resolve to a folder"
        assert (target / OUTCOME_FILE).exists(), f"{link} is not a run folder"


def test_the_table_is_stamped_with_when_the_sweep_ran(tmp_path: Path) -> None:
    [table, _] = write_results([make_sweep(tmp_path / "runs")], tmp_path / "results")
    text = table.read_text()
    assert "sweep started 2026-10-12T01:00:00+00:00" in text
    assert "last trial marked 2026-10-12T02:00:03+00:00" in text


def test_failures_are_counted_by_kind(tmp_path: Path) -> None:
    [table, _] = write_results([make_sweep(tmp_path / "runs")], tmp_path / "results")
    kinds = table.read_text().split("## Failures by kind")[1].split("## Every verdict")[0]
    # Three fails: one grounding, one planning, one navigation.
    assert "| 1 | 1 | 1 | 0 |" in kinds


def test_the_per_task_table_shows_passes_out_of_marked(tmp_path: Path) -> None:
    [table, _] = write_results([make_sweep(tmp_path / "runs")], tmp_path / "results")
    per_task = table.read_text().split("## Per task")[1].split("## Failures")[0]
    assert "| charlie | 1/3 |" in per_task
    assert "| delta | 2/2 |" in per_task, "the unmarked trial is not in the count"


def test_the_comparison_has_a_row_per_sweep_linking_to_its_table(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    cold = make_sweep(runs, "eval-openrouter-cold")
    warm = make_sweep(runs, "eval-openrouter-warm", library="warm")
    results = tmp_path / "results"

    written = write_results([cold, warm], results)

    comparison = (results / COMPARISON_FILE).read_text()
    assert written[-1] == results / COMPARISON_FILE
    for name in ("eval-openrouter-cold", "eval-openrouter-warm"):
        assert f"[{name}]({name}.md)" in comparison
        assert (results / f"{name}.md").exists()
    assert "| warm |" in comparison and "| cold |" in comparison


def test_a_reason_with_a_bar_does_not_break_the_table(tmp_path: Path) -> None:
    sweep = make_sweep(tmp_path / "runs")
    verdict = sweep / "alpha" / "trial-1" / VERDICT_FILE
    data = json.loads(verdict.read_text())
    data["reason"] = "answer 'a | b'\nmatched"
    verdict.write_text(json.dumps(data))
    [table, _] = write_results([sweep], tmp_path / "results")
    assert "answer 'a \\| b' matched" in table.read_text()


def test_killed_runs_are_never_read(tmp_path: Path) -> None:
    sweep = make_sweep(tmp_path / "runs")
    killed = sweep / "_interrupted" / "alpha" / "trial-1-1"
    killed.mkdir(parents=True)
    (killed / VERDICT_FILE).write_text(json.dumps({"status": "pass"}))
    assert len(load_sweep(sweep).rows) == len(GRID)


def test_the_judges_cost_is_shown_apart_from_the_agents(tmp_path: Path) -> None:
    sweep = make_sweep(tmp_path / "runs")
    record = {
        "call_id": "c",
        "started_at": "a",
        "ended_at": "b",
        "profile": "p",
        "model": "m",
        "vision": True,
        "prompt_tokens": 1,
        "completion_tokens": 1,
        "latency_s": 0.1,
        "finish_reason": "stop",
        "cost_usd": 0.01,
    }
    (sweep / JUDGE_LOG).write_text((json.dumps(record) + "\n") * 3)
    [table, _] = write_results([sweep], tmp_path / "results")
    assert "| Cost of judging (not the agent) | $0.0300 over 3 calls |" in table.read_text()


def test_a_sweep_with_no_judge_calls_says_so_rather_than_zero(tmp_path: Path) -> None:
    [table, _] = write_results([make_sweep(tmp_path / "runs")], tmp_path / "results")
    assert "| Cost of judging (not the agent) | none: no judge calls |" in table.read_text()


def test_find_sweeps_finds_only_folders_with_a_sweep_file(tmp_path: Path) -> None:
    make_sweep(tmp_path, "eval-a")
    (tmp_path / "2026-10-12T01-00-00Z").mkdir()  # an ordinary `waa run` folder
    assert [p.name for p in find_sweeps(tmp_path)] == ["eval-a"]


# --- Agreement with humans ---------------------------------------------------------


def test_agreement_is_counted_over_labelled_marked_trials(tmp_path: Path) -> None:
    sweep = make_sweep(tmp_path / "runs")
    labels_file = tmp_path / HUMAN_LABELS_FILE
    labels_file.write_text(
        "\n".join(
            json.dumps({"run": str(sweep / task / f"trial-{n}"), "passed": passed})
            for task, n, passed in [
                ("alpha", 1, True),  # verdict pass: agree
                ("bravo", 2, True),  # verdict fail: disagree
                ("charlie", 1, False),  # verdict fail: agree
                ("delta", 3, True),  # not marked: not counted
            ]
        )
    )
    rows = load_sweep(sweep).rows
    assert raw_agreement(rows, read_human_labels(labels_file)) == (2, 3)


def test_agreement_is_shown_as_not_measured_until_there_are_labels(tmp_path: Path) -> None:
    [table, _] = write_results([make_sweep(tmp_path / "runs")], tmp_path / "results")
    assert "not yet measured (no human labels)" in table.read_text()


def test_agreement_appears_in_the_table_once_labelled(tmp_path: Path) -> None:
    sweep = make_sweep(tmp_path / "runs")
    results = tmp_path / "results"
    results.mkdir()
    (results / HUMAN_LABELS_FILE).write_text(
        json.dumps({"run": str(sweep / "alpha" / "trial-1"), "passed": True}) + "\n"
    )
    [table, _] = write_results([sweep], results)
    assert "| Judge–human agreement | 100.0% (1 of 1 labelled trials) |" in table.read_text()


# --- The command -------------------------------------------------------------------


def test_waa_eval_writes_the_tables_after_the_sweep(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    runs = tmp_path / "runs"
    sweep = make_sweep(runs)

    class Finished:
        """Stands in for the harness: the sweep is already on disk."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def run(self) -> Any:
            return SimpleNamespace(
                results=(), summary=lambda: "12 trials", folder=sweep, stopped_by="", not_started=0
            )

    monkeypatch.setattr(cli, "Harness", Finished)
    monkeypatch.setattr(cli, "resolve_env", dict)
    monkeypatch.setattr(cli, "RUNS_ROOT", runs)
    monkeypatch.setattr(cli, "RESULTS_ROOT", tmp_path / "results")

    result = CliRunner().invoke(cli.app, ["eval"])

    assert "tables:" in result.output, result.output
    assert (tmp_path / "results" / f"{sweep.name}.md").exists()
    assert (tmp_path / "results" / COMPARISON_FILE).exists()
