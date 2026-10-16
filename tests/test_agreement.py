"""How often the machine's marks and a person's agree.

Two numbers, both checked here against values worked out by hand:

- **Raw agreement:** the share of trials where the two gave the same label.
- **Cohen's kappa:** the same, after discounting the agreement two markers
  would reach by chance alone. Two markers who both say "pass" to everything
  agree 100% of the time and have measured nothing; kappa says so.

Each is reported twice: for the pass-or-fail verdict, and for the kind of
failure. A sample smaller than the configured minimum is reported with its
size and a warning, never hidden.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from test_metrics import make_sweep
from typer.testing import CliRunner

from web_action_agent import cli
from web_action_agent.agreement import Agreement, kappa_band, measure
from web_action_agent.metrics import (
    COMPARISON_FILE,
    HUMAN_LABELS_FILE,
    HumanVerdict,
    judge_human_agreement,
    label_pairs,
    load_sweep,
    read_human_labels,
    write_results,
)

pytestmark = pytest.mark.unit

#: The fixed verdict set: (machine, human) -> how many trials.
#:
#: Worked by hand. 20 trials; 12 + 5 = 17 agree, so raw = 17/20 = 85%.
#: The machine said pass 14 times and fail 6; the human pass 13 and fail 7.
#: By chance: (14/20 x 13/20) + (6/20 x 7/20) = (182 + 42) / 400 = 0.56.
#: Kappa = (0.85 - 0.56) / (1 - 0.56) = 0.29 / 0.44 = 0.659...
VERDICTS = {("pass", "pass"): 12, ("pass", "fail"): 2, ("fail", "pass"): 1, ("fail", "fail"): 5}

#: The fixed kind-of-failure set, for the 5 trials both called a fail.
#:
#: Worked by hand. 4 of 5 agree, so raw = 80%. The machine said grounding 2,
#: navigation 2, planning 1; the human grounding 1, navigation 2, planning 2.
#: By chance: (2x1 + 2x2 + 1x2) / 25 = 8/25 = 0.32.
#: Kappa = (0.80 - 0.32) / (1 - 0.32) = 0.48 / 0.68 = 0.705...
KINDS = [
    ("grounding", "grounding"),
    ("grounding", "planning"),
    ("navigation", "navigation"),
    ("navigation", "navigation"),
    ("planning", "planning"),
]


def expand(counts: dict[tuple[str, str], int]) -> list[tuple[str, str]]:
    """Turn pair counts into a list of pairs.

    Args:
        counts: Each (machine, human) pair and how many trials had it.

    Returns:
        list[tuple[str, str]]: One pair per trial.
    """
    return [pair for pair, times in counts.items() for _ in range(times)]


# --- The two numbers, against hand-worked values ----------------------------------


def test_verdict_agreement_matches_the_hand_worked_values() -> None:
    result = measure(expand(VERDICTS))

    assert (result.agreed, result.n) == (17, 20)
    assert result.raw == pytest.approx(0.85)
    assert result.chance == pytest.approx(0.56)
    assert result.kappa == pytest.approx(0.29 / 0.44)
    assert result.render() == "85.0% (17 of 20 trials) · Cohen's κ 0.66 (substantial)"


def test_kind_agreement_matches_the_hand_worked_values() -> None:
    result = measure(KINDS, minimum=5)

    assert (result.agreed, result.n) == (4, 5)
    assert result.chance == pytest.approx(0.32)
    assert result.kappa == pytest.approx(0.48 / 0.68)
    assert result.render() == "80.0% (4 of 5 trials) · Cohen's κ 0.71 (substantial)"


def test_kappa_is_the_same_whichever_marker_is_named_first() -> None:
    swapped = [(human, machine) for machine, human in expand(VERDICTS)]

    assert measure(swapped).kappa == pytest.approx(measure(expand(VERDICTS)).kappa)


# --- The cases that show why kappa exists -----------------------------------------


def test_a_judge_that_passes_everything_scores_well_raw_and_zero_kappa() -> None:
    # The human passes 90 and fails 10; the judge says "pass" to all 100.
    result = measure(expand({("pass", "pass"): 90, ("pass", "fail"): 10}))

    assert result.raw == pytest.approx(0.90)
    # By chance: 100/100 x 90/100 + 0/100 x 10/100 = 0.90 -- all of it luck.
    assert result.chance == pytest.approx(0.90)
    assert result.kappa == pytest.approx(0.0)


def test_ninety_percent_from_both_sides_is_only_moderate() -> None:
    # Both say pass 90 times in 100, and they agree on 90: as in the docstring.
    result = measure(
        expand(
            {("pass", "pass"): 85, ("pass", "fail"): 5, ("fail", "pass"): 5, ("fail", "fail"): 5}
        )
    )

    assert result.raw == pytest.approx(0.90)
    assert result.chance == pytest.approx(0.82)
    assert result.kappa == pytest.approx(0.08 / 0.18)
    assert kappa_band(result.kappa) == "moderate"


def test_both_saying_pass_to_everything_agrees_fully_and_measures_nothing() -> None:
    # The degenerate all-agree case: kappa is 0 / 0, so it is undefined.
    result = measure([("pass", "pass")] * 20)

    assert result.raw == 1.0
    assert result.chance == 1.0
    assert result.kappa is None
    assert "'pass'" in result.kappa_note
    assert result.render() == (
        "100.0% (20 of 20 trials) · Cohen's κ undefined: every trial was labelled "
        "'pass' by both sides, so chance alone would agree every time"
    )


def test_full_agreement_across_both_labels_is_a_kappa_of_one() -> None:
    result = measure(expand({("pass", "pass"): 10, ("fail", "fail"): 10}))

    assert result.raw == 1.0
    assert result.kappa == pytest.approx(1.0)
    assert kappa_band(result.kappa) == "almost perfect"


def test_always_disagreeing_is_worse_than_chance() -> None:
    result = measure([("pass", "fail"), ("fail", "pass")])

    assert result.kappa == pytest.approx(-1.0)
    assert kappa_band(result.kappa) == "worse than chance"


def test_opposite_single_labels_agree_never_and_score_zero() -> None:
    # Chance agreement is 0 here, so kappa is 0 / 1 = 0, not undefined.
    result = measure([("pass", "fail")] * 4)

    assert result.raw == 0.0
    assert result.kappa == pytest.approx(0.0)


@pytest.mark.parametrize(
    ("kappa", "band"),
    [
        (1.0, "almost perfect"),
        (0.81, "almost perfect"),
        (0.80, "substantial"),
        (0.61, "substantial"),
        (0.60, "moderate"),
        (0.41, "moderate"),
        (0.40, "fair"),
        (0.21, "fair"),
        (0.20, "slight"),
        (0.0, "slight"),
        (-0.01, "worse than chance"),
    ],
)
def test_kappa_is_named_on_the_usual_scale(kappa: float, band: str) -> None:
    assert kappa_band(kappa) == band


def test_the_band_matches_the_rounded_figure_shown() -> None:
    # 0.6099 is printed as 0.61, so it must be named as 0.61 is.
    assert kappa_band(0.6099) == "substantial"


# --- Small samples are shown, with their size --------------------------------------


def test_a_sample_below_the_minimum_is_reported_with_its_n() -> None:
    text = measure(KINDS, minimum=20).render()

    # The numbers are all still there...
    assert text.startswith("80.0% (4 of 5 trials) · Cohen's κ 0.71 (substantial)")
    # ...followed by how few trials they rest on.
    assert "only 5 labelled, fewer than the 20 needed to rely on this" in text


def test_a_sample_at_the_minimum_carries_no_warning() -> None:
    result = measure(expand(VERDICTS), minimum=20)

    assert result.enough
    assert "only" not in result.render()


def test_no_labels_is_not_yet_measured() -> None:
    result = measure([])

    assert (result.n, result.raw, result.kappa) == (0, None, None)
    assert result.render() == "not yet measured (no human labels)"
    assert result.render("nothing to compare") == "nothing to compare"


def test_an_agreement_is_a_plain_frozen_record() -> None:
    result = measure(KINDS)

    with pytest.raises(AttributeError):
        result.n = 99  # type: ignore[misc]
    assert isinstance(result, Agreement)


# --- From the labels file to the tables ---------------------------------------------


def write_labels(path: Path, sweep: Path, labels: dict[tuple[str, int], tuple]) -> None:
    """Write a labels file the way the review tool does.

    Args:
        path: The labels file.
        sweep: The sweep the trials are in.
        labels: (task, trial) -> (passed, category).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as out:
        for (task, trial), (passed, category) in labels.items():
            record = {"run": str(sweep / task / f"trial-{trial}"), "passed": passed}
            record["category"] = category
            out.write(json.dumps(record) + "\n")


#: Labels for the made-up sweep in ``test_metrics``. Its marks are noted on
#: the right. Worked by hand:
#:
#: Verdicts (delta-3 is unmarked, so left out): 4 trials, 3 agree.
#: Machine pass 1, fail 3; human pass 2, fail 2.
#: By chance: (1x2 + 3x2) / 16 = 0.5. Kappa = (0.75 - 0.5) / 0.5 = 0.50.
#:
#: Kinds, over the 2 trials both called a fail: 1 agrees. Machine grounding
#: 1, planning 1; human grounding 1, navigation 1. By chance: 1x1 / 4 = 0.25.
#: Kappa = (0.5 - 0.25) / 0.75 = 0.33.
SWEEP_LABELS = {
    ("alpha", 1): (True, None),  # machine: pass
    ("bravo", 2): (False, "grounding"),  # machine: fail, grounding
    ("charlie", 1): (False, "navigation"),  # machine: fail, planning
    ("charlie", 2): (True, None),  # machine: fail, navigation
    ("delta", 3): (True, None),  # machine: judging_error (unmarked)
}


def test_the_labels_file_gives_verdict_and_kind(tmp_path: Path) -> None:
    sweep = make_sweep(tmp_path / "runs")
    path = tmp_path / HUMAN_LABELS_FILE
    write_labels(path, sweep, SWEEP_LABELS)

    labels = read_human_labels(path)

    assert labels[str((sweep / "bravo" / "trial-2").resolve())] == HumanVerdict(False, "grounding")
    assert labels[str((sweep / "alpha" / "trial-1").resolve())] == HumanVerdict(True, None)


def test_a_verdict_that_is_not_true_or_false_is_ignored(tmp_path: Path) -> None:
    path = tmp_path / HUMAN_LABELS_FILE
    path.write_text(json.dumps({"run": "x", "passed": "false"}) + "\n")

    assert read_human_labels(path) == {}


def test_kinds_are_compared_only_where_both_sides_said_fail(tmp_path: Path) -> None:
    sweep = make_sweep(tmp_path / "runs")
    path = tmp_path / HUMAN_LABELS_FILE
    write_labels(path, sweep, SWEEP_LABELS)

    verdicts, kinds = label_pairs(load_sweep(sweep).rows, read_human_labels(path))

    assert sorted(verdicts) == [
        ("fail", "fail"),
        ("fail", "fail"),
        ("fail", "pass"),
        ("pass", "pass"),
    ]
    assert sorted(kinds) == [("grounding", "grounding"), ("planning", "navigation")]


def test_the_sweep_agreement_matches_the_hand_worked_values(tmp_path: Path) -> None:
    sweep = make_sweep(tmp_path / "runs")
    path = tmp_path / HUMAN_LABELS_FILE
    write_labels(path, sweep, SWEEP_LABELS)

    verdict, kind = judge_human_agreement(load_sweep(sweep).rows, read_human_labels(path))

    assert (verdict.agreed, verdict.n) == (3, 4)
    assert verdict.kappa == pytest.approx(0.5)
    assert (kind.agreed, kind.n) == (1, 2)
    assert kind.kappa == pytest.approx(0.25 / 0.75)


def test_each_sweep_table_shows_both_agreements_with_their_n(tmp_path: Path) -> None:
    sweep = make_sweep(tmp_path / "runs")
    results = tmp_path / "results"
    write_labels(results / HUMAN_LABELS_FILE, sweep, SWEEP_LABELS)

    [table, _] = write_results([sweep], results, min_labels=20)
    text = table.read_text()

    assert (
        "| Judge–human agreement: pass or fail | 75.0% (3 of 4 trials) · Cohen's κ 0.50 "
        "(moderate) · only 4 labelled, fewer than the 20 needed to rely on this: "
        "treat as a rough guide |"
    ) in text
    assert (
        "| Judge–human agreement: kind of failure (trials both called a fail) | "
        "50.0% (1 of 2 trials) · Cohen's κ 0.33 (fair) · only 2 labelled"
    ) in text
    assert "Cohen's κ discounts the agreement" in text


def test_the_minimum_comes_from_the_caller(tmp_path: Path) -> None:
    sweep = make_sweep(tmp_path / "runs")
    results = tmp_path / "results"
    write_labels(results / HUMAN_LABELS_FILE, sweep, SWEEP_LABELS)

    [table, _] = write_results([sweep], results, min_labels=4)

    pass_fail = next(
        line for line in table.read_text().splitlines() if "agreement: pass or fail" in line
    )
    assert "only" not in pass_fail


def test_labels_but_no_shared_fail_says_so(tmp_path: Path) -> None:
    sweep = make_sweep(tmp_path / "runs")
    results = tmp_path / "results"
    write_labels(results / HUMAN_LABELS_FILE, sweep, {("alpha", 1): (True, None)})

    [table, _] = write_results([sweep], results)

    assert "no trial was called a fail by both sides yet" in table.read_text()


def test_no_labels_says_not_yet_measured_on_both_lines(tmp_path: Path) -> None:
    [table, comparison] = write_results([make_sweep(tmp_path / "runs")], tmp_path / "results")

    for path in (table, comparison):
        assert path.read_text().count("not yet measured (no human labels)") == 2


def test_the_comparison_pools_the_labels_of_every_sweep(tmp_path: Path) -> None:
    cold = make_sweep(tmp_path / "runs", "eval-openrouter-cold")
    warm = make_sweep(tmp_path / "runs", "eval-openrouter-warm", library="warm")
    results = tmp_path / "results"
    write_labels(results / HUMAN_LABELS_FILE, cold, SWEEP_LABELS)
    # Two more in the warm sweep: both agree.
    write_labels(
        results / HUMAN_LABELS_FILE,
        warm,
        {("alpha", 1): (True, None), ("bravo", 2): (False, "grounding")},
    )

    written = write_results([cold, warm], results)
    comparison = (results / COMPARISON_FILE).read_text()

    assert written[-1] == results / COMPARISON_FILE
    assert "## Judge–human agreement" in comparison
    # 4 + 2 labelled trials, 3 + 2 agreeing: only the pooled figure has all six.
    assert "| Judge–human agreement: pass or fail | 83.3% (5 of 6 trials)" in comparison
    assert "(3 of 4 trials)" in (results / "eval-openrouter-cold.md").read_text()
    assert "(2 of 2 trials)" in (results / "eval-openrouter-warm.md").read_text()


# --- The configured minimum reaches the tables ----------------------------------------


@pytest.mark.parametrize(("setting", "warned"), [(None, True), ("4", False)])
def test_waa_eval_uses_the_configured_minimum(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, setting: str | None, warned: bool
) -> None:
    runs = tmp_path / "runs"
    sweep = make_sweep(runs)
    results = tmp_path / "results"
    write_labels(results / HUMAN_LABELS_FILE, sweep, SWEEP_LABELS)

    class Finished:
        """Stands in for the harness: the sweep is already on disk."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def run(self) -> Any:
            return SimpleNamespace(
                results=(), summary=lambda: "12 trials", folder=sweep, stopped_by="", not_started=0
            )

    env = {"WAA_AGREEMENT_MIN_LABELS": setting} if setting else {}
    monkeypatch.setattr(cli, "Harness", Finished)
    monkeypatch.setattr(cli, "resolve_env", lambda: env)
    monkeypatch.setattr(cli, "RUNS_ROOT", runs)
    monkeypatch.setattr(cli, "RESULTS_ROOT", results)

    CliRunner().invoke(cli.app, ["eval"])

    # 4 labelled trials: below the default of 20, but not below a minimum of 4.
    table = (results / f"{sweep.name}.md").read_text()
    pass_fail = next(line for line in table.splitlines() if "agreement: pass or fail" in line)
    assert ("only 4 labelled" in pass_fail) is warned
