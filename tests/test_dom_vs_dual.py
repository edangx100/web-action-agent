"""DOM only versus DOM plus screenshots: what the screenshot channel buys.

The agent reads each page as text and adds a screenshot only when a cheap rule
says the text may be missing something. This comparison runs the benchmark
twice on the same planner model -- once with the screenshot channel removed,
once gated as usual -- and publishes the difference. These tests hold the
parts that keep it honest:

- **The text-only arm can send no image.** Its profile has no screenshot
  model, so the gate never asks for one and the client refuses one anyway.
- **What went to the model is what is counted.** Images are counted from each
  call's own record, worked out from the request that was sent.
- **The published table** shows each arm's success, cost and images, the
  change between them, and checks that only the screenshot channel changed.
- **The driver script** runs one arm per profile into its own folder, and
  refuses to start on uncommitted code.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any

import pytest
from test_eval_harness import FakeRunner, fake_harness
from test_metrics import GRID, make_sweep

from web_action_agent.config import Config
from web_action_agent.cost import shipped_prices
from web_action_agent.eval_harness import SWEEP_FILE, SweepSpec
from web_action_agent.judge import family_clash
from web_action_agent.metrics import (
    COLD_VS_WARM_FILE,
    DOM_VS_DUAL_FILE,
    ImageCount,
    cold_warm_pairs,
    count_images,
    dom_vs_dual_table,
    is_dom_only,
    load_sweep,
    modality_arms,
    write_results,
)
from web_action_agent.model_client import ModelClientError, load_profiles
from web_action_agent.screenshot_gate import ScreenshotGate
from web_action_agent.trace import COST_LOG

REPO = Path(__file__).parent.parent

#: The driver script under test.
SCRIPT = REPO / "scripts" / "dom_vs_dual.sh"

#: A made-up commit, as ``git rev-parse --short=12`` prints one.
COMMIT = "0123456789ab"


# --- The text-only arm can send no image -------------------------------------------------


@pytest.mark.unit
def test_the_text_only_profile_is_the_primary_one_without_its_screenshot_model() -> None:
    profiles = load_profiles()
    primary, text_only = profiles["openrouter"], profiles["openrouter-text"]

    # The same service, key, planner model and pinned host...
    assert (text_only.base_url, text_only.api_key_env) == (primary.base_url, primary.api_key_env)
    assert text_only.text == primary.text
    # ...and no screenshot model, which is the only difference.
    assert primary.vision is not None
    assert text_only.vision is None


@pytest.mark.unit
def test_the_text_only_profile_is_priced_like_the_primary_one() -> None:
    profile = load_profiles()["openrouter-text"]
    prices = shipped_prices()
    text_only = prices.price_for("openrouter-text", profile.model)
    primary = prices.price_for("openrouter", profile.model)

    assert text_only is not None and primary is not None
    assert (text_only.input_usd_per_mtok, text_only.output_usd_per_mtok) == (
        primary.input_usd_per_mtok,
        primary.output_usd_per_mtok,
    )


@pytest.mark.unit
def test_the_text_only_profile_is_not_in_the_judges_family() -> None:
    profiles = load_profiles()

    assert family_clash(profiles["openrouter-text"], (profiles["openrouter-judge"].model,)) is None


@pytest.mark.unit
def test_the_gate_never_asks_for_a_screenshot_on_the_text_only_profile() -> None:
    # Thresholds set to send one on every step, on a page with no elements:
    # the missing screenshot model still wins.
    profile = load_profiles()["openrouter-text"]
    gate = ScreenshotGate(
        min_elements=1000, max_dom_failures=0, supports_vision=profile.supports_vision
    )

    assert not gate.decide(element_count=0).send_screenshot


@pytest.mark.unit
def test_the_client_refuses_an_image_for_the_text_only_profile() -> None:
    # The second guard: even a request that reached the client with an image
    # in it would not go out.
    with pytest.raises(ModelClientError, match="no vision route"):
        load_profiles()["openrouter-text"].route_for(vision=True)


@pytest.mark.unit
def test_a_text_only_sweep_records_that_it_had_no_screenshot_model(tmp_path: Path) -> None:
    sweep = tmp_path / "sweep"
    fake_harness(sweep, FakeRunner(), trials=1, config=Config(profile="openrouter-text")).run()

    spec = SweepSpec.read(sweep / SWEEP_FILE)

    assert spec.vision_model == ""
    assert is_dom_only(json.loads((sweep / SWEEP_FILE).read_text()))


@pytest.mark.unit
@pytest.mark.parametrize(
    ("vision_model", "expected"),
    [("", True), ("nvidia/nemotron-3.5-super-vl", False), (None, False)],
)
def test_only_an_empty_screenshot_model_means_dom_only(
    vision_model: str | None, expected: bool
) -> None:
    # None is a sweep from before the screenshot model was recorded: it may
    # well have sent images, so it is not counted as text-only.
    spec: dict[str, Any] = {} if vision_model is None else {"vision_model": vision_model}

    assert is_dom_only(spec) is expected


# --- Counting the images that were sent --------------------------------------------------


def a_call(number: int, *, vision: bool) -> dict[str, Any]:
    """One call record, as a trial's call log stores it."""
    return {
        "call_id": f"call-{number}",
        "started_at": "2026-10-20T01:00:00.000Z",
        "ended_at": "2026-10-20T01:00:01.000Z",
        "profile": "openrouter",
        "model": "nvidia/nemotron-3.5-super-vl" if vision else "nvidia/nemotron-3-nano-30b-a3b",
        "vision": vision,
        "prompt_tokens": 1000,
        "completion_tokens": 100,
        "latency_s": 1.0,
        "finish_reason": "stop",
    }


def write_calls(trial: Path, *, images: int, texts: int = 2) -> None:
    """Write a trial's call log: some calls with an image, some without."""
    calls = [a_call(n, vision=True) for n in range(images)]
    calls += [a_call(images + n, vision=False) for n in range(texts)]
    (trial / COST_LOG).write_text("".join(json.dumps(c) + "\n" for c in calls))


def make_arm(root: Path, profile: str, *, dom_only: bool, images: int = 0, **changes: Any) -> Path:
    """Write a made-up arm: the made-up sweep, with a call log in every trial.

    Args:
        root: Where the sweep folder goes.
        profile: The arm's profile; the folder is ``modality-<profile>``.
        dom_only: True for an arm with no screenshot model.
        images: Calls with an image in every trial.
        **changes: Other ``sweep.json`` fields to change.

    Returns:
        Path: The sweep folder.
    """
    sweep = make_sweep(root, f"modality-{profile}")
    spec = json.loads((sweep / SWEEP_FILE).read_text())
    spec.update(
        profile=profile,
        vision_model="" if dom_only else "nvidia/nemotron-3.5-super-vl",
        code_versions=[COMMIT],
    )
    spec.update(changes)
    (sweep / SWEEP_FILE).write_text(json.dumps(spec))
    for task, number in GRID:
        write_calls(sweep / task / f"trial-{number}", images=images)
    return sweep


def both_arms(root: Path, *, dom_images: int = 0) -> list[Path]:
    """Write the two arms: the usual one sends one image a trial."""
    return [
        make_arm(root, "openrouter", dom_only=False, images=1),
        make_arm(root, "openrouter-text", dom_only=True, images=dom_images),
    ]


@pytest.mark.unit
def test_images_are_counted_from_each_calls_own_record(tmp_path: Path) -> None:
    sweep = load_sweep(make_arm(tmp_path / "runs", "openrouter", dom_only=False, images=1))

    # The made-up sweep has 12 trials, each with one image call and two others.
    assert count_images(sweep.rows) == ImageCount(with_image=12, calls=36, trials=12)


@pytest.mark.unit
def test_a_trial_with_no_call_log_counts_as_no_calls(tmp_path: Path) -> None:
    sweep = load_sweep(make_sweep(tmp_path / "runs", "modality-openrouter"))

    assert count_images(sweep.rows) == ImageCount(with_image=0, calls=0, trials=0)


# --- The published table -----------------------------------------------------------------


def table_for(folders: list[Path], results: Path) -> str:
    """Write the DOM-versus-dual table for some sweep folders."""
    return dom_vs_dual_table(modality_arms(load_sweep(f) for f in folders), results)


def row_of(table: str, profile: str) -> str:
    """Find an arm's row in the results table."""
    return next(line for line in table.splitlines() if line.startswith(f"| [{profile}]"))


@pytest.mark.unit
def test_each_row_says_how_the_arm_sees_the_page(tmp_path: Path) -> None:
    table = table_for(both_arms(tmp_path / "runs"), tmp_path / "results")

    assert "| DOM only |" in row_of(table, "openrouter-text")
    assert "| DOM + screenshots when gated in |" in row_of(table, "openrouter")


@pytest.mark.unit
def test_each_row_counts_the_calls_that_carried_an_image(tmp_path: Path) -> None:
    table = table_for(both_arms(tmp_path / "runs"), tmp_path / "results")

    # Over the 11 marked trials: in the usual arm one image call and two text
    # calls each; in the DOM-only arm the two text calls alone.
    assert "| 11 of 33 (in 11 trials) |" in row_of(table, "openrouter")
    assert "| 0 of 22 (in 0 trials) |" in row_of(table, "openrouter-text")


@pytest.mark.unit
def test_each_row_states_its_n_and_models(tmp_path: Path) -> None:
    table = table_for(both_arms(tmp_path / "runs"), tmp_path / "results")

    text_row = row_of(table, "openrouter-text")
    assert "11 (4 tasks × 3)" in text_row
    assert "`nvidia/nemotron-3-nano-30b-a3b`" in text_row
    assert "| none |" in text_row, "no screenshot model, said plainly"


@pytest.mark.unit
def test_the_tradeoff_is_shown_as_the_change_from_dom_only(tmp_path: Path) -> None:
    table = table_for(both_arms(tmp_path / "runs"), tmp_path / "results")

    assert "## What the screenshots bought, and what they cost" in table
    # Both made-up arms are the same sweep, so every change is nil.
    assert "| Success rate |" in table and "+0.0 points" in table
    assert "| DOM only | DOM + screenshots | Change |" in table


@pytest.mark.unit
def test_no_tradeoff_section_without_one_arm_of_each_kind(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    folders = [make_arm(runs, "openrouter", dom_only=False, images=1)]

    table = table_for(folders, tmp_path / "results")

    assert "What the screenshots bought" not in table
    assert "| No image in any request of the DOM-only arm | no DOM-only arm |" in table


@pytest.mark.unit
def test_a_clean_pair_passes_every_check(tmp_path: Path) -> None:
    table = table_for(both_arms(tmp_path / "runs"), tmp_path / "results")

    assert (
        "| No image in any request of the DOM-only arm | yes: 0 of 24 calls carried an image |"
        in table
    )
    assert (
        f"| Same code in every arm | yes: every trial of every arm ran commit `{COMMIT}`" in table
    )
    assert "| Same planner model | yes: nvidia/nemotron-3-nano-30b-a3b |" in table


@pytest.mark.unit
def test_an_image_in_the_dom_only_arm_fails_the_check(tmp_path: Path) -> None:
    table = table_for(both_arms(tmp_path / "runs", dom_images=1), tmp_path / "results")

    assert (
        "| No image in any request of the DOM-only arm | no: 12 of 36 calls carried an image |"
        in table
    )


@pytest.mark.unit
def test_a_different_planner_model_fails_the_check(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    folders = [
        make_arm(runs, "openrouter", dom_only=False, images=1),
        make_arm(runs, "openrouter-text", dom_only=True, model="qwen/qwen3.6-35b-a3b"),
    ]

    table = table_for(folders, tmp_path / "results")

    assert "| Same planner model | no: " in table


@pytest.mark.unit
def test_the_per_task_table_shows_passes_and_images(tmp_path: Path) -> None:
    table = table_for(both_arms(tmp_path / "runs"), tmp_path / "results")

    assert "| Task | openrouter | openrouter-text |" in table
    # bravo: passed 2 of 3; one image a trial in the usual arm, none in the other.
    assert "| bravo | 2/3 (3) | 2/3 (0) |" in table


@pytest.mark.unit
def test_only_modality_folders_are_arms(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    folders = [make_sweep(runs), make_sweep(runs, "modes-openrouter"), *both_arms(runs)]

    arms = modality_arms(load_sweep(f) for f in folders)

    assert [a.name for a in arms] == ["modality-openrouter", "modality-openrouter-text"]


@pytest.mark.unit
def test_an_arm_on_the_primary_model_is_not_paired_with_the_warm_sweep(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    cold = make_sweep(runs, "eval-openrouter-cold")
    warm = make_sweep(runs, "eval-openrouter-warm", library="warm")
    arm = make_sweep(runs, "modality-openrouter")

    pairs = cold_warm_pairs(load_sweep(f) for f in (cold, warm, arm))

    assert [(c.name, w.name) for c, w in pairs] == [
        ("eval-openrouter-cold", "eval-openrouter-warm")
    ]


@pytest.mark.unit
def test_the_file_is_written_beside_the_others(tmp_path: Path) -> None:
    results = tmp_path / "results"

    written = write_results(both_arms(tmp_path / "runs"), results)

    assert results / DOM_VS_DUAL_FILE in written
    assert not (results / COLD_VS_WARM_FILE).exists()


@pytest.mark.unit
def test_no_file_without_an_arm(tmp_path: Path) -> None:
    results = tmp_path / "results"

    write_results([make_sweep(tmp_path / "runs")], results)

    assert not (results / DOM_VS_DUAL_FILE).exists()


# --- The driver script -------------------------------------------------------------------

#: A stand-in for ``waa``: logs its arguments, and exits with FAKE_EXIT_eval,
#: one code per call, space-separated, the last repeating.
FAKE_WAA = """#!/usr/bin/env bash
echo "$*" >> "$FAKE_LOG"
read -r -a codes <<< "${FAKE_EXIT_eval:-0}"
counter="$FAKE_LOG.n"
n=$(cat "$counter" 2>/dev/null || echo 0)
echo $((n + 1)) > "$counter"
last=$((${#codes[@]} - 1))
exit "${codes[$((n < last ? n : last))]}"
"""

#: A stand-in for ``git``: prints FAKE_CHANGES as the uncommitted changes.
FAKE_GIT = """#!/usr/bin/env bash
printf '%s' "${FAKE_CHANGES:-}"
"""


def drive(tmp_path: Path, *args: str, changes: str = "") -> tuple[int, list[str], str]:
    """Run the driver script with stand-ins for ``waa`` and ``git``.

    Args:
        tmp_path: The working directory for the run.
        *args: Arguments to the script.
        changes: What the stand-in git reports as uncommitted changes.

    Returns:
        tuple[int, list[str], str]: The script's exit code, the commands it
        ran in order, and what it printed to stderr.
    """
    for name, text in {"fake-waa": FAKE_WAA, "fake-git": FAKE_GIT}.items():
        (tmp_path / name).write_text(text)
        (tmp_path / name).chmod(0o755)
    log = tmp_path / "calls.log"
    env = {
        **os.environ,
        "WAA": str(tmp_path / "fake-waa"),
        "GIT": str(tmp_path / "fake-git"),
        "FAKE_LOG": str(log),
        "FAKE_CHANGES": changes,
        "RETRY_WAIT_S": "0",
    }
    done = subprocess.run(
        ["bash", str(SCRIPT), *args],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    calls = log.read_text().splitlines() if log.exists() else []
    return done.returncode, calls, done.stderr


@pytest.mark.unit
def test_the_script_runs_both_arms_into_their_own_folders(tmp_path: Path) -> None:
    code, calls, _ = drive(tmp_path)

    assert code == 0
    assert calls == [
        f"eval --cold --trials 2 --profile {p} --out runs/modality-{p}"
        for p in ("openrouter", "openrouter-text")
    ]


@pytest.mark.unit
def test_the_script_passes_profiles_and_trials_through(tmp_path: Path) -> None:
    _, calls, _ = drive(tmp_path, "--profiles", "a", "--trials", "3")

    assert calls == ["eval --cold --trials 3 --profile a --out runs/modality-a"]


@pytest.mark.unit
def test_the_script_refuses_uncommitted_code(tmp_path: Path) -> None:
    code, calls, said = drive(tmp_path, changes=" M src/web_action_agent/profiles.yaml")

    assert code == 2 and calls == []
    assert "commit these changes first" in said


@pytest.mark.unit
def test_the_script_refuses_an_unknown_argument(tmp_path: Path) -> None:
    code, calls, _ = drive(tmp_path, "--no-images")

    assert code == 2 and calls == []


@pytest.mark.unit
def test_the_script_explains_itself() -> None:
    done = subprocess.run(
        ["bash", str(SCRIPT), "--help"], capture_output=True, text=True, check=False
    )

    assert done.returncode == 0
    assert "uv run scripts/dom_vs_dual.sh" in done.stdout
    assert "spends real money" in done.stdout
    assert "Exit codes" in done.stdout and "set -euo" not in done.stdout


@pytest.mark.unit
def test_the_script_is_executable() -> None:
    assert os.access(SCRIPT, os.X_OK)
