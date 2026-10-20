"""The endpoint-modes check: one task, reached through three kinds of service.

The claim under test is that the agent reaches a model through configuration
alone: the same code runs against an aggregator (OpenRouter), the model
maker's own service (build.nvidia.com), and a vLLM server the user runs, with
only ``--profile`` changed. These tests hold the parts that make the claim
checkable:

- **One task.** ``waa eval --task`` narrows a sweep to some tasks, and names
  it for them, so it can never be resumed as a sweep of all of them.
- **The service.** Every sweep records the base URL its calls went to.
- **The published table** names each arm's service and models, shows tokens
  and time always and dollars only where priced, and checks that only the
  endpoint changed.
- **The driver script** runs one arm per profile on one task, each into its
  own folder, and refuses to start on uncommitted code.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any

import pytest
from test_eval_harness import FakeRunner, fake_harness, stand_in_suite
from test_metrics import make_sweep
from typer.testing import CliRunner

from web_action_agent import cli
from web_action_agent.benchmark import BenchmarkError
from web_action_agent.config import Config
from web_action_agent.cost import shipped_prices
from web_action_agent.eval_harness import SWEEP_FILE, HarnessError, SweepSpec
from web_action_agent.metrics import (
    COLD_VS_WARM_FILE,
    ENDPOINT_MODES_FILE,
    cold_warm_pairs,
    endpoint_mode_arms,
    endpoint_modes_table,
    load_sweep,
    write_results,
)
from web_action_agent.model_client import load_profiles

REPO = Path(__file__).parent.parent

#: The driver script under test.
SCRIPT = REPO / "scripts" / "endpoint_modes.sh"

#: A made-up commit, as ``git rev-parse --short=12`` prints one.
COMMIT = "0123456789ab"

#: The three kinds of service, as the shipped profiles reach them.
SERVICES = {
    "openrouter": "https://openrouter.ai/api/v1",
    "nvidia": "https://integrate.api.nvidia.com/v1",
    "local": "http://127.0.0.1:8000/v1",
}


# --- One task ----------------------------------------------------------------------------


@pytest.mark.unit
def test_a_suite_narrowed_to_one_task_keeps_only_that_task() -> None:
    narrowed = stand_in_suite().only(("fixture-search",))

    assert [task.id for task in narrowed.tasks] == ["fixture-search"]


@pytest.mark.unit
def test_a_narrowed_suite_is_named_for_its_tasks() -> None:
    # The name is what a sweep records and checks on a restart.
    assert stand_in_suite().only(("fixture-search",)).name == "fixture-suite[fixture-search]"


@pytest.mark.unit
def test_several_tasks_keep_the_manifests_order() -> None:
    narrowed = stand_in_suite().only(("fixture-shelves", "fixture-home"))

    assert [task.id for task in narrowed.tasks] == ["fixture-home", "fixture-shelves"]
    assert narrowed.name == "fixture-suite[fixture-home,fixture-shelves]"


@pytest.mark.unit
def test_no_tasks_named_means_the_whole_suite() -> None:
    suite = stand_in_suite()

    assert suite.only(()) is suite


@pytest.mark.unit
def test_an_unknown_task_is_refused_by_name() -> None:
    with pytest.raises(BenchmarkError, match="no task named fixture-nowhere"):
        stand_in_suite().only(("fixture-search", "fixture-nowhere"))


@pytest.mark.unit
def test_a_one_task_sweep_is_never_resumed_as_the_whole_suite(tmp_path: Path) -> None:
    sweep = tmp_path / "sweep"
    harness = fake_harness(sweep, FakeRunner(), trials=1)
    harness.suite = stand_in_suite().only(("fixture-search",))
    harness.run()

    with pytest.raises(HarnessError, match="suite"):
        fake_harness(sweep, FakeRunner(), trials=1).run()


@pytest.mark.unit
def test_a_one_task_sweep_runs_only_that_task(tmp_path: Path) -> None:
    runner = FakeRunner()
    harness = fake_harness(tmp_path / "sweep", runner, trials=1)
    harness.suite = stand_in_suite().only(("fixture-search",))

    harness.run()

    assert runner.ran == ["fixture-search/trial-1"]


@pytest.mark.unit
def test_waa_eval_refuses_an_unknown_task_before_starting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    started: list[Any] = []
    monkeypatch.setattr(cli, "Harness", lambda *args, **kwargs: started.append(args))
    monkeypatch.setattr(cli, "resolve_env", dict)

    result = CliRunner().invoke(cli.app, ["eval", "--task", "no-such-task"])

    assert result.exit_code == 2, result.output
    assert "no task named no-such-task" in result.output
    assert started == [], "nothing was started"


@pytest.mark.unit
def test_waa_eval_hands_the_harness_only_the_named_task(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, Any] = {}

    class Captured:
        """Stands in for the harness, and keeps the suite it was given."""

        def __init__(self, config: Any, suite: Any, *args: Any, **kwargs: Any) -> None:
            seen["suite"] = suite

        def run(self) -> Any:
            raise HarnessError("stop here")

    monkeypatch.setattr(cli, "Harness", Captured)
    monkeypatch.setattr(cli, "resolve_env", dict)

    CliRunner().invoke(cli.app, ["eval", "--task", "cpf-employee-share"])

    assert [task.id for task in seen["suite"].tasks] == ["cpf-employee-share"]
    assert seen["suite"].name.endswith("[cpf-employee-share]")


# --- The service --------------------------------------------------------------------------


@pytest.mark.unit
def test_a_sweep_records_the_service_its_calls_went_to(tmp_path: Path) -> None:
    sweep = tmp_path / "sweep"
    fake_harness(sweep, FakeRunner(), trials=1).run()

    spec = SweepSpec.read(sweep / SWEEP_FILE)

    assert spec.base_url == load_profiles()[Config().profile].base_url


def a_spec(**changes: Any) -> SweepSpec:
    """A sweep's description, with some fields changed."""
    fields: dict[str, Any] = {
        "suite": "s",
        "profile": "p",
        "model": "m",
        "library": "cold",
        "judge_profile": "j",
        "vision_model": "v",
        "base_url": SERVICES["openrouter"],
    }
    fields.update(changes)
    return SweepSpec(**fields)


@pytest.mark.unit
def test_a_folder_holding_a_sweep_on_another_service_is_refused() -> None:
    differences = a_spec(base_url=SERVICES["local"]).differences(a_spec())

    assert len(differences) == 1 and differences[0].startswith("base_url")


@pytest.mark.unit
def test_a_sweep_from_before_the_service_was_kept_can_still_resume() -> None:
    assert a_spec().differences(a_spec(base_url=None)) == []


# --- The profiles ---------------------------------------------------------------------------


@pytest.mark.unit
def test_the_three_modes_are_three_different_services() -> None:
    profiles = load_profiles()

    assert {name: profiles[name].base_url for name in SERVICES} == SERVICES


@pytest.mark.unit
def test_the_nvidia_profile_uses_a_nano_model_it_still_serves() -> None:
    # The text-only Nano was retired on build.nvidia.com; its Omni sibling,
    # of the same Nemotron 3 Nano family, serves both routes.
    nvidia = load_profiles()["nvidia"]

    assert nvidia.model == "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning"
    assert nvidia.vision is not None and nvidia.vision.model == nvidia.model


@pytest.mark.unit
@pytest.mark.parametrize("name", ["nvidia", "local"])
def test_free_credits_and_a_self_hosted_server_have_no_price(name: str) -> None:
    # Unknown, never zero: neither is a per-token price.
    profile = load_profiles()[name]
    price = shipped_prices().price_for(name, profile.model)

    assert price is None or price.input_usd_per_mtok is None


# --- The published table --------------------------------------------------------------------


def make_mode(
    root: Path,
    profile: str,
    *,
    priced: bool = True,
    versions: list[str] | None = None,
    **changes: Any,
) -> Path:
    """Write a made-up arm of the check: the made-up sweep, on one service.

    Args:
        root: Where the sweep folder goes.
        profile: The arm's profile; the folder is ``modes-<profile>``.
        priced: False for an arm with no per-token price.
        versions: The code versions it ran. Defaults to one commit.
        **changes: Other ``sweep.json`` fields to change.

    Returns:
        Path: The sweep folder.
    """
    sweep = make_sweep(root, f"modes-{profile}", priced=priced)
    spec = json.loads((sweep / SWEEP_FILE).read_text())
    spec.update(
        profile=profile,
        base_url=SERVICES.get(profile, "https://example.test/v1"),
        vision_model="" if profile == "local" else "some/vision-model",
        code_versions=[COMMIT] if versions is None else versions,
    )
    spec.update(changes)
    (sweep / SWEEP_FILE).write_text(json.dumps(spec))
    return sweep


def three_modes(root: Path) -> list[Path]:
    """Write the three arms; the local one has no price."""
    return [
        make_mode(root, "openrouter"),
        make_mode(root, "nvidia"),
        make_mode(root, "local", priced=False),
    ]


def table_for(folders: list[Path], results: Path) -> str:
    """Write the endpoint-modes table for some sweep folders."""
    return endpoint_modes_table(endpoint_mode_arms(load_sweep(f) for f in folders), results)


def row_of(table: str, profile: str) -> str:
    """Find an arm's row in the results table."""
    return next(line for line in table.splitlines() if line.startswith(f"| [{profile}]"))


@pytest.mark.unit
def test_each_row_names_its_service(tmp_path: Path) -> None:
    table = table_for(three_modes(tmp_path / "runs"), tmp_path / "results")

    for profile, url in SERVICES.items():
        assert f"`{url}`" in row_of(table, profile)


@pytest.mark.unit
def test_the_self_hosted_arm_shows_tokens_and_time_with_unknown_dollars(tmp_path: Path) -> None:
    table = table_for(three_modes(tmp_path / "runs"), tmp_path / "results")

    local = row_of(table, "local")
    # The made-up sweep: 1,000 tokens and 20 seconds a trial, and 3 unpriced
    # calls in each of its 11 marked trials.
    assert "| 1,000 | 20.0 s | unknown (33 calls unpriced) |" in local
    assert "$" not in local, "no dollar figure, not even zero"
    assert "| none |" in local, "a profile with no screenshot model says so"


@pytest.mark.unit
def test_a_priced_arm_shows_its_dollars(tmp_path: Path) -> None:
    table = table_for(three_modes(tmp_path / "runs"), tmp_path / "results")

    assert "$0.0020" in row_of(table, "openrouter")


@pytest.mark.unit
def test_each_row_says_which_tasks_ran_and_how_many_passed(tmp_path: Path) -> None:
    table = table_for(three_modes(tmp_path / "runs"), tmp_path / "results")

    # The made-up sweep: four tasks, eight passes out of eleven marked.
    assert "| alpha, bravo, charlie, delta | 8 of 11 |" in row_of(table, "nvidia")


@pytest.mark.unit
def test_three_clean_arms_on_three_services_pass_every_check(tmp_path: Path) -> None:
    table = table_for(three_modes(tmp_path / "runs"), tmp_path / "results")

    assert (
        f"| Same code in every arm | yes: every trial of every arm ran commit `{COMMIT}`" in table
    )
    assert "| Same tasks | yes: sg-gov-v1 |" in table
    assert "| A different service in every arm | yes: 3 services |" in table


@pytest.mark.unit
def test_two_arms_on_one_service_fail_the_check(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    folders = [
        make_mode(runs, "openrouter"),
        make_mode(runs, "nvidia", base_url=SERVICES["openrouter"]),
    ]

    table = table_for(folders, tmp_path / "results")

    assert "| A different service in every arm | no: modes-nvidia and modes-openrouter" in table


@pytest.mark.unit
def test_an_arm_on_other_code_fails_the_same_code_check(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    folders = [make_mode(runs, "openrouter"), make_mode(runs, "local", versions=["fedcba987654"])]

    table = table_for(folders, tmp_path / "results")

    assert "| Same code in every arm | no: the arms ran different commits" in table


@pytest.mark.unit
def test_only_modes_folders_are_arms(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    folders = [make_sweep(runs), make_sweep(runs, "models-openrouter"), *three_modes(runs)]

    arms = endpoint_mode_arms(load_sweep(f) for f in folders)

    assert [a.name for a in arms] == ["modes-local", "modes-nvidia", "modes-openrouter"]


@pytest.mark.unit
def test_an_arm_on_the_primary_model_is_not_paired_with_the_warm_sweep(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    cold = make_sweep(runs, "eval-openrouter-cold")
    warm = make_sweep(runs, "eval-openrouter-warm", library="warm")
    arm = make_sweep(runs, "modes-openrouter")

    pairs = cold_warm_pairs(load_sweep(f) for f in (cold, warm, arm))

    assert [(c.name, w.name) for c, w in pairs] == [
        ("eval-openrouter-cold", "eval-openrouter-warm")
    ]


@pytest.mark.unit
def test_the_endpoint_modes_file_is_written_beside_the_others(tmp_path: Path) -> None:
    results = tmp_path / "results"

    written = write_results(three_modes(tmp_path / "runs"), results)

    assert results / ENDPOINT_MODES_FILE in written
    assert not (results / COLD_VS_WARM_FILE).exists()


@pytest.mark.unit
def test_no_endpoint_modes_file_without_an_arm(tmp_path: Path) -> None:
    results = tmp_path / "results"

    write_results([make_sweep(tmp_path / "runs")], results)

    assert not (results / ENDPOINT_MODES_FILE).exists()


# --- The driver script ------------------------------------------------------------------

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


def drive(
    tmp_path: Path, *args: str, changes: str = "", eval_exits: str = "0"
) -> tuple[int, list[str], str]:
    """Run the driver script with stand-ins for ``waa`` and ``git``.

    Args:
        tmp_path: The working directory for the run.
        *args: Arguments to the script.
        changes: What the stand-in git reports as uncommitted changes.
        eval_exits: Exit codes for successive ``waa eval`` calls.

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
        "FAKE_EXIT_eval": eval_exits,
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
def test_the_script_runs_one_task_per_profile_into_its_own_folder(tmp_path: Path) -> None:
    code, calls, _ = drive(tmp_path)

    assert code == 0
    assert calls == [
        f"eval --cold --trials 1 --task cpf-employee-share --profile {p} --out runs/modes-{p}"
        for p in ("openrouter", "nvidia", "local")
    ]


@pytest.mark.unit
def test_the_script_passes_profiles_task_and_trials_through(tmp_path: Path) -> None:
    _, calls, _ = drive(tmp_path, "--profiles", "a", "--task", "t", "--trials", "2")

    assert calls == ["eval --cold --trials 2 --task t --profile a --out runs/modes-a"]


@pytest.mark.unit
def test_the_script_refuses_uncommitted_code(tmp_path: Path) -> None:
    code, calls, said = drive(tmp_path, changes=" M src/web_action_agent/profiles.yaml")

    assert code == 2 and calls == []
    assert "commit these changes first" in said


@pytest.mark.unit
def test_an_arm_that_cannot_be_reached_stops_the_script(tmp_path: Path) -> None:
    # A local server that is not running fails the connection test, which
    # cannot work on a retry: the remaining arms are not started.
    code, calls, said = drive(tmp_path, "--profiles", "local openrouter", eval_exits="2")

    assert code == 2
    assert len(calls) == 1
    assert "arm local could not run" in said


@pytest.mark.unit
def test_the_script_refuses_an_unknown_argument(tmp_path: Path) -> None:
    code, calls, _ = drive(tmp_path, "--all-tasks")

    assert code == 2 and calls == []


@pytest.mark.unit
def test_the_script_explains_itself() -> None:
    done = subprocess.run(
        ["bash", str(SCRIPT), "--help"], capture_output=True, text=True, check=False
    )

    assert done.returncode == 0
    assert "uv run scripts/endpoint_modes.sh" in done.stdout
    assert "--served-model-name nvidia/nemotron-3-nano-30b-a3b" in done.stdout
    assert "Exit codes" in done.stdout, "the whole header is shown"
    assert "set -euo pipefail" not in done.stdout, "and nothing after it"


@pytest.mark.unit
def test_the_script_is_executable() -> None:
    assert os.access(SCRIPT, os.X_OK)


@pytest.mark.unit
def test_the_script_names_the_model_the_local_profile_asks_for() -> None:
    # The server must answer to the name the profile sends, or every call
    # fails; the help text is where a user copies the command from.
    local = load_profiles()["local"]

    assert f"--served-model-name {local.model}" in SCRIPT.read_text()
