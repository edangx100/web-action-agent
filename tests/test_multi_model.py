"""The multi-model comparison: the same benchmark on several AI models.

The claim under test is that the agent is tied to a model only through its
endpoint profile, so comparing models takes nothing but ``--profile``. These
tests hold the parts that let the published table show that, rather than
assert it:

- **The profiles.** The comparison models are configured like any other
  profile, priced, and kept out of the judge's family.
- **The code version.** Every sweep records the git commit it ran, marked
  when the code had uncommitted changes, and keeps every version a restart
  added.
- **The published table** states each arm's models and N, and checks that
  only the profile changed: same code, same suite, same judge, all cold.
- **The driver script** runs one cold arm per profile into its own folder,
  and refuses to start on uncommitted code.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest
from test_eval_harness import FakeRunner, fake_harness
from test_metrics import make_sweep

from web_action_agent.config import Config
from web_action_agent.cost import shipped_prices
from web_action_agent.eval_harness import (
    DIRTY,
    NOT_RECORDED,
    SWEEP_FILE,
    HarnessError,
    SweepSpec,
    code_version,
)
from web_action_agent.judge import family_clash
from web_action_agent.metrics import (
    COLD_VS_WARM_FILE,
    MULTI_MODEL_FILE,
    cold_warm_pairs,
    load_sweep,
    multi_model_arms,
    multi_model_table,
    same_code_check,
    write_results,
)
from web_action_agent.model_client import DEFAULT_PROFILES_PATH, load_profiles

REPO = Path(__file__).parent.parent

#: The driver script under test.
SCRIPT = REPO / "scripts" / "multi_model.sh"

#: The profiles the comparison adds beside the primary one.
COMPARISON_PROFILES = ("openrouter-qwen", "openrouter-glimmer")

#: A made-up commit, as ``git rev-parse --short=12`` prints one.
COMMIT = "0123456789ab"


# --- The profiles ------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize("name", COMPARISON_PROFILES)
def test_a_comparison_profile_differs_from_the_primary_only_in_its_models(name: str) -> None:
    profiles = load_profiles(DEFAULT_PROFILES_PATH)
    primary, other = profiles["openrouter"], profiles[name]

    # Same service and key: the comparison is between models, not services.
    assert (other.base_url, other.api_key_env) == (primary.base_url, primary.api_key_env)
    assert other.model != primary.model
    # One model serves both routes, so the arm measures that model alone.
    assert other.vision is not None and other.vision.model == other.model
    # A pinned host, so the number can be reproduced.
    assert other.text.provider_order


@pytest.mark.unit
@pytest.mark.parametrize("name", COMPARISON_PROFILES)
def test_a_comparison_model_is_priced(name: str) -> None:
    # An unpriced model is one whose calls the money ceiling cannot count.
    profile = load_profiles(DEFAULT_PROFILES_PATH)[name]
    price = shipped_prices().price_for(name, profile.model)

    assert price is not None and price.input_usd_per_mtok and price.output_usd_per_mtok


@pytest.mark.unit
@pytest.mark.parametrize("name", COMPARISON_PROFILES)
def test_a_comparison_model_is_not_from_the_judges_family(name: str) -> None:
    profiles = load_profiles(DEFAULT_PROFILES_PATH)
    judge = profiles["openrouter-judge"]

    assert family_clash(profiles[name], (judge.model,)) is None


# --- The code version --------------------------------------------------------------------


def git(repo: Path, *args: str) -> str:
    """Run git in a test repository.

    Args:
        repo: The repository.
        *args: The git command's arguments.

    Returns:
        str: What git printed, stripped.
    """
    done = subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@t", *args],
        cwd=repo,
        capture_output=True,
        text=True,
        check=True,
    )
    return done.stdout.strip()


@pytest.fixture
def checkout(tmp_path: Path) -> Path:
    """A git checkout laid out like this project: the package under ``src/``.

    Returns:
        Path: The package folder inside it.
    """
    repo = tmp_path / "repo"
    package = repo / "src" / "pkg"
    package.mkdir(parents=True)
    (package / "code.py").write_text("x = 1\n")
    (repo / "results").mkdir()
    (repo / "results" / "README.md").write_text("table\n")
    git(repo, "init", "-q")
    git(repo, "add", ".")
    git(repo, "commit", "-q", "-m", "start")
    return package


@pytest.mark.unit
def test_the_code_version_is_the_commit(checkout: Path) -> None:
    commit = git(checkout, "rev-parse", "--short=12", "HEAD")

    assert code_version(checkout) == commit


@pytest.mark.unit
def test_an_uncommitted_change_to_the_code_is_marked(checkout: Path) -> None:
    (checkout / "code.py").write_text("x = 2\n")

    assert code_version(checkout).endswith(DIRTY)


@pytest.mark.unit
def test_a_rewritten_results_table_is_not_a_change_to_the_code(checkout: Path) -> None:
    # Every sweep rewrites the results tables. If that counted, every arm
    # after the first would be marked as running changed code.
    (checkout.parent.parent / "results" / "README.md").write_text("new table\n")
    (checkout / "scratch.py").write_text("untracked\n")

    assert not code_version(checkout).endswith(DIRTY)


@pytest.mark.unit
def test_no_version_is_claimed_outside_this_projects_checkout(tmp_path: Path) -> None:
    # A package installed somewhere inside another repository must not take
    # that repository's commit as its own.
    repo = tmp_path / "elsewhere"
    package = repo / "lib" / "pkg"
    package.mkdir(parents=True)
    (package / "code.py").write_text("x = 1\n")
    git(repo, "init", "-q")
    git(repo, "add", ".")
    git(repo, "commit", "-q", "-m", "start")

    assert code_version(package) == ""


@pytest.mark.unit
def test_no_version_without_git(checkout: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PATH", "")

    assert code_version(checkout) == ""


@pytest.mark.unit
def test_the_running_package_names_a_version() -> None:
    # This project is tested from its own checkout, so there is an answer.
    assert code_version()


def versions_of(sweep: Path) -> list[str]:
    """Read the code versions a sweep folder recorded."""
    return json.loads((sweep / SWEEP_FILE).read_text())["code_versions"]


@pytest.mark.unit
def test_a_sweep_records_the_code_it_ran(tmp_path: Path) -> None:
    sweep = tmp_path / "sweep"
    fake_harness(sweep, FakeRunner(), trials=1, version=lambda: COMMIT).run()

    assert versions_of(sweep) == [COMMIT]


@pytest.mark.unit
def test_a_restart_on_new_code_adds_its_version(tmp_path: Path) -> None:
    sweep = tmp_path / "sweep"
    fake_harness(sweep, FakeRunner(), trials=1, version=lambda: "aaaa").run()
    fake_harness(sweep, FakeRunner(), trials=2, version=lambda: "bbbb").run()
    fake_harness(sweep, FakeRunner(), trials=3, version=lambda: "bbbb").run()

    assert versions_of(sweep) == ["aaaa", "bbbb"], "each version once, in order"


@pytest.mark.unit
def test_a_restart_with_nothing_to_do_adds_no_version(tmp_path: Path) -> None:
    # Rerunning a finished sweep only rewrites its tables; no trial runs on
    # the newer code, so it must not be named as code the sweep ran.
    sweep = tmp_path / "sweep"
    fake_harness(sweep, FakeRunner(), trials=1, version=lambda: "aaaa").run()
    fake_harness(sweep, FakeRunner(), trials=1, version=lambda: "bbbb").run()

    assert versions_of(sweep) == ["aaaa"]


@pytest.mark.unit
def test_a_sweep_from_before_versions_were_kept_is_not_credited_to_new_code(
    tmp_path: Path,
) -> None:
    sweep = tmp_path / "sweep"
    fake_harness(sweep, FakeRunner(), trials=1, version=lambda: "aaaa").run()
    # Make it look like a sweep started before versions and vision models
    # were recorded.
    spec = json.loads((sweep / SWEEP_FILE).read_text())
    del spec["code_versions"], spec["vision_model"]
    (sweep / SWEEP_FILE).write_text(json.dumps(spec))

    fake_harness(sweep, FakeRunner(), trials=2, version=lambda: "bbbb").run()

    assert versions_of(sweep) == [NOT_RECORDED, "bbbb"]


@pytest.mark.unit
def test_a_sweep_records_the_screenshot_model(tmp_path: Path) -> None:
    sweep = tmp_path / "sweep"
    fake_harness(sweep, FakeRunner(), trials=1).run()

    spec = SweepSpec.read(sweep / SWEEP_FILE)
    assert spec.vision_model == load_profiles(DEFAULT_PROFILES_PATH)["openrouter"].vision.model


@pytest.mark.unit
def test_a_folder_holding_another_models_sweep_is_refused(tmp_path: Path) -> None:
    sweep = tmp_path / "sweep"
    fake_harness(sweep, FakeRunner(), trials=1).run()

    with pytest.raises(HarnessError, match="model"):
        fake_harness(sweep, FakeRunner(), trials=1, config=Config(profile="openrouter-qwen")).run()


# --- The published table -----------------------------------------------------------------


def make_arm(
    root: Path,
    profile: str,
    model: str,
    *,
    versions: list[str] | None = None,
    **changes: str,
) -> Path:
    """Write a made-up arm of the comparison: the made-up sweep, on one model.

    Args:
        root: Where the sweep folder goes.
        profile: The arm's profile; the folder is ``models-<profile>``.
        model: Its model, for both routes.
        versions: The code versions it ran. Defaults to one commit.
        **changes: Other ``sweep.json`` fields to change.

    Returns:
        Path: The sweep folder.
    """
    sweep = make_sweep(root, f"models-{profile}")
    spec = json.loads((sweep / SWEEP_FILE).read_text())
    spec.update(
        profile=profile,
        model=model,
        vision_model=model,
        code_versions=[COMMIT] if versions is None else versions,
        **changes,
    )
    (sweep / SWEEP_FILE).write_text(json.dumps(spec))
    return sweep


def three_arms(root: Path, **second: object) -> list[Path]:
    """Write three made-up arms; ``second`` changes the second one's fields."""
    return [
        make_arm(root, "openrouter", "nvidia/nano"),
        make_arm(root, "openrouter-qwen", "qwen/qwen", **second),  # type: ignore[arg-type]
        make_arm(root, "openrouter-mistral", "mistralai/small"),
    ]


def table_for(folders: list[Path], results: Path) -> str:
    """Write the multi-model table for some sweep folders."""
    return multi_model_table(multi_model_arms(load_sweep(f) for f in folders), results)


@pytest.mark.unit
def test_the_table_states_each_arms_models_and_n(tmp_path: Path) -> None:
    table = table_for(three_arms(tmp_path / "runs"), tmp_path / "results")

    for model in ("nvidia/nano", "qwen/qwen", "mistralai/small"):
        row = next(line for line in table.splitlines() if f"`{model}`" in line)
        # The made-up sweep: 4 tasks, 3 rounds, one trial left unmarked.
        assert "11 (4 tasks × 3)" in row
        assert row.count(f"`{model}`") == 2, "planner and screenshot model both named"


@pytest.mark.unit
def test_the_table_links_each_arm_to_its_own_table(tmp_path: Path) -> None:
    table = table_for(three_arms(tmp_path / "runs"), tmp_path / "results")

    assert "[openrouter-qwen](models-openrouter-qwen.md)" in table


@pytest.mark.unit
def test_three_arms_on_one_clean_commit_pass_every_check(tmp_path: Path) -> None:
    table = table_for(three_arms(tmp_path / "runs"), tmp_path / "results")

    assert (
        f"| Same code in every arm | yes: every trial of every arm ran commit `{COMMIT}`" in table
    )
    assert "| Same benchmark suite | yes: sg-gov-v1 |" in table
    assert "| Same judge | yes: openrouter-judge |" in table
    assert "| Empty skill library in every arm | yes: cold |" in table


@pytest.mark.unit
@pytest.mark.parametrize(
    ("versions", "said"),
    [
        (["fedcba987654"], "the arms ran different commits"),
        ([COMMIT + DIRTY], "changes not yet committed"),
        ([COMMIT, "fedcba987654"], "restarted on different code partway"),
        ([NOT_RECORDED], "no recorded code version"),
        ([], "has not run any trial yet"),
    ],
)
def test_same_code_is_claimed_only_when_it_can_be_shown(
    tmp_path: Path, versions: list[str], said: str
) -> None:
    folders = three_arms(tmp_path / "runs", versions=versions)

    verdict = same_code_check(multi_model_arms(load_sweep(f) for f in folders))

    assert verdict.startswith("no: ") and said in verdict


@pytest.mark.unit
def test_an_arm_with_another_judge_fails_the_check(tmp_path: Path) -> None:
    folders = three_arms(tmp_path / "runs", judge_profile="other-judge")

    table = table_for(folders, tmp_path / "results")

    assert "| Same judge | no: " in table and "other-judge" in table


@pytest.mark.unit
def test_a_sweep_from_before_the_screenshot_model_was_kept_says_so(tmp_path: Path) -> None:
    folders = three_arms(tmp_path / "runs")
    spec = json.loads((folders[0] / SWEEP_FILE).read_text())
    del spec["vision_model"]
    (folders[0] / SWEEP_FILE).write_text(json.dumps(spec))

    table = table_for(folders, tmp_path / "results")

    row = next(line for line in table.splitlines() if "`nvidia/nano`" in line)
    assert NOT_RECORDED in row


@pytest.mark.unit
def test_the_per_task_table_has_a_column_per_arm(tmp_path: Path) -> None:
    table = table_for(three_arms(tmp_path / "runs"), tmp_path / "results")

    assert "| Task | openrouter | openrouter-mistral | openrouter-qwen |" in table, "name order"
    # bravo: passed rounds 1 and 3 of 3.
    assert "| bravo | 2/3 | 2/3 | 2/3 |" in table


@pytest.mark.unit
def test_only_models_folders_are_arms(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    folders = [make_sweep(runs), *three_arms(runs)]

    arms = multi_model_arms(load_sweep(f) for f in folders)

    assert [a.name for a in arms] == [
        "models-openrouter",
        "models-openrouter-mistral",
        "models-openrouter-qwen",
    ]


@pytest.mark.unit
def test_an_arm_on_the_primary_model_is_not_paired_with_the_warm_sweep(tmp_path: Path) -> None:
    # models-openrouter is cold, on the same profile and model as the
    # headline cold sweep; the warm sweep's skills came from the headline one.
    runs = tmp_path / "runs"
    cold = make_sweep(runs, "eval-openrouter-cold")
    warm = make_sweep(runs, "eval-openrouter-warm", library="warm")
    arm = make_sweep(runs, "models-openrouter")

    pairs = cold_warm_pairs(load_sweep(f) for f in (cold, warm, arm))

    assert [(c.name, w.name) for c, w in pairs] == [
        ("eval-openrouter-cold", "eval-openrouter-warm")
    ]


@pytest.mark.unit
def test_the_multi_model_file_is_written_beside_the_others(tmp_path: Path) -> None:
    results = tmp_path / "results"

    written = write_results(three_arms(tmp_path / "runs"), results)

    assert results / MULTI_MODEL_FILE in written
    assert not (results / COLD_VS_WARM_FILE).exists(), "the arms are not a cold-warm pair"


@pytest.mark.unit
def test_no_multi_model_file_without_an_arm(tmp_path: Path) -> None:
    results = tmp_path / "results"

    write_results([make_sweep(tmp_path / "runs")], results)

    assert not (results / MULTI_MODEL_FILE).exists()


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

#: A stand-in for ``git``: prints FAKE_CHANGES as the uncommitted changes,
#: or fails as git does outside a checkout when FAKE_NOT_A_REPO is set.
FAKE_GIT = """#!/usr/bin/env bash
if [[ -n "${FAKE_NOT_A_REPO:-}" ]]; then exit 128; fi
printf '%s' "${FAKE_CHANGES:-}"
"""


def drive(
    tmp_path: Path, *args: str, changes: str = "", repo: bool = True, eval_exits: str = "0"
) -> tuple[int, list[str], str]:
    """Run the driver script with stand-ins for ``waa`` and ``git``.

    Args:
        tmp_path: The working directory for the run.
        *args: Arguments to the script.
        changes: What the stand-in git reports as uncommitted changes.
        repo: False to make the stand-in git fail as outside a checkout.
        eval_exits: Exit codes for successive ``waa eval`` calls.

    Returns:
        tuple[int, list[str], str]: The script's exit code, the commands it
        ran in order, and what it printed to stderr.
    """
    fakes = {"fake-waa": FAKE_WAA, "fake-git": FAKE_GIT}
    for name, text in fakes.items():
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
    if not repo:
        env["FAKE_NOT_A_REPO"] = "1"
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
def test_the_script_runs_one_cold_arm_per_profile_into_its_own_folder(tmp_path: Path) -> None:
    code, calls, _ = drive(tmp_path)

    assert code == 0
    assert calls == [
        f"eval --cold --trials 2 --profile {p} --out runs/models-{p}"
        for p in ("openrouter", "openrouter-qwen", "openrouter-glimmer")
    ]


@pytest.mark.unit
def test_the_script_passes_profiles_and_trials_through(tmp_path: Path) -> None:
    _, calls, _ = drive(tmp_path, "--profiles", "a b", "--trials", "1")

    assert calls == [
        "eval --cold --trials 1 --profile a --out runs/models-a",
        "eval --cold --trials 1 --profile b --out runs/models-b",
    ]


@pytest.mark.unit
def test_the_script_refuses_uncommitted_code(tmp_path: Path) -> None:
    code, calls, said = drive(tmp_path, changes=" M src/web_action_agent/planner.py")

    assert code == 2
    assert calls == [], "nothing was spent"
    assert "commit these changes first" in said and "planner.py" in said


@pytest.mark.unit
def test_the_script_warns_outside_a_checkout_and_carries_on(tmp_path: Path) -> None:
    code, calls, said = drive(tmp_path, repo=False)

    assert code == 0 and len(calls) == 3
    assert "cannot show that every arm ran the same code" in said


@pytest.mark.unit
def test_an_unfinished_arm_is_tried_again_and_the_next_arm_follows(tmp_path: Path) -> None:
    code, calls, said = drive(tmp_path, "--profiles", "a b", eval_exits="1 0")

    assert code == 0
    assert calls == [
        "eval --cold --trials 2 --profile a --out runs/models-a",
        "eval --cold --trials 2 --profile a --out runs/models-a",
        "eval --cold --trials 2 --profile b --out runs/models-b",
    ]
    assert "arm a (try 2 of 3)" in said


@pytest.mark.unit
def test_an_arm_that_never_finishes_stops_the_script(tmp_path: Path) -> None:
    code, calls, _ = drive(tmp_path, "--profiles", "a b", "--attempts", "2", eval_exits="1")

    assert code == 1
    assert calls == ["eval --cold --trials 2 --profile a --out runs/models-a"] * 2


@pytest.mark.unit
def test_the_script_refuses_an_unknown_argument(tmp_path: Path) -> None:
    code, calls, _ = drive(tmp_path, "--warm")

    assert code == 2 and calls == []


@pytest.mark.unit
def test_the_script_explains_itself() -> None:
    done = subprocess.run(
        ["bash", str(SCRIPT), "--help"], capture_output=True, text=True, check=False
    )

    assert done.returncode == 0
    assert "uv run scripts/multi_model.sh" in done.stdout
    assert "spends real money" in done.stdout
    assert "Exit codes" in done.stdout, "the whole header is shown"


@pytest.mark.unit
def test_the_script_is_executable() -> None:
    assert os.access(SCRIPT, os.X_OK)
