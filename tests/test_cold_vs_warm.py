"""The cold-versus-warm experiment: does learning skills make the agent better?

The benchmark runs twice on one model -- once with an empty skill library
(*cold*), once with skills learnt from the cold pass (*warm*) -- and the
difference is published. These tests hold the parts that keep that difference
honest:

- **The right library, or no sweep.** A cold sweep refuses a library with any
  skill in it; a warm sweep refuses one with no active skill for the suite's
  websites. Either mistake would publish a difference that measures nothing.
- **Learning from what the benchmark passed.** ``waa induce --all`` on a sweep
  folder learns from the trials the benchmark marked a pass -- not the ones
  the agent merely believed it had done -- and stores one skill per task.
- **The published table** shows both arms side by side, with the change and
  the number of trials each rests on.
- **The driver script** runs the three steps in order, resumes safely, and
  stops at the first step that did not finish.

The live end-to-end run is at the bottom, marked ``live``: it spends money.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
from test_eval_harness import FakeRunner, fake_harness, stand_in_suite
from test_metrics import make_sweep
from test_skills_cli import (
    AN_INDUCTION,
    ScriptedEndpoint,
    _point_at,
    a_skill,
    a_workspace,
    invoke,
    put_run,
)

from web_action_agent import cli
from web_action_agent.benchmark import load_suite
from web_action_agent.config import resolve_env
from web_action_agent.eval_harness import (
    LIBRARY_DIR,
    SWEEP_FILE,
    VERDICT_FILE,
    HarnessError,
    TrialVerdict,
    library_problem,
    passing_trials,
    suite_domains,
)
from web_action_agent.judge import FAIL, PASS
from web_action_agent.metrics import (
    COLD_VS_WARM_FILE,
    cold_warm_pairs,
    load_sweep,
    write_results,
)
from web_action_agent.skill_store import SkillStore

REPO = Path(__file__).parent.parent

#: The driver script under test.
SCRIPT = REPO / "scripts" / "cold_vs_warm.sh"


def library_with(root: Path, *skills: Any) -> Path:
    """Write a skill library holding the given skills.

    Args:
        root: The library's folder.
        *skills: Skills to store.

    Returns:
        Path: The folder.
    """
    store = SkillStore(root)
    root.mkdir(parents=True, exist_ok=True)
    for skill in skills:
        store.save(skill)
    return root


# --- The right library, or no sweep -------------------------------------------------


@pytest.mark.unit
def test_the_suite_domains_are_the_ones_skills_are_matched_on() -> None:
    assert suite_domains(stand_in_suite()) == {"fixture.test"}


@pytest.mark.unit
def test_a_cold_arm_refuses_a_library_that_is_not_empty(tmp_path: Path) -> None:
    sweep = tmp_path / "sweep"
    # A sweep folder whose library copy already has a skill in it -- as when a
    # warm sweep's folder is reused for a cold one.
    library_with(sweep / LIBRARY_DIR, a_skill("search", domain="fixture.test"))
    runner = FakeRunner()

    with pytest.raises(HarnessError, match="a cold sweep must start with an empty skill library"):
        fake_harness(sweep, runner, trials=1, cold=True).run()

    assert runner.ran == [], "nothing ran"
    assert not (sweep / SWEEP_FILE).exists(), "and nothing was written"


@pytest.mark.unit
def test_a_cold_arm_refuses_even_a_retired_skill(tmp_path: Path) -> None:
    library = library_with(
        tmp_path / "lib", a_skill("old", domain="fixture.test", active=False, retired_reason="x")
    )

    assert library_problem(library, stand_in_suite(), cold=True) is not None


@pytest.mark.unit
def test_a_cold_arm_ignores_the_real_library(tmp_path: Path) -> None:
    # The real library is never used by a cold sweep, so its contents do not
    # matter: the sweep starts from an empty copy of its own.
    skills = library_with(tmp_path / "skills", a_skill("search", domain="fixture.test"))
    sweep = tmp_path / "sweep"

    fake_harness(sweep, FakeRunner(), trials=1, cold=True, skills_root=skills).run()

    assert list((sweep / LIBRARY_DIR).iterdir()) == []


@pytest.mark.unit
@pytest.mark.parametrize(
    ("skills", "said"),
    [
        ((), "has none"),
        ((a_skill("elsewhere", domain="other.example"),), "none of them active on those sites"),
        (
            (a_skill("old", domain="fixture.test", active=False, retired_reason="broke"),),
            "none of them active on those sites",
        ),
    ],
    ids=["empty", "other-sites-only", "retired-only"],
)
def test_a_warm_arm_refuses_a_library_with_nothing_for_the_suite(
    tmp_path: Path, skills: tuple[Any, ...], said: str
) -> None:
    library = library_with(tmp_path / "skills", *skills)
    sweep = tmp_path / "sweep"
    runner = FakeRunner()

    with pytest.raises(HarnessError, match="a warm sweep needs learnt skills") as caught:
        fake_harness(sweep, runner, trials=1, cold=False, skills_root=library).run()

    assert said in str(caught.value)
    assert "waa induce --all" in str(caught.value), "and says what to do"
    assert runner.ran == []
    assert not sweep.exists()


@pytest.mark.unit
def test_a_warm_arm_refuses_a_library_that_does_not_exist(tmp_path: Path) -> None:
    with pytest.raises(HarnessError, match="a warm sweep needs learnt skills"):
        fake_harness(tmp_path / "sweep", FakeRunner(), trials=1, cold=False).run()


@pytest.mark.unit
def test_a_warm_arm_starts_with_one_active_skill_for_the_suite(tmp_path: Path) -> None:
    library = library_with(
        tmp_path / "skills",
        a_skill("search", domain="fixture.test"),
        a_skill("elsewhere", domain="other.example"),
    )
    sweep = tmp_path / "sweep"

    fake_harness(sweep, FakeRunner(), trials=1, cold=False, skills_root=library).run()

    assert sorted(p.name for p in (sweep / LIBRARY_DIR).iterdir()) == [
        "elsewhere.json",
        "search.json",
    ]


@pytest.mark.unit
def test_an_unreadable_warm_library_is_refused_with_the_reason(tmp_path: Path) -> None:
    library = tmp_path / "skills"
    library.mkdir()
    (library / "broken.json").write_text("{not json")

    problem = library_problem(library, stand_in_suite(), cold=False)

    assert problem is not None and "cannot be read" in problem


@pytest.mark.unit
def test_a_restarted_warm_sweep_checks_its_own_copy(tmp_path: Path) -> None:
    # The real library was emptied after the warm sweep started: the sweep
    # carries on with the copy it started with.
    library = library_with(tmp_path / "skills", a_skill("search", domain="fixture.test"))
    sweep = tmp_path / "sweep"
    fake_harness(sweep, FakeRunner(), trials=1, cold=False, skills_root=library).run()
    shutil.rmtree(library)

    fake_harness(sweep, FakeRunner(), trials=2, cold=False, skills_root=library).run()


# --- Learning from what the benchmark passed ----------------------------------------


def a_sweep_to_learn_from(root: Path) -> Path:
    """Lay out a sweep folder with real runs, in the harness's own layout.

    Three trials, each a copy of the committed run (which ended in success):

    - ``task-a/trial-1``: marked a pass;
    - ``task-a/trial-2``: marked a pass too;
    - ``task-b/trial-1``: marked a FAIL -- the agent believed it succeeded,
      but the checks said otherwise.

    Plus the folders a sweep keeps that are not runs: ``library/`` and
    ``_interrupted/``.

    Args:
        root: Where the sweep goes.

    Returns:
        Path: The sweep folder.
    """
    sweep = root / "eval-local-cold"
    sweep.mkdir(parents=True)
    (sweep / SWEEP_FILE).write_text(json.dumps({"suite": "fixture-suite", "library": "cold"}))
    (sweep / LIBRARY_DIR).mkdir()
    (sweep / "_interrupted" / "task-a" / "trial-1").mkdir(parents=True)
    for task, trial, status in (("task-a", 1, PASS), ("task-a", 2, PASS), ("task-b", 1, FAIL)):
        folder = put_run(sweep / task, f"trial-{trial}")
        mark = TrialVerdict(status=status, reason="", agent_outcome="success")
        if status == FAIL:
            mark = TrialVerdict(
                status=FAIL, reason="wrong answer", agent_outcome="success", category="grounding"
            )
        (folder / VERDICT_FILE).write_text(json.dumps(mark.to_dict()))
    return sweep


@pytest.mark.unit
def test_passing_trials_are_the_benchmarks_passes_in_task_order(tmp_path: Path) -> None:
    sweep = a_sweep_to_learn_from(tmp_path)

    assert passing_trials(sweep) == [
        (sweep / "task-a" / "trial-1", "task-a"),
        (sweep / "task-a" / "trial-2", "task-a"),
    ]


@pytest.mark.fixture
def test_induce_all_on_a_sweep_learns_once_per_task_from_passes_only(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, playwright: Any, fixture_site: Any
) -> None:
    endpoint = ScriptedEndpoint(AN_INDUCTION)
    runs = a_workspace(monkeypatch, tmp_path, playwright, endpoint)
    sweep = a_sweep_to_learn_from(runs)
    for folder in sweep.glob("task-*/trial-*"):
        _point_at(folder, fixture_site)
    monkeypatch.setattr(cli, "resolve_env", dict)

    result = invoke(["induce", "--all", str(sweep)])

    assert result.exit_code == cli.EXIT_SUCCESS, result.output
    assert "learning from the 2 trial(s) the benchmark marked a pass" in result.output
    # The first pass of task-a is learnt from; its second is passed over.
    assert "task-a/trial-1: stored" in result.output
    assert "task-a/trial-2: skipped, task-a already has a skill" in result.output
    # task-b's run claimed success, but the benchmark failed it: never tried.
    assert "task-b" not in result.output
    assert "1 induced, 1 verified, 0 discarded" in result.output
    assert len(endpoint.calls) == 1, "one model call: one induction"
    assert len(SkillStore(tmp_path / "skills")) == 1


@pytest.mark.unit
def test_induce_all_on_a_sweep_with_no_passes_says_so(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    sweep = tmp_path / "eval-local-cold"
    sweep.mkdir()
    (sweep / SWEEP_FILE).write_text("{}")
    monkeypatch.setattr(cli, "resolve_env", dict)

    result = invoke(["induce", "--all", str(sweep)])

    assert result.exit_code == cli.EXIT_TASK_FAILURE
    assert "no passing trials to learn from" in result.output


# --- The published table --------------------------------------------------------------


def a_cold_and_a_warm_sweep(root: Path) -> tuple[Path, Path]:
    """Make a cold sweep and a warm one, the warm one with a learnt library.

    Both are the made-up sweep from ``test_metrics`` (9 of 11 marked trials
    pass); the warm one's library holds two skills, one of them retired.

    Args:
        root: Where the runs folder goes.

    Returns:
        tuple[Path, Path]: The cold and warm sweep folders.
    """
    cold = make_sweep(root / "runs", "eval-openrouter-cold")
    warm = make_sweep(root / "runs", "eval-openrouter-warm", library="warm")
    library_with(
        warm / LIBRARY_DIR,
        a_skill("search", domain="mom.gov.sg"),
        a_skill("old", domain="moe.gov.sg", active=False, retired_reason="broke"),
    )
    return cold, warm


def improve(sweep: Path) -> None:
    """Make a warm sweep better than its cold partner: fewer steps, one more pass.

    Every trial takes half the steps, and bravo's failed second trial becomes
    a pass.

    Args:
        sweep: The warm sweep.
    """
    for outcome in sweep.glob("*/trial-*/outcome.json"):
        data = json.loads(outcome.read_text())
        data["steps"] = data["steps"] // 2
        data["cost"]["cost_usd"] = 0.001
        outcome.write_text(json.dumps(data))
    verdict = sweep / "bravo" / "trial-2" / VERDICT_FILE
    data = json.loads(verdict.read_text())
    data.update(status=PASS, category=None)
    verdict.write_text(json.dumps(data))


@pytest.mark.unit
def test_cold_and_warm_sweeps_of_one_model_are_paired(tmp_path: Path) -> None:
    cold, warm = a_cold_and_a_warm_sweep(tmp_path)
    other = make_sweep(tmp_path / "runs", "eval-other-cold")
    spec = json.loads((other / SWEEP_FILE).read_text())
    (other / SWEEP_FILE).write_text(json.dumps({**spec, "model": "another/model"}))

    pairs = cold_warm_pairs([load_sweep(p) for p in (cold, warm, other)])

    assert [(c.name, w.name) for c, w in pairs] == [
        ("eval-openrouter-cold", "eval-openrouter-warm")
    ]


@pytest.mark.unit
def test_the_table_shows_both_arms_the_change_and_n(tmp_path: Path) -> None:
    cold, warm = a_cold_and_a_warm_sweep(tmp_path)
    improve(warm)

    written = write_results([cold, warm], tmp_path / "results")
    table = (tmp_path / "results" / COLD_VS_WARM_FILE).read_text()

    assert tmp_path / "results" / COLD_VS_WARM_FILE in written
    # Worked by hand. Cold: rounds 3/4, 2/4, 3/3 -> (75 + 50 + 100) / 3 = 75.0%.
    # Warm: bravo-2 now passes, so 3/4, 3/4, 3/3 -> 83.3%. Change: +8.3 points.
    assert "| Success rate | 75.0% (range 50.0%–100.0% over 3 rounds) |" in table
    assert "| 83.3% (range 75.0%–100.0% over 3 rounds) | +8.3 points |" in table
    # 12 trials, one (delta-3) unmarked: N is 11 on each side.
    assert "| Trials marked (N) | 11 (4 tasks × 3) | 11 (4 tasks × 3) | |" in table
    assert "| Trials passed | 8 | 9 | +1 |" in table
    # Cold steps over the 11 marked trials: 12 + 22 + 27 + 10 = 71, so 6.45.
    # Warm halves each, rounding down: 6 + 11 + 13 + 4 = 34, so 3.09.
    # Change: 3.09 - 6.45 = -3.36, which is -52.1% of 6.45.
    assert "| Average steps per trial | 6.5 | 3.1 | -3.4 (-52.1%) |" in table
    assert "| Average cost per trial (agent) | $0.0020 | $0.0010 | -$0.0010 (-50.0%) |" in table
    assert "| Skill library at the start | empty | 2 skills (1 active) on 2 sites | |" in table
    assert "| bravo | 2/3 | 3/3 |" in table
    assert "[eval-openrouter-cold](eval-openrouter-cold.md)" in table


@pytest.mark.unit
def test_a_change_against_an_unknown_cost_is_unknown(tmp_path: Path) -> None:
    cold = make_sweep(tmp_path / "runs", "eval-openrouter-cold", priced=False)
    warm = make_sweep(tmp_path / "runs", "eval-openrouter-warm", library="warm")

    write_results([cold, warm], tmp_path / "results")

    assert (
        "unknown (some calls had no price)"
        in (tmp_path / "results" / COLD_VS_WARM_FILE).read_text()
    )


@pytest.mark.unit
def test_no_table_without_both_arms(tmp_path: Path) -> None:
    written = write_results([make_sweep(tmp_path / "runs")], tmp_path / "results")

    assert not (tmp_path / "results" / COLD_VS_WARM_FILE).exists()
    assert len(written) == 2


# --- The driver script ------------------------------------------------------------------

#: A stand-in for ``waa``: logs its arguments, and exits with the codes named
#: for that command in FAKE_EXIT_<command> -- one per call, space-separated,
#: the last repeating ("1 0": unfinished, then finished). ``eval --warm`` also
#: makes the warm sweep's folder, as the real one does.
FAKE_WAA = """#!/usr/bin/env bash
echo "$*" >> "$FAKE_LOG"
if [[ "$1 $2" == "eval --warm" ]]; then mkdir -p runs/eval-openrouter-warm; fi
var="FAKE_EXIT_$1"
read -r -a codes <<< "${!var:-0}"
counter="$FAKE_LOG.$1"
n=$(cat "$counter" 2>/dev/null || echo 0)
echo $((n + 1)) > "$counter"
last=$((${#codes[@]} - 1))
exit "${codes[$((n < last ? n : last))]}"
"""


def drive(tmp_path: Path, *args: str, **exits: int | str) -> tuple[int, list[str], str]:
    """Run the driver script with a stand-in ``waa``, and no pauses between tries.

    Args:
        tmp_path: The working directory for the run.
        *args: Arguments to the script.
        **exits: Exit codes per subcommand, e.g. ``eval=1`` or ``eval="1 0"``.

    Returns:
        tuple[int, list[str], str]: The script's exit code, the commands it
        ran in order, and what it printed to stderr.
    """
    fake = tmp_path / "fake-waa"
    fake.write_text(FAKE_WAA)
    fake.chmod(0o755)
    log = tmp_path / "calls.log"
    env = {**os.environ, "WAA": str(fake), "FAKE_LOG": str(log), "RETRY_WAIT_S": "0"}
    env.update({f"FAKE_EXIT_{name}": str(code) for name, code in exits.items()})
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
def test_the_script_runs_cold_then_learns_then_runs_warm(tmp_path: Path) -> None:
    code, calls, _ = drive(tmp_path)

    assert code == 0
    assert calls == [
        "eval --cold --trials 3 --profile openrouter",
        "induce --all runs/eval-openrouter-cold",
        "eval --warm --trials 3 --profile openrouter",
    ]


@pytest.mark.unit
def test_the_script_passes_the_profile_and_trials_through(tmp_path: Path) -> None:
    _, calls, _ = drive(tmp_path, "--profile", "nvidia", "--trials", "1")

    assert calls == [
        "eval --cold --trials 1 --profile nvidia",
        "induce --all runs/eval-nvidia-cold",
        "eval --warm --trials 1 --profile nvidia",
    ]


@pytest.mark.unit
def test_the_script_stops_when_the_cold_pass_never_finishes(tmp_path: Path) -> None:
    code, calls, said = drive(tmp_path, eval=1)

    assert code == 1
    # Three tries, then nothing after it.
    assert calls == ["eval --cold --trials 3 --profile openrouter"] * 3
    assert "trying again in 0s (finished trials are kept)" in said
    assert "run this script again later" in said


@pytest.mark.unit
def test_an_unfinished_pass_is_tried_again_and_the_sequence_carries_on(tmp_path: Path) -> None:
    # As in the rehearsal: a few trials hit an AI-service error, so the cold
    # pass ends unfinished; the next try runs just those, and finishes.
    code, calls, said = drive(tmp_path, eval="1 0")

    assert code == 0
    assert calls == [
        "eval --cold --trials 3 --profile openrouter",
        "eval --cold --trials 3 --profile openrouter",
        "induce --all runs/eval-openrouter-cold",
        "eval --warm --trials 3 --profile openrouter",
    ]
    assert "cold pass (try 2 of 3)" in said


@pytest.mark.unit
def test_attempts_sets_how_many_tries_a_pass_gets(tmp_path: Path) -> None:
    code, calls, _ = drive(tmp_path, "--attempts", "1", eval=1)

    assert code == 1
    assert len(calls) == 1


@pytest.mark.unit
def test_learning_is_never_tried_twice(tmp_path: Path) -> None:
    # Finding nothing to learn is an answer, not a hiccup.
    code, calls, _ = drive(tmp_path, induce=1)

    assert code == 1
    assert sum(c.startswith("induce") for c in calls) == 1


@pytest.mark.unit
def test_the_script_stops_when_no_skill_is_learnt(tmp_path: Path) -> None:
    code, calls, _ = drive(tmp_path, induce=1)

    assert code == 1
    assert calls[-1] == "induce --all runs/eval-openrouter-cold", "the warm pass never starts"


@pytest.mark.unit
def test_the_script_passes_a_usage_error_straight_through(tmp_path: Path) -> None:
    code, calls, said = drive(tmp_path, eval=2)

    assert code == 2
    assert len(calls) == 1
    assert "could not run" in said


@pytest.mark.unit
def test_the_script_does_not_learn_again_once_the_warm_pass_has_started(tmp_path: Path) -> None:
    (tmp_path / "runs" / "eval-openrouter-warm").mkdir(parents=True)

    code, calls, said = drive(tmp_path)

    assert code == 0
    assert not any(c.startswith("induce") for c in calls)
    assert "learning: skipped" in said


@pytest.mark.unit
def test_the_script_refuses_an_unknown_argument(tmp_path: Path) -> None:
    code, calls, _ = drive(tmp_path, "--warm-only")

    assert code == 2
    assert calls == []


@pytest.mark.unit
def test_the_script_explains_itself(tmp_path: Path) -> None:
    done = subprocess.run(
        ["bash", str(SCRIPT), "--help"], capture_output=True, text=True, check=False
    )

    assert done.returncode == 0
    assert "Usage:" in done.stdout and "spends real money" in done.stdout


@pytest.mark.unit
def test_the_script_is_executable() -> None:
    assert os.access(SCRIPT, os.X_OK)


# --- The whole experiment, for real ------------------------------------------------------


@pytest.mark.live
def test_the_whole_sequence_runs_and_the_warm_library_fits_the_suite(tmp_path: Path) -> None:
    """Cold, learn, warm, on the real sites, one trial each. Spends real money.

    Run from a scratch folder, so the real ``runs/``, ``skills/`` and
    ``results/`` are untouched. Keys come from the project's ``.env``.
    """
    shutil.copytree(REPO / "benchmarks", tmp_path / "benchmarks")
    env = {**os.environ, **resolve_env(dotenv_path=REPO / ".env")}
    env.setdefault("WAA", str(Path(sys.executable).parent / "waa"))

    done = subprocess.run(
        ["bash", str(SCRIPT), "--trials", "1"], cwd=tmp_path, env=env, text=True, check=False
    )

    assert done.returncode == 0
    warm = tmp_path / "runs" / "eval-openrouter-warm"
    suite = load_suite(tmp_path / "benchmarks" / "suite.yaml")
    assert library_problem(warm / LIBRARY_DIR, suite, cold=False) is None, (
        "the warm arm's library is non-empty and matches the suite's sites"
    )
    assert (tmp_path / "results" / COLD_VS_WARM_FILE).exists()
