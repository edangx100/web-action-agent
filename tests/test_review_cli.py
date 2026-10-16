"""A person checking the machine's marks: ``waa judge --review``.

The machine's marks are the published results, and the only check on them is
a person labelling some of the same runs by hand. These tests drive the
review with scripted typing over three recorded runs and hold it to the three
promises that make it usable:

- **Every answer is kept.** Each label is one well-formed line in the labels
  file, written the moment it is given.
- **It carries on where it stopped.** Quitting halfway -- or the input simply
  running out -- and starting again begins at the first run with no label.
- **Nothing is labelled twice.** However many sessions it takes, each run
  gets one line.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from plain_text import plain
from typer.testing import CliRunner

from web_action_agent import cli
from web_action_agent.checkers import CheckResult
from web_action_agent.cli import EXIT_SUCCESS, EXIT_TASK_FAILURE, EXIT_USAGE, app
from web_action_agent.eval_harness import SWEEP_FILE, VERDICT_FILE, TrialVerdict
from web_action_agent.judge import FAIL, FAILURE_CATEGORIES, JUDGING_ERROR, PASS, JudgeVerdict
from web_action_agent.metrics import read_human_labels, write_results
from web_action_agent.review import (
    CATEGORY_ANSWERS,
    QUIT,
    HumanLabel,
    ReviewItem,
    append_label,
    find_reviewable,
)

pytestmark = pytest.mark.fixture

#: A real run, recorded from the practice website and committed with the tests.
TRACE = Path(__file__).parent / "fixtures" / "traces" / "search-and-open"

#: The practice website's benchmark suite, for "what counts as success".
SUITE = Path(__file__).parent / "fixtures" / "benchmark" / "suite.yaml"

#: The three runs: task, and the machine's mark on it.
MARKS = {
    "fixture-search": TrialVerdict(
        status=PASS,
        reason="every check passed",
        agent_outcome="success",
        checks=(CheckResult("answer_matches", True, "answer matched"),),
    ),
    "fixture-article": TrialVerdict(
        status=FAIL,
        reason="the answer did not match",
        agent_outcome="success",
        checks=(CheckResult("answer_matches", False, "expected 'X', got 'Y'"),),
        category="grounding",
        categorized_by=JudgeVerdict(FAIL, "it read the wrong line", "grounding"),
    ),
    "fixture-browse": TrialVerdict(
        status=FAIL,
        reason="the judge failed it",
        agent_outcome="success",
        judge=JudgeVerdict(FAIL, "it never opened the shelves page", "navigation"),
        category="navigation",
    ),
}

runner = CliRunner()


def make_sweep(
    root: Path,
    marks: dict[str, TrialVerdict] | None = None,
    *,
    name: str = "eval-openrouter-cold",
    trials: int = 1,
) -> Path:
    """Lay out a sweep of recorded runs, each with a machine mark.

    Args:
        root: Where the sweep folder goes.
        marks: Task id to mark; :data:`MARKS` when not given. Each task gets
            ``trials`` trials, each a copy of the committed run.
        name: The sweep folder's name.
        trials: Trials per task.

    Returns:
        Path: The sweep folder.
    """
    sweep = root / "runs" / name
    sweep.mkdir(parents=True)
    (sweep / SWEEP_FILE).write_text(json.dumps({"suite": "fixture-suite"}))
    for task, mark in (marks or MARKS).items():
        for number in range(1, trials + 1):
            folder = sweep / task / f"trial-{number}"
            shutil.copytree(TRACE, folder)
            (folder / VERDICT_FILE).write_text(json.dumps(mark.to_dict()))
    return sweep


def give_screenshot(folder: Path) -> Path:
    """Give a run's last step a screenshot on disk.

    The committed run is text-only, so it has none of its own.

    Args:
        folder: A run folder.

    Returns:
        Path: The screenshot's path.
    """
    picture = folder / "step-4.png"
    picture.write_bytes(b"\x89PNG\r\n\x1a\n")
    log = folder / "actions.jsonl"
    lines = log.read_text().splitlines()
    last = json.loads(lines[-1])
    last["observation"]["screenshot_on_disk"] = str(picture)
    lines[-1] = json.dumps(last)
    log.write_text("\n".join(lines) + "\n")
    return picture


def order_of(sweep: Path) -> list[str]:
    """Name the sweep's trials in the order the review shows them.

    Args:
        sweep: A sweep from :func:`make_sweep`.

    Returns:
        list[str]: Task ids, in review order.
    """
    return [item.task_id for item in find_reviewable(sweep)]


def invoke(tmp_path: Path, typed: str, *extra: str) -> Any:
    """Run ``waa judge --review`` with scripted typing.

    Args:
        tmp_path: The test's folder: it is the working directory, and holds
            the sweep and the labels file.
        typed: Everything the person types, one answer per line.
        *extra: More command-line arguments.

    Returns:
        SimpleNamespace: The exit code and the output as a person reads it.
    """
    result = runner.invoke(
        app,
        [
            "judge",
            "--review",
            str(tmp_path / "runs"),
            "--labels",
            str(tmp_path / "results" / "human_labels.jsonl"),
            "--suite",
            str(SUITE),
            *extra,
        ],
        input=typed,
        catch_exceptions=False,
    )
    return SimpleNamespace(exit_code=result.exit_code, output=plain(result.output))


def labels_in(tmp_path: Path) -> list[dict[str, Any]]:
    """Read every line of the labels file.

    Args:
        tmp_path: The test's folder.

    Returns:
        list[dict[str, Any]]: One record per line, in file order.
    """
    path = tmp_path / "results" / "human_labels.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines()]


def answers_for(sweep: Path, wanted: dict[str, str]) -> str:
    """Write the typing that gives each task its wanted answer, in review order.

    Args:
        sweep: The sweep.
        wanted: Task id to what is typed for it, such as ``"p"`` or
            ``"f\\ng"``.

    Returns:
        str: The scripted input.
    """
    return "".join(wanted[task] + "\n" for task in order_of(sweep) if task in wanted)


@pytest.fixture
def in_tmp(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Run the test from inside its own folder, as a person would from the project.

    Returns:
        Path: The test's folder.
    """
    monkeypatch.chdir(tmp_path)
    return tmp_path


# --- Every answer is kept --------------------------------------------------------


def test_three_runs_give_three_well_formed_records(in_tmp: Path) -> None:
    sweep = make_sweep(in_tmp)
    typed = answers_for(
        sweep, {"fixture-search": "p", "fixture-article": "f\ng", "fixture-browse": "f\np"}
    )

    result = invoke(in_tmp, typed)

    assert result.exit_code == EXIT_SUCCESS
    records = labels_in(in_tmp)
    assert len(records) == 3
    by_task = {r["task_id"]: r for r in records}
    assert set(by_task) == set(MARKS)
    for record in records:
        assert set(record) == set(HumanLabel.__dataclass_fields__)
        assert record["sweep"] == "eval-openrouter-cold"
        assert record["trial"] == 1
        # Stored relative to the labels file's folder, pointing at the trial.
        assert record["run"].startswith("../runs/eval-openrouter-cold/")
        assert (in_tmp / "results" / record["run"] / VERDICT_FILE).exists()
        assert record["labelled_at"]

    assert by_task["fixture-search"]["passed"] is True
    assert by_task["fixture-search"]["category"] is None
    assert by_task["fixture-article"]["passed"] is False
    assert by_task["fixture-article"]["category"] == "grounding"
    # The person's kind can differ from the machine's; both are kept.
    assert by_task["fixture-browse"]["category"] == "planning"
    assert by_task["fixture-browse"]["machine_category"] == "navigation"
    assert by_task["fixture-browse"]["machine_status"] == FAIL
    assert by_task["fixture-search"]["machine_status"] == PASS


def test_the_results_tables_read_the_labels_from_any_folder(
    in_tmp: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sweep = make_sweep(in_tmp)
    invoke(in_tmp, answers_for(sweep, {t: "p" for t in MARKS}))
    # Read back from somewhere else entirely, as a later command might be.
    monkeypatch.chdir(in_tmp / "runs")

    labels = read_human_labels(in_tmp / "results" / "human_labels.jsonl")

    assert labels == {str((sweep / t / "trial-1").resolve()): True for t in MARKS}


def test_the_labels_reach_the_agreement_line_of_the_table(in_tmp: Path) -> None:
    sweep = make_sweep(in_tmp)
    # The human passes everything; the machine passed one of the three.
    invoke(in_tmp, answers_for(sweep, {t: "p" for t in MARKS}))

    [table, _] = write_results([sweep], in_tmp / "results")

    assert "| Judge–human agreement | 33.3% (1 of 3 labelled trials) |" in table.read_text()


def test_the_screen_shows_what_a_person_needs_to_decide(in_tmp: Path) -> None:
    sweep = make_sweep(in_tmp)
    picture = give_screenshot(sweep / "fixture-article" / "trial-1")

    # Skip all three, so every screen is shown once.
    result = invoke(in_tmp, "s\ns\ns\n")

    assert "find the record about membership and open it" in result.output  # the task
    assert "Final answer:     Membership application" in result.output
    assert str(picture) in result.output
    assert "Final screenshot: (none recorded)" in result.output
    assert "the answer matches one of: 'Northgate Public Library'" in result.output
    assert "verdict:  FAIL -- the answer did not match" in result.output
    assert "check:    answer_matches failed -- expected 'X', got 'Y'" in result.output
    assert "judge:    fail -- it never opened the shelves page" in result.output
    assert "kind of failure: navigation" in result.output
    assert "kind of failure: (none: it passed)" in result.output


def test_the_evidence_comes_before_the_machines_mark(in_tmp: Path) -> None:
    make_sweep(in_tmp, {"fixture-browse": MARKS["fixture-browse"]})

    output = invoke(in_tmp, "q\n").output

    assert output.index("Final answer:") < output.index("The machine's mark:")


# --- It carries on where it stopped ----------------------------------------------


def test_a_second_session_starts_at_the_first_unlabelled_run(in_tmp: Path) -> None:
    sweep = make_sweep(in_tmp)
    first, second, third = order_of(sweep)

    invoke(in_tmp, "p\nq\n")
    assert [r["task_id"] for r in labels_in(in_tmp)] == [first]

    result = invoke(in_tmp, "p\np\n")

    # The labelled run is not shown again; the next one is shown first.
    assert f"/{first}/" not in result.output
    assert result.output.index(f"/{second}/") < result.output.index(f"/{third}/")
    assert [r["task_id"] for r in labels_in(in_tmp)] == [first, second, third]


def test_input_running_out_keeps_every_answer_given(in_tmp: Path) -> None:
    sweep = make_sweep(in_tmp)

    # Two answers, then nothing more: as when a session is closed mid-way.
    result = invoke(in_tmp, "p\nf\n")

    assert result.exit_code == EXIT_SUCCESS
    assert [r["task_id"] for r in labels_in(in_tmp)] == order_of(sweep)[:1]
    assert "Run the same command again" in result.output


def test_a_skipped_run_stays_unlabelled_and_comes_first_next_time(in_tmp: Path) -> None:
    sweep = make_sweep(in_tmp)
    first, second, _ = order_of(sweep)

    invoke(in_tmp, "s\np\nq\n")
    assert [r["task_id"] for r in labels_in(in_tmp)] == [second]

    result = invoke(in_tmp, "q\n")
    assert f"/{first}/" in result.output


def test_every_task_in_every_sweep_is_seen_once_before_any_twice(in_tmp: Path) -> None:
    for name in ("eval-openrouter-cold", "eval-openrouter-warm"):
        make_sweep(in_tmp, name=name, trials=2)

    items = find_reviewable(in_tmp / "runs")

    pairs = [(i.sweep, i.task_id) for i in items]
    # Six sweep-and-task pairs, two trials each: the first six are all different.
    assert len(set(pairs[:6])) == 6
    assert set(pairs[6:]) == set(pairs[:6])


def test_the_order_is_fixed_and_comes_from_the_trial_names(in_tmp: Path) -> None:
    for name in ("eval-openrouter-cold", "eval-openrouter-warm"):
        make_sweep(in_tmp, name=name)

    items = find_reviewable(in_tmp / "runs")

    # One trial per task, so a single round, in the order of the names' hashes.
    names = [i.name for i in items]
    assert names == sorted(names, key=lambda n: hashlib.sha256(n.encode()).hexdigest())
    assert names != sorted(names)
    assert names == [i.name for i in find_reviewable(in_tmp / "runs")]
    # The position does not depend on where the folder is on disk.
    moved = ReviewItem(Path("/elsewhere"), items[0].sweep, items[0].task_id, 1, items[0].verdict)
    assert moved.position == items[0].position


# --- Nothing is labelled twice ---------------------------------------------------


def test_running_again_when_all_are_labelled_adds_nothing(in_tmp: Path) -> None:
    sweep = make_sweep(in_tmp)
    invoke(in_tmp, answers_for(sweep, {t: "p" for t in MARKS}))

    result = invoke(in_tmp, "p\np\np\n")

    assert result.exit_code == EXIT_SUCCESS
    assert len(labels_in(in_tmp)) == 3
    assert "Your verdict?" not in result.output
    assert "3 of 3 trials now have a label" in result.output
    assert "every trial that can be labelled now has a label" in result.output


def test_many_short_sessions_never_duplicate_a_record(in_tmp: Path) -> None:
    make_sweep(in_tmp)

    for _ in range(5):
        invoke(in_tmp, "f\nn\nq\n")

    runs = [r["run"] for r in labels_in(in_tmp)]
    assert len(runs) == 3
    assert len(set(runs)) == 3


def test_a_label_written_from_another_directory_is_recognised(
    in_tmp: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sweep = make_sweep(in_tmp)
    first = order_of(sweep)[0]
    invoke(in_tmp, "p\nq\n")

    # The same sweep, reviewed from one folder further down.
    (in_tmp / "elsewhere").mkdir()
    monkeypatch.chdir(in_tmp / "elsewhere")
    result = invoke(in_tmp, "q\n")

    assert f"/{first}/" not in result.output


# --- Which runs are shown ---------------------------------------------------------


@pytest.mark.parametrize(
    "unfinished",
    [
        TrialVerdict(status=FAIL, reason="no judge yet", agent_outcome="step_cap"),
        TrialVerdict(status=JUDGING_ERROR, reason="the judge timed out", agent_outcome="success"),
    ],
    ids=["fail-without-a-kind", "judging-error"],
)
def test_a_mark_that_is_not_finished_is_not_shown(in_tmp: Path, unfinished: TrialVerdict) -> None:
    sweep = make_sweep(
        in_tmp, {"fixture-search": MARKS["fixture-search"], "fixture-home": unfinished}
    )

    assert order_of(sweep) == ["fixture-search"]


def test_an_unreadable_run_is_passed_over_with_a_reason(in_tmp: Path) -> None:
    sweep = make_sweep(in_tmp)
    (sweep / "fixture-article" / "trial-1" / "run.json").unlink()

    result = invoke(in_tmp, "p\np\np\n")

    assert "skipping eval-openrouter-cold/fixture-article/trial-1" in result.output
    assert {r["task_id"] for r in labels_in(in_tmp)} == {"fixture-search", "fixture-browse"}
    assert "1 could not be read" in result.output


def test_the_session_stops_at_the_target(in_tmp: Path) -> None:
    make_sweep(in_tmp)

    result = invoke(in_tmp, "p\np\np\n", "--target", "2")

    assert len(labels_in(in_tmp)) == 2
    assert "target of 2 labels reached" in result.output


# --- Typing ------------------------------------------------------------------------


def test_an_answer_that_is_not_a_choice_is_asked_again(in_tmp: Path) -> None:
    sweep = make_sweep(in_tmp)

    result = invoke(in_tmp, "maybe\nf\nx\nS\nq\n")

    assert result.output.count("please answer one of") == 2
    assert labels_in(in_tmp)[0]["category"] == "site_change"
    assert labels_in(in_tmp)[0]["task_id"] == order_of(sweep)[0]


def test_quitting_at_the_kind_question_labels_nothing(in_tmp: Path) -> None:
    make_sweep(in_tmp)

    invoke(in_tmp, "f\nq\n")

    assert labels_in(in_tmp) == []


def test_every_kind_of_failure_can_be_typed_and_nothing_else() -> None:
    kinds = {meaning for meaning in CATEGORY_ANSWERS.values() if meaning != QUIT}

    assert kinds == set(FAILURE_CATEGORIES)


# --- The labels file ----------------------------------------------------------------


def test_a_line_cut_off_by_a_crash_does_not_swallow_the_next_label(in_tmp: Path) -> None:
    sweep = make_sweep(in_tmp)
    path = in_tmp / "results" / "human_labels.jsonl"
    path.parent.mkdir()
    path.write_text('{"run": "runs/x", "pass')  # half a line, no newline

    item = find_reviewable(sweep)[0]
    append_label(path, HumanLabel.for_item(item, True, None, path.parent))

    assert read_human_labels(path) == {str(item.folder.resolve()): True}


def test_a_pass_never_carries_a_kind(in_tmp: Path) -> None:
    item = find_reviewable(make_sweep(in_tmp))[0]

    assert HumanLabel.for_item(item, True, "planning", in_tmp).category is None


# --- The command ---------------------------------------------------------------------


def test_a_folder_with_no_sweeps_is_a_usage_error(in_tmp: Path) -> None:
    (in_tmp / "runs").mkdir()

    result = invoke(in_tmp, "")

    assert result.exit_code == EXIT_USAGE
    assert "no benchmark sweeps" in result.output


def test_a_folder_that_does_not_exist_is_a_usage_error(in_tmp: Path) -> None:
    assert invoke(in_tmp, "").exit_code == EXIT_USAGE


def test_sweeps_with_nothing_finished_say_there_is_nothing_to_check(in_tmp: Path) -> None:
    make_sweep(
        in_tmp,
        {"fixture-home": TrialVerdict(status=FAIL, reason="x", agent_outcome="step_cap")},
    )

    result = invoke(in_tmp, "")

    assert result.exit_code == EXIT_TASK_FAILURE
    assert "nothing to check" in result.output


def test_review_reads_no_settings_and_calls_no_ai(
    in_tmp: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def refuse(**_: object) -> None:
        raise AssertionError("the review read the settings")

    # Reading settings is where keys are checked and models are chosen; a
    # review that never gets there cannot spend anything.
    monkeypatch.setattr(cli, "_startup", refuse)
    make_sweep(in_tmp)

    assert invoke(in_tmp, "q\n").exit_code == EXIT_SUCCESS


def test_a_missing_suite_only_leaves_out_what_counts_as_success(in_tmp: Path) -> None:
    make_sweep(in_tmp)

    result = runner.invoke(
        app,
        ["judge", "--review", str(in_tmp / "runs"), "--suite", str(in_tmp / "nope.yaml")],
        input="q\n",
    )

    assert result.exit_code == EXIT_SUCCESS
    assert "what counts as success will not be shown" in plain(result.output)
    assert "What counts as success" not in plain(result.output)
