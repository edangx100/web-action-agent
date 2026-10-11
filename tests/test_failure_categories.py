"""Every failure says what kind it was, or there is no results table.

The four kinds -- grounding, planning, navigation, site change -- are what
turn "the agent failed 9 times" into "it clicked the wrong thing 5 times and
got lost 4 times", which is the part of the results someone can act on. So a
failure without one is not a small gap: it is a row the failure analysis
silently does not cover. These tests hold the rule at both ends.

**Where categories come from.** A fail the judge decided carries the judge's
category. A fail decided by something else -- the agent ran out of steps, a
machine check caught a wrong answer -- is categorised by asking the judge the
narrower question, with the fail already settled, so it cannot be argued back
into a pass. A fail that cannot be categorised yet is an unfinished mark, which
a restart with a judge completes without running the trial again.

**Where the rule is enforced.** The results tables refuse to be written while
any failure is uncategorised, and say which and how to fix it -- loudly, rather
than printing a table with a blank cell that nobody chases.
"""

from __future__ import annotations

import json
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import pytest
from test_eval_harness import (
    FakeJudge,
    FakeRunner,
    fake_harness,
    finished_run,
    judged_task,
    stand_in_suite,
)
from test_judge import a_judge, a_task, the_run
from test_metrics import make_sweep

from web_action_agent.eval_harness import (
    DONE,
    REMARKED,
    UNMARKED,
    VERDICT_FILE,
    mark_trial,
    trial_state,
)
from web_action_agent.judge import (
    FAIL,
    FAILURE_CATEGORIES,
    JUDGING_ERROR,
    PASS,
    JudgeVerdict,
    category_schema,
)
from web_action_agent.metrics import MetricsError, check_categories, load_sweep, write_results


def uncategorise(sweep: Path, task: str = "charlie", trial: int = 1) -> Path:
    """Remove the category from one failed trial's mark.

    Args:
        sweep: A sweep from :func:`test_metrics.make_sweep`.
        task: The task.
        trial: The trial, which must be a fail.

    Returns:
        Path: The trial's folder.
    """
    folder = sweep / task / f"trial-{trial}"
    verdict = json.loads((folder / VERDICT_FILE).read_text())
    assert verdict["status"] == FAIL
    verdict["category"] = None
    (folder / VERDICT_FILE).write_text(json.dumps(verdict))
    return folder


# --- The tables refuse -----------------------------------------------------------


@pytest.mark.unit
def test_an_uncategorised_failure_stops_the_tables_loudly(tmp_path: Path) -> None:
    sweep = make_sweep(tmp_path / "runs")
    folder = uncategorise(sweep)
    results = tmp_path / "results"

    with pytest.raises(MetricsError) as raised:
        write_results([sweep], results)

    message = str(raised.value)
    assert str(folder) in message, "it names the trial"
    assert "no category" in message
    assert "WAA_JUDGE_PROFILE" in message, "and says how to fix it"
    assert not results.exists(), "no table was written, not even a partial one"


@pytest.mark.unit
def test_nothing_is_written_if_any_sweep_has_the_problem(tmp_path: Path) -> None:
    # Two sweeps, the second with the problem: the first's table must not be
    # updated either, or the folder would hold a mix of fresh and stale tables.
    good = make_sweep(tmp_path / "runs", "eval-a")
    bad = make_sweep(tmp_path / "runs", "eval-b")
    uncategorise(bad)
    with pytest.raises(MetricsError):
        write_results([good, bad], tmp_path / "results")
    assert not (tmp_path / "results").exists()


@pytest.mark.unit
def test_every_problem_is_listed_not_just_the_first(tmp_path: Path) -> None:
    sweep = make_sweep(tmp_path / "runs")
    uncategorise(sweep, "charlie", 1)
    uncategorise(sweep, "bravo", 2)
    with pytest.raises(MetricsError, match=r"^2 trial\(s\)"):
        check_categories([load_sweep(sweep)])


@pytest.mark.unit
@pytest.mark.parametrize("bad", ["luck", "none", "Grounding", ""])
def test_a_category_that_is_not_one_of_the_four_is_refused(tmp_path: Path, bad: str) -> None:
    sweep = make_sweep(tmp_path / "runs")
    folder = sweep / "charlie" / "trial-1"
    verdict = json.loads((folder / VERDICT_FILE).read_text())
    verdict["category"] = bad
    (folder / VERDICT_FILE).write_text(json.dumps(verdict))
    with pytest.raises(MetricsError, match="category"):
        write_results([sweep], tmp_path / "results")


@pytest.mark.unit
def test_a_pass_carrying_a_category_is_refused(tmp_path: Path) -> None:
    sweep = make_sweep(tmp_path / "runs")
    folder = sweep / "alpha" / "trial-1"
    verdict = json.loads((folder / VERDICT_FILE).read_text())
    verdict["category"] = "planning"
    (folder / VERDICT_FILE).write_text(json.dumps(verdict))
    with pytest.raises(MetricsError, match="a pass that carries"):
        write_results([sweep], tmp_path / "results")


@pytest.mark.unit
def test_a_fully_categorised_sweep_writes_its_tables(tmp_path: Path) -> None:
    written = write_results([make_sweep(tmp_path / "runs")], tmp_path / "results")
    text = written[0].read_text()
    assert "uncategorized" not in text, "there is no column for a gap that cannot exist"


# --- Marking gives every failure a category ------------------------------------------


@pytest.mark.unit
def test_a_failure_the_judge_did_not_decide_is_categorised_by_asking(tmp_path: Path) -> None:
    finished_run(tmp_path / "r", agent_ok=False)  # failed: the agent ran out of steps
    judge = FakeJudge(JudgeVerdict(FAIL, justification="Lost in menus.", category="navigation"))
    verdict = mark_trial(stand_in_suite().tasks[0], tmp_path / "r", judge)  # type: ignore[arg-type]
    assert verdict.category == "navigation"
    assert verdict.complete


@pytest.mark.unit
def test_when_the_judge_says_pass_but_a_check_failed_the_check_decides(tmp_path: Path) -> None:
    # The judge thought it passed; a machine check disagreed. The check is
    # the stricter authority, so the run fails -- and still needs a category,
    # which the judge gives when asked the narrower question.
    finished_run(tmp_path / "r")
    task = judged_task()
    task = task.model_copy(
        update={
            "verdicts": (
                task.verdicts[0].model_copy(update={"accepted": ("Something else",)}),
                *task.verdicts[1:],
            )
        }
    )

    class SaysPassThenCategorises:
        def judge(self, task: Any, run: Any) -> JudgeVerdict:
            return JudgeVerdict(PASS, justification="Looks right.")

        def categorize(self, task: Any, run: Any, reason: str) -> JudgeVerdict:
            assert "answer_matches failed" in reason, "told why it failed"
            return JudgeVerdict(FAIL, justification="Read the wrong row.", category="grounding")

    verdict = mark_trial(task, tmp_path / "r", SaysPassThenCategorises())  # type: ignore[arg-type]
    assert verdict.status == FAIL
    assert verdict.category == "grounding"
    assert verdict.judge is not None and verdict.judge.outcome == PASS, "its pass is kept, too"


@pytest.mark.unit
def test_with_no_judge_a_failure_is_an_unfinished_mark(tmp_path: Path) -> None:
    finished_run(tmp_path / "r", agent_ok=False)
    verdict = mark_trial(stand_in_suite().tasks[0], tmp_path / "r", None)
    assert verdict.status == FAIL and verdict.category is None
    assert not verdict.complete
    assert "no judge is configured" in verdict.reason


@pytest.mark.unit
def test_a_judge_that_cannot_categorise_leaves_the_mark_unfinished(tmp_path: Path) -> None:
    finished_run(tmp_path / "r", agent_ok=False)
    judge = FakeJudge(JudgeVerdict(JUDGING_ERROR, error="service down"))
    verdict = mark_trial(stand_in_suite().tasks[0], tmp_path / "r", judge)  # type: ignore[arg-type]
    assert verdict.status == FAIL, "still a fail: that was decided without the judge"
    assert not verdict.complete


@pytest.mark.unit
def test_a_restart_with_a_judge_categorises_without_running_again(tmp_path: Path) -> None:
    sweep = tmp_path / "sweep"
    fake_harness(sweep, FakeRunner(agent_ok=False), trials=1).run()  # no judge
    folder = sweep / "fixture-home" / "trial-1"
    assert trial_state(folder) == UNMARKED, "a fail with no category is not finished"

    again = FakeRunner()
    judge = FakeJudge(JudgeVerdict(FAIL, justification="Gave up.", category="planning"))
    report = fake_harness(sweep, again, trials=1, judge=judge).run()

    assert again.ran == [], "nothing ran again"
    assert report.count(REMARKED) == 5
    assert trial_state(folder) == DONE
    assert json.loads((folder / VERDICT_FILE).read_text())["category"] == "planning"


# --- The judge's narrower question ---------------------------------------------------


@pytest.mark.mock_llm
def test_categorize_tells_the_judge_the_fail_is_settled() -> None:
    judge, endpoint = a_judge({"category": "planning", "justification": "Stopped early."})
    verdict = judge.categorize(a_task(), the_run(), "the agent did not finish the task (step_cap)")
    # The case itself, not the standing instructions -- those describe this
    # situation too, so a check of everything sent would pass even if the
    # case never said it.
    content = endpoint.request["messages"][1]["content"]
    case = content if isinstance(content, str) else content[0]["text"]
    assert "Already decided: this run failed" in case
    assert "the agent did not finish the task (step_cap)" in case
    assert "Do not decide pass or fail again" in case
    assert endpoint.request["response_format"]["json_schema"]["schema"] == category_schema()
    assert (verdict.outcome, verdict.category) == (FAIL, "planning")


@pytest.mark.unit
def test_the_categorize_answer_cannot_say_pass_or_none() -> None:
    schema = category_schema()
    assert "passed" not in schema["properties"]
    assert schema["properties"]["category"]["enum"] == list(FAILURE_CATEGORIES)
    assert "none" not in schema["properties"]["category"]["enum"]


@pytest.mark.mock_llm
@pytest.mark.parametrize(
    "reply",
    [
        {"category": "none", "justification": "x"},
        {"category": "luck", "justification": "x"},
        {"justification": "x"},
        {"category": "planning", "justification": ""},
        "not json at all",
    ],
)
def test_a_bad_category_answer_is_a_judging_error(reply: Any) -> None:
    judge, _ = a_judge(reply)
    verdict = judge.categorize(a_task(), the_run(), "a check failed")
    assert verdict.outcome == JUDGING_ERROR
    assert verdict.category is None


@pytest.mark.unit
def test_the_harness_counts_both_judge_calls_in_the_budget(tmp_path: Path) -> None:
    # A judge-marked task where the judge says pass but a check fails: two
    # paid calls, the verdict and the category. Both must reach the budget.
    from test_budget import a_record

    from web_action_agent.budget import sweep_spend
    from web_action_agent.config import Config
    from web_action_agent.eval_harness import Harness

    class TwoCalls:
        def judge(self, task: Any, run: Any) -> JudgeVerdict:
            return JudgeVerdict(PASS, justification="x", record=a_record(0.01))

        def categorize(self, task: Any, run: Any, reason: str) -> JudgeVerdict:
            return JudgeVerdict(
                FAIL, justification="y", category="grounding", record=a_record(0.02)
            )

    suite = stand_in_suite()
    first = judged_task()
    first = first.model_copy(
        update={
            "verdicts": (
                first.verdicts[0].model_copy(update={"accepted": ("Something else",)}),
                *first.verdicts[1:],
            )
        }
    )
    one = type(suite)(name=suite.name, path=suite.path, sites=suite.sites, tasks=(first,))
    sweep = tmp_path / "sweep"
    Harness(
        Config(),
        one,
        sweep,
        trials=1,
        cold=True,
        workers=1,
        skills_root=tmp_path / "none",
        judge=TwoCalls(),  # type: ignore[arg-type]
        client_factory=lambda cfg: None,  # type: ignore[arg-type,return-value]
        browser_factory=lambda cfg: nullcontext(),
        runner=lambda trial, config, folder, browser: finished_run(folder),
        log=lambda line: None,
    ).run()
    assert sweep_spend(sweep).known_usd == pytest.approx(0.03)
