"""The committed example run: complete, readable, and safe to publish.

``runs/example/`` is the one run folder this repository commits, so that
anyone can see what a run leaves behind without running anything. It is only
worth committing if it is the real thing, so these tests hold it to what every
run folder promises:

- **It reads back.** The trace reader rebuilds the whole run from the folder
  alone, and what it rebuilds agrees with the run's own summary: the same
  number of steps, the same final action and answer, the same cost.
- **Nothing is missing.** The action log, the cost log, the outcome, the
  verdict, the settings, and a screenshot for every step are all there, and
  every action-log line passes the schema.
- **It is fit to publish.** Every file in it survives the ignore rules, its
  paths work from a fresh clone, and nothing in it names this machine.

These read the committed files directly, so they need no browser, network or
model.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import pytest
from test_repo_hygiene import is_ignored

from web_action_agent.eval_harness import VERDICT_FILE, TrialVerdict
from web_action_agent.trace import (
    ACTION_LOG,
    CONFIG_FILE,
    COST_LOG,
    OUTCOME_FILE,
    RUN_FILE,
    SCREENSHOT_DIR,
    read_action_log,
    validate_record,
)
from web_action_agent.trace_reader import ReplayedRun, cost_log_agrees_with_steps, read_run

REPO_ROOT = Path(__file__).resolve().parent.parent
"""Path: The repository's top folder, where recorded paths are relative to."""

EXAMPLE = REPO_ROOT / "runs" / "example"
"""Path: The committed example run."""

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
"""bytes: The eight bytes every PNG file starts with."""


@pytest.fixture(scope="module")
def example() -> ReplayedRun:
    """The example run, as the trace reader rebuilds it."""
    return read_run(EXAMPLE)


def summary() -> dict[str, Any]:
    """The run's own summary of itself, written when it ended.

    Returns:
        dict[str, Any]: The parsed ``outcome.json``.
    """
    return json.loads((EXAMPLE / OUTCOME_FILE).read_text(encoding="utf-8"))


# --- It reads back -----------------------------------------------------------------------


@pytest.mark.unit
def test_the_reader_rebuilds_a_finished_run(example: ReplayedRun) -> None:
    assert example.finished, "the example reached a verdict"
    assert example.outcome is not None and example.outcome.ok
    assert example.run_id == summary()["run_id"]


@pytest.mark.unit
def test_the_step_count_and_answer_match_the_runs_summary(example: ReplayedRun) -> None:
    recorded = summary()

    assert len(example.steps) == recorded["steps"]
    assert example.outcome is not None
    assert example.outcome.answer == recorded["outcome"]["answer"]


@pytest.mark.unit
def test_the_action_sequence_matches_the_log(example: ReplayedRun) -> None:
    # The reader's actions, step by step, against the log's raw lines: every
    # click on the way, then the ending the summary names.
    raw = [record["action"] for record in read_action_log(EXAMPLE)]

    assert [step.action.to_dict() for step in example.steps] == raw
    assert raw[-1] == summary()["final_action"]
    assert [a["action"] for a in raw] == ["click", "click", "click", "click", "done"]


@pytest.mark.unit
def test_the_cost_adds_up_to_the_runs_summary(example: ReplayedRun) -> None:
    # The reader adds the cost up again from the calls, rather than repeating
    # the total the run wrote down, so agreeing here means the two match.
    recorded = summary()["cost"]
    cost = example.cost

    assert cost.calls == recorded["calls"]
    assert cost.prompt_tokens == recorded["prompt_tokens"]
    assert cost.completion_tokens == recorded["completion_tokens"]
    assert cost.cost_usd == pytest.approx(recorded["cost_usd"], abs=1e-6)


@pytest.mark.unit
def test_the_cost_log_agrees_with_the_steps(example: ReplayedRun) -> None:
    assert cost_log_agrees_with_steps(example)


# --- Nothing is missing ------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize(
    "name", [RUN_FILE, CONFIG_FILE, ACTION_LOG, COST_LOG, OUTCOME_FILE, VERDICT_FILE]
)
def test_every_record_file_is_there(name: str) -> None:
    assert (EXAMPLE / name).is_file(), f"runs/example/{name} is missing"


@pytest.mark.unit
def test_every_action_log_line_passes_the_schema() -> None:
    records = read_action_log(EXAMPLE)

    assert records, "the action log has steps in it"
    for record in records:
        validate_record(record)  # Raises, naming the field, if one is wrong.


@pytest.mark.unit
def test_every_step_has_its_screenshot(example: ReplayedRun) -> None:
    # Each step records where its picture is, relative to the repository's top
    # folder, so the path has to work from a fresh clone as written.
    pictures = [REPO_ROOT / path for path in example.screenshots]

    assert len(pictures) == len(example.steps), "one picture per step"
    for picture in pictures:
        assert picture.parent == EXAMPLE / SCREENSHOT_DIR, f"{picture} is outside the example"
        assert picture.read_bytes().startswith(PNG_SIGNATURE), f"{picture.name} is not a PNG"
    # And no stray picture that no step owns.
    assert sorted(p.name for p in pictures) == sorted(
        p.name for p in (EXAMPLE / SCREENSHOT_DIR).iterdir()
    )


@pytest.mark.unit
def test_the_verdict_is_a_pass_on_every_check() -> None:
    data = json.loads((EXAMPLE / VERDICT_FILE).read_text(encoding="utf-8"))
    verdict = TrialVerdict.from_dict(data)

    assert verdict.status == "pass"
    assert data["checks"] and all(check["passed"] for check in data["checks"])


# --- It is fit to publish ----------------------------------------------------------------


@pytest.mark.unit
def test_every_file_in_the_example_is_committed_not_ignored() -> None:
    files = [p for p in EXAMPLE.rglob("*") if p.is_file()]

    assert files
    for path in files:
        relative = path.relative_to(REPO_ROOT).as_posix()
        assert not is_ignored(relative), f"{relative} would be left out of the commit"


@pytest.mark.unit
def test_the_example_is_not_a_cpf_task(example: ReplayedRun) -> None:
    # The screenshots are copies of the site's pages. CPF's terms of use are
    # the only ones among the benchmark's sites that speak to robots copying
    # its pages, so the example comes from a different site.
    host = urlparse(example.start_url).hostname or ""

    assert not host.endswith("cpf.gov.sg"), f"the example starts on {host}"


@pytest.mark.unit
def test_nothing_in_the_example_names_this_machine() -> None:
    # Absolute paths would give away whose computer the run was made on, and
    # would not work anywhere else anyway.
    for path in EXAMPLE.rglob("*"):
        if path.suffix not in {".json", ".jsonl", ".md"}:
            continue
        text = path.read_text(encoding="utf-8")
        for marker in ("/home/", "/Users/", "C:\\"):
            assert marker not in text, f"{path.name} contains {marker!r}"
