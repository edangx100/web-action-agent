"""The commands that drive the skill library, and the rule that a run never does.

Six tasks built the parts: a format, durable element descriptions, the thing
that learns a recipe from a run, the check that proves one, the listing that
offers them, and the replay that runs one for free. Each was deliberately left
disconnected, because wiring half a library into a real run would have been
worse than leaving it out. This is where they go in together.

**The rule that gets its own test is that `waa run` never induces.** Learning is
a command somebody types, and that is not a stylistic preference. The project's
headline result is two benchmark passes compared against each other -- the same
tasks with an empty library and with a learned one -- and that comparison is
only meaningful if the library holds still while each pass runs. A run that
quietly wrote a skill would change the conditions of the experiment partway
through it, and nothing in the results table would show that it had.

**The order inside `induce` is the other thing worth pinning.** Learn, then
check against a live page with *different* values, then store. A candidate that
skipped the middle step would be a recipe believed on the strength of the single
run it came from, which is the failure the whole verification layer exists for.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from plain_text import plain
from typer.testing import CliRunner

from web_action_agent import cli
from web_action_agent.cli import EXIT_SUCCESS, EXIT_TASK_FAILURE, EXIT_USAGE, app
from web_action_agent.config import Config
from web_action_agent.model_client import ModelClient, load_profile
from web_action_agent.skill_schema import Skill
from web_action_agent.skill_store import SkillStore

TRACE = Path(__file__).resolve().parent / "fixtures" / "traces" / "search-and-open"
"""Path: The committed successful run every induction here learns from."""

#: What the model is told to answer when asked to turn the committed run into a
#: recipe. Scripted, because what a real model picks out is a question for the
#: benchmark; what matters here is that the command does the right things with
#: an answer, in the right order.
AN_INDUCTION: dict[str, Any] = {
    "name": "search_and_open_record",
    "description": "search the catalogue for a term and open the matching record",
    "parameters": [
        {
            "name": "query",
            "value": "membership",
            "description": "the subject to search the catalogue for",
            "required": True,
        }
    ],
    "verification": {
        "params": [{"name": "query", "value": "privacy"}],
        "expected_outcome": "a record page about privacy, opened from the search results",
    },
}

runner = CliRunner()
"""CliRunner: Runs the command line in-process and captures what it printed."""


def invoke(argv: list[str]) -> Any:
    """Run the command line with the given arguments, as a shell would.

    Args:
        argv: The arguments after the program name.

    Returns:
        SimpleNamespace: The exit code and the output as a person reads it.
    """
    result = runner.invoke(app, argv, catch_exceptions=False)
    return SimpleNamespace(exit_code=result.exit_code, output=plain(result.output))


class ScriptedEndpoint:
    """A model endpoint answering from a script, counting what it was asked.

    Attributes:
        calls: Every request made, so a test can prove a command spent nothing.
    """

    def __init__(self, *replies: dict[str, Any]) -> None:
        """Set up the script.

        Args:
            *replies: Objects to answer with, in order; the last repeats.
        """
        self._replies = [json.dumps(reply) for reply in replies] or ["{}"]
        self.calls: list[dict[str, Any]] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **request: Any) -> SimpleNamespace:
        """Answer one call.

        Args:
            **request: The call body, recorded.

        Returns:
            SimpleNamespace: A completion shaped like the SDK's.
        """
        self.calls.append(request)
        reply = self._replies[min(len(self.calls) - 1, len(self._replies) - 1)]
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=reply), finish_reason="stop")],
            usage=SimpleNamespace(prompt_tokens=900, completion_tokens=40),
        )


def a_workspace(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    playwright: Any,
    endpoint: ScriptedEndpoint,
) -> Path:
    """Point the command line at a temporary library, runs folder and endpoint.

    Three swaps, and only the first is pretending. The endpoint answers from a
    script because the point is the command's wiring rather than a model's
    judgement; the folders are temporary so no test writes into the working
    tree; and the browser borrows the suite's driver because Playwright allows
    one synchronous driver per thread and the suite already has one running.

    Args:
        monkeypatch: The patcher.
        tmp_path: The test's own directory.
        playwright: The suite's driver, lent to the command.
        endpoint: The scripted model.

    Returns:
        Path: The runs folder, for a test that wants to put traces in it.
    """
    starts_its_own = cli.BrowserProcess
    monkeypatch.setattr(
        cli, "BrowserProcess", lambda settings: starts_its_own(settings, playwright=playwright)
    )

    class Client:
        """Stands in for the model client's own constructor."""

        @staticmethod
        def from_config(config: Config, **kwargs: Any) -> Any:
            """Build a real client over the scripted endpoint.

            Args:
                config: The run's settings, which the local profile ignores.
                **kwargs: Ignored.

            Returns:
                ModelClient: A client that talks to the script.
            """
            return ModelClient(load_profile("local"), transport=endpoint, env={})

    monkeypatch.setattr(cli, "ModelClient", Client)
    monkeypatch.setattr(cli, "SKILLS_ROOT", tmp_path / "skills")
    monkeypatch.setattr(cli, "RUNS_ROOT", tmp_path / "runs")
    (tmp_path / "runs").mkdir()
    return tmp_path / "runs"


def put_run(runs: Path, name: str, *, successful: bool = True) -> Path:
    """Copy the committed run into a runs folder, optionally spoiling it.

    Args:
        runs: Where the run folders live.
        name: What to call this copy.
        successful: False to rewrite its verdict as a failure, so the folder is
            a real trace of a run that did not work.

    Returns:
        Path: The run folder.
    """
    folder = runs / name
    shutil.copytree(TRACE, folder)
    if not successful:
        verdict = json.loads((folder / "outcome.json").read_text(encoding="utf-8"))
        verdict["outcome"] = {**verdict["outcome"], "ok": False, "category": "step_cap"}
        (folder / "outcome.json").write_text(json.dumps(verdict), encoding="utf-8")
    return folder


def a_library(tmp_path: Path, *skills: Skill) -> SkillStore:
    """Build a library holding the given skills.

    Args:
        tmp_path: The test's own directory.
        *skills: What to put in it.

    Returns:
        SkillStore: The library the commands will read.
    """
    store = SkillStore(tmp_path / "skills")
    for skill in skills:
        store.save(skill)
    return store


def a_skill(name: str = "search_catalogue", **overrides: Any) -> Skill:
    """Build one stored skill.

    Args:
        name: What it is called.
        **overrides: Fields to replace.

    Returns:
        Skill: The skill.
    """
    skill = {
        "name": name,
        "description": "search the catalogue and open the matching record",
        "domain": "northgate.example",
        "params": [{"name": "query", "description": "what to search for", "example": "membership"}],
        "steps": [
            {
                "action": "fill",
                "selector": {"name": "Search terms", "role": "searchbox"},
                "text": "{query}",
            },
            {"action": "press", "key": "Enter"},
        ],
        "verification": {"params": {"query": "privacy"}, "expected_outcome": "a record page"},
        "induced_from": "run-2b110b92b259",
    }
    return Skill.from_dict({**skill, **overrides})


# --- The rule: a run never writes a skill -----------------------------------


@pytest.mark.unit
def test_an_ordinary_run_never_induces_a_skill(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Learning is a command somebody types, and that is not a style choice.

    The project's headline result is two benchmark passes compared against each
    other — the same tasks with an empty library and with a learned one. That
    comparison only means anything if the library holds still while each pass
    runs. A run that quietly wrote a skill would change the conditions of the
    experiment halfway through it, and nothing in the results table would show
    that it had.

    Asserted by watching the induction machinery rather than the folder: a
    library that happened to stay empty because induction failed would pass a
    "no new files" check while the rule was being broken.
    """
    induced: list[Any] = []
    monkeypatch.setattr(cli, "SKILLS_ROOT", tmp_path / "skills")
    monkeypatch.setattr(
        cli.Inducer, "induce", lambda self, run: induced.append(run) or pytest.fail("induced")
    )

    invoke(["run", "find the opening hours"])

    assert induced == [], "a run may read the library and must never add to it"
    assert not (tmp_path / "skills").exists()


@pytest.mark.unit
def test_induce_refuses_a_request_it_cannot_honour() -> None:
    """One run folder or --all, never both and never neither.

    Both would be asking for two different things; neither asks for nothing.
    Refused before any work starts, because a command that guessed which was
    meant would guess wrong in front of a model call.
    """
    assert invoke(["induce"]).exit_code == EXIT_USAGE
    assert invoke(["induce", "runs/x", "--all", "runs"]).exit_code == EXIT_USAGE


# --- Learning from a directory of runs --------------------------------------


@pytest.mark.fixture
def test_induce_all_learns_only_from_the_runs_that_worked(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, playwright: Any, fixture_site: Any
) -> None:
    """Three run folders, one of which failed, and only two candidates come out.

    A recipe made from a failure is a recipe for failing, so a failed run is
    skipped rather than attempted. The count is reported because somebody
    pointing this at a directory of two hundred runs needs to know what became
    of them without reading two hundred lines.
    """
    # No judge is configured below, so every call the command makes is an
    # induction -- and the script's last entry repeats, covering both runs.
    endpoint = ScriptedEndpoint(AN_INDUCTION)
    runs = a_workspace(monkeypatch, tmp_path, playwright, endpoint)
    for name in ("first", "second"):
        _point_at(put_run(runs, name), fixture_site)
    put_run(runs, "failed-one", successful=False)
    monkeypatch.setattr(cli, "resolve_env", dict)

    result = invoke(["induce", "--all", str(runs)])

    assert "2 induced, 2 verified" in result.output
    assert "nothing to learn" in result.output, "and it says why the third was skipped"
    assert "step_cap" in result.output, "naming the verdict that made it ineligible"
    assert result.exit_code == EXIT_SUCCESS


@pytest.mark.fixture
def test_a_candidate_is_checked_against_a_live_page_before_it_is_stored(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, playwright: Any, fixture_site: Any
) -> None:
    """Learn, check with different values, store — and in that order.

    The committed run searched for "membership"; the check replays with
    "privacy" against the real practice site. A candidate stored without that
    step would be a recipe believed on the strength of the one run it came
    from.
    """
    endpoint = ScriptedEndpoint(AN_INDUCTION, {"passed": True, "reason": "a privacy record"})
    runs = a_workspace(monkeypatch, tmp_path, playwright, endpoint)
    folder = put_run(runs, "first")
    _point_at(folder, fixture_site)
    # A judge from another family than the planner's, or the command refuses
    # to start. The model behind it is the script either way.
    monkeypatch.setattr(cli, "resolve_env", lambda: {"WAA_JUDGE_PROFILE": "openrouter-judge"})

    result = invoke(["induce", str(folder)])

    assert "1 induced, 1 verified, 0 discarded" in result.output
    stored = SkillStore(tmp_path / "skills").find("search_and_open_record")
    assert stored is not None, result.output
    assert stored.verification.params == {"query": "privacy"}, "checked with other values"
    assert len(endpoint.calls) == 2, "one call to learn it, one to judge the page it reached"


@pytest.mark.fixture
def test_a_candidate_that_fails_its_check_is_discarded_with_a_reason(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, playwright: Any, fixture_site: Any
) -> None:
    """Nothing unverified reaches the library, and the discard says why.

    The project reports a discard rate. A rate with no reasons behind it says
    only that something is wrong somewhere.
    """
    endpoint = ScriptedEndpoint(AN_INDUCTION, {"passed": False, "reason": "this is the home page"})
    runs = a_workspace(monkeypatch, tmp_path, playwright, endpoint)
    folder = put_run(runs, "first")
    _point_at(folder, fixture_site)
    # A judge from another family than the planner's, or the command refuses
    # to start. The model behind it is the script either way.
    monkeypatch.setattr(cli, "resolve_env", lambda: {"WAA_JUDGE_PROFILE": "openrouter-judge"})

    result = invoke(["induce", str(folder)])

    assert "1 induced, 0 verified, 1 discarded" in result.output
    assert "this is the home page" in result.output
    assert SkillStore(tmp_path / "skills").names() == (), "and nothing was stored"
    assert result.exit_code == EXIT_TASK_FAILURE, "a pass that stored nothing is worth noticing"


@pytest.mark.fixture
def test_induction_says_whether_a_judge_was_involved(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, playwright: Any, fixture_site: Any
) -> None:
    """A library checked without a judge must not look like one checked with.

    Running the mechanical layer alone is allowed — it is a stated position,
    with retirement as the quality filter instead — but it is a weaker check,
    and which one happened should not have to be inferred later.
    """
    endpoint = ScriptedEndpoint(AN_INDUCTION)
    runs = a_workspace(monkeypatch, tmp_path, playwright, endpoint)
    folder = put_run(runs, "first")
    _point_at(folder, fixture_site)
    monkeypatch.setattr(cli, "resolve_env", dict)

    result = invoke(["induce", str(folder)])

    assert "no judge configured" in result.output
    assert len(endpoint.calls) == 1, "and no judge call was made"


def _point_at(folder: Path, site: Any) -> None:
    """Rewrite a committed run's address to the practice site that is running.

    The committed trace records a stable fake host, so that the folder reads
    the same on every machine. Checking a candidate means replaying it against
    a page that actually exists, so for these tests the address is pointed back
    at the live site.

    Args:
        folder: The copied run folder.
        site: The running practice site.
    """
    for name in ("run.json", "config.json", "actions.jsonl"):
        path = folder / name
        path.write_text(
            path.read_text(encoding="utf-8").replace("http://northgate.example/", site.base_url),
            encoding="utf-8",
        )


# --- Looking at what the agent has learnt -----------------------------------


@pytest.mark.unit
def test_an_empty_library_says_how_to_fill_it(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The ordinary state of a fresh install, and not a failure.

    The message names the command that would change it, because "no skills yet"
    with no next step is a dead end for somebody who has just installed this.
    """
    monkeypatch.setattr(cli, "SKILLS_ROOT", tmp_path / "skills")
    monkeypatch.setattr(cli, "RUNS_ROOT", tmp_path / "runs")

    result = invoke(["skills", "list"])

    assert result.exit_code == EXIT_SUCCESS
    assert "no skills yet" in result.output
    assert "waa induce --all" in result.output


@pytest.mark.unit
def test_the_listing_shows_retired_skills_and_marks_them(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Hiding them would leave somebody wondering where a skill had gone.

    They are not offered to the agent any more, and they are still part of what
    the library knows.
    """
    a_library(
        tmp_path,
        a_skill(),
        a_skill(
            "old_search",
            active=False,
            retired_reason="2 of 9 replays worked",
            retired_at="2026-10-01T09:00:00.000Z",
            stats={"uses": 9, "successes": 2},
        ),
    )
    monkeypatch.setattr(cli, "SKILLS_ROOT", tmp_path / "skills")

    result = invoke(["skills", "list"])

    assert "search_catalogue" in result.output
    assert "old_search" in result.output and "retired" in result.output
    assert "1 in service, 1 retired" in result.output


@pytest.mark.unit
def test_showing_a_skill_explains_what_it_does_in_words(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The file is readable and right there; this answers a different question.

    "What does this thing actually do" is answered faster by a numbered list of
    steps in words than by a nested document — and the check it passed is shown
    too, because that is the reason it is trusted at all.
    """
    a_library(tmp_path, a_skill())
    monkeypatch.setattr(cli, "SKILLS_ROOT", tmp_path / "skills")

    result = invoke(["skills", "show", "search_catalogue"])

    assert result.exit_code == EXIT_SUCCESS
    assert "northgate.example" in result.output, "which site it is for"
    assert "query" in result.output and "what to search for" in result.output, "what it needs"
    assert '1. fill searchbox "Search terms" with "{query}"' in result.output, "and what it does"
    assert "2. press Enter" in result.output
    assert "checked" in result.output and "query='privacy'" in result.output
    assert "learnt   from run run-2b110b92b259" in result.output, "traceable to its evidence"


@pytest.mark.unit
def test_showing_a_skill_that_is_not_there_says_so(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """And points at the command that would list what is."""
    a_library(tmp_path, a_skill())
    monkeypatch.setattr(cli, "SKILLS_ROOT", tmp_path / "skills")

    result = invoke(["skills", "show", "no_such_skill"])

    assert result.exit_code == EXIT_USAGE
    assert "no skill named" in result.output
    assert "waa skills list" in result.output


# --- Retiring one by hand ---------------------------------------------------


@pytest.mark.unit
def test_a_person_can_retire_a_skill_and_say_why(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """For the reasons a success rate cannot see.

    A skill that works perfectly and does the wrong thing has a flawless record
    and should still go. The reason is recorded exactly as given, because a
    human explanation is almost always better than anything the code could
    write.
    """
    a_library(tmp_path, a_skill())
    monkeypatch.setattr(cli, "SKILLS_ROOT", tmp_path / "skills")

    result = invoke(
        ["skills", "retire", "search_catalogue", "--reason", "it opens the wrong record"]
    )

    assert result.exit_code == EXIT_SUCCESS
    stored = SkillStore(tmp_path / "skills").find("search_catalogue")
    assert stored is not None
    assert stored.active is False
    assert stored.retired_reason == "it opens the wrong record"
    assert "kept at" in result.output, "and it says the file is still there"


@pytest.mark.unit
def test_retiring_a_skill_that_is_not_there_is_a_usage_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Nothing was changed, so the fix is to the command rather than to the run."""
    monkeypatch.setattr(cli, "SKILLS_ROOT", tmp_path / "skills")

    result = invoke(["skills", "retire", "never_existed"])

    assert result.exit_code == EXIT_USAGE
    assert "no such skill" in result.output
