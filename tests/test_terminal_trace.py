"""What the terminal shows while a run happens, pinned to a committed transcript.

Most tests here assert one fact each. This file's main test asserts *all* of
them at once, by comparing a whole run's output against a file checked into the
repository. That is deliberate, and it is the only place in the suite that works
this way.

**Why a golden transcript.** The terminal trace is one artefact, not a list of
independent facts. A test per fact -- the step number is there, the phase is
there, the element is named -- passes happily while the thing as a whole reads
badly: columns that no longer line up, a stutter, a blank line in the wrong
place, an answer printed twice. None of that is a property you can assert; all
of it is obvious the moment you look at the file. So the file is the assertion,
and a diff on it is the review.

**Why it needs normalising.** Three things in a real run differ every time and
none of them is what the test is about: the clock, the fixture site's port, and
the folder named after the moment the run started. :func:`normalise` replaces
each with a fixed stand-in, and the list of what it replaces is the list of
things this test deliberately does not check. Everything else -- every word,
every column, every blank line -- is compared exactly.

**Why the answer appears in it.** The golden file is what a person sees on
screen, which is both channels interleaved the way a terminal shows them -- so
the answer is in there, on the last line. That the answer is *also* alone on
standard output, with the whole trace on standard error, cannot be read off a
merged transcript and is pinned in ``test_cli_run.py`` instead.
"""

from __future__ import annotations

import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from plain_text import plain
from typer.testing import CliRunner

from web_action_agent import cli
from web_action_agent.actions import parse_action
from web_action_agent.cli import EXIT_SUCCESS, EXIT_TASK_FAILURE, app
from web_action_agent.config import Config
from web_action_agent.controller import StepReport
from web_action_agent.model_client import ModelClient, load_profile
from web_action_agent.observation import Observation
from web_action_agent.observer import Box, Element
from web_action_agent.reporting import (
    ACT,
    DONE,
    PHASES,
    PLAN,
    SKILL,
    TerminalTrace,
    describe_action,
    phase_of,
    step_line,
)
from web_action_agent.trace import new_id, utc_now

GOLDEN = Path(__file__).resolve().parent / "golden" / "terminal_trace.txt"
"""Path: The committed transcript the scripted run is compared against."""

TASK = "how many records mention membership"
"""str: The task the golden run is given."""

SITE = "http://site.test/"
"""str: What the fixture site's real address is rewritten to.

The site binds an ephemeral port so that several test sessions can run at once,
which means its address is different on every run and can never appear in a
committed file.
"""

RUN_FOLDER = "runs/RUN"
"""str: What the run's folder is rewritten to. Named after the moment the run
started, so it is different every time by design."""

SECONDS = "0.00s"
"""str: What every elapsed time is rewritten to. A click takes as long as the
browser takes."""


def normalise(output: str) -> str:
    """Replace everything about a run that is different every time.

    The list is short on purpose. Each entry is something this test is
    deliberately not checking, and anything not listed here is compared
    character for character against the committed file.

    Args:
        output: The captured terminal output, already stripped of styling.

    Returns:
        str: The same output with the clock, the fixture site's address, the
        run folder's path, and the elapsed times replaced by fixed stand-ins.
    """
    # The site's port, and therefore its whole address, changes per test
    # session. Rewritten before the folder, because neither contains the other.
    output = re.sub(r"http://127\.0\.0\.1:\d+/", SITE, output)
    # The run folder is a temporary directory plus a timestamp; the line is
    # worth keeping (a reader needs to be told where the evidence went) and its
    # contents are worth nothing to compare.
    output = re.sub(r"^trace: .*$", f"trace: {RUN_FOLDER}", output, flags=re.MULTILINE)
    # How long a browser took, and how long a model took. Both are facts about
    # the machine the test ran on.
    output = re.sub(r"\(\d+\.\d\ds\)", f"({SECONDS})", output)
    return re.sub(r"[\d.]+s thinking", "0.0s thinking", output)


def transcript(output: str) -> str:
    """Cut the captured output down to the run's own trace.

    Every command prints a capabilities report first, and what it says depends
    on which API keys the machine running the test happens to have. That is not
    part of the run, and it could never be committed to a file, so the
    transcript starts where the run does.

    Args:
        output: The captured output, already stripped of styling.

    Returns:
        str: From the ``task:`` line onwards.

    Raises:
        AssertionError: If there is no ``task:`` line, which would mean the run
            never announced itself -- a real failure rather than a missing
            fixture.
    """
    start = output.find("task:")
    assert start != -1, f"the run never announced itself:\n{output}"
    return output[start:]


# --- Running the real command against the practice site ----------------------


class ScriptedEndpoint:
    """A model endpoint that answers with a fixed list of actions.

    Attributes:
        calls: How many times it was asked.
    """

    def __init__(self, *replies: str) -> None:
        """Set up the script.

        Args:
            *replies: Message contents to return, one per call, the last
                repeating if the run asks more often than the script provides.
        """
        self._replies = list(replies)
        self.calls = 0
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **request: Any) -> SimpleNamespace:
        """Answer one call.

        Args:
            **request: The call body, ignored -- the script does not depend on
                what it was asked.

        Returns:
            SimpleNamespace: A completion shaped like the SDK's.
        """
        self.calls += 1
        reply = self._replies[min(self.calls - 1, len(self._replies) - 1)]
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=reply), finish_reason="stop")],
            usage=SimpleNamespace(prompt_tokens=903, completion_tokens=21),
        )


def run_the_command(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    playwright: Any,
    fixture_site: Any,
    *replies: str,
) -> Any:
    """Run ``waa run`` in this process against the practice site.

    The endpoint is scripted and the browser borrows the suite's driver; both
    swaps are explained at length in ``test_cli_run.py``, which does the same
    thing for the same reasons.

    Args:
        monkeypatch: The patcher.
        tmp_path: Where the run's folder may go.
        playwright: The suite's driver, lent to the command.
        fixture_site: The practice site.
        *replies: The actions the endpoint answers with, in order.

    Returns:
        SimpleNamespace: The exit code, and the whole output as a person would
        read it -- both channels, the way a terminal interleaves them.
    """
    endpoint = ScriptedEndpoint(*replies)
    starts_its_own = cli.BrowserProcess
    monkeypatch.setattr(
        cli,
        "BrowserProcess",
        lambda settings: starts_its_own(settings, playwright=playwright),
    )

    class Client:
        """Stands in for the model client's own constructor."""

        @staticmethod
        def from_config(config: Config) -> Any:
            """Build a real client over the scripted endpoint.

            Args:
                config: The run's settings, which the local profile ignores.

            Returns:
                ModelClient: A client that talks to the script.
            """
            return ModelClient(load_profile("local"), transport=endpoint, env={})

    monkeypatch.setattr(cli, "ModelClient", Client)
    monkeypatch.setattr(cli, "RUNS_ROOT", tmp_path / "runs")

    runner = CliRunner()
    result = runner.invoke(
        app,
        ["run", TASK, "--start-url", fixture_site.url_for("search.html")],
        catch_exceptions=False,
    )
    return SimpleNamespace(exit_code=result.exit_code, output=plain(result.output))


@pytest.mark.mock_llm
def test_a_whole_run_reads_the_way_the_committed_transcript_says(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    playwright: Any,
    fixture_site: Any,
) -> None:
    """The golden test: a real run's terminal output, word for word.

    The script is chosen to put four different kinds of step in one transcript:
    two ordinary actions, a repeat that the controller allows, a third repeat
    that it replaces with a scroll, and the agent ending the run itself. If the
    layout, the wording, or the element naming changes, this is where it shows
    -- and the diff is the review.
    """
    result = run_the_command(
        monkeypatch,
        tmp_path,
        playwright,
        fixture_site,
        '{"action": "fill", "element_id": 7, "text": "membership"}',
        '{"action": "press", "key": "Enter"}',
        '{"action": "click", "element_id": 3}',
        '{"action": "click", "element_id": 3}',
        '{"action": "click", "element_id": 3}',
        '{"action": "done", "status": "success", "answer": "17 records"}',
    )

    assert result.exit_code == EXIT_SUCCESS, result.output
    assert normalise(transcript(result.output)) == GOLDEN.read_text(encoding="utf-8")


@pytest.mark.mock_llm
def test_the_last_line_names_the_exit_code_the_process_uses(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    playwright: Any,
    fixture_site: Any,
) -> None:
    """The verdict on screen and the verdict a script reads must be one fact.

    A trace that says a run failed while the process exits 0 is worse than
    either on its own: a person and their build server would disagree about
    what happened, and only one of them would be in the room.
    """
    result = run_the_command(
        monkeypatch,
        tmp_path,
        playwright,
        fixture_site,
        '{"action": "done", "status": "failed", "reason": "the catalogue has no such record"}',
    )

    assert result.exit_code == EXIT_TASK_FAILURE
    last = plain(result.output).strip().splitlines()[-1]
    assert last.endswith(f"(exit {result.exit_code})"), last
    assert "gave_up" in last and "the catalogue has no such record" in last


@pytest.mark.mock_llm
def test_a_step_that_chose_nothing_says_so_and_names_the_refusals(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    playwright: Any,
    fixture_site: Any,
) -> None:
    """The one phase the golden run does not reach, and the most confusing to omit.

    A step where every attempt was refused has no action and no result. Left to
    render itself, it would be a blank line under a step number -- which reads
    as "nothing happened" when what happened is that the agent proposed three
    things and the gate refused all of them.
    """
    result = run_the_command(
        monkeypatch,
        tmp_path,
        playwright,
        fixture_site,
        # Element 99 is not on the page, so the gate refuses it -- every time,
        # until the step's retries run out.
        '{"action": "click", "element_id": 99}',
    )

    assert result.exit_code == EXIT_TASK_FAILURE
    assert f"  1  {PLAN}" in result.output, "the phase says the step never chose anything"
    assert "no action: every attempt refused (unknown_element" in result.output
    assert "validation_exhausted" in result.output
    # The category is in brackets on the verdict line and must not also open
    # the reason beside it.
    assert "(validation_exhausted): validation_exhausted" not in result.output


# --- The pieces, without a browser -------------------------------------------


def a_page() -> Observation:
    """Build an observation with two named elements on it.

    Returns:
        Observation: A page offering a link and a search box, which between
        them cover both ways an action can name an element.
    """
    return Observation(
        url="http://site.test/search.html",
        title="Search",
        elements=(
            Element(
                id=3,
                tag="a",
                role="link",
                name="Next page",
                text="Next page",
                box=Box(x=0, y=20, width=80, height=18),
            ),
            Element(
                id=7,
                tag="input",
                role="searchbox",
                name="Search terms",
                text="",
                box=Box(x=0, y=60, width=200, height=24),
            ),
        ),
        text="Records are matched on title and subject.",
    )


def a_step(action: Any = None, result: Any = None, **overrides: Any) -> StepReport:
    """Build one finished step, for the rendering tests.

    Args:
        action: What was done, or None for a step that did nothing.
        result: What came of it, or None.
        **overrides: Any other field of the step report.

    Returns:
        StepReport: A step, with the bookkeeping fields filled in.
    """
    fields: dict[str, Any] = {
        "step_id": new_id("step"),
        "started_at": utc_now(),
        "ended_at": utc_now(),
        "number": 1,
        "observation": a_page(),
        "action": action,
        "result": result,
    }
    return StepReport(**{**fields, **overrides})


@pytest.mark.unit
def test_an_action_names_the_element_rather_than_numbering_it() -> None:
    """``click 3`` is meaningless a minute later; ``click [3] link "Next page"`` is not.

    The number is kept alongside the name rather than replaced by it, because
    the number is what the trace file and the validator both speak, and a
    terminal line that could not be matched back to them would be a third
    vocabulary.
    """
    click = parse_action({"action": "click", "element_id": 3})

    assert describe_action(click, a_page()) == 'click [3] link "Next page"'


@pytest.mark.unit
def test_filling_a_field_keeps_the_field_and_the_text_apart() -> None:
    """Two quoted strings in a row would read as one thing typed into another."""
    fill = parse_action({"action": "fill", "element_id": 7, "text": "membership"})

    assert describe_action(fill, a_page()) == 'fill [7] searchbox "Search terms" with "membership"'


@pytest.mark.unit
def test_an_element_the_observation_does_not_have_falls_back_to_its_number() -> None:
    """A refused action names an element that was never there, and that is the finding.

    Inventing a name for it, or dropping the action from the line, would both
    hide the one thing worth seeing: the agent asked for something that does
    not exist.
    """
    click = parse_action({"action": "click", "element_id": 99})

    assert describe_action(click, a_page()) == "click 99"


@pytest.mark.unit
def test_an_action_with_no_element_is_left_as_it_describes_itself() -> None:
    """Scrolling and key presses name nothing, and need no special case here."""
    assert describe_action(parse_action({"action": "press", "key": "Enter"}), a_page()) == (
        "press Enter"
    )
    assert describe_action(parse_action({"action": "scroll", "direction": "down"}), a_page()) == (
        "scroll down"
    )


@pytest.mark.unit
def test_every_phase_is_one_of_the_four_and_each_is_reachable() -> None:
    """A closed list, like the outcome categories, and for the same reason.

    A phase invented in one place would appear in a trace that nothing else
    knows how to read -- and, being a word in a column, would look entirely
    normal while doing it.
    """
    reached = {
        phase_of(a_step()): None,
        phase_of(a_step(action=parse_action({"action": "click", "element_id": 3}))): None,
        phase_of(a_step(action=parse_action({"action": "use_skill", "skill": "search"}))): None,
        phase_of(a_step(action=parse_action({"action": "done", "status": "success"}))): None,
    }

    assert set(reached) == set(PHASES) == {ACT, SKILL, DONE, PLAN}


@pytest.mark.unit
def test_the_phase_column_is_not_repeated_by_the_decision_beside_it() -> None:
    """``done  done success`` is a stutter; the phase is the half that is a column."""
    done = a_step(action=parse_action({"action": "done", "status": "success", "answer": "17"}))
    skill = a_step(action=parse_action({"action": "use_skill", "skill": "search"}))

    assert step_line(done).split() == ["1", DONE, "success:", "17"]
    assert step_line(skill).split() == ["1", SKILL, "search()"]


@pytest.mark.unit
def test_a_retried_step_says_how_many_attempts_it_took() -> None:
    """Shown only when it was more than one, so the steps where it matters stand out."""
    once = a_step(action=parse_action({"action": "scroll", "direction": "down"}))
    twice = a_step(action=parse_action({"action": "scroll", "direction": "down"}), attempts=2)

    assert "attempts" not in step_line(once)
    assert "[2 attempts]" in step_line(twice)


@pytest.mark.unit
def test_a_forced_action_says_that_it_was_forced_and_why() -> None:
    """The one place a step's action is not the agent's own choice.

    Left unmarked, the transcript would show the agent doing something it never
    proposed, and the next question -- "why did it scroll?" -- would have no
    answer on screen.
    """
    forced = a_step(
        action=parse_action({"action": "scroll", "direction": "down"}),
        forced="click 3 was proposed 3 times in a row",
    )

    assert "[forced: click 3 was proposed 3 times in a row]" in step_line(forced)


@pytest.mark.unit
def test_the_terminal_trace_writes_wherever_it_is_pointed() -> None:
    """A stream, a list, a log file -- all the same thing to the renderer.

    This is what lets the command line hand it standard error and the
    controller hand it steps, without either of them knowing about the other.
    """
    lines: list[str] = []
    trace = TerminalTrace(lines.append)

    trace.opening(TASK, "http://site.test/search.html")
    trace.step(a_step(action=parse_action({"action": "scroll", "direction": "down"})))

    assert lines[0] == f"task:  {TASK}"
    assert lines[1] == "start: http://site.test/search.html"
    assert lines[2] == "", "a blank line before the steps, so the two do not run together"
    assert lines[3].startswith("  1  act")
