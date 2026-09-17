"""The ``run`` command really running the agent.

Separate from the rest of the command-line tests because these need a browser.
Those tests are pure Python -- they check flags, exit codes and help text with
nothing behind them -- and this one drives the whole program: a real Chromium,
the local fixture site, a real form submission, and the command's own exit
code and output at the end of it.

The model endpoint is the only stand-in, and it is scripted rather than
stubbed-shallow: it answers with the same JSON actions a real endpoint would.
The point being tested is the *wiring* -- that the command assembles fourteen
components in the right order and passes each of them the right settings -- and
that is exactly what no unit test of any one component can catch.
"""

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from plain_text import plain
from typer.testing import CliRunner

from web_action_agent import cli
from web_action_agent.cli import EXIT_SUCCESS, EXIT_TASK_FAILURE, app
from web_action_agent.config import Config
from web_action_agent.model_client import ModelClient, load_profile
from web_action_agent.trace import OUTCOME_FILE, read_action_log

runner = CliRunner()
"""CliRunner: Runs the command line in-process and captures what it printed."""


def invoke(argv: list[str]) -> Any:
    """Run the command line with the given arguments, as a shell would.

    Args:
        argv: The arguments after the program name.

    Returns:
        Any: The runner's result, with the output already reduced to plain
        words so an assertion means the same thing on a laptop and on a build
        server.
    """
    result = runner.invoke(app, argv, catch_exceptions=False)
    return SimpleNamespace(exit_code=result.exit_code, output=plain(result.output))


class ScriptedEndpoint:
    """A model endpoint that answers with a fixed list of actions.

    Attributes:
        calls: How many times it was asked, so a test can tell a run that did
            three steps from one that did the same step three times.
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
            usage=SimpleNamespace(prompt_tokens=900, completion_tokens=20),
        )


def a_run_in_this_process(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    playwright: Any,
    *replies: str,
) -> ScriptedEndpoint:
    """Make the command runnable inside the test process, and script its model.

    Three things are swapped, and only the first is about pretending:

    * **The endpoint** answers from a script, because the point of these tests
      is the command's wiring rather than a model's judgement.
    * **The run folder** is a temporary one, so no test writes into the working
      tree.
    * **The browser process borrows the suite's driver.** Playwright allows one
      synchronous driver per thread, and the test suite already has one
      running. A real ``waa run`` is its own process and starts its own, which
      is why the command does exactly that -- so the swap here is what makes
      the command testable in-process, not a fix for anything wrong with it.

    Args:
        monkeypatch: The patcher.
        tmp_path: Where the run's files may go.
        playwright: The suite's driver, lent to the command.
        *replies: The actions the endpoint answers with, in order.

    Returns:
        ScriptedEndpoint: The endpoint, so a test can count the calls.
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
    return endpoint


@pytest.mark.fixture
def test_a_finished_task_exits_zero_and_prints_the_answer(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    playwright: Any,
    fixture_site: Any,
) -> None:
    """The whole command, end to end: a real browser, a real page, a real form.

    If the command's assembly were wrong -- a component built in the wrong
    order, a setting not passed through, the executor given a page instead of a
    session -- this is where it would show, and nowhere else would.
    """
    endpoint = a_run_in_this_process(
        monkeypatch,
        tmp_path,
        playwright,
        '{"action": "fill", "element_id": 7, "text": "membership"}',
        '{"action": "press", "key": "Enter"}',
        '{"action": "done", "status": "success", "answer": "17 records"}',
    )

    result = invoke(["run", "how many records", "--start-url", fixture_site.url_for("search.html")])

    assert result.exit_code == EXIT_SUCCESS, result.output
    assert "17 records" in result.output
    assert "success in 3 steps" in result.output
    assert endpoint.calls == 3, "one model call per step, and no step taken twice"


@pytest.mark.fixture
def test_the_answer_is_the_only_thing_on_standard_output(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    playwright: Any,
    fixture_site: Any,
) -> None:
    """So that ``waa run ... > answer.txt`` gives the answer and not a log.

    The steps, the verdict and the cost go to standard error, where a person
    watching the run sees them and a pipe does not.
    """
    a_run_in_this_process(
        monkeypatch,
        tmp_path,
        playwright,
        '{"action": "done", "status": "success", "answer": "09:00 to 17:30"}',
    )

    result = runner.invoke(
        app,
        ["run", "the hours", "--start-url", fixture_site.url_for("search.html")],
        catch_exceptions=False,
    )

    assert result.stdout.strip() == "09:00 to 17:30"


@pytest.mark.fixture
def test_a_task_the_agent_cannot_finish_exits_one(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    playwright: Any,
    fixture_site: Any,
) -> None:
    """A run that ends in a failure category is a failed run, and says which."""
    a_run_in_this_process(
        monkeypatch,
        tmp_path,
        playwright,
        '{"action": "done", "status": "failed", "reason": "the catalogue has no such record"}',
    )

    result = invoke(["run", "find nothing", "--start-url", fixture_site.url_for("search.html")])

    assert result.exit_code == EXIT_TASK_FAILURE
    assert "gave_up" in result.output
    assert "the catalogue has no such record" in result.output


@pytest.mark.fixture
def test_a_run_that_runs_out_of_steps_says_so(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    playwright: Any,
    fixture_site: Any,
) -> None:
    """The step cap reaching the terminal, with the category a reader can act on."""
    a_run_in_this_process(
        monkeypatch,
        tmp_path,
        playwright,
        '{"action": "scroll", "direction": "down"}',
    )
    monkeypatch.setattr(cli, "resolve_env", lambda: {"WAA_MAX_STEPS": "3"})

    result = invoke(["run", "scroll forever", "--start-url", fixture_site.url_for("ledger.html")])

    assert result.exit_code == EXIT_TASK_FAILURE
    assert "step_cap" in result.output
    assert "failed in 3 steps" in result.output


@pytest.mark.fixture
def test_the_command_leaves_a_run_folder_and_says_where(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    playwright: Any,
    fixture_site: Any,
) -> None:
    """Every run leaves its evidence behind, and names the folder it left it in.

    Tested here rather than only in the trace writer's own module because this
    is the wiring: the folder, the screenshot directory the observer writes
    into, and the writer the controller flushes to all have to be the same
    place, and nothing but the command assembles all three.
    """
    a_run_in_this_process(
        monkeypatch,
        tmp_path,
        playwright,
        '{"action": "done", "status": "success", "answer": "09:00 to 17:30"}',
    )

    result = invoke(["run", "the hours", "--start-url", fixture_site.url_for("search.html")])

    assert result.exit_code == EXIT_SUCCESS, result.output
    folders = list((tmp_path / "runs").iterdir())
    assert len(folders) == 1, "one run, one folder"
    folder = folders[0]
    assert str(folder) in result.output, "and the reader is told where it is"
    steps = read_action_log(folder)
    assert [step["number"] for step in steps] == [1]
    assert (
        json.loads((folder / OUTCOME_FILE).read_text(encoding="utf-8"))["outcome"]["answer"]
        == "09:00 to 17:30"
    )
