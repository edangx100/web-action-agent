"""The command surface, and the promise made by the number it exits with.

Two different readers depend on this module, and they need different things.

A **person** needs every command to explain itself: ``--help`` has to work on a
fresh checkout with nothing configured, name every flag, and print prose rather
than the Args and Raises notes that belong in the source. Those notes leaking
into a terminal is the kind of thing nobody notices for months, so it is
asserted here rather than left to whoever next reads the output.

A **script** needs the exit code to mean something fixed. Zero worked, one the
work failed, two the command was unusable. A benchmark runner records a failed
task and carries on, but stops dead on a mistyped flag of its own, because
every remaining invocation carries the same typo. Collapsing both into
"non-zero" takes that choice away, so each documented case is pinned below.

Tests here call the command line the way a shell does -- through the argument
parser, checking the code that comes back -- rather than calling the functions
underneath. The parsing is most of what this module is.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import pytest
import typer
from plain_text import plain
from typer.testing import CliRunner

from web_action_agent import cli
from web_action_agent.cli import (
    COMMANDS,
    EXIT_SUCCESS,
    EXIT_TASK_FAILURE,
    EXIT_USAGE,
    app,
)
from web_action_agent.config import Config

pytestmark = pytest.mark.unit

#: Every command, written out as a shell would type it. The nested skills
#: commands are included, because "every subcommand's --help works" has to mean
#: the ones a user reaches through a group too.
EVERY_COMMAND = [
    ["run"],
    ["eval"],
    ["induce"],
    ["judge"],
    ["skills"],
    ["skills", "list"],
    ["skills", "show"],
    ["skills", "retire"],
]

#: The flags the command surface promises, and where each one lives. Written
#: out here rather than read from the code, so that deleting a flag fails this
#: test instead of quietly agreeing with itself.
DOCUMENTED_FLAGS = [
    (["run"], "--start-url"),
    (["run"], "--profile"),
    (["run"], "--headed"),
    (["eval"], "--suite"),
    (["eval"], "--trials"),
    (["eval"], "--cold"),
    (["eval"], "--warm"),
    (["eval"], "--profile"),
    (["induce"], "--all"),
    (["judge"], "--review"),
]

runner = CliRunner()


@pytest.fixture(autouse=True)
def _no_configuration_needed(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Give every test an empty environment and a working default config.

    Without this, these tests would read the developer's own ``.env`` and their
    exported keys. A machine with a key set would then exercise a different
    path from a machine without one, and the suite would pass or fail depending
    on whose laptop it ran on.

    Yields:
        None: with the environment stubbed for the duration of one test.
    """
    monkeypatch.setattr(cli, "resolve_env", dict)
    yield


@dataclass(frozen=True)
class Invocation:
    """What one run of the command line produced.

    Attributes:
        exit_code: The number the command exited with.
        output: Everything it printed, with terminal styling stripped -- see
            :mod:`plain_text` for why that has to happen before any assertion
            touches it.
    """

    exit_code: int
    output: str


def invoke(argv: list[str]) -> Invocation:
    """Run the command line with the given arguments, as a shell would.

    Args:
        argv: The arguments after the program name, e.g. ``["run", "--help"]``.

    Returns:
        Invocation: The exit code and the printed output as words, so that a
        test asserting on the output means the same thing on a laptop and on a
        build server -- which style the output differently.
    """
    result = runner.invoke(app, argv, catch_exceptions=False)
    return Invocation(exit_code=result.exit_code, output=plain(result.output))


# --- Every command explains itself ------------------------------------------


@pytest.mark.parametrize("command", EVERY_COMMAND, ids=lambda c: " ".join(c))
def test_every_command_explains_itself(command: list[str]) -> None:
    """``--help`` works, exits 0, and says the command's own name.

    This has to hold with nothing configured: asking what a command does is the
    first thing a new reader does, before they have a key or a settings file.
    """
    result = invoke([*command, "--help"])

    assert result.exit_code == EXIT_SUCCESS
    assert command[-1] in result.output


def test_the_top_level_help_lists_every_command() -> None:
    """Someone typing ``waa --help`` sees all five commands, not a subset."""
    result = invoke(["--help"])

    assert result.exit_code == EXIT_SUCCESS
    for command in COMMANDS:
        assert command in result.output


@pytest.mark.parametrize("command", EVERY_COMMAND, ids=lambda c: " ".join(c))
def test_help_prints_prose_not_source_notes(command: list[str]) -> None:
    """The ``Args:`` and ``Raises:`` sections stay in the source.

    Those are notes for whoever reads the code. The help screen is for whoever
    is trying to use the command, and the two audiences want different text --
    so every command passes an explicit ``help=`` rather than letting its
    docstring be printed. This is the test that keeps that true: forget the
    ``help=`` on a new command and its Args section lands in the terminal.
    """
    output = invoke([*command, "--help"]).output

    assert "Args:" not in output
    assert "Raises:" not in output
    assert "Returns:" not in output


@pytest.mark.parametrize(
    ("command", "flag"),
    DOCUMENTED_FLAGS,
    ids=[f"{' '.join(c)} {f}" for c, f in DOCUMENTED_FLAGS],
)
def test_each_documented_flag_is_offered(command: list[str], flag: str) -> None:
    """Every flag the project promises is really there, on the right command."""
    assert flag in invoke([*command, "--help"]).output


def test_the_command_list_is_taken_from_the_commands_themselves() -> None:
    """``COMMANDS`` is derived, not typed out twice.

    Other modules use it to tell a task string from a mistyped command, so a
    hand-maintained copy drifting from the real list would misroute arguments.
    """
    assert COMMANDS == {"run", "eval", "induce", "skills", "judge", "export"}


# --- The exit-code contract -------------------------------------------------


def test_a_bad_flag_is_a_usage_error() -> None:
    """A flag that does not exist stops the command before anything is spent."""
    assert invoke(["run", "a task", "--nonesuch"]).exit_code == EXIT_USAGE


def test_a_missing_argument_is_a_usage_error() -> None:
    """``run`` without a task cannot be guessed at, so it is refused."""
    assert invoke(["run"]).exit_code == EXIT_USAGE


def test_an_unknown_command_is_a_usage_error() -> None:
    """A misspelled command is a typo, not a task to attempt."""
    assert invoke(["frobnicate"]).exit_code == EXIT_USAGE


def test_a_value_outside_its_range_is_a_usage_error() -> None:
    """Zero trials would run the benchmark and measure nothing.

    Caught by the flag's own bounds rather than downstream, so the message
    names the flag while the person who typed it is still watching.
    """
    assert invoke(["eval", "--trials", "0"]).exit_code == EXIT_USAGE


def test_asking_for_help_succeeds() -> None:
    """Help is what was asked for, so delivering it is a success."""
    assert invoke(["--help"]).exit_code == EXIT_SUCCESS


def test_work_that_did_not_happen_is_a_task_failure_not_a_usage_error() -> None:
    """A valid command whose machinery is unbuilt exits 1, not 2.

    The distinction is the whole point of having two codes: the request was
    understood and accepted, and it is the work that did not succeed. A script
    should treat this like any other failed run, not like its own typo.

    ``run`` was the example here, then ``skills list``, and both are now built.
    The example moves on each time; the rule does not. It is about the exit
    codes, not about which command happens to be unfinished this month.
    """
    result = invoke(["judge"])

    assert result.exit_code == EXIT_TASK_FAILURE
    assert "not built yet" in result.output


def test_a_run_with_no_endpoint_configured_is_a_usage_error() -> None:
    """No key means nothing was attempted and nothing spent -- so 2, not 1.

    The message names the variable to set, because "the run failed" would send
    somebody looking at the agent for a problem that is in their .env.
    """
    result = invoke(["run", "find the opening hours"])

    assert result.exit_code == EXIT_USAGE
    assert "WAA_OPENROUTER_API_KEY" in result.output


def test_the_three_exit_codes_are_distinct() -> None:
    """Nothing has quietly collapsed two meanings into one number."""
    assert len({EXIT_SUCCESS, EXIT_TASK_FAILURE, EXIT_USAGE}) == 3


def test_main_returns_a_number_rather_than_raising(monkeypatch: pytest.MonkeyPatch) -> None:
    """``main`` hands back an exit code instead of letting the raise escape.

    Typer signals an exit by raising, which suits a program and not a caller.
    Both the packaged ``waa`` command and these tests want a number, so the
    conversion happens in one place.
    """
    monkeypatch.setattr("sys.argv", ["waa", "--help"])
    assert cli.main() == EXIT_SUCCESS


# --- Choices that cannot be expressed wrongly -------------------------------


def test_cold_and_warm_cannot_both_be_asked_for() -> None:
    """One sweep is cold or warm. Written as one flag, so "both" is unsayable.

    The last one typed wins rather than erroring, which is click's rule for a
    switch pair and the one people expect from a shell.
    """
    assert invoke(["eval", "--cold", "--warm", "--help"]).exit_code == EXIT_SUCCESS
    assert "--cold" in invoke(["eval", "--help"]).output
    assert "--warm" in invoke(["eval", "--help"]).output


def test_induce_needs_exactly_one_source() -> None:
    """Neither a run folder nor ``--all`` is a request for nothing."""
    assert invoke(["induce"]).exit_code == EXIT_USAGE


def test_induce_refuses_two_sources_at_once() -> None:
    """A folder and ``--all`` are two different jobs, so the command asks which."""
    result = invoke(["induce", "runs/one", "--all", "runs"])

    assert result.exit_code == EXIT_USAGE
    assert "not both" in result.output


# --- Flags are the top configuration layer ----------------------------------


def test_a_flag_beats_the_configuration_files(monkeypatch: pytest.MonkeyPatch) -> None:
    """``--profile local`` reaches the resolved settings for this run only.

    The three layers are config.yaml, then .env, then flags. This pins that
    flags really are the last of them: what the command line says is what the
    rest of the program sees.
    """
    seen: dict[str, Config] = {}

    def record(config: Config, **_: object) -> Any:
        seen["config"] = config
        raise typer.Exit(EXIT_TASK_FAILURE)

    monkeypatch.setattr(cli, "report_capabilities", record)
    invoke(["run", "a task", "--profile", "local"])

    assert seen["config"].profile == "local"


def test_an_unpassed_flag_leaves_the_configured_value(monkeypatch: pytest.MonkeyPatch) -> None:
    """Not passing a flag is not the same as passing its default.

    A flag that always overrode would make the lower layers unreachable: a
    setting put in .env would be silently replaced on every run by the flag's
    own default, which nobody chose.
    """
    seen: dict[str, Config] = {}

    def record(config: Config, **_: object) -> Any:
        seen["config"] = config
        raise typer.Exit(EXIT_TASK_FAILURE)

    monkeypatch.setattr(cli, "report_capabilities", record)
    monkeypatch.setattr(cli, "resolve_env", lambda: {"WAA_PROFILE": "nvidia"})
    invoke(["run", "a task"])

    assert seen["config"].profile == "nvidia"


def test_the_browser_window_can_be_turned_off_as_well_as_on(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``--headed`` has three states, so a configured value can be overridden.

    A plain on/off switch could only turn the window on. A machine that had set
    it in .env could then never turn it off for one run -- which would make the
    flag layer the weakest one for that setting instead of the strongest.
    """
    seen: list[Config] = []

    def record(config: Config, **_: object) -> Any:
        seen.append(config)
        raise typer.Exit(EXIT_TASK_FAILURE)

    monkeypatch.setattr(cli, "report_capabilities", record)
    monkeypatch.setattr(cli, "resolve_env", lambda: {"WAA_HEADED": "true"})

    invoke(["run", "a task"])
    invoke(["run", "a task", "--no-headed"])

    assert [config.headed for config in seen] == [True, False]


def test_an_unreadable_setting_is_a_usage_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """A bad value in .env stops the command with 2, before any work starts.

    It is not a task failure: nothing was attempted, nothing was spent, and the
    fix is to correct the configuration rather than to retry.
    """
    monkeypatch.setattr(cli, "resolve_env", lambda: {"WAA_MAX_STEPS": "not-a-number"})
    result = invoke(["run", "a task"])

    assert result.exit_code == EXIT_USAGE
    assert "max_steps" in result.output


# --- The startup report still happens ---------------------------------------


def test_a_command_reports_what_the_configured_keys_allow() -> None:
    """Every command that does work still prints the capability report first."""
    assert "capabilities:" in invoke(["run", "a task"]).output


def test_help_does_not_need_a_working_configuration(monkeypatch: pytest.MonkeyPatch) -> None:
    """``--help`` reads no settings at all, so it works when they are broken.

    Configuration is resolved inside each command rather than once for the
    whole program precisely so this holds: the first thing a new reader types
    must not depend on the setup they have not done yet.
    """
    monkeypatch.setattr(cli, "resolve_env", lambda: {"WAA_MAX_STEPS": "not-a-number"})

    assert invoke(["run", "--help"]).exit_code == EXIT_SUCCESS
    assert "capabilities:" not in invoke(["run", "--help"]).output
