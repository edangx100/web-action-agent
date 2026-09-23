"""The commands a user types, and the number the program exits with.

Five subcommands, one job each::

    waa run "<task>"        do one task, start to finish
    waa eval                run the whole benchmark
    waa induce <run>        turn successful runs into reusable skills
    waa skills              list, inspect, and retire those skills
    waa judge <runs>        grade finished runs, and review the grades

**The exit code is a promise.** A person reads the messages; a script reads the
number. There are exactly three, and they mean::

    0   it worked
    1   the task ran and did not succeed
    2   the command could not be run at all -- a bad flag, a bad setting

The distinction between 1 and 2 is the one that matters in practice. A
benchmark script wants to record a failed task and keep going, but stop
immediately on a typo in its own arguments, because every remaining run would
have the same typo. Collapsing both into "non-zero" takes that choice away.

**Flags are the top configuration layer.** Settings resolve as
``config.yaml`` -> ``.env`` -> flags, and the flags here are the last of those,
so ``--profile local`` beats both files for one invocation and changes nothing
for the next. Flag names are not spelled out here: they are derived from the
settings themselves by :func:`~web_action_agent.config.flag_for`, so a flag
cannot come to be named differently from the setting it sets.

**Nothing here runs the agent yet.** Each command reports what it would do and
exits 1. The surface, the flags, and the exit codes are settled first because
they are what everything else is written against -- the test suite, the
benchmark scripts, and the README.
"""

from __future__ import annotations

import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated

import typer

from web_action_agent.allowlist import Allowlist, AllowlistError
from web_action_agent.browser import BrowserProcess, BrowserSettings
from web_action_agent.capabilities import report_capabilities
from web_action_agent.config import Config, ConfigError, flag_for, load_config, resolve_env
from web_action_agent.controller import Controller, RunReport
from web_action_agent.executor import Executor
from web_action_agent.memory import Memory
from web_action_agent.model_client import ModelClient, ModelClientError
from web_action_agent.planner import Planner
from web_action_agent.politeness import politeness_for_run
from web_action_agent.reporting import TerminalTrace
from web_action_agent.screenshot_gate import ScreenshotGate
from web_action_agent.timeouts import RunBudget
from web_action_agent.trace import TraceError, TraceWriter
from web_action_agent.validator import Validator

#: The command did what was asked.
EXIT_SUCCESS = 0

#: The command ran, and the work did not succeed: a task the agent could not
#: complete, a benchmark that failed. The program worked; the run did not.
EXIT_TASK_FAILURE = 1

#: The command could not be run at all -- an unknown flag, a missing argument,
#: an unreadable setting. Nothing was attempted, so nothing was spent.
EXIT_USAGE = 2

#: Where a run puts what it leaves behind. One folder per run, named for the
#: moment it started. The trace writer will own the contents; for now the
#: run's screenshots go here, which is why the folder exists this early.
RUNS_ROOT = Path("runs")

#: Where the benchmark's task definitions live when ``--suite`` is not given.
DEFAULT_SUITE = Path("benchmark/tasks.yaml")

#: Where finished runs are written, and therefore where the commands that read
#: runs back look first.
DEFAULT_RUNS_DIR = Path("runs")

#: How many times the benchmark repeats each task when ``--trials`` is not
#: given. More trials narrow the error bars and cost proportionally more.
DEFAULT_TRIALS = 3


app = typer.Typer(
    name="waa",
    help="A browser agent that reads pages, decides what to do, and does it.",
    # A bare `waa` prints the help rather than a bare error, since someone
    # typing it is asking what this is. It still exits 2: no command was named,
    # so nothing was run. Asking outright with `waa --help` exits 0.
    no_args_is_help=True,
    # No shell-completion machinery. It would add two flags to every --help
    # screen for a convenience this project does not need.
    add_completion=False,
    context_settings={"help_option_names": ["-h", "--help"]},
)

skills_app = typer.Typer(
    name="skills",
    help="Inspect and curate the learned skill library.",
    no_args_is_help=True,
)
app.add_typer(skills_app)

# Every command in this module passes an explicit `help=`. Typer falls back to
# the function's docstring otherwise, which would print this project's Args and
# Raises sections straight to the terminal -- notes for someone reading the
# code, not for someone asking what a command does. A test pins the rule, so
# forgetting `help=` on a new command fails the suite rather than the reader.


# --- Shared flag definitions ------------------------------------------------
#
# Declared once and reused, so that --profile means the same thing and reads
# the same way on every command that offers it. The names come from
# flag_for(), which derives them from the settings schema.

ProfileOption = Annotated[
    str | None,
    typer.Option(
        flag_for("profile"),
        help="Endpoint profile to use for this run: openrouter, nvidia, or local.",
    ),
]

StartUrlOption = Annotated[
    str | None,
    typer.Option(
        flag_for("start_url"),
        help="The page the run opens on. Also fixes which sites the run may visit.",
    ),
]

# Three states, not two: --headed on, --no-headed off, and neither, which
# leaves whatever config.yaml or .env says. A plain on/off flag could only ever
# turn the window on, so a machine that had set it in .env could not turn it
# off again for one run -- and flags are supposed to be the layer that wins.
HeadedOption = Annotated[
    bool | None,
    typer.Option(
        f"{flag_for('headed')}/--no-headed",
        help="Show the browser window. Off by default, so runs work over SSH and in CI.",
    ),
]


def _startup(**overrides: object) -> Config:
    """Resolve this run's settings and print what the configured keys allow.

    Called at the top of every command that does real work, rather than once
    for the whole program, so that ``--help`` never needs a valid
    configuration. Asking what a command does must work on a fresh checkout
    with nothing set up.

    The report goes to standard error. Piping a command's output to a file
    should capture the command's output, not this.

    Args:
        **overrides: Flag values for this invocation, keyed by setting name.
            A ``None`` means the flag was not passed, and the ``.env`` and
            ``config.yaml`` layers below it stand.

    Returns:
        Config: The resolved settings, with the flags applied on top.

    Raises:
        typer.Exit: With :data:`EXIT_USAGE` if any layer is unreadable or
            names something that is not a setting. The message says which.
    """
    try:
        env = resolve_env()
        config = load_config(env=env, cli=overrides)
    except ConfigError as error:
        typer.echo(f"waa: {error}", err=True)
        raise typer.Exit(EXIT_USAGE) from error

    typer.echo(report_capabilities(config, env=env).render(), err=True)
    return config


def _do_one_task(config: Config, task: str) -> int:
    """Assemble the agent, run one task, and report how it went.

    This is the wiring: every component the run needs, built once, in the one
    place that knows the run's settings. The order is forced by what depends on
    what -- the model client needs a profile before the planner can exist, the
    gate needs to know whether that profile can see images, and the validator
    needs the allowlist the start URL implies.

    Args:
        config: The resolved settings for this run.
        task: The task in the user's own words.

    Returns:
        int: :data:`EXIT_SUCCESS` if the agent finished the task,
        :data:`EXIT_TASK_FAILURE` if it did not, or :data:`EXIT_USAGE` if the
        run could not be set up -- which is not a failed run but an unstarted
        one.
    """
    folder = RUNS_ROOT / datetime.now(UTC).strftime("%Y-%m-%dT%H-%M-%SZ")
    try:
        client = ModelClient.from_config(config)
        allowlist = Allowlist.derive(config.start_url, config_extra=config.allowlist_extra)
        # Opened before the browser starts, and before anything is spent: a run
        # whose folder cannot be written is a run that would produce no
        # evidence, and finding that out after paying for twenty model calls
        # would be finding it out too late.
        trace = TraceWriter.open(folder, task=task, start_url=config.start_url, config=config)
    except (ModelClientError, AllowlistError, TraceError) as error:
        # Nothing has been attempted and nothing spent: the fix is to the
        # configuration, which is what exit code 2 means.
        typer.echo(f"waa: {error}", err=True)
        return EXIT_USAGE

    throttle, robots = politeness_for_run(config)
    # Everything the watcher sees goes to standard error, so that standard
    # output carries the answer alone and `waa run ... > answer.txt` is useful.
    watching = TerminalTrace(lambda line: typer.echo(line, err=True))
    watching.opening(task, config.start_url)

    # One browser process, one task, one context inside it -- the same shape a
    # benchmark worker uses, so an ad-hoc run and a measured one behave alike.
    with (
        BrowserProcess(BrowserSettings.from_config(config)) as browser,
        browser.task_session() as session,
    ):
        controller = Controller(
            planner=Planner(client),
            validator=Validator(allowlist, robots=robots),
            executor=Executor(
                session,
                action_timeout_s=config.action_timeout_s,
                throttle=throttle,
            ),
            memory=Memory.from_config(config),
            max_steps=config.max_steps,
            retry_budget=config.retry_budget,
            gate=ScreenshotGate.from_config(config, supports_vision=client.supports_vision),
            budget=RunBudget.from_config(config),
            screenshot_dir=trace.screenshot_dir,
            config=config,
            trace=trace,
            on_step=watching.step,
        )
        report = controller.run(task, start_url=config.start_url)

    exit_code = EXIT_SUCCESS if report.outcome.ok else EXIT_TASK_FAILURE
    _report_run(report, folder, watching, exit_code)
    return exit_code


def _report_run(
    report: RunReport,
    folder: Path,
    watching: TerminalTrace,
    exit_code: int,
) -> None:
    """Close off the run's terminal trace, and print the answer.

    The steps are already on screen: they were streamed as they happened. What
    is left is what could only be known at the end -- the cost, where the
    evidence went, and the verdict with the exit code beside it.

    The answer, and only the answer, goes to standard output, so that
    ``waa run ... > answer.txt`` gives the answer alone while the reader still
    watches the run happen. A failed run prints no answer at all rather than an
    empty line, because a blank stdout is easier to check for than a blank line.

    Args:
        report: What the run produced.
        folder: Where the run's files are, for the closing line.
        watching: The terminal trace that has been streaming the steps.
        exit_code: What the process is about to exit with, named on the last
            line so a person and a shell read the same verdict.
    """
    watching.closing(report, folder, exit_code)
    if report.outcome.ok and report.outcome.answer:
        typer.echo(report.outcome.answer)


def _not_built_yet(what: str) -> None:
    """Report that a command's machinery does not exist yet, and stop.

    The command surface is settled before the components behind it, so every
    command currently ends here. It exits 1 rather than 2 because the request
    itself was valid and understood -- it is the work that did not happen.

    Args:
        what: What this command would have done, in a few words, so the
            message says more than "not implemented".

    Raises:
        typer.Exit: Always, with :data:`EXIT_TASK_FAILURE`.
    """
    typer.echo(f"waa: {what} is not built yet", err=True)
    raise typer.Exit(EXIT_TASK_FAILURE)


@app.command(
    help=(
        "Do one task, from the start URL, and report what happened.\n\n"
        "The agent never begins on a blank page and guesses an address. Every "
        "run starts somewhere named -- by --start-url, or by the default in "
        "config.yaml -- and that page's site fixes where the run may go next."
    )
)
def run(
    task: Annotated[
        str,
        typer.Argument(help='What to do, in plain language: "find the opening hours".'),
    ],
    start_url: StartUrlOption = None,
    profile: ProfileOption = None,
    headed: HeadedOption = None,
) -> None:
    """Do one task, from the start URL, and report what happened.

    The user-facing description lives in the decorator's ``help``, so that
    these Args and Raises sections stay where they are useful -- in the code --
    rather than being printed to the terminal by ``--help``.

    Args:
        task: The task, in plain language.
        start_url: Page to open first. Defaults to the configured start URL.
        profile: Endpoint profile for this run. Defaults to the configured one.
        headed: True to show the browser window, False to hide it, None to
            leave the setting as the configuration files have it.

    Raises:
        typer.Exit: With :data:`EXIT_SUCCESS` when the agent completed the
            task, :data:`EXIT_TASK_FAILURE` when it did not, and
            :data:`EXIT_USAGE` when the run could not be set up at all -- no
            API key, or a start URL that is not a usable address. The last of
            those is not a task failure: nothing was attempted and nothing was
            spent.
    """
    config = _startup(start_url=start_url, profile=profile, headed=headed)
    raise typer.Exit(_do_one_task(config, task))


@app.command(
    name="eval",
    help=(
        "Run the benchmark and write a folder of results.\n\n"
        "Cold and warm are the headline comparison: the same tasks with an "
        "empty skill library, and with a learned one."
    ),
)
def eval_command(
    suite: Annotated[
        Path,
        typer.Option("--suite", help="File of benchmark task definitions."),
    ] = DEFAULT_SUITE,
    trials: Annotated[
        int,
        typer.Option("--trials", min=1, help="Times to repeat each task."),
    ] = DEFAULT_TRIALS,
    cold: Annotated[
        bool,
        typer.Option(
            "--cold/--warm",
            help="Run with an empty skill library (cold) or the learned one (warm).",
        ),
    ] = True,
    profile: ProfileOption = None,
    headed: HeadedOption = None,
) -> None:
    """Run the benchmark and write a folder of results.

    Cold and warm are one flag rather than two because a sweep is one or the
    other, never both and never neither. Writing it ``--cold/--warm`` makes
    that true by construction, so there is no invalid combination to check for.

    Args:
        suite: File listing the benchmark tasks.
        trials: How many times to repeat each task.
        cold: True for an empty skill library, False for the learned one.
        profile: Endpoint profile for the sweep. This is the only thing that
            changes between arms of the multi-model comparison.
        headed: True to show the browser windows, False to hide them, None
            to leave the setting as the configuration files have it.
    """
    _startup(profile=profile, headed=headed)
    _not_built_yet(f"the benchmark harness ({'cold' if cold else 'warm'}, {trials} trials)")


@app.command(
    help=(
        "Turn successful runs into reusable skills.\n\n"
        "Deliberately a command you type, never something `waa run` does on "
        "its own: two benchmark sweeps only compare if the skill library did "
        "not change between them.\n\n"
        "Give it one run folder, or --all and a directory of them."
    )
)
def induce(
    run_folder: Annotated[
        Path | None,
        typer.Argument(help="One finished run to learn a skill from."),
    ] = None,
    all_runs: Annotated[
        Path | None,
        typer.Option("--all", help="Learn from every successful run in this directory."),
    ] = None,
) -> None:
    """Turn successful runs into reusable skills.

    One run folder or ``--all``, never both and never neither: those would ask
    for two different things, and for nothing, respectively. Neither is a
    request the command can honour, so both are refused before any work starts.

    Args:
        run_folder: A single run to learn from.
        all_runs: A directory of runs to learn from instead.

    Raises:
        typer.Exit: With :data:`EXIT_USAGE` if neither or both were given.
    """
    if run_folder is None and all_runs is None:
        typer.echo("waa: name a run folder, or pass --all with a directory of runs", err=True)
        raise typer.Exit(EXIT_USAGE)
    if run_folder is not None and all_runs is not None:
        typer.echo("waa: give a run folder or --all, not both", err=True)
        raise typer.Exit(EXIT_USAGE)

    _startup()
    _not_built_yet("skill induction")


@skills_app.command(
    "list",
    help="List every skill in the library, with how well each one has done.",
)
def skills_list() -> None:
    """List every skill in the library, with how well each one has done."""
    _startup()
    _not_built_yet("the skill library")


@skills_app.command(
    "show",
    help="Show one skill in full: its steps, its parameters, and its record.",
)
def skills_show(
    name: Annotated[str, typer.Argument(help="Which skill to show.")],
) -> None:
    """Show one skill in full: its steps, its parameters, and its record.

    Args:
        name: The skill to show.
    """
    _startup()
    _not_built_yet(f"showing skill {name!r}")


@skills_app.command(
    "retire",
    help=(
        "Retire a skill so the agent stops using it.\n\n"
        "Retired, not deleted: a skill that stopped working because a site "
        "changed is evidence about that site, and worth keeping."
    ),
)
def skills_retire(
    name: Annotated[str, typer.Argument(help="Which skill to retire.")],
) -> None:
    """Retire a skill so the agent stops using it.

    Args:
        name: The skill to retire.
    """
    _startup()
    _not_built_yet(f"retiring skill {name!r}")


@app.command(
    help=(
        "Grade finished runs, or check those grades by hand.\n\n"
        "Without --review, a second and stronger model marks each run. With "
        "--review, it shows you those runs one at a time and records your "
        "verdict, so the machine's marking can be measured rather than "
        "trusted."
    )
)
def judge(
    runs_dir: Annotated[
        Path,
        typer.Argument(help="Directory of finished runs to grade."),
    ] = DEFAULT_RUNS_DIR,
    review: Annotated[
        bool,
        typer.Option("--review", help="Review the machine grades yourself, one run at a time."),
    ] = False,
) -> None:
    """Grade finished runs, or check the grades by hand.

    Args:
        runs_dir: Directory of finished runs.
        review: Review the machine grades by hand instead of producing them.
    """
    _startup()
    _not_built_yet("human review" if review else "the judge")


#: The subcommand names, taken from the commands actually registered above
#: rather than written out a second time. ``python -m web_action_agent`` uses
#: this to tell a task string from a mistyped command.
COMMANDS = frozenset(typer.main.get_command(app).commands)


def main() -> int:
    """Run the command line and return the process exit code.

    Typer signals an exit by raising, which is right for a program but awkward
    for everything that calls this: the packaged ``waa`` command and the tests
    both want a number back. So the raise is caught here and turned into one,
    and this function is the single place the exit code is decided.

    Returns:
        int: :data:`EXIT_SUCCESS`, :data:`EXIT_TASK_FAILURE`, or
        :data:`EXIT_USAGE`.
    """
    try:
        app()
    except SystemExit as request:
        code = request.code
        if code is None:
            return EXIT_SUCCESS
        if isinstance(code, int):
            return code
        # A non-numeric exit value is a message, which some libraries use to
        # mean "stop, and say this". Print it and report a usage error.
        print(code, file=sys.stderr)
        return EXIT_USAGE
    return EXIT_SUCCESS


if __name__ == "__main__":
    raise SystemExit(main())
