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
from pydantic import ValidationError

from web_action_agent.agent_run import RunSetupError, run_agent
from web_action_agent.allowlist import Allowlist
from web_action_agent.benchmark import BenchmarkError, load_suite
from web_action_agent.browser import BrowserProcess, BrowserSettings
from web_action_agent.budget import unpriced_routes
from web_action_agent.capabilities import report_capabilities
from web_action_agent.config import Config, ConfigError, flag_for, load_config, resolve_env
from web_action_agent.controller import RunReport
from web_action_agent.eval_harness import (
    SWEEP_FILE,
    Harness,
    HarnessError,
    ServiceBusyError,
    default_sweep_dir,
    passing_trials,
)
from web_action_agent.executor import Executor
from web_action_agent.inducer import Inducer, InductionError
from web_action_agent.judge import Judge, independence_problem
from web_action_agent.metrics import HUMAN_LABELS_FILE, MetricsError, find_sweeps, write_results
from web_action_agent.model_client import (
    ModelClient,
    ModelClientError,
    ProfileError,
    resolve_profile,
)
from web_action_agent.reporting import TerminalTrace
from web_action_agent.review import DEFAULT_TARGET, ReviewError
from web_action_agent.review import review as review_trials
from web_action_agent.skill_schema import Skill
from web_action_agent.skill_store import SkillStore, SkillStoreError
from web_action_agent.trace import TraceError
from web_action_agent.trace_reader import read_run
from web_action_agent.validator import Validator
from web_action_agent.verifier import SkillVerifier, Verification, VerificationError

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

#: Where the screenshots of each skill check go, inside the runs folder: one
#: folder per check, named for when it ran and the skill it checked.
VERIFICATION_DIR = "_verification"

#: Where the learned skills live. One folder of readable files, beside the run
#: folders, so that what the agent has learnt is as easy to look at as what it
#: did.
SKILLS_ROOT = Path("skills")

#: The benchmark suite manifest used when ``--suite`` is not given. It lists
#: the sites and the tasks; each task is its own file in ``benchmarks/tasks/``.
DEFAULT_SUITE = Path("benchmarks/suite.yaml")

#: Where finished runs are written, and therefore where the commands that read
#: runs back look first.
DEFAULT_RUNS_DIR = Path("runs")

#: Where the results tables are written: one Markdown file per sweep, and a
#: README comparing them. Committed, unlike the run folders they summarise.
RESULTS_ROOT = Path("results")

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
        help="Endpoint profile to use: a name from profiles.yaml, such as openrouter or nvidia.",
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

    # A judge from the planner's own family of models is refused here, before
    # anything runs, rather than when the first run is marked: by then a
    # whole sweep may have been paid for with no way to mark it fairly.
    problem = independence_problem(config)
    if problem is not None:
        typer.echo(f"waa: {problem}", err=True)
        raise typer.Exit(EXIT_USAGE)
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
    except ModelClientError as error:
        typer.echo(f"waa: {error}", err=True)
        return EXIT_USAGE

    # One browser process, one task, one context inside it -- the same shape a
    # benchmark worker uses, and the same assembly, so an ad-hoc run and a
    # measured one are the same agent. Everything the watcher sees goes to
    # standard error, so that standard output carries the answer alone.
    try:
        with BrowserProcess(BrowserSettings.from_config(config)) as browser:
            finished = run_agent(
                config,
                task,
                client=client,
                folder=folder,
                browser=browser,
                library=SkillStore.from_config(config, SKILLS_ROOT),
                write=lambda line: typer.echo(line, err=True),
            )
    except RunSetupError as error:
        # Nothing has been attempted and nothing spent: the fix is to the
        # configuration, which is what exit code 2 means.
        typer.echo(f"waa: {error}", err=True)
        return EXIT_USAGE
    report, watching = finished.report, finished.watching

    exit_code = EXIT_SUCCESS if report.outcome.ok else EXIT_TASK_FAILURE
    _report_run(report, folder, watching, exit_code)
    return exit_code


def _judge_for(config: Config) -> ModelClient | None:
    """Build the model that decides whether a replayed skill landed correctly.

    Deliberately a *different* endpoint from the one that drove the run, when
    one is configured: a model marking work it produced is marking its own
    homework, and the project's whole output is a table of marks.

    Args:
        config: The run's resolved settings, for the timeout.

    Returns:
        ModelClient | None: The judge, or None when none is configured. None is
        a legitimate answer rather than an error -- verification then runs its
        mechanical layer alone, which the project allows as a stated position
        with the retirement threshold as the quality filter instead. The caller
        says out loud which of the two happened, so a library checked without a
        judge cannot be mistaken for one checked with.
    """
    name = (config.judge_profile or "").strip()
    if not name:
        return None
    try:
        # The profile is chosen through the config, which is the one place
        # profile selection happens -- so the judge gets the run's timeouts and
        # backoff policy without this having to know about either.
        return ModelClient.from_config(config.model_copy(update={"profile": name}))
    except (ModelClientError, ProfileError):
        # Named but unusable -- a profile that is not there, or a key that is
        # not set. Treated as "no judge" rather than fatal, because the
        # mechanical layer still catches most bad candidates and refusing to
        # induce at all would be a worse answer than inducing carefully.
        return None


def _induce_from(
    config: Config, folders: list[Path], task_of: dict[Path, str] | None = None
) -> int:
    """Learn a skill from each finished run, keeping only the ones that check out.

    The order is the point: induce, then *verify against a live page with
    different values*, then store. A candidate that skipped the middle step
    would be a recipe believed on the strength of the one run it came from.

    Args:
        config: The run's resolved settings.
        folders: The run folders to learn from.
        task_of: For runs from a benchmark sweep, each folder's task. Once a
            task has a stored skill, its other trials are passed over: three
            passing trials of one task would give three copies of one route,
            each costing a model call and a live check.

    Returns:
        int: :data:`EXIT_SUCCESS` if at least one skill was stored, or if there
        was nothing eligible to learn from -- a directory of failed runs is not
        a failed command. :data:`EXIT_TASK_FAILURE` when every candidate that
        was produced went on to be discarded, which is a result worth an exit
        code because it usually means the site has moved on.
    """
    try:
        client = ModelClient.from_config(config)
    except ModelClientError as error:
        typer.echo(f"waa: {error}", err=True)
        return EXIT_USAGE

    judge = _judge_for(config)
    typer.echo(
        "checking candidates by replay"
        + (" and one judge call each" if judge else " alone (no judge configured)"),
        err=True,
    )

    library = SkillStore.from_config(config, SKILLS_ROOT)
    inducer = Inducer(client)
    induced = verified = discarded = 0

    learned: set[str] = set()
    task_of = task_of or {}
    with BrowserProcess(BrowserSettings.from_config(config)) as browser:
        for folder in folders:
            task = task_of.get(folder)
            if task is not None and task in learned:
                typer.echo(f"  {_name(folder)}: skipped, {task} already has a skill", err=True)
                continue
            try:
                run = read_run(folder)
            except TraceError as error:
                typer.echo(f"  {_name(folder)}: not a run folder ({error})", err=True)
                continue

            try:
                # The result wraps the candidate with what the call cost and
                # what was left out of the route; the skill is one field of it.
                induction = inducer.induce(run)
                candidate = induction.skill
            except InductionError as error:
                # Not an error in the ordinary sense: most run folders are not
                # eligible, because most runs fail, and only successful ones
                # may be learnt from.
                typer.echo(f"  {_name(folder)}: nothing to learn ({error})", err=True)
                continue
            except (ModelClientError, ValidationError) as error:
                typer.echo(
                    f"  {_name(folder)}: could not be turned into a skill ({error})", err=True
                )
                continue

            induced += 1
            # What was left out -- failed steps, parameters no step could use
            # -- is said, so a corrected candidate is never a silent one.
            for note in induction.notes:
                typer.echo(f"  {_name(folder)}: note: {note}", err=True)
            # The replay's screenshots are kept, so a candidate that is
            # discarded can be looked into rather than taken on trust.
            evidence = _evidence_dir(candidate)
            verdict = _check(config, browser, candidate, run.start_url, judge, evidence)
            if verdict is None:
                continue
            if verdict.passed:
                library.save(candidate)
                verified += 1
                if task is not None:
                    learned.add(task)
                typer.echo(
                    f"  {_name(folder)}: stored {candidate.name!r} - {verdict.reason}", err=True
                )
            else:
                discarded += 1
                typer.echo(f"  {_name(folder)}: {verdict.summary()}", err=True)
                typer.echo(f"  {_name(folder)}: the check's screenshots: {evidence}", err=True)

    typer.echo(f"{induced} induced, {verified} verified, {discarded} discarded", err=True)
    if verified:
        typer.echo(f"library: {SKILLS_ROOT}", err=True)
    return EXIT_TASK_FAILURE if induced and not verified else EXIT_SUCCESS


def _evidence_dir(candidate: Skill) -> Path:
    """Choose the folder for one skill check's screenshots.

    Args:
        candidate: The skill about to be checked.

    Returns:
        Path: ``runs/_verification/<time>-<skill name>``, new for each check.
    """
    stamp = datetime.now(UTC).strftime("%Y-%m-%dT%H-%M-%SZ")
    return RUNS_ROOT / VERIFICATION_DIR / f"{stamp}-{candidate.name}"


def _name(folder: Path) -> str:
    """Name a run folder in a progress line.

    Args:
        folder: A run folder.

    Returns:
        str: ``<task>/trial-<n>`` for a benchmark trial, whose own name alone
        (``trial-1``) would not say which task it was; the folder's name
        otherwise.
    """
    if folder.name.startswith("trial-"):
        return f"{folder.parent.name}/{folder.name}"
    return folder.name


def _check(
    config: Config,
    browser: BrowserProcess,
    candidate: Skill,
    start_url: str,
    judge: ModelClient | None,
    evidence: Path | None = None,
) -> Verification | None:
    """Replay one candidate with different values, and say whether it worked.

    Each candidate gets its own browser context, so a page left over from
    checking one skill cannot be what the next one is checked against.

    Args:
        config: The run's resolved settings.
        browser: The browser process to open a context in.
        candidate: The skill to check.
        start_url: The page its run started from, which is where the check
            begins. A skill has no starting page of its own: it is a move the
            planner may make from wherever the agent already is.
        judge: The model that decides whether the final page is right, or None
            for the mechanical layer alone.
        evidence: Where the replay's screenshots go, or None for nowhere.

    Returns:
        Verification | None: The verdict, or None when the check could not be
        carried out -- a judge that was unreachable, a page that would not
        open. Those are not the candidate's fault and it is not written off for
        them; it is simply not stored this time.
    """
    with browser.task_session() as session:
        executor = Executor(session, action_timeout_s=config.action_timeout_s)
        verifier = SkillVerifier(
            executor=executor,
            validator=Validator(Allowlist.derive(start_url)),
            config=config,
            judge=judge,
            screenshot_dir=evidence,
        )
        try:
            return verifier.verify(candidate, start_url=start_url)
        except VerificationError as error:
            typer.echo(f"  {candidate.name}: could not be checked ({error})", err=True)
            return None


def _describe_skill(skill: Skill) -> list[str]:
    """Write one skill out in full, for a person deciding whether to trust it.

    Args:
        skill: The skill to describe.

    Returns:
        list[str]: The lines to print, in order: what it is, what it needs,
        what it does step by step, how it has done, and how it was checked --
        which is the order somebody reads them in when asking whether to keep
        it.
    """
    lines = [
        f"{skill.name} - {skill.description}",
        f"  site     {skill.domain}",
        f"  record   {skill.stats.successes}/{skill.stats.uses} replays worked"
        if skill.stats.uses
        else "  record   never replayed",
    ]
    if not skill.active:
        lines.append(f"  retired  {skill.retired_at}: {skill.retired_reason}")
    if skill.induced_from:
        lines.append(f"  learnt   from run {skill.induced_from}")

    if skill.params:
        lines.append("  needs")
        for param in skill.params:
            optional = "" if param.required else " (optional)"
            example = f" e.g. {param.example!r}" if param.example else ""
            lines.append(f"    {param.name}{optional} - {param.description}{example}")

    lines.append("  does")
    lines.extend(f"    {number}. {step.describe()}" for number, step in enumerate(skill.steps, 1))

    # The check is shown because it is the reason the skill is trusted at all,
    # and because a reader who disagrees with it has found a real problem.
    values = ", ".join(
        f"{name}={value!r}" for name, value in sorted(skill.verification.params.items())
    )
    lines.append("  checked  by replaying it with " + (values or "no values"))
    lines.append(f"           expecting {skill.verification.expected_outcome}")
    return lines


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
        typer.Option("--suite", help="Benchmark suite manifest listing the tasks."),
    ] = DEFAULT_SUITE,
    trials: Annotated[
        int,
        typer.Option("--trials", min=1, help="Times to repeat each task."),
    ] = DEFAULT_TRIALS,
    tasks: Annotated[
        list[str] | None,
        typer.Option(
            "--task",
            help="Run only this task, by id. Repeat for several. Default: every task.",
        ),
    ] = None,
    cold: Annotated[
        bool,
        typer.Option(
            "--cold/--warm",
            help="Run with an empty skill library (cold) or the learned one (warm).",
        ),
    ] = True,
    workers: Annotated[
        int | None,
        typer.Option("--workers", min=1, help="Trials to run at once. Default from config."),
    ] = None,
    out: Annotated[
        Path | None,
        typer.Option(
            "--out",
            help="Sweep folder. Default runs/eval-<profile>-<cold|warm>; rerun to resume.",
        ),
    ] = None,
    budget_usd: Annotated[
        float | None,
        typer.Option(
            "--budget-usd", min=0.0, help="Money ceiling for the sweep, in dollars. Default $15."
        ),
    ] = None,
    budget_hours: Annotated[
        float | None,
        typer.Option(
            "--budget-hours", min=0.0, help="Time ceiling for the sweep, in hours. Default 6."
        ),
    ] = None,
    profile: ProfileOption = None,
    headed: HeadedOption = None,
) -> None:
    """Run the benchmark and write a folder of results.

    Cold and warm are one flag rather than two because a sweep is one or the
    other, never both and never neither. Writing it ``--cold/--warm`` makes
    that true by construction, so there is no invalid combination to check for.

    Running the same command again carries on where it stopped: trials already
    finished and marked are skipped, and a run that was killed partway is done
    again from the start.

    Args:
        suite: The suite manifest, which lists the benchmark tasks.
        trials: How many times to repeat each task.
        tasks: The ids of the tasks to run, or None for every task. A sweep
            of some tasks is named for them, so it cannot later be resumed
            as a sweep of all of them.
        cold: True for an empty skill library, False for the learned one.
        workers: How many trials run at once, or None for the configured
            number.
        out: The sweep's folder, or None for the default, which is named
            after what the sweep measures so that a rerun finds it.
        budget_usd: The money ceiling, or None for the configured one. Once
            the sweep has spent this much, no new trial starts.
        budget_hours: The time ceiling, or None for the configured one.
        profile: Endpoint profile for the sweep. This is the only thing that
            changes between arms of the multi-model comparison.
        headed: True to show the browser windows, False to hide them, None
            to leave the setting as the configuration files have it.

    Raises:
        typer.Exit: :data:`EXIT_SUCCESS` when every trial has a mark (pass or
            fail -- a failed task is a result, not an error),
            :data:`EXIT_TASK_FAILURE` when some trial was left without one
            and a rerun is needed -- including when a ceiling stopped the
            sweep early -- and :data:`EXIT_USAGE` when the sweep could not
            start at all.
    """
    config = _startup(
        profile=profile, headed=headed, budget_usd=budget_usd, budget_hours=budget_hours
    )
    try:
        loaded = load_suite(suite).only(tuple(tasks or ()))
        judge = Judge.from_config(config) if config.judge_profile else None
    except (BenchmarkError, ConfigError, ModelClientError) as error:
        typer.echo(f"waa: {error}", err=True)
        raise typer.Exit(EXIT_USAGE) from error
    if judge is None:
        typer.echo(
            "waa: no judge configured: tasks marked by the judge will wait for one",
            err=True,
        )

    # Said before anything is spent: a model with no price is a model whose
    # calls the money ceiling cannot count.
    profiles = [resolve_profile(config)] + ([judge.client.profile] if judge else [])
    for route in unpriced_routes(profiles):
        typer.echo(
            f"waa: warning: {route} has no price in pricing.yaml, so the money ceiling "
            "cannot count its calls (the time ceiling still applies)",
            err=True,
        )
    typer.echo(
        f"budget: ${config.budget_usd:.2f} and {config.budget_hours:g} hours for this sweep",
        err=True,
    )

    harness = Harness(
        config,
        loaded,
        out or default_sweep_dir(RUNS_ROOT, config.profile, cold),
        trials=trials,
        cold=cold,
        workers=workers,
        skills_root=SKILLS_ROOT,
        judge=judge,
        log=lambda line: typer.echo(line, err=True),
    )
    try:
        report = harness.run()
    except ServiceBusyError as error:
        # Not a mistake, and not permanent: the same command later may work,
        # so it is reported as unfinished, which a retrying script retries.
        typer.echo(f"waa: {error}. Nothing has run; try again later.", err=True)
        raise typer.Exit(EXIT_TASK_FAILURE) from error
    except HarnessError as error:
        typer.echo(f"waa: {error}", err=True)
        raise typer.Exit(EXIT_USAGE) from error

    typer.echo(report.summary())
    typer.echo(f"runs: {report.folder}")
    # The tables are rewritten after every sweep -- this one and every other
    # sweep found beside it -- so the comparison is never out of date.
    if (report.folder / SWEEP_FILE).exists():
        sweeps = sorted({*find_sweeps(RUNS_ROOT), report.folder})
        try:
            written = write_results(sweeps, RESULTS_ROOT, min_labels=config.agreement_min_labels)
        except MetricsError as error:
            # Loud on purpose: a table with a blank where a failure's kind
            # should be is worse than no table, because nobody chases a blank.
            typer.echo(f"waa: the results tables were not written: {error}", err=True)
            raise typer.Exit(EXIT_TASK_FAILURE) from error
        else:
            typer.echo(f"tables: {', '.join(str(p) for p in written)}")
    if report.stopped_by:
        typer.echo(f"stopped early: {report.stopped_by}")
    unmarked = [r for r in report.results if r.verdict is None or not r.verdict.complete]
    unfinished = unmarked or report.not_started
    raise typer.Exit(EXIT_TASK_FAILURE if unfinished else EXIT_SUCCESS)


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

    config = _startup()
    task_of: dict[Path, str] = {}
    if all_runs is not None and (all_runs / SWEEP_FILE).exists():
        # A benchmark sweep keeps its runs two levels down, beside folders
        # that are not runs. Learn only from the trials it marked a pass.
        passing = passing_trials(all_runs)
        task_of = dict(passing)
        folders: list[Path | None] = [folder for folder, _ in passing]
        typer.echo(
            f"sweep: learning from the {len(passing)} trial(s) the benchmark marked a pass, "
            "at most one skill per task",
            err=True,
        )
        if not passing:
            typer.echo(f"waa: {all_runs} has no passing trials to learn from", err=True)
            raise typer.Exit(EXIT_TASK_FAILURE)
    else:
        folders = sorted(f for f in all_runs.iterdir() if f.is_dir()) if all_runs else [run_folder]
        if all_runs is not None and not folders:
            typer.echo(f"waa: {all_runs} holds no run folders", err=True)
            raise typer.Exit(EXIT_USAGE)
    raise typer.Exit(_induce_from(config, [f for f in folders if f is not None], task_of))


@skills_app.command(
    "list",
    help="List every skill in the library, with how well each one has done.",
)
def skills_list() -> None:
    """List every skill in the library, with how well each one has done.

    Retired skills are shown too, and marked. They are not offered to the agent
    any more, but they are part of what the library knows -- and a list that
    hid them would leave somebody wondering where a skill had gone.

    Raises:
        typer.Exit: With :data:`EXIT_SUCCESS` always; an empty library is the
            ordinary state of a fresh install rather than a failure.
    """
    config = _startup()
    library = SkillStore.from_config(config, SKILLS_ROOT)
    try:
        skills = library.load_all()
    except SkillStoreError as error:
        typer.echo(f"waa: {error}", err=True)
        raise typer.Exit(EXIT_USAGE) from error

    if not skills:
        typer.echo(f"no skills yet. run some tasks, then: waa induce --all {RUNS_ROOT}", err=True)
        raise typer.Exit(EXIT_SUCCESS)

    for skill in skills:
        typer.echo(skill.summary())
    active = sum(1 for skill in skills if skill.active)
    typer.echo(f"{active} in service, {len(skills) - active} retired", err=True)
    raise typer.Exit(EXIT_SUCCESS)


@skills_app.command(
    "show",
    help="Show one skill in full: its steps, its parameters, and its record.",
)
def skills_show(
    name: Annotated[str, typer.Argument(help="Which skill to show.")],
) -> None:
    """Show one skill in full: its steps, its parameters, and its record.

    Written out as prose rather than as the file's own JSON. The file is
    readable and sitting right there for anybody who wants it; what this is for
    is the question "what does this thing actually do", and a numbered list of
    steps in words answers that faster than a nested document.

    Args:
        name: The skill to show.

    Raises:
        typer.Exit: With :data:`EXIT_USAGE` if the library has no such skill.
    """
    config = _startup()
    library = SkillStore.from_config(config, SKILLS_ROOT)
    try:
        skill = library.find(name)
    except SkillStoreError as error:
        typer.echo(f"waa: {error}", err=True)
        raise typer.Exit(EXIT_USAGE) from error
    if skill is None:
        typer.echo(f"waa: no skill named {name!r}. try: waa skills list", err=True)
        raise typer.Exit(EXIT_USAGE)

    for line in _describe_skill(skill):
        typer.echo(line)
    raise typer.Exit(EXIT_SUCCESS)


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
    reason: Annotated[
        str,
        typer.Option("--reason", help="Why, for whoever reads the library later."),
    ] = "retired by hand",
) -> None:
    """Retire a skill so the agent stops using it.

    For the reasons a success rate cannot see. A skill that works perfectly and
    does the wrong thing has a flawless record and should still go, and no
    amount of counting will ever notice.

    Args:
        name: The skill to retire.
        reason: Why, recorded in the file for whoever reads it later.

    Raises:
        typer.Exit: With :data:`EXIT_USAGE` if the library has no such skill.
    """
    config = _startup()
    library = SkillStore.from_config(config, SKILLS_ROOT)
    try:
        retired = library.retire(name, reason)
    except SkillStoreError as error:
        typer.echo(f"waa: {error}", err=True)
        raise typer.Exit(EXIT_USAGE) from error

    typer.echo(f"retired {retired.name!r}: {retired.retired_reason}")
    typer.echo(
        f"kept at {library.path_for(name)} -- its record is the evidence for why it went",
        err=True,
    )
    raise typer.Exit(EXIT_SUCCESS)


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
    target: Annotated[
        int,
        typer.Option(
            "--target", min=1, help="With --review: stop once this many runs have your label."
        ),
    ] = DEFAULT_TARGET,
    labels: Annotated[
        Path | None,
        typer.Option(
            "--labels",
            help="With --review: the file your labels go in. Default results/human_labels.jsonl.",
        ),
    ] = None,
    suite: Annotated[
        Path,
        typer.Option("--suite", help="With --review: the suite, to show what counts as success."),
    ] = DEFAULT_SUITE,
) -> None:
    """Grade finished runs, or check the grades by hand.

    Args:
        runs_dir: Directory of finished runs.
        review: Review the machine grades by hand instead of producing them.
        target: For a review, how many runs should end up with a label.
        labels: For a review, the labels file, or None for the default.
        suite: For a review, the benchmark suite the runs came from.

    Raises:
        typer.Exit: For a review, :data:`EXIT_SUCCESS` when the session ended
            normally (quitting part-way is normal: it resumes next time),
            :data:`EXIT_TASK_FAILURE` when there was nothing to label, and
            :data:`EXIT_USAGE` when ``runs_dir`` holds no sweeps.
    """
    if not review:
        _startup()
        _not_built_yet("the judge")
    _review(runs_dir, labels or RESULTS_ROOT / HUMAN_LABELS_FILE, suite, target)


def _ask(question: str) -> str:
    """Ask the person a question at the terminal.

    Args:
        question: The question, shown before the cursor.

    Returns:
        str: What they typed. ``"q"`` (quit) when the input runs out or they
        press Ctrl-C, so either ends the session cleanly, keeping every label
        already given.
    """
    try:
        return input(question)
    except (EOFError, KeyboardInterrupt):
        typer.echo("")
        return "q"


def _review(runs_dir: Path, labels_path: Path, suite_path: Path, target: int) -> None:
    """Run a review session at the terminal, and exit.

    No settings or keys are read: reviewing calls no AI, so it works on a
    machine with nothing configured.

    Args:
        runs_dir: A sweep folder, or a folder of them.
        labels_path: Where the labels are kept.
        suite_path: The benchmark suite manifest.
        target: How many runs should end up with a label.

    Raises:
        typer.Exit: Always; see :func:`judge` for the codes.
    """
    # The suite only adds "what counts as success" to each screen. Without it
    # the review still works, so a suite that cannot be read is a warning.
    try:
        loaded = load_suite(suite_path)
    except BenchmarkError as error:
        typer.echo(f"waa: warning: {error}: what counts as success will not be shown", err=True)
        loaded = None
    try:
        summary = review_trials(
            runs_dir, labels_path, ask=_ask, say=typer.echo, suite=loaded, target=target
        )
    except ReviewError as error:
        typer.echo(f"waa: {error}", err=True)
        raise typer.Exit(EXIT_USAGE) from error

    typer.echo(summary.render())
    typer.echo(f"labels: {labels_path}")
    if summary.labelled_now:
        # The review writes labels, not tables; the tables are rewritten by
        # the eval command, which is the one that knows the settings.
        typer.echo("agreement figures: updated the next time `waa eval` writes the tables")
    if summary.reviewable == 0:
        typer.echo("waa: no trial has a finished mark yet, so there is nothing to check", err=True)
        raise typer.Exit(EXIT_TASK_FAILURE)
    raise typer.Exit(EXIT_SUCCESS)


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
