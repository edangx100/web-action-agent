"""Putting the agent together and running one task, in one place.

Two commands run the agent: ``waa run`` does one task you type, and ``waa
eval`` runs every task in the benchmark, several at once. Both must build the
agent *the same way*. If they did not, the benchmark would be measuring a
slightly different agent from the one people use, and its numbers would be
about the wrong thing. So the building happens here, once, and both commands
call :func:`run_agent`.

What differs between the two callers is passed in rather than decided here:

* **the model client** -- each caller builds its own, which is also how a
  test hands in a stand-in model;
* **the browser** -- ``waa run`` starts one for its single task; a benchmark
  worker starts one and reuses it for many tasks, each in a fresh context;
* **the skill library** -- ``waa run`` uses the real one; a benchmark trial
  uses its own copy, so that one trial's results cannot change the next's;
* **where progress lines go** -- the terminal for ``waa run``; a worker's
  quieter log for the benchmark.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .allowlist import Allowlist, AllowlistError
from .browser import BrowserProcess
from .config import Config
from .controller import Controller, RunReport
from .executor import Executor
from .memory import Memory
from .model_client import ModelClient
from .planner import Planner
from .politeness import politeness_for_run
from .replay import SkillReplayer
from .reporting import StepListener, TerminalTrace
from .retrieval import for_url
from .screenshot_gate import ScreenshotGate
from .skill_store import SkillStore, SkillStoreError
from .timeouts import RunBudget
from .trace import TraceError, TraceWriter
from .validator import Validator


class RunSetupError(Exception):
    """The run could not be started, so nothing was attempted or spent.

    Raised for a missing API key, an unusable start URL, or a run folder that
    cannot be written. Distinct from a run that started and failed: that is
    an ordinary result with its own folder and verdict.
    """


@dataclass(frozen=True)
class AgentRun:
    """What one finished run produced.

    Attributes:
        report: The run's steps and verdict.
        folder: Where its evidence was written.
        watching: The progress printer it used, so the caller can close off
            the display in its own way.
    """

    report: RunReport
    folder: Path
    watching: TerminalTrace


def run_agent(
    config: Config,
    task: str,
    *,
    client: ModelClient,
    folder: Path,
    browser: BrowserProcess,
    library: SkillStore,
    write: Callable[[str], None],
    task_extra: Iterable[str] = (),
) -> AgentRun:
    """Assemble the agent and run one task from ``config.start_url``.

    The order is forced by what depends on what: the model client needs a
    profile before the planner can exist, the screenshot gate needs to know
    whether that profile can see images, and the validator needs the
    allowlist the start URL implies.

    Args:
        config: The settings for this run. Its ``start_url`` is where the run
            opens.
        task: The task in plain language.
        client: The model client the planner calls.
        folder: Where the run's evidence goes. Must not hold another run.
        browser: A running browser. The task gets a fresh context in it,
            closed when the task ends, so nothing carries over between tasks.
        library: The skill library this run may use and records its replays
            in.
        write: Where each progress line goes.
        task_extra: Extra sites this task may visit, from its task file.

    Returns:
        AgentRun: The run's report, folder, and progress printer.

    Raises:
        RunSetupError: If the run could not be set up -- an unusable start
            URL, or a folder that cannot be written. Raised before the task's
            browser context opens and before any model call, so nothing is
            spent.
    """
    try:
        allowlist = Allowlist.derive(
            config.start_url, config_extra=config.allowlist_extra, task_extra=task_extra
        )
        # Opened before anything is spent: a run whose folder cannot be
        # written is a run that would produce no evidence, and finding that
        # out after paying for twenty model calls would be too late.
        trace = TraceWriter.open(
            folder,
            task=task,
            start_url=config.start_url,
            config=config,
            # A fact about the endpoint, not a setting: the reader of a trace
            # should be told once that the model could not see.
            text_only=not client.supports_vision,
        )
    except (AllowlistError, TraceError) as error:
        raise RunSetupError(str(error)) from error

    throttle, robots = politeness_for_run(config)
    watching = TerminalTrace(write)
    watching.opening(task, config.start_url)

    # The skills belonging to this site, read once before the browser
    # starts, so the run works from a fixed list.
    offered = for_url(library.load_all(), config.start_url)
    if offered.skills:
        watching.write(f"skills: {offered.summary()}")

    with browser.task_session() as session:
        executor = Executor(session, action_timeout_s=config.action_timeout_s, throttle=throttle)
        # The gate is told about the library so that `use_skill` is an action
        # at all: without one it refuses every invocation.
        validator = Validator(allowlist, robots=robots, skills=library)
        controller = Controller(
            planner=Planner(client),
            validator=validator,
            executor=executor,
            memory=Memory.from_config(config),
            max_steps=config.max_steps,
            retry_budget=config.retry_budget,
            gate=ScreenshotGate.from_config(config, supports_vision=client.supports_vision),
            budget=RunBudget.from_config(config),
            screenshot_dir=trace.screenshot_dir,
            config=config,
            trace=trace,
            skills=offered.lines,
            replay=SkillReplayer(
                executor=executor,
                validator=validator,
                config=config,
                library=library,
                screenshot_dir=trace.screenshot_dir,
            ),
            on_step=watch_and_tally(watching, library),
        )
        report = controller.run(task, start_url=config.start_url)
    return AgentRun(report=report, folder=folder, watching=watching)


def watch_and_tally(watching: TerminalTrace, library: SkillStore) -> StepListener:
    """Show each finished step, and write down how any replay in it went.

    Two things happen to a step and they belong to different components, so
    they are joined here rather than inside either: the display shows it, and
    the library counts it. The library's count is what eventually retires a
    skill that a site has changed under.

    What is counted is whether the *recipe* worked, not whether the run did. A
    replay that handed control back after two steps may be part of a run that
    goes on to succeed by hand -- a good outcome for the task, and still a
    failure of the skill.

    Args:
        watching: The progress printer.
        library: The skill library to record the use in.

    Returns:
        StepListener: The callback the loop calls as each step completes.
    """

    def on_step(step: Any) -> None:
        """Show one step, and tally it if it replayed a skill.

        Args:
            step: The step the controller has just finished.
        """
        watching.step(step)
        if step.replay is None:
            return
        try:
            library.record_use(step.replay.skill, succeeded=step.replay.ok)
        except SkillStoreError as error:
            # Bookkeeping that cannot be written must not end a run that is
            # otherwise working. Said out loud rather than swallowed, because
            # a library silently stopping counting is how a skill that has
            # stopped working stays in service forever.
            watching.write(f"warning: could not record the use of {step.replay.skill!r}: {error}")

    return on_step
