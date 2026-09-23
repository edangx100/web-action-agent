"""What you watch in the terminal while a run happens.

The run folder (see :mod:`~web_action_agent.trace`) is written for a reader
coming back afterwards with a question. This module is for the person sitting in
front of the run *now*, and the two audiences want different things: the folder
wants everything, and a terminal wants one line per step that can be understood
at a glance, printed the moment the step finishes rather than when the run does.

**Streamed, not summarised.** The lines are emitted per step, as each step
completes. That is the whole difference between a tool you can debug and one
that stares back at you for ninety seconds and then tells you it failed: a run
that is stuck on a page is obvious from a stream and invisible from a summary.
It also means a run you interrupt has still told you where it got to, which is
the same argument the trace writer makes about the folder, applied to the
screen.

**Elements are named, not numbered.** Inside the loop an element is a number --
``click 4`` -- because numbers are what the planner is given and what the
validator re-checks. But a number is assigned per observation and means nothing
outside the one it came from, so a person reading ``click 4`` a minute later
cannot tell whether the agent clicked the right thing. Here the same action
reads ``click [4] link "Next page"``: the number, kept so it can be matched
against the trace, plus the role and accessible name that say what it actually
was.

**Four phases, and they are a closed list.** Every recorded step ends in one of
them, and the word sits in a fixed column so a run can be skimmed down that
column alone -- a column of ``act`` with one ``plan`` in it locates the
interesting step without reading a single line in full.

**The last line is the one a script reads.** It carries the verdict, the answer
or the reason, and the process's exit code, in that order, so that a person and
a shell get their answer from the same place. Everything on the way there goes
to standard error; only the answer itself goes to standard output, so
``waa run ... > answer.txt`` yields the answer and not a log.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from .actions import Action, Done, Fill, UseSkill

if TYPE_CHECKING:
    # Annotations only. The controller builds these and hands them here to be
    # rendered; importing it at runtime would be a cycle.
    from pathlib import Path

    from .controller import RunReport, StepReport
    from .observation import Observation

#: An ordinary action was chosen and performed. The common case, and the one
#: the eye should skip over.
ACT = "act"

#: A stored skill was replayed instead of a single action.
SKILL = "skill"

#: The agent ended the run itself, with an answer or an admission. The last
#: line of a run that reached its own conclusion.
DONE = "done"

#: The step never got past deciding: every attempt was refused, or the endpoint
#: broke. Nothing reached the browser, which is exactly what makes it worth a
#: word of its own rather than an empty action column.
PLAN = "plan"

#: Every phase a recorded step can be in. Closed, like the outcome categories,
#: so that a phase invented in one place cannot quietly appear in a trace
#: nothing else knows how to read.
PHASES = (ACT, SKILL, DONE, PLAN)

#: How wide the phase column is. One space wider than the longest phase, so the
#: decisions line up and can be read as a column.
_PHASE_WIDTH = max(len(phase) for phase in PHASES) + 1

#: How wide the step-number column is. Three digits covers any run: the step cap
#: is configurable but a run of a thousand steps is not a run anybody watches.
_NUMBER_WIDTH = 3

#: Verbs the phase column already says, so the decision half does not repeat
#: them: ``done  done success: 17 records`` is a stutter, and the phase is the
#: half that belongs in a column. Keyed by phase, so a phase that does not
#: shadow its verb -- ``act`` shadows nothing -- simply is not in here.
_VERB_THE_PHASE_ALREADY_SAYS = {DONE: "done ", SKILL: "use_skill "}

#: Written where a step ended before it chose anything. Spelled out rather than
#: left blank, because a blank column reads as "nothing happened" when what
#: happened is that everything proposed was refused.
_NO_ACTION = "no action"

#: The type of the per-step callback the controller calls as each step is
#: recorded. A plain function rather than an interface, because the controller
#: needs exactly one thing from the terminal -- "here is a finished step" -- and
#: anything larger would make the loop depend on how a run is displayed.
StepListener = Callable[["StepReport"], None]

#: Where each line goes. A function taking one line, so that the terminal, a
#: test's list, and a log file are all the same thing to this module.
LineWriter = Callable[[str], None]


def phase_of(step: StepReport) -> str:
    """Say which phase a finished step ended in.

    Derived from the step rather than recorded on it, because the phase is not
    a separate fact: it is what the presence of an action, and the kind of
    action it was, already say. Recording it separately would create a second
    version of the truth that could disagree with the first.

    Args:
        step: A step the controller has finished.

    Returns:
        str: One of :data:`PHASES`.
    """
    if step.action is None:
        # No action means the step never got past the planner -- every attempt
        # refused, or the model layer failed before one arrived.
        return PLAN
    if isinstance(step.action, Done):
        return DONE
    if isinstance(step.action, UseSkill):
        return SKILL
    return ACT


def describe_action(action: Action, observation: Observation) -> str:
    """Render one action for a person, naming its element rather than numbering it.

    The action's own :meth:`~web_action_agent.actions.BaseAction.describe` is
    kept compact on purpose -- it is what the stuck detector compares for
    identity and what memory records, and both want a short stable string. This
    is the other rendering: the same action, written out for somebody reading
    over the agent's shoulder, with the element resolved through the
    observation the action was planned against.

    Only the actions that carry an element are treated specially. Anything else
    falls back to its own ``describe()``, so an action added later reads
    correctly here without this function knowing about it.

    Args:
        action: The action to render.
        observation: The page it was planned against -- the one whose numbering
            the action's element id refers to.

    Returns:
        str: e.g. ``click [4] link "Next page"`` or
        ``fill [7] searchbox "Search the catalogue" with "membership"``. Falls
        back to the bare ``click 4`` form when the action names no element, or
        when the element is not in this observation -- which happens on a
        refused action, and where the number really is all that is known.
    """
    element_id = getattr(action, "element_id", None)
    if element_id is None:
        return action.describe()

    element = observation.element_by_id(element_id)
    if element is None:
        return action.describe()

    if isinstance(action, Fill):
        # "with" rather than a bare second quoted string, so the field's name
        # and the text typed into it cannot be read as one thing.
        return f'fill {element.label()} with "{action.text}"'
    # `action.action` is the verb from the action's own schema, so a future
    # element-bearing action with no extra parameters reads correctly here
    # without being named above.
    return f"{action.action} {element.label()}"


def step_line(step: StepReport) -> str:
    """Render one finished step as the single line the terminal shows.

    Args:
        step: The step, as the controller finished it.

    Returns:
        str: e.g.
        ``  2  act    press Enter -> ok (0.08s)``, with the refusals counted
        where there were any and the forcing named where there was one. The
        number and the phase sit in fixed columns so a run can be skimmed down
        them.
    """
    number = f"{step.number:>{_NUMBER_WIDTH}}"
    phase = f"{phase_of(step):<{_PHASE_WIDTH}}"
    line = f"{number}  {phase} {_what_happened(step)}"

    if step.attempts > 1:
        # Only shown when there was more than one, because "1 attempt" on every
        # ordinary step is noise that hides the steps where it was not 1.
        line += f" [{step.attempts} attempts]"
    if step.forced:
        line += f" [forced: {step.forced}]"
    return line


def opening_lines(task: str, start_url: str) -> list[str]:
    """Say what the run is about to try, before it tries it.

    Printed before the first step so the transcript describes itself: a
    terminal scrollback, or a log somebody pasted into a bug report, should not
    need the command that produced it to be readable.

    Args:
        task: The task in the user's own words.
        start_url: The page the run opens on.

    Returns:
        list[str]: Two lines and a blank one, ready to write in order.
    """
    return [f"task:  {task}", f"start: {start_url}", ""]


def closing_lines(report: RunReport, folder: Path, exit_code: int) -> list[str]:
    """Render the three lines a run ends with.

    Args:
        report: What the run produced.
        folder: The run's trace folder.
        exit_code: What the process is about to exit with.

    Returns:
        list[str]: The cost, the folder, and the final status line -- in that
        order, so the status line is last and a person reading the bottom of a
        scrollback finds the verdict rather than an accounting total.
    """
    return [
        "",
        report.cost.summary(),
        f"trace: {folder}",
        status_line(report, exit_code),
    ]


def status_line(report: RunReport, exit_code: int) -> str:
    """Render the last line of a run: the verdict, and what the process exits with.

    The two belong on one line because they are one fact stated twice, once for
    a person and once for a shell, and putting them apart is how they come to
    disagree. The exit code goes last so it is the final thing in the
    scrollback.

    Args:
        report: What the run produced.
        exit_code: The process's exit code.

    Returns:
        str: e.g. ``success in 3 steps: 17 records mention membership
        (exit 0)`` or ``failed in 25 steps (step_cap): the step cap of 25 was
        reached without finishing (exit 1)``.
    """
    return f"{report.outcome.summary()} (exit {exit_code})"


class TerminalTrace:
    """The run as it appears on screen, written line by line as it happens.

    Holds nothing but where to write. That is what lets the command line hand
    it a real stream, a test hand it a list, and neither of them be a special
    case.

    Attributes:
        write: Where each line goes.
    """

    def __init__(self, write: LineWriter) -> None:
        """Point the trace at somewhere to write.

        Args:
            write: Called once per line, without a trailing newline. In the
                command line this is a write to standard error: the steps are
                commentary, and standard output is reserved for the answer so
                that redirecting it gives the answer alone.
        """
        self.write = write

    def opening(self, task: str, start_url: str) -> None:
        """Announce the run before it starts.

        Args:
            task: The task in the user's own words.
            start_url: The page the run opens on.
        """
        self._write_all(opening_lines(task, start_url))

    def step(self, step: StepReport) -> None:
        """Show one finished step.

        This is the method handed to the controller as its per-step callback,
        which is why it takes a step and returns nothing: the loop reports that
        a step happened, and what that looks like is entirely this module's
        business.

        Args:
            step: The step, as the controller finished it.
        """
        self.write(step_line(step))

    def closing(self, report: RunReport, folder: Path, exit_code: int) -> None:
        """Show the verdict, the cost, and where the evidence went.

        Args:
            report: What the run produced.
            folder: The run's trace folder.
            exit_code: What the process is about to exit with.
        """
        self._write_all(closing_lines(report, folder, exit_code))

    def _write_all(self, lines: list[str]) -> None:
        """Write several lines in order.

        Args:
            lines: The lines to write.
        """
        for line in lines:
            self.write(line)


def _what_happened(step: StepReport) -> str:
    """Render the decision half of a step's line.

    Args:
        step: The step to describe.

    Returns:
        str: The action and what came of it -- ``press Enter -> ok (0.08s)`` --
        or the action alone where the browser was never asked to perform it,
        which is the case for the final ``done`` and for a replayed skill.
    """
    if step.action is None:
        return _refusal_note(step)

    described = describe_action(step.action, step.observation)
    shadowed = _VERB_THE_PHASE_ALREADY_SAYS.get(phase_of(step))
    if shadowed:
        described = described.removeprefix(shadowed)
    if step.result is None:
        return described

    # The result owns how a status and a complaint are written; only the action
    # is re-rendered here, because only the action gains from being named.
    line = f"{described} -> {step.result.status}"
    if step.result.detail:
        line += f": {step.result.detail}"
    line += f" ({step.result.elapsed_s:.2f}s)"
    if step.result.events:
        # A tab that opened, a dialog that was dismissed, a download refused.
        # Appended rather than dropped: handled silently, these are
        # indistinguishable on screen from a page that did nothing.
        line += " | " + "; ".join(event.summary() for event in step.result.events)
    return line


def _refusal_note(step: StepReport) -> str:
    """Explain a step that chose nothing at all.

    Args:
        step: A step whose action is None.

    Returns:
        str: ``no action``, with the refusal codes named when the validator was
        the reason. A step that reached this point because the endpoint broke
        has no rejections to name, and the run's final status line carries that
        reason instead.
    """
    if not step.rejections:
        return _NO_ACTION
    codes = ", ".join(rejection.code for rejection in step.rejections)
    return f"{_NO_ACTION}: every attempt refused ({codes})"
