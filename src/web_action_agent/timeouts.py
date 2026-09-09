"""The three ceilings a run cannot exceed, and what reaching one produces.

Something has to stop a run that is stuck. A page that never finishes loading,
an endpoint that accepts a request and goes quiet, an agent circling a site it
cannot make progress on -- none of these ends on its own, and a benchmark that
hangs is worse than one that fails, because a failure is a row in a table and a
hang is somebody noticing an hour later.

So there are three, nested, each with its own reason:

* **One action** -- 30 seconds. The browser is asked to do something and does
  not manage it. Reported by the executor as a timed-out step.
* **One model call** -- 90 seconds. The endpoint took the request and did not
  answer. Reported by the model client as its own kind of failure, so that a
  slow endpoint is never mistaken for a broken one.
* **One whole run** -- 5 minutes of wall clock. The catch-all: everything the
  first two do not cover, including all the small waits between them, and any
  loop that is technically making progress and getting nowhere.

**A ceiling produces a result, not an exception.** Every one of them ends in
``done(failed, "<category>: <what happened>")`` -- the same explicit terminal
outcome a successful run produces, with a different verdict. That is what makes
"how many runs died of a slow site?" a question the results table can answer.
An exception, by contrast, produces a stack trace, no verdict, and a run
missing from the table for reasons nobody wrote down.

**Where each one is enforced.** The action ceiling belongs to the executor,
because only it knows when a browser operation started. The model ceiling
belongs to the model client, for the same reason. The run ceiling belongs to
whoever owns the loop, and is checked between steps rather than during them --
which is why :class:`RunBudget` is a small value that can be handed around
rather than a timer that interrupts anything.

**What the run ceiling deliberately cannot do.** It does not stop work already
underway. A step that begins with four minutes gone will finish, and the run
ends after it rather than in the middle of it. Interrupting a step would mean
killing a browser call halfway and having nothing coherent to write in the
trace; overshooting by one step is the cheaper mistake, and
:meth:`RunBudget.clamp` keeps that overshoot small by shrinking the next
operation's own ceiling to whatever is left.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import TYPE_CHECKING

from .actions import Done

if TYPE_CHECKING:
    # For the annotation only; nothing here reads settings at runtime, and the
    # budget has to be constructible from a plain number in a test.
    from .config import Config

#: The browser did not finish one operation in time.
ACTION_TIMEOUT = "action_timeout"

#: The model endpoint took a request and did not answer in time.
LLM_TIMEOUT = "llm_timeout"

#: The whole run ran out of wall clock.
RUN_TIMEOUT = "run_timeout"

#: The three, from the smallest scope to the largest. Tests assert the list has
#: not grown quietly, and an evaluation counts failed runs by these names --
#: "slow site" and "slow endpoint" are different findings and get different
#: fixes, which is the whole reason they are not one category called "timeout".
TIMEOUT_CATEGORIES = (ACTION_TIMEOUT, LLM_TIMEOUT, RUN_TIMEOUT)


def timed_out(category: str, detail: str) -> Done:
    """Turn one expired ceiling into the run's terminal outcome.

    Args:
        category: One of :data:`TIMEOUT_CATEGORIES`.
        detail: What ran out, in a few words, for whoever reads the trace --
            ``"click 4 took longer than 30.0s"`` rather than ``"timeout"``.

    Returns:
        Done: ``done(failed, "<category>: <detail>")``. The category comes
        first so that counting failures by cause is a prefix match rather than
        a search through prose.

    Raises:
        ValueError: If the category is not one of the three. A misspelt
            category would produce a run whose failure reason nothing counts,
            which is worse than a loud stop here.
    """
    if category not in TIMEOUT_CATEGORIES:
        raise ValueError(
            f"{category!r} is not a timeout category; the categories are "
            f"{', '.join(TIMEOUT_CATEGORIES)}"
        )
    return Done(action="done", status="failed", reason=f"{category}: {detail}")


class RunBudget:
    """The wall clock for one run, and the verdict when it runs out.

    Deliberately passive. It does not interrupt anything, run a timer, or hold
    a thread: it answers "how much time is left" and "is it over", and the
    caller decides what to do about it. A budget that interrupted work would
    have to interrupt it *somewhere*, and every somewhere is a half-finished
    browser call with nothing coherent to record.

    The clock is injectable, which is not a detail: the alternative is a test
    of the five-minute ceiling that takes five minutes, so the ceiling ends up
    tested at one second and hoped about at five.

    Attributes:
        seconds: The ceiling this run was given, in seconds.
    """

    def __init__(self, seconds: float, *, clock: Callable[[], float] = time.monotonic) -> None:
        """Start the clock.

        The run begins now, not at some later ``start()``: a budget that had to
        be started separately would eventually be created early, forgotten, and
        quietly give a run more time than it was configured for.

        Args:
            seconds: How long the whole run may take.
            clock: What to read the time from. Defaults to a monotonic clock,
                which cannot go backwards when the machine's clock is adjusted
                mid-run.

        Raises:
            ValueError: If the ceiling is not positive. A run with no time
                could never do anything, and treating that as "no limit" --
                the other plausible reading -- would silently remove the
                ceiling instead of reporting a misconfigured one.
        """
        if seconds <= 0:
            raise ValueError(f"a run budget must be positive, not {seconds}")
        self.seconds = seconds
        self._clock = clock
        self._started = clock()

    @classmethod
    def from_config(cls, config: Config) -> RunBudget:
        """Start a budget from a run's settings.

        Args:
            config: The resolved configuration for this run.

        Returns:
            RunBudget: A budget on the configured wall-clock ceiling, started
            now.
        """
        return cls(config.run_timeout_s)

    @property
    def elapsed_s(self) -> float:
        """float: Seconds since the run began."""
        return self._clock() - self._started

    @property
    def remaining_s(self) -> float:
        """float: Seconds left, never negative -- an overrun reads as zero."""
        return max(0.0, self.seconds - self.elapsed_s)

    @property
    def expired(self) -> bool:
        """bool: Whether the run has used its whole budget."""
        return self.remaining_s <= 0

    def clamp(self, seconds: float) -> float:
        """Shrink one operation's own ceiling to fit inside what is left.

        A ninety-second model call started with ten seconds of run left would
        overshoot the run's ceiling by more than a minute. Clamping it means
        the call fails as a model timeout at ten seconds instead -- sooner, and
        under a name that says what actually happened.

        Args:
            seconds: The operation's own ceiling.

        Returns:
            float: The smaller of that and the time left in the run. Zero when
            the run is already over, which callers should read as "do not start
            this at all".
        """
        return min(seconds, self.remaining_s)

    def check(self) -> Done | None:
        """Ask whether the run may continue, and get the verdict if not.

        Written to be the first line of a loop iteration::

            ended = budget.check()
            if ended is not None:
                return ended

        Returns:
            Done | None: The terminal outcome when the budget is spent, or
            None while there is time left.
        """
        if not self.expired:
            return None
        return timed_out(RUN_TIMEOUT, f"{self.seconds:.0f}s of wall clock used")

    def summary(self) -> str:
        """Render the budget as the line a step log records.

        Returns:
            str: e.g. ``42s of 300s used``. Whole seconds, because nobody
            reading a run's progress needs milliseconds of a five-minute cap.
        """
        return f"{self.elapsed_s:.0f}s of {self.seconds:.0f}s used"
