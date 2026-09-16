"""How a run ends, and the fixed list of ways it is allowed to.

Every run produces exactly one of these, and producing it is the controller's
job alone. That is the point of the module: not the dataclass, but the closed
list of *categories* beside it.

**Why a category and not just a message.** "Failed: the button wasn't there" is
readable and useless. Nobody can count it, group it, or put it in a column. The
evaluation's whole output is a table of a few hundred runs, and the questions
asked of that table are "how many failed because the site was slow?" and "did
the warm pass run out of steps less often than the cold one?" -- questions a
sentence cannot answer and a category can. So the sentence stays, for the
person reading one run, and the category leads, for the program reading them
all.

**Why the list is closed.** An open list of strings drifts: two places invent
``rate_limited`` and ``rate-limited``, a third writes ``ratelimit``, and the
table silently splits one finding into three. Here the categories are named
constants, :data:`TERMINAL_CATEGORIES` holds all of them, and a test asserts
that every component able to end a run names one from that list.

**Where the categories come from.** Four are this module's own, because only
the controller can decide them -- it is the only component that knows how many
steps have passed or that a page has become unusable. The rest are imported
from the components that already own them: the three clocks, the model layer's
three, and the validator's. Importing rather than restating is deliberate: a
constant defined twice is a constant that will eventually disagree with itself.

**One run, one ending.** The controller has one exit, and it returns one of
these. There is no path out of a run that leaves the outcome unset, which is
enforced both by the shape of the loop and by a test that reads the loop's own
source.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .actions import Done
from .model_client import (
    ENDPOINT_UNAVAILABLE,
    MODEL_ERROR,
    RATE_LIMITED,
    ModelClientError,
)
from .timeouts import ACTION_TIMEOUT, LLM_TIMEOUT, RUN_TIMEOUT
from .validator import VALIDATION_EXHAUSTED

#: The task was done, and the answer -- where the task asked for one -- is in
#: the outcome. The only category that is not a failure.
SUCCESS = "success"

#: The step cap was reached. The run was still going; it simply ran out of the
#: moves it was allowed. Distinct from every other failure because it means
#: "not finished", not "went wrong".
STEP_CAP = "step_cap"

#: The browser could not give the run a usable page: the start URL was
#: unreachable, or navigation kept failing, or the page went away underneath
#: the agent. There is nowhere to act, so there is nothing to retry.
NAVIGATION_DEAD_END = "navigation_dead_end"

#: The agent said it was beaten, in its own words. A real answer to some tasks,
#: and much better than looping until the cap: a run that stops early with a
#: reason costs less and says more than one that stops late with none.
GAVE_UP = "gave_up"

#: Every way a run can end, success first. An evaluation counts by these, so a
#: category that is not in this list is a finding that will not appear in any
#: table -- which is why the list is asserted rather than assumed.
TERMINAL_CATEGORIES = (
    SUCCESS,
    STEP_CAP,
    VALIDATION_EXHAUSTED,
    NAVIGATION_DEAD_END,
    GAVE_UP,
    ACTION_TIMEOUT,
    LLM_TIMEOUT,
    RUN_TIMEOUT,
    RATE_LIMITED,
    ENDPOINT_UNAVAILABLE,
    MODEL_ERROR,
)

#: The failures, for a report that counts them. Derived rather than written out
#: again, so the two lists cannot fall out of step.
FAILURE_CATEGORIES = tuple(name for name in TERMINAL_CATEGORIES if name != SUCCESS)


@dataclass(frozen=True)
class RunOutcome:
    """The single verdict on one run.

    Frozen, like every other record in this project: it is the run's result,
    and a result that can be edited afterwards is worth less as evidence.

    Attributes:
        category: One of :data:`TERMINAL_CATEGORIES`. The field a results table
            groups by.
        answer: What the task asked for, when it asked for something. None for
            a task that asked for an action rather than a fact, and None on
            every failure -- a failed run has no answer, and an empty string
            would read as one.
        reason: Why it failed, in a sentence written for a person. None on
            success, where the category and the answer say everything.
        steps: How many steps the run took. Recorded on the outcome rather than
            only in the step list because the two most common questions about a
            finished run are "did it work" and "how long did it take".
    """

    category: str
    answer: str | None = None
    reason: str | None = None
    steps: int = 0

    def __post_init__(self) -> None:
        """Refuse an outcome nothing can count.

        Raises:
            ValueError: If the category is not one of the known ones. This is
                a programming mistake rather than something a website can
                cause, and it is worth stopping for: a run that ended in a
                category no table has a column for has, for reporting
                purposes, not ended at all.
        """
        if self.category not in TERMINAL_CATEGORIES:
            raise ValueError(
                f"{self.category!r} is not a run outcome; the categories are "
                f"{', '.join(TERMINAL_CATEGORIES)}"
            )

    @property
    def ok(self) -> bool:
        """bool: Whether the task was completed."""
        return self.category == SUCCESS

    @classmethod
    def success(cls, *, answer: str | None = None, steps: int = 0) -> RunOutcome:
        """Build the outcome of a run that did what it was asked.

        Args:
            answer: The answer, for a task that asked a question.
            steps: Steps taken.

        Returns:
            RunOutcome: A successful outcome.
        """
        return cls(category=SUCCESS, answer=answer, steps=steps)

    @classmethod
    def failure(cls, category: str, reason: str, *, steps: int = 0) -> RunOutcome:
        """Build the outcome of a run that did not.

        Args:
            category: Which kind of failure, from :data:`FAILURE_CATEGORIES`.
            reason: The sentence for whoever reads this one run.
            steps: Steps taken.

        Returns:
            RunOutcome: A failed outcome.
        """
        return cls(category=category, reason=reason, steps=steps)

    @classmethod
    def from_done(cls, done: Done, *, steps: int = 0) -> RunOutcome:
        """Turn the agent's own ``done`` action into the run's verdict.

        The agent says how it thinks the run went; this decides what that means
        for the record. A successful ``done`` is taken at its word -- whether
        the answer is *correct* is the judge's question, not the controller's,
        and conflating the two would let the agent be marked by the component
        that ran it.

        Args:
            done: The ``done`` action the planner produced.
            steps: Steps taken.

        Returns:
            RunOutcome: Success with the answer, or a categorized give-up with
            the agent's own reason.
        """
        if done.status == "success":
            return cls.success(answer=done.answer, steps=steps)
        return cls.failure(
            GAVE_UP, done.reason or "the agent gave up without saying why", steps=steps
        )

    @classmethod
    def from_model_error(cls, error: ModelClientError, *, steps: int = 0) -> RunOutcome:
        """Turn a model-layer failure into the run's verdict.

        The category comes off the error itself, which is why those errors
        carry one: the controller should not be reading exception messages to
        work out whether an endpoint was slow, busy, or broken.

        Args:
            error: Anything the model client raised.
            steps: Steps taken.

        Returns:
            RunOutcome: A failure in the error's own category.
        """
        return cls.failure(error.category, str(error), steps=steps)

    def as_done(self) -> Done:
        """Render this outcome as the ``done`` action that expresses it.

        The action log is a list of actions, and a run's ending is part of it.
        A run that stopped for a reason the agent never chose -- the step cap,
        a dead end -- still gets a final ``done`` in that log, built here, so
        the log always ends explicitly instead of simply stopping.

        Returns:
            Done: ``done(success, answer)`` or ``done(failed,
            "<category>: <reason>")``.
        """
        if self.ok:
            return Done(action="done", status="success", answer=self.answer)
        return Done(action="done", status="failed", reason=f"{self.category}: {self.reason}")

    def summary(self) -> str:
        """Render the outcome as the line a run's report ends with.

        Returns:
            str: e.g. ``success in 6 steps: 09:00 to 17:30`` or ``failed in 25
            steps (step_cap): the step cap of 25 was reached``.
        """
        plural = "step" if self.steps == 1 else "steps"
        if self.ok:
            answer = f": {self.answer}" if self.answer else ""
            return f"success in {self.steps} {plural}{answer}"
        return f"failed in {self.steps} {plural} ({self.category}): {self.reason}"

    def to_dict(self) -> dict[str, Any]:
        """Render the outcome as plain data, for the run's trace.

        Returns:
            dict[str, Any]: JSON-safe values, with ``ok`` written out rather
            than left to be derived -- a reader of the folder should not have
            to know which categories count as success.
        """
        return {
            "ok": self.ok,
            "category": self.category,
            "answer": self.answer,
            "reason": self.reason,
            "steps": self.steps,
        }
