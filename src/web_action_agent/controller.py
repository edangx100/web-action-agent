"""The loop: look, decide, check, act, repeat -- and the only place a run ends.

Everything else in this agent is a part. The observer describes a page, the
planner chooses a move, the validator refuses a bad one, the executor performs
it, memory remembers, the clocks bound it, and none of them knows whether the
run is over. This module is what makes those parts into a run, and it holds two
responsibilities that are deliberately nowhere else.

**It is the only component that ends a run.** Every other part reports: a step
timed out, an endpoint was busy, an action was refused. None of them decides
what that means for the task as a whole, because a component that can end a run
is a component that can end it *quietly*, and "the run stopped somewhere in the
executor" is the kind of thing nobody finds until the results table has a hole
in it. Here there is one loop with one exit, and it returns a
:class:`~web_action_agent.outcomes.RunOutcome` on every path.

**It notices when the agent is going in circles.** A model at temperature zero
shown the same page twice will answer the same way twice -- which is a feature
until the answer does not work, at which point it is a loop that only the step
cap can stop. So two identical actions in a row are allowed and a third is
not: the controller substitutes a scroll, writes down why, and lets the changed
page change the next prompt. Re-asking the planner instead would cost a call to
get the same answer back, because nothing in the question would have changed.

**Step 0 is not step 1.** The run opens by navigating to the start URL and
looking at what arrives. That navigation is the controller's own, not the
planner's -- there is nothing to plan from yet -- so it is numbered zero, it is
not counted against the step cap, and it is not written into memory: the
planner is about to see that page in full, and "you are at the page you started
at" is not news. If it fails, the run is over before it begins, which is its
own outcome rather than a first bad step.

**One controller per run.** It keeps the run's state -- where it is, how many
steps have passed, what it just did -- so a second run needs a second
controller, in the same way a task needs its own browser context and its own
memory.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol

from playwright.sync_api import Error as BrowserFailure

from .actions import Action, Done, Navigate, Scroll, UseSkill, parse_action
from .config import Config
from .cost import RunCost, total_cost
from .executor import NAVIGATION_ERROR, TIMEOUT, ExecutionResult, Executor
from .memory import Memory
from .model_client import CallRecord, ModelClientError
from .observation import Observation, capture
from .outcomes import (
    ACTION_TIMEOUT,
    NAVIGATION_DEAD_END,
    RUN_TIMEOUT,
    STEP_CAP,
    VALIDATION_EXHAUSTED,
    RunOutcome,
)
from .planner import Planner, PlannerDecision
from .reporting import StepListener, step_line
from .screenshot_gate import GateDecision, ScreenshotGate
from .timeouts import RunBudget
from .trace import TraceWriter, new_id, utc_now
from .validator import Rejection, StepOutcome, Validator, plan_until_valid

#: How many identical actions in a row are allowed before one is replaced. Two,
#: because the second is often right -- a click that missed the first time
#: because the page was still settling deserves another go. The third is where
#: a repeat stops being persistence and starts being a loop.
IDENTICAL_ACTIONS_ALLOWED = 2

#: Consecutive failures to reach a page before the run is called a dead end.
#: One is ordinary -- a link to a page that has moved. Two in a row means the
#: agent cannot get anywhere from here, and further steps would spend model
#: calls discovering that again.
NAVIGATION_FAILURES_ALLOWED = 2

#: The note memory keeps when an action is forced. Keyed, so a second forcing
#: replaces the first rather than filling the notepad with near-copies.
STUCK_NOTE = "repeated action"

#: Taking one look at the current page. Given a label for the step's
#: screenshot, it returns the observation and the screenshot decision behind
#: it.
#:
#: Injectable for the same reason the run's clock and the model transport are:
#: this is the boundary between the loop and a real browser, and a loop that
#: could only be tested with a browser attached would be a loop whose branches
#: mostly went untested. The default is the real thing.
PageObserver = Callable[[str], tuple[Observation, "GateDecision | None"]]


class SkillReplay(Protocol):
    """Replaying a stored skill, for the run to fall back from.

    Declared as a shape rather than imported, because the skill library does
    not exist yet and this module should not have to change when it does. What
    the controller needs of a replay is only this: do it, and tell me whether
    it worked and in a few words what happened.
    """

    def __call__(self, action: UseSkill, observation: Observation) -> ReplayResult:
        """Replay one skill against the current page."""
        ...


class ReplayResult(Protocol):
    """What a replay reports back.

    Attributes:
        ok: Whether every step of the skill ran.
        detail: What happened, in a few words -- for the trace, and for the
            planner's history when control comes back to it.
    """

    ok: bool
    detail: str


@dataclass(frozen=True)
class StepReport:
    """One step of a run, as the trace will record it.

    Attributes:
        step_id: A stable id for this step, minted when the step is recorded.
            Every model call the step made carries it, which is what lets the
            run's flattened cost log be put back under its steps -- and what
            keeps a step identifiable once it has been lifted out of its folder
            into a benchmark's table, where its number alone would collide with
            the same-numbered step of every other run.
        started_at: When the step began, RFC 3339 in UTC: the moment the
            planner was first asked. Recorded as a timestamp rather than a
            duration, because a duration cannot be turned back into a position
            on the run's timeline.
        ended_at: When the step was recorded, after its action was executed.
        number: The step's position in the run, from 1. Step 0's navigation is
            not one of these; it is reported on the run instead.
        observation: The page this step was decided against.
        action: What was done, or None for a step that never got as far as
            doing anything -- a step whose every planning attempt was refused.
        result: What came of doing it, or None for a step that did nothing and
            for the final ``done``, which the browser is never asked to perform.
        attempts: Planning attempts this step took. One on a good step; more
            when the validator sent the planner back.
        rejections: Why each refused attempt was refused, in order. Empty on a
            step that was approved first time.
        forced: Why the controller replaced the planner's action, or empty when
            it did not. The one place a step's action is not the planner's
            choice, so it is recorded rather than left to be inferred.
        gate: The screenshot decision for this step's observation, with the
            rule that made it.
        records: What this step's model calls cost, refused attempts included.
    """

    step_id: str
    started_at: str
    ended_at: str
    number: int
    observation: Observation
    action: Action | None = None
    result: ExecutionResult | None = None
    attempts: int = 1
    rejections: tuple[Rejection, ...] = ()
    forced: str = ""
    gate: GateDecision | None = None
    records: tuple[CallRecord, ...] = ()

    def summary(self) -> str:
        """Render this step as the single line a run log prints.

        Delegated rather than written here, so that a step reads the same way
        in a terminal, in a test's failure message, and anywhere else it is
        printed. The rendering itself belongs to
        :mod:`~web_action_agent.reporting`, which is the module that knows what
        a person needs to see; this class knows what happened.

        Returns:
            str: e.g. ``  3  act    click [4] link "Next page" -> ok (0.31s)``.
        """
        return step_line(self)

    def to_dict(self) -> dict[str, Any]:
        """Render this step as plain data, for the run's trace.

        Returns:
            dict[str, Any]: The step, with each part rendered by whichever
            component owns it -- so the trace and the run agree by
            construction rather than by two pieces of code being kept in step.
            The keys are the ones ``schemas/action_log.schema.json`` requires,
            and the trace writer validates against it on the way out.
        """
        return {
            "step_id": self.step_id,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "number": self.number,
            "observation": self.observation.to_dict(),
            "action": self.action.to_dict() if self.action else None,
            "result": self.result.to_dict() if self.result else None,
            "attempts": self.attempts,
            "rejections": [rejection.to_dict() for rejection in self.rejections],
            "forced": self.forced,
            "screenshot_decision": self.gate.to_dict() if self.gate else None,
            "calls": [
                {
                    **asdict(record),
                    # `asdict` leaves a tuple field a tuple, and the trace is
                    # JSON -- where the only sequence is a list. Rendered by
                    # the backoff record itself, like every other part of this
                    # dictionary, so there is one place that decides what a
                    # pause looks like on disk.
                    "backoffs": [event.to_dict() for event in record.backoffs],
                    # The parent link is added here rather than carried on the
                    # record itself: the model client makes the call and has no
                    # idea which step it is inside, so the step is the first
                    # thing that knows.
                    "parent_step_id": self.step_id,
                }
                for record in self.records
            ],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> StepReport:
        """Rebuild one step from the plain data of a run's action log.

        The inverse of :meth:`to_dict`, and assembled the same way: each part is
        rebuilt by whichever component owns it, so a step read back agrees with
        the step written out by construction rather than by two pieces of code
        being kept in step.

        Args:
            data: One record, as the action log stores it.

        Returns:
            StepReport: The same step the trace was written from.

        Raises:
            KeyError: If the record is missing a field that is always written.
                The timestamps and the id are among them, which is the point:
                a step that cannot say when it happened is one the schema was
                written to refuse.
            ActionError: If a stored action does not fit the action schema.
            ValueError: If the record is not a step at all.
        """
        action = data.get("action")
        result = data.get("result")
        gate = data.get("screenshot_decision")
        return cls(
            step_id=data["step_id"],
            started_at=data["started_at"],
            ended_at=data["ended_at"],
            number=data["number"],
            observation=Observation.from_dict(data["observation"]),
            # The three optional parts are optional in the run, not just in the
            # record: a step whose every attempt was refused chose no action,
            # and the final `done` is never handed to the browser.
            action=parse_action(action) if action else None,
            result=ExecutionResult.from_dict(result) if result else None,
            attempts=data.get("attempts", 1),
            rejections=tuple(Rejection.from_dict(item) for item in data.get("rejections", ())),
            forced=data.get("forced", ""),
            gate=GateDecision.from_dict(gate) if gate else None,
            records=tuple(CallRecord.from_dict(call) for call in data.get("calls", ())),
        )


@dataclass(frozen=True)
class RunReport:
    """Everything one run produced: its verdict, its steps, and what it cost.

    This is what the trace writer turns into a folder and what the eval harness
    counts. The controller builds exactly one per run, at its single exit.

    Attributes:
        task: The task in the user's own words.
        start_url: Where the run opened.
        outcome: The verdict. Always set -- there is no path out of the loop
            that leaves it unset, which is the point of the single exit.
        started_at: When the run began, RFC 3339 in UTC -- before the opening
            navigation, which is part of the run even though it is not a step.
        ended_at: When the verdict was reached.
        steps: Every step, in order.
        cost: The roll-up of every model call the run made, refused attempts
            and rate-limit waiting included.
    """

    task: str
    start_url: str
    outcome: RunOutcome
    started_at: str = ""
    ended_at: str = ""
    steps: tuple[StepReport, ...] = ()
    cost: RunCost = field(default_factory=RunCost)

    def summary(self) -> str:
        """Render the run as the few lines a terminal prints at the end.

        Returns:
            str: One line per step, then the outcome, then the cost.
        """
        lines = [step.summary() for step in self.steps]
        lines.append(self.outcome.summary())
        lines.append(self.cost.summary())
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Render the run as plain data, for the run's trace.

        Returns:
            dict[str, Any]: The whole run, JSON-safe. The final ``done`` is
            written out even when the agent never chose one, so that an action
            log always ends explicitly rather than simply stopping.
        """
        return {
            "task": self.task,
            "start_url": self.start_url,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "outcome": self.outcome.to_dict(),
            "final_action": self.outcome.as_done().to_dict(),
            "steps": [step.to_dict() for step in self.steps],
            "cost": self.cost.to_dict(),
        }


class Controller:
    """Runs one task from start to finish, and says how it went.

    Attributes:
        max_steps: The hard ceiling on steps, from the run's settings.
    """

    def __init__(
        self,
        *,
        planner: Planner,
        validator: Validator,
        executor: Executor,
        memory: Memory,
        max_steps: int,
        retry_budget: int,
        gate: ScreenshotGate | None = None,
        budget: RunBudget | None = None,
        screenshot_dir: Path | None = None,
        replay: SkillReplay | None = None,
        skills: Sequence[str] = (),
        config: Config | None = None,
        observe: PageObserver | None = None,
        trace: TraceWriter | None = None,
        on_step: StepListener | None = None,
    ) -> None:
        """Assemble a run.

        Every part is passed in rather than built here. That is what lets a
        test drive the loop with a stub planner and no browser, and it is also
        the honest division of labour: the controller decides *when* things
        happen, not how any of them work.

        Args:
            planner: Chooses one action per step.
            validator: Refuses actions that make no sense here.
            executor: Performs them. Also the source of the page to look at,
                since a click that opened a tab changes which page that is.
            memory: What the planner is told about earlier steps.
            max_steps: Hard ceiling on steps. The loop's guarantee that it ends.
            retry_budget: How many extra planning attempts one step may have
                after a refusal.
            gate: Decides which steps carry a screenshot. None means none do,
                which is a legitimate run and the shape a test takes.
            budget: The run's wall clock. None means no clock, for a test that
                is not testing the clock.
            screenshot_dir: Where annotated screenshots go. None means no
                screenshots, whatever the gate says.
            replay: How to replay a stored skill, when the run has a library.
                None leaves ``use_skill`` to the executor, which reports it as
                something the browser does not do.
            skills: Skills retrieved for this site, already rendered as lines
                for the prompt. Empty on a cold run.
            config: The run's settings, needed for the text budget when a page
                is observed. Required unless ``observe`` is given, which is the
                only other way the loop can see a page.
            observe: How to look at the page, for a test that has no browser.
                Defaults to really looking at the executor's current page.
            trace: Where each step is written as it completes. None means the
                run leaves nothing behind, which is what a unit test wants and
                what no real run should do.
            on_step: Called with each step as it completes, for whoever is
                watching the run happen. The loop knows nothing about what is
                done with it -- a terminal line, a progress bar, a test's list
                -- which is why it is a plain callback and not a display.

        Raises:
            ValueError: If neither ``config`` nor ``observe`` is given. A loop
                with no way to see the page could still be constructed and
                would fail on its first step, which is a worse place to find
                out than here.
        """
        if config is None and observe is None:
            raise ValueError("give the controller a config to observe with, or an observer")
        self._planner = planner
        self._validator = validator
        self._executor = executor
        self._memory = memory
        self.max_steps = max_steps
        self._retry_budget = retry_budget
        self._gate = gate
        self._budget = budget
        self._screenshot_dir = screenshot_dir
        self._replay = replay
        self._skills = tuple(skills)
        self._config = config
        self._observe_page = observe or self._really_observe
        self._trace = trace
        self._on_step = on_step

        self._step = 0
        self._step_started_at = utc_now()
        self._steps: list[StepReport] = []
        self._observation: Observation | None = None
        # What was actually executed, most recent last, for spotting a loop.
        # The *executed* action rather than the proposed one, so a forced
        # alternative breaks the streak it was forced to break.
        self._executed: list[str] = []
        self._navigation_failures = 0

    def run(self, task: str, *, start_url: str) -> RunReport:
        """Do one task, and report what happened.

        The single exit. Every way the loop can end comes back through here, so
        a run always produces a report with a verdict in it -- including the
        ways nobody planned for.

        Args:
            task: The task in the user's own words.
            start_url: The page to open first. Already checked against the
                allowlist by whoever resolved it, since the run is not allowed
                to begin somewhere it would not be allowed to go.

        Returns:
            RunReport: The verdict, every step, and what the run cost.
        """
        started_at = utc_now()
        outcome = self._drive(task, start_url)
        report = RunReport(
            task=task,
            start_url=start_url,
            outcome=outcome,
            started_at=started_at,
            ended_at=utc_now(),
            steps=tuple(self._steps),
            cost=total_cost(record for step in self._steps for record in step.records),
        )
        if self._trace is not None:
            # The verdict is the one thing that can only be written at the end,
            # because it is the one thing that is not known until then. Every
            # step is already on disk, written as it finished -- so a run that
            # never reaches this line still leaves a readable folder, minus
            # exactly the file whose absence means "this run was killed".
            self._trace.finish(report)
        return report

    def _drive(self, task: str, start_url: str) -> RunOutcome:
        """Open the start page, then step until something ends the run.

        Args:
            task: The task in the user's own words.
            start_url: The page to open first.

        Returns:
            RunOutcome: Always. Every ``return`` in this method and in the ones
            it calls carries a verdict; the loop cannot fall off its end,
            because the line after it is the step cap.
        """
        opening = self._open(start_url)
        if opening is not None:
            return opening

        while self._step < self.max_steps:
            out_of_time = self._budget.check() if self._budget else None
            if out_of_time is not None:
                # Checked between steps rather than during one: a step already
                # underway finishes, because interrupting it would leave a
                # half-done action with nothing coherent to record.
                return RunOutcome.failure(
                    RUN_TIMEOUT,
                    f"the run's {self._budget.seconds:.0f}s wall clock ran out",
                    steps=self._step,
                )

            self._step += 1
            ended = self._take_a_step(task)
            if ended is not None:
                return ended

        return RunOutcome.failure(
            STEP_CAP,
            f"the step cap of {self.max_steps} was reached without finishing",
            steps=self._step,
        )

    def _open(self, start_url: str) -> RunOutcome | None:
        """Go to the start URL and take the first look at it.

        Args:
            start_url: Where to open.

        Returns:
            RunOutcome | None: None when the page is open and observed, or the
            run's verdict when it is not. A run that cannot reach its own
            starting page has not failed at the task -- it never started, and
            the outcome says so rather than blaming the agent.
        """
        arrival = self._executor.execute(Navigate(action="navigate", url=start_url))
        if not arrival.ok:
            category = ACTION_TIMEOUT if arrival.status == TIMEOUT else NAVIGATION_DEAD_END
            return RunOutcome.failure(
                category,
                f"could not open the start page {start_url}: {arrival.detail or arrival.status}",
                steps=0,
            )

        observation = self._observe("step-00")
        if observation is None:
            return RunOutcome.failure(
                NAVIGATION_DEAD_END,
                f"opened {start_url} but could not read the page",
                steps=0,
            )
        self._observation = observation
        return None

    def _take_a_step(self, task: str) -> RunOutcome | None:
        """Plan one action, do it, and look again.

        Args:
            task: The task in the user's own words.

        Returns:
            RunOutcome | None: The run's verdict when this step ended it, or
            None to carry on.
        """
        observation = self._observation
        assert observation is not None, "a step is only taken after a page has been observed"

        # Read once here and kept for whichever path ends up recording the
        # step. Taking it again inside `_add_step` would time the recording
        # rather than the step, and would quietly exclude the planning that is
        # usually most of a step's wall clock.
        self._step_started_at = utc_now()

        try:
            plan = plan_until_valid(
                attempt=lambda complaints: self._ask(task, complaints),
                validator=self._validator,
                observation=observation,
                retry_budget=self._retry_budget,
            )
        except ModelClientError as error:
            # A broken or overloaded endpoint is not a bad step, and the model
            # layer has already waited out whatever was worth waiting for. The
            # category comes off the error, so a busy endpoint and a dead one
            # are different findings.
            self._add_step(observation, attempts=1)
            return RunOutcome.from_model_error(error, steps=self._step)

        if self._gate is not None:
            self._gate.record_planner_outcome(
                succeeded=plan.ok, had_screenshot=observation.has_screenshot
            )

        if plan.giving_up is not None:
            self._add_step(observation, plan=plan)
            return RunOutcome.failure(
                VALIDATION_EXHAUSTED,
                plan.giving_up.reason or "every planning attempt was refused",
                steps=self._step,
            )

        action = plan.action
        assert action is not None, "a plan that did not give up has an action"

        if isinstance(action, Done):
            # The agent's own ending. Recorded as a step so the action log ends
            # with the action that ended it, then turned into the verdict.
            self._memory.record(action, observation=observation)
            self._add_step(observation, action=action, plan=plan)
            return RunOutcome.from_done(action, steps=self._step)

        if isinstance(action, UseSkill) and self._replay is not None:
            return self._replay_skill(action, observation, plan)

        action, forced = self._choose_alternative_if_stuck(action)
        result = self._executor.execute(action, observation=observation)
        self._executed.append(action.describe())
        self._memory.record(action, observation=observation, result=result)
        self._add_step(observation, action=action, result=result, plan=plan, forced=forced)

        stuck = self._dead_end(result)
        if stuck is not None:
            return stuck

        return self._look_again()

    def _replay_skill(
        self,
        action: UseSkill,
        observation: Observation,
        plan: StepOutcome,
    ) -> RunOutcome | None:
        """Replay a stored skill, and hand control back if it breaks.

        The fallback is the whole reason this lands in the controller: a replay
        that fails halfway has left the browser somewhere nobody planned for,
        so the run takes a *fresh* observation before the planner resumes.
        Letting the planner reason from the page as it was before the replay
        would be asking it about a page that no longer exists.

        Args:
            action: The skill invocation the planner chose.
            observation: The page before the replay.
            plan: This step's planning record.

        Returns:
            RunOutcome | None: The run's verdict if looking again fails, else
            None to carry on.
        """
        replayed = self._replay(action, observation) if self._replay else None
        detail = replayed.detail if replayed else "no replay configured"
        self._executed.append(action.describe())
        self._memory.record(action, observation=observation)
        self._memory.note("last skill", f"{action.describe()} - {detail}")
        self._add_step(
            observation, action=action, plan=plan, forced="" if replayed and replayed.ok else detail
        )
        return self._look_again()

    def _ask(self, task: str, complaints: tuple[str, ...]) -> PlannerDecision:
        """Ask the planner for one action.

        Args:
            task: The task in the user's own words.
            complaints: What was wrong with this step's earlier attempts, if
                any, so the retry is told what to fix.

        Returns:
            PlannerDecision: The action, or the failure to produce one.
        """
        observation = self._observation
        assert observation is not None, "the planner is only asked about a page already observed"
        return self._planner.decide(
            task=task,
            observation=observation,
            history=self._memory.history(),
            scratchpad=self._memory.scratchpad(),
            skills=self._skills,
            feedback=complaints,
            screenshot_png=_screenshot_bytes(observation),
        )

    def _choose_alternative_if_stuck(self, action: Action) -> tuple[Action, str]:
        """Replace an action that is about to be repeated once too often.

        Two identical actions in a row are allowed: the second is often right,
        because a click that missed while the page was still settling deserves
        another go. A third is a loop, and at temperature zero the planner
        cannot talk itself out of one -- shown the same page and the same
        history it will answer the same way, so asking again would spend a call
        to receive the same action. What breaks the loop is changing the page,
        so the controller scrolls instead and writes down why: the next
        observation differs, and the next question is therefore a different
        question.

        Args:
            action: What the planner proposed.

        Returns:
            tuple[Action, str]: The action to perform, and why it was
            substituted -- empty when it was not.
        """
        described = action.describe()
        recent = self._executed[-IDENTICAL_ACTIONS_ALLOWED:]
        if len(recent) < IDENTICAL_ACTIONS_ALLOWED or any(step != described for step in recent):
            return action, ""

        # Scrolling down is the alternative, except when down is what is being
        # repeated -- an agent stuck at the bottom of a page needs to go back up,
        # not press against the end again.
        direction = "up" if isinstance(action, Scroll) and action.direction == "down" else "down"
        alternative = Scroll(action="scroll", direction=direction)
        forced = (
            f"{described} was proposed {IDENTICAL_ACTIONS_ALLOWED + 1} times in a row; "
            f"{alternative.describe()} substituted to change the page"
        )
        # Written into the notepad, not just the trace: the planner reads the
        # notepad, and being told it is repeating itself is the part that might
        # change its mind on the next step.
        self._memory.note(STUCK_NOTE, forced)
        return alternative, forced

    def _dead_end(self, result: ExecutionResult) -> RunOutcome | None:
        """Decide whether the run has nowhere left to go.

        A single navigation failure is ordinary -- a link to a page that has
        moved -- and is reported to the planner like any other failed step. Two
        in a row is different: the agent cannot get anywhere from here, and
        further steps would spend model calls finding that out again.

        Note what is *not* here. A step that times out mid-run does not end the
        run: it is a result the planner can act on, and what stops a run that
        keeps timing out is the step cap or the wall clock. Only the start
        page's own timeout ends a run, because then there is nothing to act on.

        Args:
            result: What the step's execution produced.

        Returns:
            RunOutcome | None: The verdict, or None to carry on.
        """
        if result.status != NAVIGATION_ERROR:
            self._navigation_failures = 0
            return None

        self._navigation_failures += 1
        if self._navigation_failures < NAVIGATION_FAILURES_ALLOWED:
            return None
        return RunOutcome.failure(
            NAVIGATION_DEAD_END,
            f"{self._navigation_failures} navigations failed in a row; last was {result.detail}",
            steps=self._step,
        )

    def _look_again(self) -> RunOutcome | None:
        """Take the observation the next step will be decided from.

        Returns:
            RunOutcome | None: None when the page was read, or the verdict when
            it could not be. A page that cannot be looked at cannot be acted
            on either, so that is a dead end rather than a step to retry.
        """
        observation = self._observe(f"step-{self._step:02d}")
        if observation is None:
            return RunOutcome.failure(
                NAVIGATION_DEAD_END,
                "the page could not be read after the last action",
                steps=self._step,
            )
        self._observation = observation
        return None

    def _observe(self, label: str) -> Observation | None:
        """Look at the page the executor is currently on.

        Asked of the executor rather than remembered, because a click that
        opened a tab moved the run to a different page, and the observation has
        to be of the page the next action will be performed on.

        Args:
            label: Filename stem for this step's screenshot.

        Returns:
            Observation | None: The observation, or None when the page could
            not be read at all.
        """
        try:
            observation, decision = self._observe_page(label)
        except (BrowserFailure, OSError, ValueError):
            # The page has closed, navigated away mid-read, or cannot be
            # photographed. Any of those leaves the run with nothing to decide
            # from, and the caller turns it into a dead end -- a verdict, where
            # an exception here would be a run with no verdict at all.
            return None
        self._last_gate = decision
        return observation

    def _really_observe(self, label: str) -> tuple[Observation, GateDecision | None]:
        """Look at the executor's current page, for real.

        Args:
            label: Filename stem for this step's screenshot.

        Returns:
            tuple[Observation, GateDecision | None]: The observation and the
            screenshot decision behind it.
        """
        assert self._config is not None, "a controller without a config was given an observer"
        return capture(
            self._executor.page,
            config=self._config,
            gate=self._gate,
            screenshot_dir=self._screenshot_dir,
            label=label,
        )

    def _add_step(
        self,
        observation: Observation,
        *,
        action: Action | None = None,
        result: ExecutionResult | None = None,
        plan: StepOutcome | None = None,
        forced: str = "",
        attempts: int | None = None,
    ) -> None:
        """Record one step for the trace.

        Args:
            observation: The page the step was decided against.
            action: What was done, if anything.
            result: What came of it, if it was done.
            plan: The planning record, for the attempt count, the refusals and
                the call costs.
            forced: Why the action was substituted, if it was.
            attempts: Overrides the attempt count, for a step that never got as
                far as a plan.

        Raises:
            TraceError: If the step cannot be written to the run folder, or
                does not fit the action log schema. Not caught here: a trace
                that is silently missing steps is believed exactly as much as a
                complete one, and every reader downstream would be reading a
                run that did not happen the way the file says it did.
        """
        rejections = ()
        records: tuple[CallRecord, ...] = ()
        if plan is not None:
            rejections = tuple(
                attempt.rejection for attempt in plan.attempts if attempt.rejection is not None
            )
            records = plan.records
        step = StepReport(
            step_id=new_id("step"),
            started_at=self._step_started_at,
            ended_at=utc_now(),
            number=self._step,
            observation=observation,
            action=action,
            result=result,
            attempts=attempts or (len(plan.attempts) if plan else 1),
            rejections=rejections,
            forced=forced,
            gate=getattr(self, "_last_gate", None),
            records=records,
        )
        self._steps.append(step)
        if self._trace is not None:
            # Written now, not at the end of the run. This is the flush-per-step
            # contract: when this returns, the step is on disk, so whatever
            # happens during the next one cannot take this one with it.
            self._trace.write_step(step)
        if self._on_step is not None:
            # Told now, for the same reason and to a different audience: a run
            # that reports its steps only at the end is a run nobody can watch,
            # and a step is at its most useful on screen while the next one is
            # still being decided. After the trace write, so that anything the
            # watcher is told about has already been recorded.
            self._on_step(step)


def _screenshot_bytes(observation: Observation) -> bytes | None:
    """Read this step's annotated screenshot, when there is one to read.

    Args:
        observation: The step's observation.

    Returns:
        bytes | None: The PNG, or None when this observation carries no image
        or the file has gone. A missing file is not worth ending a run over:
        the two text channels are always there, which is the whole reason the
        image channel is allowed to be optional.
    """
    if observation.screenshot is None:
        return None
    try:
        return observation.screenshot.read_bytes()
    except OSError:
        return None
