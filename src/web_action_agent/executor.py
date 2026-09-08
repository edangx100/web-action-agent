"""The part that actually does it: one approved action, one browser command.

Everything upstream of here is talk. The planner proposes, the validator
approves, and nothing has happened yet -- this module is where a decision
becomes a click, and it is the only place in the agent that *changes* a page.
The observer reads pages and this writes to them, and keeping that division
sharp is worth a rule rather than a habit: when the answer to "what did the run
do to that page?" has one file to look in, a step that did something surprising
is a bug with an address.

**Failures are results, not exceptions.** A web page can refuse an action in a
dozen ordinary ways -- the site is slow, the button moved, the link is dead --
and none of them is a crash. A run that dies on the first stale button is a run
with no verdict and no trace worth reading, so every one of those endings comes
back as an :class:`ExecutionResult` carrying one of a fixed set of statuses.
The controller then decides what to do next, and the planner is told what
happened in a sentence it can act on.

**Four failures worth telling apart.** They are different problems with
different responses, and collapsing them into "the step failed" throws away the
part that says what to do:

* :data:`TIMEOUT` -- it is there, it did not finish in time. Slow site, or an
  element that never became clickable. Usually worth another try.
* :data:`NAVIGATION_ERROR` -- the page could not be reached at all: no such
  host, connection refused, request blocked. Trying again rarely helps; a
  different route might. Note that a 404 is *not* this. A 404 is a page that
  loaded and says it has nothing, which the agent can read and reason about.
* :data:`ELEMENT_DETACHED` -- the thing the planner named is gone. The page
  moved on between being observed and being acted on, so the fix is to look
  again, not to try harder.
* :data:`BROWSER_ERROR` -- everything else the browser objected to, kept as a
  named bucket so that an unfamiliar failure is reported rather than mistaken
  for one of the three above.

**Acting on the same element that was seen.** A planner says ``click 4``, and
between the observation and the click the page may have re-rendered. Finding
"the fourth element" again would then act on whatever is fourth *now*, and
finding it by its label would act on whatever wears that label now -- both of
which are wrong in the same quiet way, and both of which look like a successful
step in the trace. Instead the observer writes each element's number onto the
element itself (:data:`~web_action_agent.observer.ELEMENT_MARK`), and this
module acts through that mark. If the marked node is gone, the mark is gone
with it, and the step is reported as :data:`ELEMENT_DETACHED` instead of
succeeding at the wrong thing.

**What is deliberately not here.** Whether the action was allowed is the
validator's business and has already been settled -- this module does not
re-check the allowlist, and reading it as a second gate would be a mistake.
Whether the action *worked*, in the sense of getting the task closer to done,
is nobody's business here either: this reports that the browser accepted the
command, and the next observation shows what it did to the page.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from playwright.sync_api import Error as BrowserFailure
from playwright.sync_api import Locator, Page
from playwright.sync_api import TimeoutError as BrowserTimeout

from .actions import Action, Click, Done, Fill, Navigate, Press, Scroll, UseSkill
from .browser import ACTION_TIMEOUT_S
from .observation import Observation
from .observer import ELEMENT_MARK

#: The browser accepted the command. Not a claim that it achieved anything --
#: a click on the wrong button succeeds. The next observation says what changed.
OK = "ok"

#: The action did not finish inside the configured ceiling.
TIMEOUT = "timeout"

#: The page could not be loaded at all: unresolvable host, refused connection,
#: blocked request. A page that loads and returns 404 is not this.
NAVIGATION_ERROR = "navigation_error"

#: The element the planner named is no longer in the page. Between the
#: observation and this moment the page changed underneath the plan.
ELEMENT_DETACHED = "element_detached"

#: Anything else the browser refused. A named bucket rather than a re-raise,
#: so an unfamiliar failure is still a reported step with a trace entry.
BROWSER_ERROR = "browser_error"

#: The action is not something the browser does. ``done`` ends the run and
#: ``use_skill`` expands into other actions; both are the controller's to
#: handle, and both reaching here means a caller skipped a stage.
NOT_EXECUTABLE = "not_executable"

#: Every status an execution can report, success first. Tests assert the set
#: has not quietly grown, and a run's summary counts steps by it.
EXECUTION_STATUSES = (
    OK,
    TIMEOUT,
    NAVIGATION_ERROR,
    ELEMENT_DETACHED,
    BROWSER_ERROR,
    NOT_EXECUTABLE,
)

#: How much of the viewport one scroll moves, as a fraction. Slightly less than
#: a full screen on purpose: the overlap means the two observations either side
#: of a scroll share a strip of page, so nothing can fall between them unseen.
SCROLL_FRACTION = 0.9

#: Scrolls the window by a number of pixels, immediately. Run in the page
#: rather than sent as a wheel event, for two reasons: a wheel event goes to
#: whatever the pointer happens to be over, and it is processed asynchronously,
#: so the observation that follows can be taken halfway through the movement.
#: ``instant`` also overrides a site's own smooth scrolling, which is the same
#: problem wearing a stylesheet.
SCROLL_SCRIPT = "(top) => window.scrollBy({ top, left: 0, behavior: 'instant' })"

#: Viewport height assumed when the page cannot report its own, in pixels.
#: Only reachable for a page with no fixed viewport, which a configured run
#: never has.
FALLBACK_VIEWPORT_HEIGHT = 800

#: How long to give a key press to *start* a navigation, in seconds. Playwright
#: waits out a navigation a click begins, but a raw key press is only an input
#: event: pressing Enter in a search box returns before the browser has left the
#: old page, and an observation taken then describes the page the agent was
#: trying to leave. So a press waits this long for a navigation to begin.
#:
#: To *begin*, not to finish: waiting for the new page to load is
#: :data:`SETTLE_TIMEOUT_S`'s job, and giving a slow site only this long to
#: finish would be the same bug in a smaller form. Nothing having begun is the
#: ordinary outcome, so this is what most key presses cost.
NAVIGATION_GRACE_S = 1.0

#: How long to wait, after a click or a key press, for any navigation it
#: started to finish, in seconds. Shorter than the action timeout because this
#: is not the action -- it is the pause before observing, and a page that keeps
#: a connection open forever (a live feed, an advert frame) would otherwise
#: spend the whole action budget on every step for the rest of the run.
SETTLE_TIMEOUT_S = 5.0


def classify_error(error: BaseException) -> str:
    """Decide which failure a browser complaint is.

    Playwright reports everything as one exception type with a message, so the
    message is what there is to go on. The order matters: a navigation that
    ran out of time raises a timeout whose message also mentions the URL, and
    it is a timeout -- a site that is merely slow is a different problem from a
    site that is not there.

    Args:
        error: The exception the browser call raised.

    Returns:
        str: One of :data:`EXECUTION_STATUSES`, never :data:`OK`.
    """
    if isinstance(error, BrowserTimeout):
        return TIMEOUT
    message = str(error).lower()
    if "not attached" in message or "element is detached" in message:
        # Playwright's own words for a node that left the document while it was
        # being acted on -- the race the mark cannot close, because the element
        # was there when the click started.
        return ELEMENT_DETACHED
    if "net::" in message or "navigation failed" in message:
        return NAVIGATION_ERROR
    return BROWSER_ERROR


def describe_error(error: BaseException) -> str:
    """Reduce a browser complaint to the one line worth reporting.

    Playwright messages carry a call log -- every internal retry, with
    timestamps -- which is invaluable when debugging the browser and noise
    everywhere else, including in a prompt that is charged by the token. The
    first line carries the actual complaint.

    Args:
        error: The exception the browser call raised.

    Returns:
        str: The first non-empty line of the message, trimmed. Empty only for
        an exception with no message at all.
    """
    for line in str(error).splitlines():
        stripped = line.strip()
        if stripped:
            return stripped
    return ""


@dataclass(frozen=True)
class ExecutionResult:
    """What happened when one action was handed to the browser.

    Frozen, like the observation and the action it sits between: this is the
    record of a step, and a record that can be edited afterwards is worth less
    as evidence of why the run went the way it did.

    Attributes:
        action: The action that was attempted, kept alongside the outcome so a
            result reads on its own in a log.
        status: One of :data:`EXECUTION_STATUSES`.
        detail: One sentence saying what went wrong, written to be shown to the
            planner on the next step. Empty on success, where there is nothing
            to say beyond the action itself.
        url: The page's address afterwards. Recorded because the most common
            effect of a step is that it moved, and because a *failed* step that
            moved the page anyway is worth being able to see.
        elapsed_s: How long the attempt took, in seconds. The timeout budget is
            spent here, so this is where a run that is slow rather than broken
            shows itself.
    """

    action: Action
    status: str
    detail: str = ""
    url: str = ""
    elapsed_s: float = 0.0

    @property
    def ok(self) -> bool:
        """bool: Whether the browser accepted the command."""
        return self.status == OK

    def summary(self) -> str:
        """Render this result as the one line the step log records.

        Returns:
            str: The action, its status, and the complaint if there was one --
            for example ``click 4 -> ok (0.31s)`` or ``click 4 ->
            element_detached: element 4 is no longer on the page (0.02s)``.
        """
        line = f"{self.action.describe()} -> {self.status}"
        if self.detail:
            line += f": {self.detail}"
        return f"{line} ({self.elapsed_s:.2f}s)"

    def to_dict(self) -> dict[str, Any]:
        """Render this result as plain data, for the run's trace.

        Returns:
            dict[str, Any]: JSON-safe values. The action is written in the same
            shape :func:`~web_action_agent.actions.parse_action` reads back, so
            a trace can be replayed rather than only read.
        """
        return {
            "action": self.action.to_dict(),
            "status": self.status,
            "detail": self.detail,
            "url": self.url,
            "elapsed_s": round(self.elapsed_s, 3),
        }


class Executor:
    """Carries out validated actions on one page, and reports what happened.

    One executor belongs to one task, because one page belongs to one task.
    Nothing is remembered between calls: each action is performed against the
    page as it is at that moment, which is the only state that matters and the
    only state that can be trusted.

    Attributes:
        page: The page this executor acts on.
    """

    def __init__(
        self,
        page: Page,
        *,
        action_timeout_s: float = ACTION_TIMEOUT_S,
        settle_timeout_s: float = SETTLE_TIMEOUT_S,
        navigation_grace_s: float = NAVIGATION_GRACE_S,
    ) -> None:
        """Bind an executor to a page.

        Args:
            page: The task's page, from
                :meth:`~web_action_agent.browser.BrowserProcess.task_page`.
            action_timeout_s: Ceiling on any one browser operation. The task's
                context sets the same value as its default; passing it
                explicitly on every call as well means a page handed in from
                somewhere else still obeys the run's configuration.
            settle_timeout_s: How long to wait for a navigation started by a
                click or a key press to finish. See :data:`SETTLE_TIMEOUT_S`.
            navigation_grace_s: How long to give a key press to start a
                navigation. See :data:`NAVIGATION_GRACE_S`.
        """
        self.page = page
        self._action_timeout_ms = action_timeout_s * 1000
        self._settle_timeout_ms = settle_timeout_s * 1000
        self._navigation_grace_ms = navigation_grace_s * 1000

    def execute(
        self,
        action: Action,
        *,
        observation: Observation | None = None,
    ) -> ExecutionResult:
        """Perform one action and report the outcome.

        This does not raise for anything a website can do. A browser complaint
        is caught here, classified, and returned as a result -- which is what
        lets the controller keep a run going, and lets the trace record a
        failed step rather than losing the run at it.

        Args:
            action: An action the validator has already approved. Passing an
                unvalidated action is a bug: this module does no policy checks
                of its own, so an unchecked URL would simply be visited.
            observation: The observation the action was planned against, used
                only to name an element that has since vanished: ``element 4
                (button "Search")`` reads better in a trace, and in the retry
                prompt, than ``element 4`` alone. Optional, because the outcome
                does not depend on it.

        Returns:
            ExecutionResult: The status, a sentence where something went wrong,
            the page's address afterwards, and how long it took.
        """
        started = time.monotonic()
        try:
            status, detail = self._perform(action, observation)
        except BrowserFailure as error:
            status, detail = classify_error(error), describe_error(error)
        return ExecutionResult(
            action=action,
            status=status,
            detail=detail,
            url=self._current_url(),
            elapsed_s=time.monotonic() - started,
        )

    def _perform(
        self,
        action: Action,
        observation: Observation | None,
    ) -> tuple[str, str]:
        """Route one action to the browser command that carries it out.

        Args:
            action: The action to perform.
            observation: The observation it was planned against, or None.

        Returns:
            tuple[str, str]: The status and its detail sentence.

        Raises:
            playwright.sync_api.Error: Whatever the browser raised. Caught and
                classified by :meth:`execute`, which is the only caller.
        """
        if isinstance(action, Click):
            return self._click(action, observation)
        if isinstance(action, Fill):
            return self._fill(action, observation)
        if isinstance(action, Press):
            return self._press(action)
        if isinstance(action, Scroll):
            return self._scroll(action)
        if isinstance(action, Navigate):
            return self._navigate(action)
        # `done` and `use_skill` land here, and so would an eighth action added
        # to the vocabulary without a branch above. Reporting it beats a silent
        # `pass`, which would record a step that did nothing as a success.
        return NOT_EXECUTABLE, _not_executable_detail(action)

    def _click(self, action: Click, observation: Observation | None) -> tuple[str, str]:
        """Click the element the planner numbered.

        Args:
            action: The click.
            observation: The observation it was planned against, or None.

        Returns:
            tuple[str, str]: The status and its detail sentence.
        """
        element = self._element(action.element_id)
        if element is None:
            return ELEMENT_DETACHED, _gone_detail(action.element_id, observation)
        element.click(timeout=self._action_timeout_ms)
        self._settle()
        return OK, ""

    def _fill(self, action: Fill, observation: Observation | None) -> tuple[str, str]:
        """Enter text into the element the planner numbered.

        Playwright's fill clears the field first, so filling with an empty
        string is how a field is emptied -- which is what the action schema
        promises, and why it is not done as a click plus keystrokes.

        Args:
            action: The fill.
            observation: The observation it was planned against, or None.

        Returns:
            tuple[str, str]: The status and its detail sentence.
        """
        element = self._element(action.element_id)
        if element is None:
            return ELEMENT_DETACHED, _gone_detail(action.element_id, observation)
        element.fill(action.text, timeout=self._action_timeout_ms)
        # No settle: typing does not navigate. Submitting the form is a
        # separate `press Enter` or a click, and each is one recorded step.
        return OK, ""

    def _press(self, action: Press) -> tuple[str, str]:
        """Press a key in whatever currently has focus.

        No element is named, which is the point of the action: on many pages
        Enter submits a form that has no button to click.

        That same case is why the press is wrapped in a wait for navigation:
        submitting a form is the most common reason to press a key, and a bare
        key press does not wait for the page it just asked for. The wait is for
        the new page to be *committed* -- to exist at all -- after which the
        usual settle waits for it to load. Nothing having navigated is the
        ordinary outcome, for arrow keys and Escape and most of the rest, so
        the wait running out is not a failure and is not reported as one.

        Args:
            action: The key press.

        Returns:
            tuple[str, str]: The status and its detail sentence.
        """
        try:
            with self.page.expect_navigation(
                wait_until="commit", timeout=self._navigation_grace_ms
            ):
                self.page.keyboard.press(action.key)
        except BrowserTimeout:
            # The key did not take the page anywhere, which is most keys.
            pass
        self._settle()
        return OK, ""

    def _scroll(self, action: Scroll) -> tuple[str, str]:
        """Move the viewport up or down by most of a screen.

        Scrolling past either end is not an error. The browser stops at the
        edge, this reports success, and the next observation shows a page that
        did not move -- which is exactly what a person scrolling at the bottom
        of a page experiences, and enough for the planner to conclude there is
        nothing further down.

        Only the window is scrolled; a page that puts its content in an inner
        scrolling panel is not reached this way. That is a known v0 limit,
        chosen over wheel events because those land wherever the pointer is and
        settle whenever they like. See :data:`SCROLL_SCRIPT`.

        Args:
            action: The scroll.

        Returns:
            tuple[str, str]: The status and its detail sentence.
        """
        viewport = self.page.viewport_size or {}
        height = viewport.get("height", FALLBACK_VIEWPORT_HEIGHT)
        distance = int(height * SCROLL_FRACTION)
        self.page.evaluate(SCROLL_SCRIPT, distance if action.direction == "down" else -distance)
        return OK, ""

    def _navigate(self, action: Navigate) -> tuple[str, str]:
        """Go to a URL the validator has already cleared.

        A response is not inspected. A 404 or a 500 is a page that loaded and
        says something, and the agent reads it in the next observation like any
        other page; only a failure to reach the site at all is a failed step.

        Args:
            action: The navigation.

        Returns:
            tuple[str, str]: The status and its detail sentence.
        """
        self.page.goto(action.url, timeout=self._action_timeout_ms)
        return OK, ""

    def _element(self, element_id: int) -> Locator | None:
        """Find the marked element the planner referred to by number.

        The check for absence is made here, before acting, rather than left to
        the browser: clicking an element that is not there is a wait that ends
        in a timeout, which would spend the whole action budget and then report
        the wrong failure. Asking first turns it into an immediate, correctly
        named result.

        Args:
            element_id: The number from the observation the planner was shown.

        Returns:
            Locator | None: A handle on the marked element, or None if no
            element in the page carries that mark any more.
        """
        marked = self.page.locator(f'[{ELEMENT_MARK}="{element_id}"]')
        if marked.count() == 0:
            return None
        # `.first` rather than the bare locator: a page that cloned a marked
        # node -- a carousel duplicating a slide, say -- would otherwise fail
        # Playwright's strictness check with an error about ambiguity, when
        # acting on the first match is both harmless and what was meant.
        return marked.first

    def _settle(self) -> None:
        """Wait for any navigation the last action started to finish.

        Without this, an observation taken immediately after a click that
        followed a link can describe the old page, the new one, or a half-built
        version of either. Reaching the timeout is not a failure: some pages
        never finish loading, and the agent still has to be able to look at
        them.
        """
        try:
            self.page.wait_for_load_state("load", timeout=self._settle_timeout_ms)
        except BrowserFailure:
            # Both endings are fine here. A timeout means the page is still
            # loading, and the observation will describe it as it stands; any
            # other complaint belongs to the action that follows, which will
            # report it properly rather than have it surface as a settle.
            pass

    def _current_url(self) -> str:
        """Read the page's address, without letting the read fail a step.

        Returns:
            str: The current URL, or an empty string if the page is gone --
            which happens when a step closed it, and is worth recording as
            "unknown" rather than as an exception raised while reporting an
            exception.
        """
        try:
            return self.page.url
        except BrowserFailure:
            return ""


def _gone_detail(element_id: int, observation: Observation | None) -> str:
    """Write the sentence handed to the planner when an element has vanished.

    Named where possible, because "element 4" means nothing to a model on the
    next step -- the numbering will have been reissued by then -- while
    ``button "Search"`` is something it can look for in the new list.

    Args:
        element_id: The number the planner used.
        observation: The observation it came from, or None.

    Returns:
        str: One sentence, ready to quote into the next prompt.
    """
    element = observation.element_by_id(element_id) if observation else None
    named = f' ({element.role} "{element.name}")' if element else ""
    return (
        f"element {element_id}{named} is no longer on the page; "
        "it was removed or the page re-rendered. Observe again before acting"
    )


def _not_executable_detail(action: Action) -> str:
    """Explain why an action reached the browser that should not have.

    Args:
        action: The action that has no browser command.

    Returns:
        str: One sentence naming what should have handled it instead.
    """
    if isinstance(action, Done):
        return "'done' ends the run; the controller records it, the browser is not involved"
    if isinstance(action, UseSkill):
        return (
            "'use_skill' is not a browser command; skill replay expands it into "
            "concrete actions and hands those here one at a time"
        )
    return f"no browser command is defined for {action.action!r}"
