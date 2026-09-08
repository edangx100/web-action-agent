"""Where the browser comes from, and how long each piece of it lives.

Two lifetimes, deliberately different, and almost every browser-automation bug
that looks like flakiness is one of them being confused for the other.

**The process lives as long as the worker.** Launching Chromium costs roughly a
second and a hundred megabytes. A benchmark runs a few dozen tasks, so paying
that once per worker rather than once per task is most of the difference
between an evaluation that finishes over lunch and one that does not.

**The context lives as long as one task.** A *context* is the browser's own
isolation boundary: its own cookies, its own local storage, its own history. A
fresh one per task is what makes the second task a fair test. Without it the
first task's login, dismissed banner, and search history are still there, and
the agent looks better than it is -- in a way that would never show up as a
failure, only as a success it did not earn. Worse, it would be irreproducible:
the same task run alone and run seventh would behave differently.

So: one :class:`BrowserProcess` per worker, one
:meth:`BrowserProcess.task_session` per task. The nesting is enforced by the shape of the API rather than by a
convention -- a task gets its page from a context manager, and the context
closes when the block ends whether the task succeeded, failed, or raised.

**Only two callers.** The observer reads the page and the executor acts on it.
Nothing else in the agent holds a page, which is what keeps "what the browser
did" a question with one place to look.

**The surprises a page can spring.** Three things happen on real sites that no
action asked for, and each one ends a run that is not ready for it:

* A link opens a **second tab**. The agent goes on looking at the tab it left,
  sees a page that never changes, and concludes it is stuck.
* The page opens one of the browser's own **dialogs** -- ``alert``,
  ``confirm``. These are modal in the strongest sense: while one is open the
  page runs no code and answers no automation. A run that ignores it does not
  fail, it hangs, until something times out and reports the wrong reason.
* A click starts a **download**. Files are out of scope, but the click still
  happened and the agent still needs to know why no page arrived.

:class:`TaskSession` handles all three, for every page a task opens, and writes
each one down as a :class:`BrowserEvent`. Handling them anywhere else would
mean handling them once per caller and forgetting one; writing them down is
what turns "the agent went quiet for a step" into a line in the trace.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from typing import Any, Self

from playwright.sync_api import (
    Browser,
    BrowserContext,
    Dialog,
    Download,
    Page,
    Playwright,
    sync_playwright,
)
from playwright.sync_api import Error as BrowserFailure

from .config import Config
from .politeness import DEFAULT_USER_AGENT

#: Seconds any single browser operation may take before it is called a timeout.
#: Mirrors the configured default; a browser started without a Config is a test
#: or a one-off script, and should behave like a real run rather than hang.
ACTION_TIMEOUT_S = 30.0


#: A link or a script opened a second tab, and the agent is now looking at it.
POPUP_ADOPTED = "popup_adopted"

#: The active tab closed and the agent fell back to the one underneath.
PAGE_CLOSED = "page_closed"

#: One of the browser's own dialogs appeared and was dismissed.
DIALOG_DISMISSED = "dialog_dismissed"

#: A click offered a file. It was declined; files are out of scope.
DOWNLOAD_IGNORED = "download_ignored"

#: Everything that can happen to a task's pages without any action asking for
#: it. Tests assert the list has not grown quietly, and a run's summary counts
#: events by it -- "how often does this site open tabs?" is a question the
#: trace should be able to answer.
BROWSER_EVENTS = (
    POPUP_ADOPTED,
    PAGE_CLOSED,
    DIALOG_DISMISSED,
    DOWNLOAD_IGNORED,
)


class BrowserError(RuntimeError):
    """The browser was asked for something its current state cannot give.

    Raised for a programming mistake -- asking a stopped process for a page --
    rather than for anything a website did. A page that fails to load is an
    ordinary result the executor reports; this is a bug in the caller.
    """


@dataclass(frozen=True)
class BrowserSettings:
    """The handful of browser choices a run makes, in one value.

    Separate from :class:`~web_action_agent.config.Config` so that a browser
    can be started in a test or a script from four plain values, without
    building the whole configuration -- and so that this module depends on
    configuration in exactly one place, :meth:`from_config`.

    Attributes:
        headed: Show the browser window. Off by default, because a run has to
            work over SSH and in continuous integration, where there is no
            screen to draw on.
        viewport_width: Window width in pixels.
        viewport_height: Window height in pixels. Fixed rather than maximised,
            because element numbering is viewport-only: the same page observed
            at two sizes offers different elements, and a benchmark whose
            numbers depended on the reviewer's monitor would not be a
            benchmark.
        action_timeout_s: How long any one browser operation may take. Reaching
            it is a reported timeout, never an exception that ends the run.
        user_agent: What the browser tells every site it is. Sent on every
            request the page makes, so it is the one setting here that a
            stranger's server sees.
    """

    headed: bool = False
    viewport_width: int = 1280
    viewport_height: int = 800
    action_timeout_s: float = ACTION_TIMEOUT_S
    user_agent: str = DEFAULT_USER_AGENT

    @classmethod
    def from_config(cls, config: Config) -> BrowserSettings:
        """Take the browser settings out of a whole run configuration.

        Args:
            config: The resolved configuration for this run.

        Returns:
            BrowserSettings: The four values this module needs, copied out.
        """
        return cls(
            headed=config.headed,
            viewport_width=config.viewport_width,
            viewport_height=config.viewport_height,
            action_timeout_s=config.action_timeout_s,
            user_agent=config.user_agent,
        )

    @property
    def viewport(self) -> dict[str, int]:
        """dict[str, int]: The viewport in the shape Playwright wants it."""
        return {"width": self.viewport_width, "height": self.viewport_height}

    @property
    def action_timeout_ms(self) -> float:
        """float: The action timeout in milliseconds, which is Playwright's unit."""
        return self.action_timeout_s * 1000


@dataclass(frozen=True)
class BrowserEvent:
    """Something the browser did that no action asked for.

    Frozen, like every other record in a step: this is evidence of what the
    page did while the agent was doing something else, and it is written into
    the trace beside the action it happened during.

    Attributes:
        kind: One of :data:`BROWSER_EVENTS`.
        detail: One sentence saying what happened and what was done about it,
            written for whoever reads the run afterwards.
        url: The page the event concerns, where there is one. Empty for an
            event about no particular address, such as a dialog.
    """

    kind: str
    detail: str
    url: str = ""

    def summary(self) -> str:
        """Render this event as the fragment a step log appends.

        Returns:
            str: e.g. ``popup_adopted: a new tab opened and is now the active
            page``.
        """
        return f"{self.kind}: {self.detail}"

    def to_dict(self) -> dict[str, Any]:
        """Render this event as plain data, for the run's trace.

        Returns:
            dict[str, Any]: JSON-safe values, under the same keys every time.
        """
        return {"kind": self.kind, "detail": self.detail, "url": self.url}


class TaskSession:
    """One task's pages: which one is active, and what they have done to us.

    A task starts with one page and may end up with several, because a link
    can open a tab. Something has to hold the answer to "which page is the
    agent looking at *now*", and it cannot be the executor, which is handed a
    page once and would keep acting on it long after the site moved the
    conversation to another tab.

    So the session owns that answer, the executor asks it every time, and the
    three surprises in this module's docstring are handled here once for every
    page the task opens -- including tabs that did not exist when the task
    started.

    Given a page but no context, the session is a plain holder: nothing is
    watched, because there is nothing to watch new tabs *on*. That is the shape
    a script or a test gets, and it is why an executor can still be handed a
    bare page.

    Attributes:
        page: The page the agent is looking at now. Changes when a tab opens.
    """

    def __init__(self, page: Page, *, context: BrowserContext | None = None) -> None:
        """Start a session on one page.

        Args:
            page: The task's first page.
            context: The browsing context that page belongs to, when the
                session should watch for new tabs, dialogs and downloads.
                Omitted for a bare page nobody is watching.
        """
        self._active = page
        self._context = context
        self._events: list[BrowserEvent] = []
        if context is not None:
            # Registered after the first page exists, because the context
            # announces *every* page it opens -- including that one, which is
            # not a popup and must not be reported as one.
            context.on("page", self._adopt)
            self._watch(page)

    @property
    def page(self) -> Page:
        """Page: The page the agent is looking at now."""
        return self._active

    def drain_events(self) -> tuple[BrowserEvent, ...]:
        """Take everything recorded since the last time this was called.

        Draining rather than accumulating is what lets each step's result carry
        exactly the events that belong to it. Anything that happened between
        two actions -- while the page was being observed, say -- is reported
        with the next action, which is where a reader looking for it would go.

        Returns:
            tuple[BrowserEvent, ...]: The events, oldest first. Empty when
            nothing has happened, which is most steps.
        """
        drained = tuple(self._events)
        self._events.clear()
        return drained

    def _watch(self, page: Page) -> None:
        """Subscribe to the surprises one page can spring.

        Args:
            page: A page belonging to this task, new or original.
        """
        page.on("dialog", self._dismiss)
        page.on("download", self._decline)
        page.on("close", self._release)

    def _adopt(self, page: Page) -> None:
        """Make a newly opened tab the active page.

        Following the new tab is right far more often than not: a site that
        opens one has moved the conversation there, and the page left behind is
        finished with. The alternative -- staying put -- looks to the agent
        exactly like a click that did nothing, which is the most expensive
        thing a page can look like.

        Args:
            page: The tab the browser has just opened.
        """
        if page is self._active:
            return
        self._watch(page)
        self._active = page
        self._record(
            POPUP_ADOPTED,
            "a new tab opened and is now the page being acted on",
            url=_url_of(page),
        )

    def _release(self, page: Page) -> None:
        """Fall back to another tab when the active one closes.

        A background tab closing changes nothing. The active one closing would
        otherwise leave every later action failing against a page that no
        longer exists, reporting a closed target rather than the plain fact
        that the site shut the tab.

        Args:
            page: The page that has just closed.
        """
        if page is not self._active:
            return
        remaining = [open_page for open_page in self._pages() if open_page is not page]
        if not remaining:
            # Nothing to fall back to. The closed page is kept as the active
            # one so that the next action fails honestly against it, rather
            # than this method inventing a page that was never part of the task.
            self._record(PAGE_CLOSED, "the last tab closed; the task has no page left")
            return
        self._active = remaining[-1]
        self._record(
            PAGE_CLOSED,
            "the tab being acted on closed; back to the one underneath",
            url=_url_of(self._active),
        )

    def _dismiss(self, dialog: Dialog) -> None:
        """Answer one of the browser's own dialogs, and write down what it said.

        Dismissed rather than accepted, deliberately. Accepting means answering
        "yes" to a question nobody read -- delete the account, place the order,
        leave the page and lose the form -- and an agent that quietly agrees to
        things is worse than one that quietly declines them. The message is
        recorded either way, so a run blocked by a dialog it needed to accept
        shows exactly that in its trace instead of looking like a dead end.

        Args:
            dialog: The dialog the page opened.
        """
        self._record(DIALOG_DISMISSED, f"{dialog.type} dismissed: {dialog.message}")
        # A dialog that has already been handled -- by a page that closed under
        # it, say -- refuses a second answer, and that must not become the
        # exception that ends the run.
        with suppress(BrowserFailure):
            dialog.dismiss()

    def _decline(self, download: Download) -> None:
        """Refuse a file the page offered, and write down that it was offered.

        Downloads are out of scope for this agent: it reads pages. Cancelling
        rather than saving keeps a large file from being fetched in full for
        nothing, and recording it is what explains the step afterwards -- a
        click that produced no new page is otherwise a mystery.

        Args:
            download: The download the page started.
        """
        self._record(
            DOWNLOAD_IGNORED,
            f"the page offered a file, {download.suggested_filename!r}; downloads are not handled",
            url=download.url,
        )
        with suppress(BrowserFailure):
            download.cancel()

    def _pages(self) -> list[Page]:
        """List the task's pages that are still open, oldest first.

        Returns:
            list[Page]: The context's pages, or just the active one when this
            session has no context to ask.
        """
        if self._context is None:
            return [self._active]
        return [page for page in self._context.pages if not page.is_closed()]

    def _record(self, kind: str, detail: str, *, url: str = "") -> None:
        """Add one event to the list the next action will carry.

        Args:
            kind: One of :data:`BROWSER_EVENTS`.
            detail: The sentence for whoever reads the trace.
            url: The page the event concerns, if any.
        """
        self._events.append(BrowserEvent(kind=kind, detail=detail, url=url))


def _url_of(page: Page) -> str:
    """Read a page's address without letting the read raise.

    Event handlers run at awkward moments -- a tab may be closing as it is
    being asked about -- and an exception from a handler would surface far from
    here, as a failed action with nothing to do with the action.

    Args:
        page: The page to ask.

    Returns:
        str: Its address, or an empty string if it cannot be read.
    """
    with suppress(BrowserFailure):
        return page.url
    return ""


class BrowserProcess:
    """One running browser, handing out one fresh context per task.

    Use it as a context manager, which is the only way to be sure the browser
    is closed on every path out::

        with BrowserProcess(settings) as browser:
            for task in tasks:
                with browser.task_session() as session:
                    run(task, session)

    Not thread-safe, and not meant to be: Playwright's synchronous API belongs
    to the thread that started it. Parallel benchmark workers each run their
    own process, which is also what keeps one worker's crash from taking the
    others down.

    Attributes:
        settings: The viewport, timeout and headedness this browser was
            started with.
    """

    def __init__(
        self,
        settings: BrowserSettings | None = None,
        *,
        playwright: Playwright | None = None,
    ) -> None:
        """Prepare a browser, without starting anything yet.

        Nothing is launched until :meth:`start`, so constructing one is cheap
        and safe to do while working out whether a run should happen at all.

        Args:
            settings: The browser choices for this run. Defaults to the
                documented defaults, which match a fresh configuration.
            playwright: A driver that is already running, to launch this
                browser from. Playwright allows one synchronous driver per
                thread, so a process started inside something that already has
                one -- a test suite, a host application -- has to borrow it
                rather than start a second. A borrowed driver is left running
                by :meth:`stop`, since whoever started it is still using it.
        """
        self.settings = settings or BrowserSettings()
        self._playwright = playwright
        self._owns_driver = playwright is None
        self._browser: Browser | None = None

    @property
    def running(self) -> bool:
        """bool: Whether the browser process is up and can hand out pages."""
        return self._browser is not None

    def start(self) -> Self:
        """Launch the browser process.

        Starting twice is a no-op rather than an error: it means the caller
        already has what it asked for, and raising would only invite defensive
        checks at call sites that do not need them.

        Returns:
            Self: This process, so that ``start()`` can be chained.
        """
        if self._browser is not None:
            return self
        # Playwright's driver is a separate process, started here and stopped
        # in `stop`. Keeping the handle is what makes shutdown possible; a
        # driver left running outlives the run and is the classic way a test
        # suite ends up with orphaned browsers.
        if self._playwright is None:
            self._playwright = sync_playwright().start()
        self._browser = self._playwright.chromium.launch(headless=not self.settings.headed)
        return self

    def stop(self) -> None:
        """Close the browser and its driver.

        Safe to call twice, and safe to call on a process that never started,
        so that cleanup paths need no conditions of their own.
        """
        # Closed innermost first, and each independently: a browser that has
        # already died on its own must not stop the driver from being shut
        # down, which is the leak that outlives the run.
        try:
            if self._browser is not None:
                self._browser.close()
        finally:
            self._browser = None
            # A borrowed driver belongs to whoever lent it, and stopping it
            # here would take their browsers down along with this one.
            if self._owns_driver and self._playwright is not None:
                self._playwright.stop()
                self._playwright = None

    @contextmanager
    def task_session(self) -> Iterator[TaskSession]:
        """Open one task's browser context, and close it when the task ends.

        The context is the isolation boundary, so this is the guarantee that
        one task's cookies, storage and history are not part of the next task's
        starting conditions. The page inside it starts blank; navigating to the
        task's start URL is the controller's first step, not this method's, so
        that step 0 is recorded like any other.

        A session rather than a page, because a task may end up with more than
        one: the session is what knows which of them the agent is looking at,
        and it is watching for tabs, dialogs and downloads from this moment
        rather than from the first action.

        Yields:
            TaskSession: The task's pages, watched. Its ``page`` is blank,
            sized to the configured viewport, with the configured action
            timeout already applied as its default.

        Raises:
            BrowserError: If the process is not running.
        """
        if self._browser is None:
            raise BrowserError(
                "browser is not running: call start(), or use it as a context manager"
            )

        # The user agent is set on the context rather than per request, so
        # every request a page makes -- its images, its scripts, its background
        # fetches -- says the same true thing about who is asking.
        context = self._browser.new_context(
            viewport=self.settings.viewport,
            user_agent=self.settings.user_agent,
        )
        # Set once on the context rather than passed to every call, so that an
        # operation nobody thought to pass a timeout to still cannot hang the
        # run for longer than the configured ceiling.
        context.set_default_timeout(self.settings.action_timeout_ms)
        try:
            yield TaskSession(context.new_page(), context=context)
        finally:
            # Closing the context closes every page in it, including any tab a
            # click opened. Nothing about the task survives this line.
            context.close()

    def __enter__(self) -> Self:
        """Start the browser on entry to a ``with`` block.

        Returns:
            Self: This process, started.
        """
        return self.start()

    def __exit__(self, *exception: object) -> None:
        """Stop the browser on the way out, whatever the reason for leaving.

        Args:
            *exception: The exception triple, if the block is leaving because
                of one. Not inspected: the browser is closed either way, and
                the exception continues on its way.
        """
        self.stop()
