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

So: one :class:`BrowserProcess` per worker, one :meth:`BrowserProcess.task_page`
per task. The nesting is enforced by the shape of the API rather than by a
convention -- a task gets its page from a context manager, and the context
closes when the block ends whether the task succeeded, failed, or raised.

**Only two callers.** The observer reads the page and the executor acts on it.
Nothing else in the agent holds a page, which is what keeps "what the browser
did" a question with one place to look.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Self

from playwright.sync_api import Browser, Page, Playwright, sync_playwright

from .config import Config

#: Seconds any single browser operation may take before it is called a timeout.
#: Mirrors the configured default; a browser started without a Config is a test
#: or a one-off script, and should behave like a real run rather than hang.
ACTION_TIMEOUT_S = 30.0


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
    """

    headed: bool = False
    viewport_width: int = 1280
    viewport_height: int = 800
    action_timeout_s: float = ACTION_TIMEOUT_S

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
        )

    @property
    def viewport(self) -> dict[str, int]:
        """dict[str, int]: The viewport in the shape Playwright wants it."""
        return {"width": self.viewport_width, "height": self.viewport_height}

    @property
    def action_timeout_ms(self) -> float:
        """float: The action timeout in milliseconds, which is Playwright's unit."""
        return self.action_timeout_s * 1000


class BrowserProcess:
    """One running browser, handing out one fresh context per task.

    Use it as a context manager, which is the only way to be sure the browser
    is closed on every path out::

        with BrowserProcess(settings) as browser:
            for task in tasks:
                with browser.task_page() as page:
                    run(task, page)

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
    def task_page(self) -> Iterator[Page]:
        """Open one task's browser context, and close it when the task ends.

        The context is the isolation boundary, so this is the guarantee that
        one task's cookies, storage and history are not part of the next task's
        starting conditions. The page inside it starts blank; navigating to the
        task's start URL is the controller's first step, not this method's, so
        that step 0 is recorded like any other.

        Yields:
            Page: A blank page, sized to the configured viewport, with the
            configured action timeout already applied as its default.

        Raises:
            BrowserError: If the process is not running.
        """
        if self._browser is None:
            raise BrowserError(
                "browser is not running: call start(), or use it as a context manager"
            )

        context = self._browser.new_context(viewport=self.settings.viewport)
        # Set once on the context rather than passed to every call, so that an
        # operation nobody thought to pass a timeout to still cannot hang the
        # run for longer than the configured ceiling.
        context.set_default_timeout(self.settings.action_timeout_ms)
        try:
            yield context.new_page()
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
