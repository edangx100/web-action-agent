"""What happens when an approved action is handed to a real browser.

Everything here is about one seam: the moment a plan stops being a sentence and
becomes something that changed a page. Three properties carry the weight, and
none of them announces itself when broken.

**The right element.** ``click 4`` has to reach the element the observer
numbered 4, not the element sitting fourth on the page a second later. A page
that re-renders between observation and click is ordinary, and an executor that
matched on position would click the wrong thing and report success -- a failure
that reads, in the trace, as a planner that cannot reason. So the mark is
tested directly, and again through a page that shifts underneath a plan.

**Failures with the right name.** A vanished element, a slow site, a dead host
and a hidden button are four different problems with four different responses,
and they arrive from Playwright as one exception type with a message. The tests
put the browser into each state for real -- removing a node, hanging a request,
aborting one, hiding an element -- rather than asserting against hand-made
exceptions, because the classification is only worth anything if it matches
what the browser actually says.

**Nothing raises.** A run that dies at a stale button has no verdict and no
usable trace. Every failure below is asserted to come back as a result.

The unit tests at the end cover the parts that need no browser: what each
action turns into, and what a result reads like in a log.
"""

from contextlib import contextmanager

import pytest
from playwright.sync_api import Error as BrowserFailure
from playwright.sync_api import TimeoutError as BrowserTimeout

from web_action_agent.actions import parse_action
from web_action_agent.browser import BrowserError, BrowserProcess, BrowserSettings
from web_action_agent.config import Config
from web_action_agent.executor import (
    BROWSER_ERROR,
    ELEMENT_DETACHED,
    EXECUTION_STATUSES,
    FALLBACK_VIEWPORT_HEIGHT,
    NAVIGATION_ERROR,
    NOT_EXECUTABLE,
    OK,
    SCROLL_FRACTION,
    SCROLL_SCRIPT,
    TIMEOUT,
    ExecutionResult,
    Executor,
    classify_error,
    describe_error,
)
from web_action_agent.observation import Observation
from web_action_agent.observer import ELEMENT_MARK, extract_elements

VIEWPORT_HEIGHT = 800
"""int: The height the conftest viewport uses, repeated for the scroll maths."""


def observe(page):
    """Look at the page the way the agent does, marks and all.

    Extraction is what writes the marks the executor acts through, so calling
    this before executing is not test scaffolding -- it is the same order the
    real loop runs in, and skipping it would leave nothing for a click to find.

    Args:
        page: The live page.

    Returns:
        Observation: The elements and the address. The text channel is left
        empty; nothing in this module reads it.
    """
    return Observation(
        url=page.url,
        title=page.title(),
        elements=extract_elements(page),
        text="",
    )


def element_id_named(observation, name):
    """Find the number the observation gave to the element with a given name.

    Args:
        observation: The observation to look in.
        name: The element's accessible name.

    Returns:
        int: Its number.

    Raises:
        AssertionError: If the page offers no such element, which would make
            the rest of the test assert nothing.
    """
    for element in observation.elements:
        if element.name == name:
            return element.id
    raise AssertionError(f"no element named {name!r} in {[e.name for e in observation.elements]}")


def act(page, proposal, *, observation=None, **kwargs):
    """Execute one proposed action against a page.

    Args:
        page: The live page.
        proposal: The action as plain data, parsed here so that every test
            names its action in the same shape the planner emits.
        observation: The observation to plan against. Taken fresh when not
            given; passed explicitly by tests that change the page in between.
        **kwargs: Passed to :class:`Executor`, for the tests that need a short
            timeout.

    Returns:
        ExecutionResult: What the browser did.
    """
    if observation is None:
        observation = observe(page)
    return Executor(page, **kwargs).execute(parse_action(proposal), observation=observation)


# --- The marks the executor acts through ------------------------------------


@pytest.mark.fixture
def test_every_listed_element_is_marked_with_its_own_number(page, fixture_site):
    """Each number the planner is shown names exactly one node in the page."""
    page.goto(fixture_site.url_for("search.html"))
    observation = observe(page)

    assert observation.elements, "the search page should offer elements to mark"
    for element in observation.elements:
        marked = page.locator(f'[{ELEMENT_MARK}="{element.id}"]')
        assert marked.count() == 1, f"element {element.id} is marked {marked.count()} times"


@pytest.mark.fixture
def test_the_mark_lands_on_the_element_the_listing_describes(page, fixture_site):
    """The marked node is the one the listing says it is, not its neighbour."""
    page.goto(fixture_site.url_for("search.html"))
    observation = observe(page)

    field = element_id_named(observation, "Search terms")
    assert page.locator(f'[{ELEMENT_MARK}="{field}"]').get_attribute("id") == "query"


@pytest.mark.fixture
def test_observing_twice_does_not_leave_two_sets_of_marks(page, fixture_site):
    """Marks from the previous look are cleared, so no number means two things."""
    page.goto(fixture_site.url_for("search.html"))
    observe(page)
    observation = observe(page)

    marked = page.locator(f"[{ELEMENT_MARK}]").count()
    assert marked == len(observation.elements)


# --- Each action against the fixture site -----------------------------------


@pytest.mark.fixture
def test_click_follows_the_link_that_was_numbered(page, fixture_site):
    """A click moves the page, and the result reports where it ended up."""
    page.goto(fixture_site.url_for("index.html"))
    observation = observe(page)

    result = act(
        page,
        {"action": "click", "element_id": element_id_named(observation, "Reading room news")},
        observation=observation,
    )

    assert result.ok
    assert result.status == OK
    assert page.url.endswith("article.html")
    assert result.url == page.url


@pytest.mark.fixture
def test_click_acts_on_the_numbered_element_after_the_page_shifts(page, fixture_site):
    """The mark survives a re-render that moves everything down by one.

    This is the failure the mark exists to prevent. A new link is inserted
    above the one that was numbered, so anything resolving by position now
    resolves to the wrong element -- and would still report a cheerful success.
    """
    page.goto(fixture_site.url_for("index.html"))
    observation = observe(page)
    target = element_id_named(observation, "Reading room news")

    page.evaluate(
        """() => document.querySelector('nav.site').insertAdjacentHTML(
            'afterbegin', '<a href="ledger.html">Archive ledger</a>')"""
    )
    result = act(page, {"action": "click", "element_id": target}, observation=observation)

    assert result.ok
    assert page.url.endswith("article.html"), (
        "the click landed on the element that shifted into place"
    )


@pytest.mark.fixture
def test_fill_enters_the_text_into_the_field(page, fixture_site):
    """After a fill, the field shows what was typed."""
    page.goto(fixture_site.url_for("search.html"))
    observation = observe(page)
    field = element_id_named(observation, "Search terms")

    result = act(
        page,
        {"action": "fill", "element_id": field, "text": "membership"},
        observation=observation,
    )

    assert result.ok
    assert page.locator("#query").input_value() == "membership"


@pytest.mark.fixture
def test_fill_with_empty_text_clears_the_field(page, fixture_site):
    """Filling with nothing is how a field is emptied, not a no-op."""
    page.goto(fixture_site.url_for("search.html"))
    observation = observe(page)
    field = element_id_named(observation, "Search terms")
    act(page, {"action": "fill", "element_id": field, "text": "archive"}, observation=observation)

    result = act(
        page,
        {"action": "fill", "element_id": field, "text": ""},
        observation=observe(page),
    )

    assert result.ok
    assert page.locator("#query").input_value() == ""


@pytest.mark.fixture
def test_press_enter_submits_the_form(page, fixture_site):
    """A key press acts on whatever has focus, with no element named."""
    page.goto(fixture_site.url_for("search.html"))
    observation = observe(page)
    field = element_id_named(observation, "Search terms")
    act(
        page, {"action": "fill", "element_id": field, "text": "membership"}, observation=observation
    )

    result = act(page, {"action": "press", "key": "Enter"})

    assert result.ok
    assert "results.html" in page.url
    assert "q=membership" in page.url


@pytest.mark.fixture
def test_scroll_moves_the_viewport_down_and_back_up(page, fixture_site):
    """Down moves most of a screen; up returns what down moved."""
    page.goto(fixture_site.url_for("ledger.html"))
    assert page.evaluate("window.scrollY") == 0

    assert act(page, {"action": "scroll", "direction": "down"}).ok
    scrolled = page.evaluate("window.scrollY")
    assert scrolled > 0

    assert act(page, {"action": "scroll", "direction": "up"}).ok
    assert page.evaluate("window.scrollY") == 0


@pytest.mark.fixture
def test_scrolling_past_the_end_is_not_a_failure(page, fixture_site):
    """At the top of the page, scrolling up succeeds and changes nothing.

    The planner learns there is nothing above from the unchanged observation,
    which is what a person scrolling at the top of a page also gets.
    """
    page.goto(fixture_site.url_for("index.html"))

    result = act(page, {"action": "scroll", "direction": "up"})

    assert result.ok
    assert page.evaluate("window.scrollY") == 0


@pytest.mark.fixture
def test_navigate_opens_the_url(page, fixture_site):
    """Navigation is a step like any other, and reports the address it reached."""
    page.goto(fixture_site.url_for("index.html"))

    result = act(page, {"action": "navigate", "url": fixture_site.url_for("article.html")})

    assert result.ok
    assert page.url.endswith("article.html")
    assert result.url.endswith("article.html")


@pytest.mark.fixture
def test_a_page_that_returns_404_is_a_page_not_a_failure(page, fixture_site):
    """A missing page loaded successfully and says so; the agent can read it."""
    result = act(page, {"action": "navigate", "url": fixture_site.url_for("no-such-page.html")})

    assert result.ok, result.summary()
    assert result.url.endswith("no-such-page.html")


# --- The four ways a step fails ---------------------------------------------


@pytest.mark.fixture
def test_a_removed_element_is_reported_detached_rather_than_raising(page, fixture_site):
    """The element vanished between the plan and the click: say so, do not crash."""
    page.goto(fixture_site.url_for("search.html"))
    observation = observe(page)
    button = element_id_named(observation, "Search")
    page.evaluate("() => document.querySelector('#search-submit').remove()")

    result = act(
        page,
        {"action": "click", "element_id": button},
        observation=observation,
        action_timeout_s=5.0,
    )

    assert result.status == ELEMENT_DETACHED
    assert not result.ok


@pytest.mark.fixture
def test_a_detached_element_is_reported_at_once_not_after_the_timeout(page, fixture_site):
    """Asking first, rather than clicking and waiting, is what keeps it quick.

    Waiting the full action timeout for an element that is provably gone would
    spend the step budget and then report a timeout -- the wrong name for it.
    """
    page.goto(fixture_site.url_for("search.html"))
    observation = observe(page)
    button = element_id_named(observation, "Search")
    page.evaluate("() => document.querySelector('#search-submit').remove()")

    result = act(
        page,
        {"action": "click", "element_id": button},
        observation=observation,
        action_timeout_s=10.0,
    )

    assert result.elapsed_s < 2.0, result.summary()


@pytest.mark.fixture
def test_the_detached_complaint_names_the_element_that_went(page, fixture_site):
    """The next prompt gets a description, because the numbering will have moved."""
    page.goto(fixture_site.url_for("search.html"))
    observation = observe(page)
    button = element_id_named(observation, "Search")
    page.evaluate("() => document.querySelector('#search-submit').remove()")

    result = act(page, {"action": "click", "element_id": button}, observation=observation)

    assert f"element {button}" in result.detail
    assert "Search" in result.detail
    assert "Observe again" in result.detail


@pytest.mark.fixture
def test_an_element_that_is_present_but_hidden_is_a_timeout(page, fixture_site):
    """Still there, not operable: a different problem from a vanished element."""
    page.goto(fixture_site.url_for("search.html"))
    observation = observe(page)
    button = element_id_named(observation, "Search")
    page.evaluate("() => { document.querySelector('#search-submit').style.display = 'none'; }")

    result = act(
        page,
        {"action": "click", "element_id": button},
        observation=observation,
        action_timeout_s=0.5,
    )

    assert result.status == TIMEOUT
    assert result.detail


@pytest.mark.fixture
def test_an_unreachable_page_is_a_navigation_error(page, fixture_site):
    """A request that never arrives is a navigation error, not a timeout."""
    page.route("**/*", lambda route: route.abort())

    result = act(page, {"action": "navigate", "url": fixture_site.url_for("index.html")})

    assert result.status == NAVIGATION_ERROR
    assert not result.ok


@pytest.mark.fixture
def test_a_page_that_never_answers_is_a_timeout(page, fixture_site):
    """A site that is slow is a different problem from a site that is not there."""
    # The handler neither fulfils nor aborts, so the request simply hangs --
    # the behaviour of a server that accepted the connection and went quiet.
    page.route("**/*", lambda route: None)

    result = act(
        page,
        {"action": "navigate", "url": fixture_site.url_for("index.html")},
        action_timeout_s=0.5,
    )

    assert result.status == TIMEOUT
    assert not result.ok


@pytest.mark.fixture
def test_a_failed_step_still_reports_where_the_page_is(page, fixture_site):
    """A failure is a step in the trace, so it carries the same fields as any other."""
    page.goto(fixture_site.url_for("search.html"))
    observation = observe(page)
    button = element_id_named(observation, "Search")
    page.evaluate("() => document.querySelector('#search-submit').remove()")

    result = act(page, {"action": "click", "element_id": button}, observation=observation)

    assert result.url.endswith("search.html")
    assert result.elapsed_s >= 0.0
    assert result.to_dict()["status"] == ELEMENT_DETACHED


# --- The browser process and its per-task contexts --------------------------


@pytest.mark.fixture
def test_two_tasks_in_one_process_do_not_share_cookies(browser_process, fixture_site):
    """The whole point of a context per task: task two starts clean.

    Sharing cookies between tasks would not fail anything. It would quietly
    hand the second task a site that already knows it -- logged in, banner
    dismissed, searches remembered -- and report the result as though the agent
    had earned it.
    """
    with browser_process.task_page() as first:
        first.goto(fixture_site.url_for("index.html"))
        first.evaluate("document.cookie = 'waa_seen=yes; path=/'")
        first.evaluate("window.localStorage.setItem('waa_seen', 'yes')")
        assert "waa_seen=yes" in first.evaluate("document.cookie")

    with browser_process.task_page() as second:
        second.goto(fixture_site.url_for("index.html"))
        assert second.evaluate("document.cookie") == ""
        assert second.evaluate("window.localStorage.getItem('waa_seen')") is None


@pytest.mark.fixture
def test_one_browser_process_serves_every_task(browser_process):
    """Fresh context per task, same browser throughout: the two lifetimes."""
    with browser_process.task_page() as first:
        first_context = first.context
        process = first.context.browser

    with browser_process.task_page() as second:
        assert second.context is not first_context, "each task needs its own context"
        assert second.context.browser is process, "the process is shared, not relaunched"


@pytest.mark.fixture
def test_the_task_context_uses_the_configured_viewport(playwright):
    """Element numbering is viewport-only, so the size is not left to the browser."""
    settings = BrowserSettings(viewport_width=1024, viewport_height=768)
    with (
        BrowserProcess(settings, playwright=playwright) as browser,
        browser.task_page() as page,
    ):
        assert page.viewport_size == {"width": 1024, "height": 768}


@pytest.mark.fixture
def test_stopping_twice_is_harmless(playwright):
    """Cleanup paths need no conditions of their own."""
    browser = BrowserProcess(playwright=playwright).start()
    assert browser.running

    browser.stop()
    browser.stop()

    assert not browser.running


@pytest.mark.fixture
def test_a_borrowed_driver_outlives_the_process_that_used_it(playwright):
    """Stopping a browser must not take down a driver somebody else is using.

    The session's own browser is still running on this driver; if ``stop`` had
    shut the driver down, every browser test after this one would fail.
    """
    BrowserProcess(playwright=playwright).start().stop()

    with BrowserProcess(playwright=playwright) as second:
        assert second.running


@pytest.mark.unit
def test_a_page_cannot_be_taken_from_a_browser_that_is_not_running():
    """Asking a stopped process for a page is a bug, and says so."""
    browser = BrowserProcess()

    with pytest.raises(BrowserError, match="not running"), browser.task_page():
        pass


@pytest.mark.unit
def test_settings_come_out_of_the_run_configuration():
    """One place reads configuration, so there is one place to change it."""
    config = Config(headed=True, viewport_width=1440, viewport_height=900, action_timeout_s=12.0)

    settings = BrowserSettings.from_config(config)

    assert settings.headed is True
    assert settings.viewport == {"width": 1440, "height": 900}
    assert settings.action_timeout_ms == 12000.0


# --- Actions the browser is not asked to perform ----------------------------


class StubPage:
    """A page that records what it was asked to do, and does nothing.

    Enough of Playwright's surface for the routing tests below, which are about
    which command an action becomes rather than about what a browser does with
    it. Anything the executor calls that is missing here would fail loudly, so
    the stub cannot quietly diverge from the real thing without a test noticing.

    Attributes:
        url: The address reported back in the result.
        calls: Every command asked of it, in order, as ``(name, arguments)``.
    """

    def __init__(self, *, url="http://localhost/page.html", viewport=None, fails=None):
        """Build a stub page.

        Args:
            url: The address to report.
            viewport: The viewport size to report, or None to report none,
                which is the case the fallback height exists for.
            fails: An exception ``goto`` should raise, for the test that a
                browser complaint becomes a result rather than an escape.
        """
        self.url = url
        self.calls = []
        self._viewport = viewport
        self._fails = fails
        self.keyboard = _Recorder(self.calls, "keyboard")

    @property
    def viewport_size(self):
        """dict[str, int] | None: The viewport this page reports."""
        return self._viewport

    def goto(self, url, **kwargs):
        """Record a navigation, or raise the failure this stub was built with."""
        self.calls.append(("goto", url))
        if self._fails is not None:
            raise self._fails

    def locator(self, selector):
        """Record a lookup and answer that nothing matches."""
        self.calls.append(("locator", selector))
        return _EmptyLocator()

    def wait_for_load_state(self, state, **kwargs):
        """Record the settle wait."""
        self.calls.append(("wait_for_load_state", state))

    def evaluate(self, expression, arg=None):
        """Record a script run in the page, and return nothing."""
        self.calls.append(("evaluate", (expression, arg)))

    @contextmanager
    def expect_navigation(self, **kwargs):
        """Stand in for the wait around a key press, which nothing navigates.

        The real one raises a timeout when the block leaves without the page
        having gone anywhere, which is the case worth testing: the press must
        still be reported as a success.
        """
        self.calls.append(("expect_navigation", kwargs.get("timeout")))
        yield
        raise BrowserTimeout("Timeout exceeded while waiting for navigation")


class _Recorder:
    """Records any method call made on it, under a prefix."""

    def __init__(self, calls, prefix):
        """Args: calls: The shared call list. prefix: The name to record under."""
        self._calls = calls
        self._prefix = prefix

    def __getattr__(self, name):
        """Returns: A callable that appends its arguments to the call list."""

        def record(*args, **kwargs):
            self._calls.append((f"{self._prefix}.{name}", args))

        return record


class _EmptyLocator:
    """A locator that matches nothing, which is the detached case."""

    def count(self):
        """Returns: int: Zero, always."""
        return 0


@pytest.mark.unit
def test_done_is_not_something_the_browser_does():
    """The controller ends the run; asking the browser would be a bug upstream."""
    page = StubPage()

    result = Executor(page).execute(parse_action({"action": "done", "status": "success"}))

    assert result.status == NOT_EXECUTABLE
    assert "controller" in result.detail
    assert page.calls == [], "the browser must not be touched"


@pytest.mark.unit
def test_use_skill_is_expanded_elsewhere_not_executed_here():
    """A skill is a sequence of actions; replay hands them here one at a time."""
    page = StubPage()

    result = Executor(page).execute(
        parse_action({"action": "use_skill", "skill": "search", "params": {"query": "rust"}})
    )

    assert result.status == NOT_EXECUTABLE
    assert "skill replay" in result.detail
    assert page.calls == []


@pytest.mark.unit
def test_scroll_moves_most_of_a_screen_at_a_time():
    """The overlap is deliberate: two observations either side share a strip."""
    page = StubPage(viewport={"width": 1280, "height": VIEWPORT_HEIGHT})
    executor = Executor(page)

    executor.execute(parse_action({"action": "scroll", "direction": "down"}))
    executor.execute(parse_action({"action": "scroll", "direction": "up"}))

    expected = int(VIEWPORT_HEIGHT * SCROLL_FRACTION)
    assert [arg for name, (_, arg) in page.calls if name == "evaluate"] == [expected, -expected]


@pytest.mark.unit
def test_scroll_falls_back_to_a_screen_height_when_the_page_reports_none():
    """A page with no fixed viewport still scrolls by something sensible."""
    page = StubPage(viewport=None)

    Executor(page).execute(parse_action({"action": "scroll", "direction": "down"}))

    expected = int(FALLBACK_VIEWPORT_HEIGHT * SCROLL_FRACTION)
    assert page.calls == [("evaluate", (SCROLL_SCRIPT, expected))]


@pytest.mark.unit
def test_a_key_press_that_takes_the_page_nowhere_still_succeeds():
    """Most keys do not navigate; waiting for one that never comes is not a failure."""
    page = StubPage()

    result = Executor(page).execute(parse_action({"action": "press", "key": "Escape"}))

    assert result.ok
    assert ("keyboard.press", ("Escape",)) in page.calls


@pytest.mark.unit
def test_a_click_on_a_missing_mark_never_reaches_the_browser():
    """Nothing carries the mark, so there is nothing to click and nothing to wait for."""
    page = StubPage()

    result = Executor(page).execute(parse_action({"action": "click", "element_id": 3}))

    assert result.status == ELEMENT_DETACHED
    assert page.calls == [("locator", f'[{ELEMENT_MARK}="3"]')]


@pytest.mark.unit
def test_a_browser_complaint_comes_back_as_a_result():
    """Nothing a website does may end the run by exception."""
    page = StubPage(fails=BrowserFailure("net::ERR_NAME_NOT_RESOLVED at http://nope.invalid/"))

    result = Executor(page).execute(
        parse_action({"action": "navigate", "url": "http://nope.invalid/"})
    )

    assert result.status == NAVIGATION_ERROR
    assert "ERR_NAME_NOT_RESOLVED" in result.detail


# --- Statuses, classification, and how a result reads -----------------------


@pytest.mark.unit
def test_there_are_six_statuses_and_they_are_distinct():
    """A status is counted in run summaries, so the set may not drift quietly."""
    assert len(EXECUTION_STATUSES) == len(set(EXECUTION_STATUSES)) == 6
    assert EXECUTION_STATUSES[0] == OK


@pytest.mark.unit
@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (BrowserTimeout("Timeout 30000ms exceeded."), TIMEOUT),
        (BrowserFailure("Element is not attached to the DOM"), ELEMENT_DETACHED),
        (BrowserFailure("page.goto: net::ERR_CONNECTION_REFUSED"), NAVIGATION_ERROR),
        (BrowserFailure("Unknown key: 'Enterr'"), BROWSER_ERROR),
    ],
)
def test_each_browser_complaint_is_given_its_own_name(error, expected):
    """One exception type, four problems; the message is what tells them apart."""
    assert classify_error(error) == expected


@pytest.mark.unit
def test_a_timeout_during_navigation_is_a_timeout_not_a_navigation_error():
    """The order of the checks matters: slow and unreachable are not the same."""
    slow = BrowserTimeout(
        "page.goto: Timeout 500ms exceeded.\nCall log:\n  - navigating to http://x/"
    )

    assert classify_error(slow) == TIMEOUT


@pytest.mark.unit
def test_the_complaint_keeps_the_first_line_and_drops_the_call_log():
    """Playwright's call log is for debugging a browser, not for a prompt."""
    error = BrowserFailure(
        "page.click: Target closed\nCall log:\n  - waiting for locator\n  - retry"
    )

    assert describe_error(error) == "page.click: Target closed"


@pytest.mark.unit
def test_a_result_reads_as_one_line_in_a_log():
    """A hundred steps have to stay readable, so a step is one line."""
    action = parse_action({"action": "click", "element_id": 4})

    succeeded = ExecutionResult(action=action, status=OK, elapsed_s=0.25).summary()
    failed = ExecutionResult(
        action=action, status=ELEMENT_DETACHED, detail="element 4 is gone", elapsed_s=0.02
    ).summary()

    assert succeeded == "click 4 -> ok (0.25s)"
    assert failed == "click 4 -> element_detached: element 4 is gone (0.02s)"


@pytest.mark.unit
def test_a_result_writes_itself_as_plain_data():
    """A trace is read after the run, so a result has to survive being written down."""
    action = parse_action({"action": "navigate", "url": "http://localhost/a.html"})

    record = ExecutionResult(
        action=action, status=OK, url="http://localhost/a.html", elapsed_s=1.23456
    ).to_dict()

    assert record["action"] == {"action": "navigate", "url": "http://localhost/a.html"}
    assert record["status"] == OK
    assert record["elapsed_s"] == 1.235
    assert parse_action(record["action"]).describe() == action.describe()
