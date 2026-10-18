"""Fixtures and collection rules shared by the whole test suite.

pytest loads this file automatically, so anything defined here is available to
every test module by name, without an import.
"""

import os
from collections.abc import Iterator
from pathlib import Path

import pytest
from fixtures.site_server import StaticSite
from marker_policy import policy_violations, violation_report
from playwright.sync_api import Page, Playwright, sync_playwright

from web_action_agent import config
from web_action_agent.browser import BrowserProcess, BrowserSettings, TaskSession

VIEWPORT = {"width": 1280, "height": 800}
"""dict[str, int]: The viewport every browser test runs at.

Mirrors the committed default in ``config.yaml``, and deliberately fixed rather
than read from it: element extraction is viewport-only and the annotated
screenshot is numbered to match, so a test whose viewport came from local
configuration would be testing a different page than continuous integration
was.
"""

BROWSER_SETTINGS = BrowserSettings(
    viewport_width=VIEWPORT["width"],
    viewport_height=VIEWPORT["height"],
)
"""BrowserSettings: How the suite's browser is started.

The suite drives the agent's own browser lifecycle rather than a second copy of
it, so a test page is opened exactly the way a task's page is: one process for
the session, one fresh context per test. Anything that broke isolation for a
run would break it here too, in front of every browser test at once.
"""


@pytest.fixture(autouse=True)
def _no_local_secrets(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Keep every test away from this machine's own ``.env`` and ``WAA_`` variables.

    A developer's ``.env`` holds their keys and choices -- a judge profile,
    a different planner -- and a test that read it would pass on one machine
    and fail on the next, or behave differently once keys are added. So each
    test sees no ``.env`` file and no ``WAA_`` variable unless it supplies its
    own. A test that needs the real keys (a ``live`` one) reads the project's
    ``.env`` by its path, which this leaves alone.

    Args:
        monkeypatch: Undoes both changes when the test ends.
        tmp_path: Where the absent ``.env`` would be.
    """
    monkeypatch.setattr(config, "DEFAULT_DOTENV_PATH", tmp_path / "absent.env")
    for name in [n for n in os.environ if n.startswith("WAA_")]:
        monkeypatch.delenv(name)


@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(
    config: pytest.Config,
    items: list[pytest.Item],
) -> None:
    """Refuse to run a suite in which any test fails to say what it needs.

    ``tryfirst`` puts this ahead of the plugin that applies ``-m``, so the
    whole suite is checked on every invocation -- including one that selects a
    narrow slice of it. Without that ordering, a test could be missing its
    marker for weeks and only be noticed by the run meant to execute it.

    Args:
        config: The active pytest configuration. Unused, but part of the hook
            signature.
        items: Every test collected for this session.

    Raises:
        pytest.UsageError: If any test carries no test-type marker, or more
            than one. The message names each offending test.
    """
    violations = policy_violations(items)
    if violations:
        raise pytest.UsageError(violation_report(violations))


@pytest.fixture(scope="session")
def fixture_site() -> Iterator[StaticSite]:
    """Serve the local static site for the length of the test session.

    Session scope is deliberate: the server keeps no per-test state, so
    starting a fresh one for every test would buy nothing but a slower suite.
    Test isolation comes from a fresh browser context instead.

    Yields:
        StaticSite: The running server. Use ``site.url_for("search.html")`` to
        address a page and ``site.base_url`` for the site root.
    """
    with StaticSite() as site:
        yield site


@pytest.fixture(scope="session")
def playwright() -> Iterator[Playwright]:
    """Run one Playwright driver for the whole test session.

    Exposed as a fixture of its own because the synchronous API allows exactly
    one driver per thread: a test that starts its own browser process has to
    borrow this one rather than start a second, which would fail outright.

    Yields:
        Playwright: The running driver.
    """
    with sync_playwright() as driver:
        yield driver


@pytest.fixture(scope="session")
def browser_process(playwright: Playwright) -> Iterator[BrowserProcess]:
    """Run one headless Chromium for the whole test session.

    Launching a browser costs a second or so, which is worth paying once and
    not once per test. Nothing leaks between tests as a result, because a test
    never touches this directly -- it asks for ``page``, which brings its own
    context.

    Args:
        playwright: The session's driver, borrowed rather than started again.

    Yields:
        BrowserProcess: The running browser.
    """
    with BrowserProcess(BROWSER_SETTINGS, playwright=playwright) as process:
        yield process


@pytest.fixture
def task_session(browser_process: BrowserProcess) -> Iterator[TaskSession]:
    """Open a fresh browser context, watched, for one test.

    The context is the isolation boundary: its own cookies, its own storage,
    its own history. One per test is the same rule the agent follows for one
    per task, and for the same reason -- what an earlier run left behind must
    not decide what a later one sees.

    Args:
        browser_process: The session's browser.

    Yields:
        TaskSession: The test's pages, with tabs, dialogs and downloads
        watched exactly as they are during a run.
    """
    with browser_process.task_session() as session:
        yield session


@pytest.fixture
def page(task_session: TaskSession) -> Page:
    """The page a test starts on.

    Deliberately the *first* page rather than ``task_session.page``: a test
    that opens a tab wants to keep talking about the page it started on, while
    the session moves on to the new one.

    Args:
        task_session: The test's watched context.

    Returns:
        Page: A blank page sized to :data:`VIEWPORT`.
    """
    return task_session.page
