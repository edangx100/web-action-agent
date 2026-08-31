"""Fixtures and collection rules shared by the whole test suite.

pytest loads this file automatically, so anything defined here is available to
every test module by name, without an import.
"""

from collections.abc import Iterator

import pytest
from fixtures.site_server import StaticSite
from marker_policy import policy_violations, violation_report
from playwright.sync_api import Browser, Page, sync_playwright

VIEWPORT = {"width": 1280, "height": 800}
"""dict[str, int]: The viewport every browser test runs at.

Mirrors the committed default in ``config.yaml``, and deliberately fixed rather
than read from it: element extraction is viewport-only and the annotated
screenshot is numbered to match, so a test whose viewport came from local
configuration would be testing a different page than continuous integration
was.
"""


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
def browser() -> Iterator[Browser]:
    """Run one headless Chromium for the whole test session.

    Launching a browser costs a second or so, which is worth paying once and
    not once per test. Nothing leaks between tests as a result, because a test
    never touches this directly -- it asks for ``page``, which brings its own
    context.

    Yields:
        Browser: The running browser.
    """
    with sync_playwright() as playwright:
        instance = playwright.chromium.launch()
        try:
            yield instance
        finally:
            instance.close()


@pytest.fixture
def page(browser: Browser) -> Iterator[Page]:
    """Open a fresh page, in a fresh browser context, for one test.

    The context is the isolation boundary: its own cookies, its own storage,
    its own history. One per test is the same rule the agent follows for one
    per task, and for the same reason -- what an earlier run left behind must
    not decide what a later one sees.

    Args:
        browser: The session's browser.

    Yields:
        Page: A blank page sized to :data:`VIEWPORT`.
    """
    context = browser.new_context(viewport=VIEWPORT)
    try:
        yield context.new_page()
    finally:
        context.close()
