"""Fixtures and collection rules shared by the whole test suite.

pytest loads this file automatically, so anything defined here is available to
every test module by name, without an import.
"""

from collections.abc import Iterator

import pytest
from fixtures.site_server import StaticSite
from marker_policy import policy_violations, violation_report


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
