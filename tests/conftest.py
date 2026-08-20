"""Fixtures shared by the whole test suite.

pytest loads this file automatically, so anything defined here is available to
every test module by name, without an import.
"""

from collections.abc import Iterator

import pytest
from fixtures.site_server import StaticSite


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
