"""Fixtures shared by the whole test suite."""

from collections.abc import Iterator

import pytest
from fixtures.site_server import StaticSite


@pytest.fixture(scope="session")
def fixture_site() -> Iterator[StaticSite]:
    """The local static site, served for the length of the test session.

    Session scope on purpose: the server holds no per-test state, so starting
    one per test would only buy a slower suite.
    """
    with StaticSite() as site:
        yield site
