"""Smoke tests for the local static site the agent is developed against.

Every later browser test leans on these pages being reachable and on their
marker elements keeping their identifiers. That makes the site worth testing in
its own right rather than checking it incidentally, through whichever test
happens to use it first: a broken page then fails here, with a plain message,
instead of surfacing downstream as an unexplained browser timeout.

The whole module runs over plain HTTP with no browser involved, which keeps it
fast enough to be worth running on every change.
"""

import urllib.error
import urllib.request

import pytest
from fixtures.site_server import SITE_ROOT

PAGES = [
    ("index.html", 'id="home-heading"'),
    ("search.html", 'id="search-form"'),
    ("browse.html", 'id="browse-list"'),
    ("results.html", 'id="results-list"'),
    ("form-step-1.html", 'id="applicant-name"'),
    ("form-step-2.html", 'id="applicant-email"'),
    ("form-step-3.html", 'id="applicant-branch"'),
    ("form-complete.html", 'id="reference-number"'),
    ("consent.html", 'id="consent-modal"'),
    ("article.html", 'id="article-heading"'),
    ("ledger.html", 'id="ledger-heading"'),
]
"""list[tuple[str, str]]: Each page, paired with an element only that page has.

The markers are the identifiers later browser tests click and read, so a
redesign that renames one is caught here first.
"""

ASSETS = [
    ("static/site.css", "text/css"),
    ("static/results.js", "javascript"),
    ("static/form.js", "javascript"),
    ("static/complete.js", "javascript"),
    ("static/consent.js", "javascript"),
]
"""list[tuple[str, str]]: Each supporting asset, paired with its content type.

A stylesheet served as ``text/plain`` still returns 200 but is ignored by the
browser, so the type is worth asserting alongside the status.
"""


def fetch(url: str) -> tuple[int, str, str]:
    """Fetch a URL over HTTP.

    Args:
        url: Absolute URL to request.

    Returns:
        A ``(status, content_type, body)`` tuple, where ``body`` is decoded as
        UTF-8.

    Raises:
        urllib.error.HTTPError: If the server answers 4xx or 5xx.
    """
    with urllib.request.urlopen(url, timeout=10) as response:
        body = response.read().decode("utf-8")
        return response.status, response.headers.get("Content-Type", ""), body


@pytest.mark.fixture
@pytest.mark.parametrize(("path", "marker"), PAGES, ids=[path for path, _ in PAGES])
def test_every_page_serves_with_its_marker_element(fixture_site, path, marker):
    """Assert each page returns 200 as HTML and still carries its marker.

    Args:
        fixture_site: The running static site.
        path: Site-relative path of the page under test.
        marker: Fragment of HTML that identifies the page's marker element.
    """
    status, content_type, body = fetch(fixture_site.url_for(path))

    assert status == 200
    assert content_type.startswith("text/html")
    assert marker in body, f"{path} is missing its marker element {marker}"


@pytest.mark.fixture
@pytest.mark.parametrize(("path", "expected_type"), ASSETS, ids=[path for path, _ in ASSETS])
def test_static_assets_serve(fixture_site, path, expected_type):
    """Assert the stylesheet and scripts are served, and served as themselves.

    Args:
        fixture_site: The running static site.
        path: Site-relative path of the asset under test.
        expected_type: Substring the ``Content-Type`` header must contain.
    """
    status, content_type, _ = fetch(fixture_site.url_for(path))

    assert status == 200
    assert expected_type in content_type


@pytest.mark.fixture
def test_site_root_serves_the_home_page(fixture_site):
    """Assert a bare start URL lands on the home page rather than a 404.

    A task hands the agent a start URL, and the site root is the shortest one
    it can be given.

    Args:
        fixture_site: The running static site.
    """
    status, _, body = fetch(fixture_site.base_url)

    assert status == 200
    assert 'data-page="home"' in body


@pytest.mark.fixture
def test_unknown_path_is_a_404(fixture_site):
    """Assert a mistyped path fails visibly instead of serving another page.

    An agent that follows a broken link has to see the failure; a server that
    quietly fell back to the home page would let that mistake pass unnoticed.

    Args:
        fixture_site: The running static site.
    """
    with pytest.raises(urllib.error.HTTPError) as raised:
        fetch(fixture_site.url_for("no-such-page.html"))

    assert raised.value.code == 404


@pytest.mark.fixture
def test_pages_link_only_within_the_site(fixture_site):
    """Assert no page reaches off-box, so the suite stays entirely offline.

    A single stray font or analytics tag would make continuous integration
    depend on the public internet and fail whenever that host is slow or
    unreachable. Scanning the source for absolute URLs catches such a reference
    the moment it is added.

    Args:
        fixture_site: The running static site, requested so this test runs
            alongside the others even though it reads the files directly.
    """
    offenders = []
    for source in sorted(SITE_ROOT.rglob("*")):
        if source.suffix not in {".html", ".css", ".js"}:
            continue
        text = source.read_text(encoding="utf-8")
        if "http://" in text or "https://" in text:
            offenders.append(source.relative_to(SITE_ROOT).as_posix())

    assert offenders == [], f"absolute URLs found in {offenders}"
