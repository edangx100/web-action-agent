"""Smoke tests for the local static site the agent is developed against.

Every later browser test leans on these pages being reachable and on their
marker elements keeping their identifiers, so the site gets its own test rather
than being checked incidentally by whatever happens to use it first.
"""

import urllib.error
import urllib.request

import pytest
from fixtures.site_server import SITE_ROOT

# Each page, and one element that only that page has. If a marker disappears in
# a redesign, the tests that click it should fail here first, with a clearer
# message than a browser timeout.
PAGES = [
    ("index.html", 'id="home-heading"'),
    ("search.html", 'id="search-form"'),
    ("results.html", 'id="results-list"'),
    ("form-step-1.html", 'id="applicant-name"'),
    ("form-step-2.html", 'id="applicant-email"'),
    ("form-step-3.html", 'id="applicant-branch"'),
    ("form-complete.html", 'id="reference-number"'),
    ("consent.html", 'id="consent-modal"'),
    ("article.html", 'id="article-heading"'),
]

ASSETS = [
    ("static/site.css", "text/css"),
    ("static/results.js", "javascript"),
    ("static/form.js", "javascript"),
    ("static/complete.js", "javascript"),
    ("static/consent.js", "javascript"),
]


def fetch(url: str) -> tuple[int, str, str]:
    """Return status, content type, and body text for a URL."""
    with urllib.request.urlopen(url, timeout=10) as response:
        body = response.read().decode("utf-8")
        return response.status, response.headers.get("Content-Type", ""), body


@pytest.mark.fixture
@pytest.mark.parametrize(("path", "marker"), PAGES, ids=[path for path, _ in PAGES])
def test_every_page_serves_with_its_marker_element(fixture_site, path, marker):
    status, content_type, body = fetch(fixture_site.url_for(path))

    assert status == 200
    assert content_type.startswith("text/html")
    assert marker in body, f"{path} is missing its marker element {marker}"


@pytest.mark.fixture
@pytest.mark.parametrize(("path", "expected_type"), ASSETS, ids=[path for path, _ in ASSETS])
def test_static_assets_serve(fixture_site, path, expected_type):
    status, content_type, _ = fetch(fixture_site.url_for(path))

    assert status == 200
    assert expected_type in content_type


@pytest.mark.fixture
def test_site_root_serves_the_home_page(fixture_site):
    """A bare start URL has to land somewhere, not 404."""
    status, _, body = fetch(fixture_site.base_url)

    assert status == 200
    assert 'data-page="home"' in body


@pytest.mark.fixture
def test_unknown_path_is_a_404(fixture_site):
    """A mistyped link must fail visibly rather than serving another page."""
    with pytest.raises(urllib.error.HTTPError) as raised:
        fetch(fixture_site.url_for("no-such-page.html"))

    assert raised.value.code == 404


@pytest.mark.fixture
def test_pages_link_only_within_the_site(fixture_site):
    """No page may reach off-box: the suite runs with no network at all."""
    offenders = []
    for source in sorted(SITE_ROOT.rglob("*")):
        if source.suffix not in {".html", ".css", ".js"}:
            continue
        text = source.read_text(encoding="utf-8")
        if "http://" in text or "https://" in text:
            offenders.append(source.relative_to(SITE_ROOT).as_posix())

    assert offenders == [], f"absolute URLs found in {offenders}"
