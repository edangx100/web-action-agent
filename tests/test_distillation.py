"""What the agent reads when it reads a page's words rather than its controls.

Two things are being defended, and they fail in different ways.

The first is that **the content survives and the furniture does not**. The
failure here is quiet: nothing errors, the extract is full of text, and the text
is the navigation menu repeated on every page of the site. So the assertions are
made both ways round -- named facts from the article body must be present, and
named strings from the header, nav and footer must be absent. Checking only the
first would pass on an extract that kept everything.

The second is that **an over-long page is cut, and says so**. The failure there
is worse than quiet: a model shown two thirds of a page, with nothing to suggest
anything is missing, answers "the page does not mention it" with full
confidence. So the truncation is checked for its flag, its marker, its size, and
where exactly it fell -- the fixture numbers every entry so the last one kept
and the first one dropped can both be named.

The size arithmetic is separated out and tested with no browser at all, because
it is the part most likely to be tuned and the part that needs no page to be
wrong.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest
from playwright.sync_api import Page

from web_action_agent.config import Config
from web_action_agent.distill import (
    CHARS_PER_TOKEN,
    TRUNCATION_MARKER,
    Distillation,
    distill_from_config,
    distill_text,
    estimate_tokens,
    fit_to_budget,
)

#: Sentences the article fixture contains, each specific enough that finding it
#: means the body was read rather than guessed at.
ARTICLE_FACTS = [
    "214 seats",
    "41,000 catalogued items",
    "2700 kelvin",
    "£2.4 million",
    "Halloway",
]

#: Strings that appear on the article page only in its header, nav or footer.
#: Every one of them must be absent from the extract.
BOILERPLATE_STRINGS = [
    "Browse the catalogue",
    "Join the library",
    "Registered charity",
    "Closed Sundays",
    "Privacy and cookies",
]

#: A budget large enough that nothing is cut, for tests about selection rather
#: than about size.
GENEROUS_BUDGET = 100_000


def read(page: Page, fixture_site, path: str, *, token_budget: int = GENEROUS_BUDGET):
    """Load a fixture page and distil its text.

    Args:
        page: A fresh browser page.
        fixture_site: The running static site.
        path: Site-relative path of the page to read.
        token_budget: Ceiling for the extract, in estimated tokens.

    Returns:
        Distillation: The extract and how it was obtained.
    """
    page.goto(fixture_site.url_for(path))
    return distill_text(page, token_budget=token_budget)


# --- Selecting the content --------------------------------------------------


@pytest.mark.fixture
def test_the_article_body_survives(page: Page, fixture_site) -> None:
    """Every named fact from the article is in the extract.

    These are the facts a task would ask about, so an extract missing them is
    an extract that cannot answer anything, however tidy it looks.
    """
    result = read(page, fixture_site, "article.html")

    missing = [fact for fact in ARTICLE_FACTS if fact not in result.text]
    assert missing == [], f"article facts missing from the extract: {missing}"


@pytest.mark.fixture
def test_the_furniture_does_not(page: Page, fixture_site) -> None:
    """No header, nav or footer string reaches the extract.

    This is the half that catches an extractor which kept everything. The
    strings chosen appear on the page *only* as boilerplate, so finding one
    means a whole region was not stripped.
    """
    result = read(page, fixture_site, "article.html")

    kept = [string for string in BOILERPLATE_STRINGS if string in result.text]
    assert kept == [], f"boilerplate strings survived: {kept}"


@pytest.mark.fixture
def test_the_page_saying_where_its_content_is_gets_believed(page: Page, fixture_site) -> None:
    """A page with a <main> is read from it, and the record says so.

    The strategy is recorded rather than inferred so that a page which read
    oddly can be diagnosed without re-running it.
    """
    result = read(page, fixture_site, "article.html")

    assert result.root == "main"


@pytest.mark.fixture
def test_paragraph_structure_survives_the_tidy_up(page: Page, fixture_site) -> None:
    """Headings and paragraphs stay separated, and blank runs are collapsed.

    Whitespace is normalised aggressively, and the risk in that is arriving at
    one undifferentiated block in which a model cannot tell a heading from the
    sentence after it.
    """
    result = read(page, fixture_site, "article.html")

    assert "\n" in result.text
    # Tidied, not merely passed through: no run of three or more newlines, and
    # no line padded with leading or trailing spaces.
    assert "\n\n\n" not in result.text
    assert all(line == line.strip() for line in result.text.splitlines())


@pytest.mark.fixture
def test_a_page_of_controls_yields_little_and_does_not_pretend_otherwise(
    page: Page, fixture_site
) -> None:
    """A form page has almost no prose, and an honest extract is nearly empty.

    Worth pinning because the tempting failure is a fallback that "finds"
    content by widening until it has swept in the navigation. An extract that
    is short on a short page is correct.
    """
    result = read(page, fixture_site, "form-step-1.html")

    for string in ("Browse the catalogue", "Join the library"):
        assert string not in result.text


# --- Choosing a root on pages that say nothing ------------------------------
#
# Every page of the fixture site has a <main>, which is the easy case and the
# one already covered above. These build their own markup, because the shapes
# worth testing here are the shapes a well-built site does not have.


def with_markup(page: Page, html: str):
    """Distil a page built from markup written inline.

    A real browser is still doing the work -- the visibility and layout
    questions have no answer outside one -- but the document is written here so
    that a shape the fixture site deliberately lacks can still be tested.

    Args:
        page: A fresh browser page.
        html: The document body to load.

    Returns:
        Distillation: The extract and how it was obtained.
    """
    page.set_content(html)
    return distill_text(page, token_budget=GENEROUS_BUDGET)


@pytest.mark.fixture
def test_an_article_is_believed_when_there_is_no_main(page: Page) -> None:
    """``<article>`` is the second-best thing a page can say about itself."""
    result = with_markup(
        page,
        """
        <nav><a href="/a">Home</a><a href="/b">Catalogue</a></nav>
        <article><p>The reading room reopened with 214 seats.</p></article>
        """,
    )

    assert result.root == "article"
    assert "214 seats" in result.text


@pytest.mark.fixture
def test_a_page_with_no_semantic_markup_falls_back_to_the_biggest_block(page: Page) -> None:
    """The guess that carries most of the real web.

    Nothing here says where the content is, so the containers are scored and
    the prose wins. This is the path with the most room to be wrong, which is
    why it is worth pinning on a page shaped to make it wrong.
    """
    result = with_markup(
        page,
        """
        <div id="chrome"><a href="/a">Home</a><a href="/b">Catalogue</a></div>
        <div id="content">
          <p>The reading room reopened with 214 seats and a repaired lantern.</p>
          <p>The archive holds 41,000 catalogued items and opens on Tuesdays.</p>
        </div>
        """,
    )

    assert result.root == "largest_block"
    assert "214 seats" in result.text
    assert "41,000 catalogued items" in result.text


@pytest.mark.fixture
def test_a_long_menu_does_not_beat_a_short_article(page: Page) -> None:
    """Link text is discounted, which is the whole reason the scoring works.

    By raw character count the menu here wins comfortably. Subtracting the text
    inside anchors is what makes the comparison mean something -- without it,
    any site with a large enough navigation would have its menu read back as
    the page's content.
    """
    links = "".join(f'<a href="/{n}">Department of Something Number {n}</a>' for n in range(30))
    result = with_markup(
        page,
        f"""
        <div id="menu">{links}</div>
        <div id="content">
          <p>The reading room reopened with 214 seats and a repaired lantern.</p>
        </div>
        """,
    )

    assert "214 seats" in result.text
    assert "Department of Something" not in result.text


@pytest.mark.fixture
def test_a_wrapper_around_everything_does_not_win(page: Page) -> None:
    """Nesting resolves inward, which is what keeps the menu out.

    Almost every real page wraps its whole body in a container div. That
    wrapper holds the article's prose *and* the menu, so it scores about the
    same as the article alone -- the menu being nearly all link text, which is
    discounted. Preferring the outer of two equals would therefore pick the
    wrapper and carry the menu into the extract on every page built this way.
    """
    result = with_markup(
        page,
        """
        <div id="wrapper">
          <div id="menu"><a href="/a">Home</a><a href="/b">Catalogue</a></div>
          <div id="content">
            <p>The reading room reopened with 214 seats and a repaired lantern.</p>
          </div>
        </div>
        """,
    )

    assert result.root == "largest_block"
    assert "214 seats" in result.text
    assert "Catalogue" not in result.text


@pytest.mark.fixture
def test_a_page_with_no_real_content_gives_up_honestly(page: Page) -> None:
    """No container clears the floor, so the whole body is read.

    Falling back to the body is the honest answer to "I could not tell": it
    fails visibly, with everything included, rather than confidently returning
    whichever fragment happened to score highest among equals.
    """
    result = with_markup(page, "<div><p>Hello.</p></div>")

    assert result.root == "body"


@pytest.mark.fixture
def test_hidden_text_is_not_read(page: Page) -> None:
    """What the reader cannot see, the agent does not report.

    Pages routinely carry collapsed menus and inactive tab panels whose text is
    in the document and not on the screen. Reading it back would describe a
    page nobody is looking at.
    """
    result = with_markup(
        page,
        """
        <main>
          <p>The reading room reopened with 214 seats.</p>
          <p style="display:none">The reading room is permanently closed.</p>
          <p hidden>Members are no longer admitted.</p>
          <p aria-hidden="true">Decorative filler.</p>
        </main>
        """,
    )

    assert "214 seats" in result.text
    assert "permanently closed" not in result.text
    assert "no longer admitted" not in result.text
    assert "Decorative filler" not in result.text


# --- Cutting to the budget --------------------------------------------------


@pytest.mark.fixture
def test_an_over_long_page_is_cut_and_flagged(page: Page, fixture_site) -> None:
    """The ledger page is far over any sane budget, so it is truncated."""
    result = read(page, fixture_site, "ledger.html", token_budget=400)

    assert result.truncated is True
    assert result.text.endswith(TRUNCATION_MARKER)
    assert result.estimated_tokens <= 400


@pytest.mark.fixture
def test_the_cut_falls_where_the_budget_runs_out(page: Page, fixture_site) -> None:
    """Early entries are kept, late ones are not.

    The fixture numbers every entry, so "where was it cut" is a question with
    an exact answer rather than an impression. Without this, a truncation that
    silently returned the *last* 400 tokens would pass every other assertion
    here.
    """
    result = read(page, fixture_site, "ledger.html", token_budget=400)

    assert "Entry 001 of the ledger." in result.text
    assert "Entry 060 of the ledger." not in result.text


@pytest.mark.fixture
def test_the_same_page_within_budget_is_not_cut(page: Page, fixture_site) -> None:
    """Given room, nothing is removed and nothing is flagged.

    The negative case for the flag: if it were set on every page it would tell
    a trace reader nothing.
    """
    result = read(page, fixture_site, "ledger.html")

    assert result.truncated is False
    assert TRUNCATION_MARKER not in result.text
    assert "Entry 060 of the ledger." in result.text


@pytest.mark.fixture
def test_the_configured_budget_is_the_one_that_applies(page: Page, fixture_site) -> None:
    """Reading through the config path uses ``text_token_budget``.

    The budget is a setting, so the path that reads it has to be exercised --
    a default hard-coded somewhere would pass every test that passes a budget
    in by hand.
    """
    page.goto(fixture_site.url_for("ledger.html"))

    result = distill_from_config(page, Config(text_token_budget=300))

    assert result.truncated is True
    assert result.estimated_tokens <= 300


# --- The arithmetic, with no browser in sight -------------------------------


@pytest.mark.unit
def test_an_empty_extract_costs_nothing() -> None:
    """Zero, not one. An empty page should not be reported as spending tokens."""
    assert estimate_tokens("") == 0


@pytest.mark.unit
def test_the_estimate_rounds_up() -> None:
    """A partial token is a token: budgets are ceilings, so estimates round up.

    Rounding down would let a text one character over the budget report itself
    as fitting, which is the one direction the error must not go.
    """
    assert estimate_tokens("a") == 1
    assert estimate_tokens("a" * CHARS_PER_TOKEN) == 1
    assert estimate_tokens("a" * (CHARS_PER_TOKEN + 1)) == 2


@pytest.mark.unit
def test_text_inside_the_budget_is_returned_untouched() -> None:
    """No marker, no flag, and not so much as a stripped space."""
    text = "The reading room reopened on 12 March 2026.\n\nIt has 214 seats."

    fitted, truncated = fit_to_budget(text, token_budget=1000)

    assert fitted == text
    assert truncated is False


@pytest.mark.unit
def test_the_marker_is_paid_for_out_of_the_budget() -> None:
    """The announcement of the cut fits inside the budget it announces.

    Otherwise a page cut to its ceiling would come back one marker over it,
    which is exactly the overflow the budget exists to prevent.
    """
    text = "word " * 4000

    fitted, truncated = fit_to_budget(text, token_budget=100)

    assert truncated is True
    assert fitted.endswith(TRUNCATION_MARKER)
    assert estimate_tokens(fitted) <= 100


@pytest.mark.unit
def test_the_cut_lands_on_a_paragraph_break_when_one_is_near() -> None:
    """A paragraph boundary is preferred to a word boundary.

    Ending an extract mid-sentence is legible; ending it at a paragraph is
    better, and costs only the few words between the two.
    """
    first = "alpha " * 50
    text = first.strip() + "\n\n" + ("bravo " * 50)

    fitted, truncated = fit_to_budget(text, token_budget=91)

    assert truncated is True
    assert fitted == first.strip() + TRUNCATION_MARKER


@pytest.mark.unit
def test_retreating_to_a_boundary_never_costs_half_the_extract() -> None:
    """A break too far back is ignored, and the plain cut stands.

    Otherwise a short extract that opens with a heading gets cut back to the
    heading: the paragraph break after it is, technically, the nearest one to
    the end. Tidiness is not worth content.
    """
    text = "Heading\n\n" + ("x" * 400)

    fitted, truncated = fit_to_budget(text, token_budget=60)

    assert truncated is True
    assert fitted.startswith("Heading\n\nxxxx")


@pytest.mark.unit
def test_no_word_is_cut_in_half() -> None:
    """With no paragraph break to find, the cut retreats to a word boundary.

    A word sliced down the middle costs nothing in tokens and makes the whole
    extract look corrupted, which invites a reader to distrust all of it rather
    than just its ending.
    """
    text = "alpha bravo charlie delta echo foxtrot golf hotel india juliet " * 40

    fitted, truncated = fit_to_budget(text, token_budget=20)

    assert truncated is True
    body = fitted[: -len(TRUNCATION_MARKER)]
    assert body == body.rstrip()
    assert text.startswith(body)
    # The character just past what we kept is a space, so a whole word was the
    # unit dropped.
    assert text[len(body)] == " "


@pytest.mark.unit
def test_unbroken_text_is_cut_where_it_must_be() -> None:
    """Text with no boundary at all still fits, rather than overflowing.

    A minified script or a base64 blob has no word breaks. Preferring a tidy
    cut must not become a refusal to cut.
    """
    text = "x" * 10_000

    fitted, truncated = fit_to_budget(text, token_budget=50)

    assert truncated is True
    assert estimate_tokens(fitted) <= 50


@pytest.mark.unit
def test_a_budget_too_small_for_the_marker_yields_nothing_rather_than_nonsense() -> None:
    """An absurd budget returns empty text, still flagged as truncated.

    Returning the marker alone would be worse than useless: it would spend the
    entire budget announcing that there was no budget.
    """
    fitted, truncated = fit_to_budget("some real page text here", token_budget=1)

    assert fitted == ""
    assert truncated is True


# --- The record ------------------------------------------------------------


@pytest.mark.unit
def test_the_record_serializes_for_the_trace() -> None:
    """Every field reaches the trace under the name it has here."""
    record = Distillation(text="hello", truncated=False, root="main", estimated_tokens=2)

    assert record.to_dict() == {
        "text": "hello",
        "truncated": False,
        "root": "main",
        "estimated_tokens": 2,
    }


@pytest.mark.unit
def test_the_summary_line_says_size_source_and_whether_it_was_cut() -> None:
    """The one line a step log carries, readable at a glance."""
    intact = Distillation(text="x" * 412, truncated=False, root="main", estimated_tokens=103)
    cut = Distillation(text="x" * 412, truncated=True, root="largest_block", estimated_tokens=103)

    assert intact.summary() == "412 chars (~103 tokens) from main"
    assert cut.summary() == "412 chars (~103 tokens) from largest_block, truncated"


@pytest.mark.unit
def test_the_record_is_evidence_and_cannot_be_edited() -> None:
    """Frozen, like every other record of what the agent was shown."""
    record = Distillation(text="hello", truncated=False, root="main", estimated_tokens=2)

    with pytest.raises(FrozenInstanceError):
        record.text = "something else"  # type: ignore[misc]
