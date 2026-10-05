"""What the agent is offered to act on, and what it is deliberately not.

Two properties carry the rest here, and neither shows itself as a failure
where it is broken.

**The right set.** An element wrongly offered becomes a step spent pressing
something that cannot be pressed. An element wrongly withheld becomes a task
the agent cannot finish and cannot explain. Both surface much later, as a
planner that looks bad at reasoning, so the set is asserted exactly rather than
by spot check: on the search page, *these* elements, with *these* names, and
nothing else.

**Stable numbering.** The planner says ``click 4``, so 4 has to mean the same
thing to the validator and the executor that it meant to the observer. Two
observations of a page nobody touched must agree.

The rest of the module is the filters, each on the piece of ordinary page
furniture that motivates it: the consent overlay for occlusion, the tall
catalogue for the viewport, and the browse page's skip link, disabled
pagination button, hidden block, and decorative duplicate link for the four
things that are drawn but not actionable.
"""

import pytest

from web_action_agent.observer import Box, Element, describe_elements, extract_elements

SEARCH_PAGE_ELEMENTS = [
    (1, "link", "Home"),
    (2, "link", "Search the catalogue"),
    (3, "link", "Browse the catalogue"),
    (4, "link", "Join the library"),
    (5, "link", "Reading room news"),
    (6, "link", "Member notices"),
    (7, "searchbox", "Search terms"),
    (8, "button", "Search"),
]
"""list[tuple[int, str, str]]: Every element of the search page, in order.

The six navigation links, the query field, and the submit button — the whole
of what a person could do on that page, which is what the agent should be
offered and all of it. The field is a ``searchbox`` rather than a ``textbox``
because it is ``<input type="search">``, and its name comes from its
``<label>`` rather than its placeholder.
"""


def named(elements, name):
    """Find the one element with a given accessible name.

    Args:
        elements: Elements from one observation.
        name: The accessible name to look for.

    Returns:
        Element | None: The match, or None when there is none.

    Raises:
        AssertionError: If more than one element carries the name, which would
            make any assertion about "the" element with it meaningless.
    """
    matches = [element for element in elements if element.name == name]
    assert len(matches) <= 1, f"{name!r} matched {len(matches)} elements"
    return matches[0] if matches else None


def names(elements):
    """List the accessible names of an observation, in order.

    Args:
        elements: Elements from one observation.

    Returns:
        list[str]: One name per element.
    """
    return [element.name for element in elements]


@pytest.mark.fixture
def test_the_search_page_yields_exactly_its_interactive_elements(page, fixture_site):
    """Assert the whole element set of a known page, names and numbering.

    Args:
        page: A blank page at the fixed viewport.
        fixture_site: The running static site.
    """
    page.goto(fixture_site.url_for("search.html"))

    elements = extract_elements(page)

    assert [(e.id, e.role, e.name) for e in elements] == SEARCH_PAGE_ELEMENTS


@pytest.mark.fixture
def test_repeat_observation_of_an_unchanged_page_yields_identical_ids(page, fixture_site):
    """Assert nothing about an untouched page renumbers between observations.

    Whole-value equality, not just the numbering: two observations of a page
    nobody touched should be indistinguishable, boxes included.

    Args:
        page: A blank page at the fixed viewport.
        fixture_site: The running static site.
    """
    page.goto(fixture_site.url_for("search.html"))

    first = extract_elements(page)
    second = extract_elements(page)

    assert first == second


@pytest.mark.fixture
def test_ids_run_from_one_without_gaps(page, fixture_site):
    """Assert the numbering is dense, so every number in range names something.

    A gap would be a number the planner could plausibly emit and the validator
    would have to reject for no reason a reader could see.

    Args:
        page: A blank page at the fixed viewport.
        fixture_site: The running static site.
    """
    page.goto(fixture_site.url_for("browse.html"))

    elements = extract_elements(page)

    assert [element.id for element in elements] == list(range(1, len(elements) + 1))


@pytest.mark.fixture
def test_only_what_is_in_the_viewport_is_listed(page, fixture_site):
    """Assert scrolling is what reaches an element further down the page.

    The catalogue is far taller than the viewport, so its pagination sits well
    below the fold and the site header well above it once scrolled. Neither is
    hidden, disabled, or covered — the only thing keeping each out of an
    observation is where the page is scrolled to.

    Args:
        page: A blank page at the fixed viewport.
        fixture_site: The running static site.
    """
    page.goto(fixture_site.url_for("browse.html"))

    at_top = names(extract_elements(page))
    assert "Home" in at_top
    assert "Next page" not in at_top

    page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
    at_bottom = names(extract_elements(page))

    assert "Next page" in at_bottom
    assert "Home" not in at_bottom


@pytest.mark.fixture
def test_every_box_lies_inside_the_viewport(page, fixture_site):
    """Assert boxes are clipped to the screen, since the screenshot is too.

    A box running off the bottom of the viewport would put an element's mark
    outside the annotated screenshot, breaking the one guarantee the two
    channels have to keep: that they number the same things.

    Args:
        page: A blank page at the fixed viewport.
        fixture_site: The running static site.
    """
    page.goto(fixture_site.url_for("browse.html"))
    viewport = page.viewport_size

    for element in extract_elements(page):
        box = element.box
        assert box.x >= 0 and box.y >= 0, element
        assert box.x + box.width <= viewport["width"], element
        assert box.y + box.height <= viewport["height"], element
        assert box.width > 0 and box.height > 0, element


@pytest.mark.fixture
def test_drawn_but_unusable_furniture_is_not_offered(page, fixture_site):
    """Assert the four kinds of element that must never reach the planner.

    Each is real page furniture rather than a contrivance: a visually hidden
    skip link, a block hidden until a filter matches nothing, an anchor that is
    a scroll target rather than a link, and — after scrolling to the
    pagination — a disabled button and a decorative duplicate link marked
    ``aria-hidden``.

    Args:
        page: A blank page at the fixed viewport.
        fixture_site: The running static site.
    """
    page.goto(fixture_site.url_for("browse.html"))

    at_top = names(extract_elements(page))
    assert "Skip to the catalogue" not in at_top  # One pixel square.
    assert "Clear the filter" not in at_top  # Inside a `hidden` block.
    assert "" not in at_top  # The `<a id="catalogue">` has no href.

    page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
    at_bottom = names(extract_elements(page))

    assert "Next page" in at_bottom  # The pagination is genuinely on screen.
    assert "Previous page" not in at_bottom  # Disabled on the first page.
    assert "›" not in at_bottom  # aria-hidden, and duplicates "Next page".


@pytest.mark.fixture
def test_a_span_with_a_button_role_is_offered_as_a_button(page, fixture_site):
    """Assert a control is recognized by its role, not by its tag.

    Sites build buttons out of ``<span>`` and ``<div>`` constantly. An
    extractor that trusted tags would leave the agent unable to see them.

    Args:
        page: A blank page at the fixed viewport.
        fixture_site: The running static site.
    """
    page.goto(fixture_site.url_for("browse.html"))

    saved = named(extract_elements(page), "Save this search")

    assert saved is not None
    assert (saved.tag, saved.role) == ("span", "button")


@pytest.mark.fixture
def test_an_aria_label_names_an_element_its_own_text_does_not(page, fixture_site):
    """Assert the accessible name wins over visible text where they differ.

    The current page number reads as "1" on screen and announces itself as
    "Page 1, current page". The planner needs the second to tell it from the
    links either side; a screenshot reader needs the first to find it.

    Args:
        page: A blank page at the fixed viewport.
        fixture_site: The running static site.
    """
    page.goto(fixture_site.url_for("browse.html"))
    page.evaluate("window.scrollTo(0, document.body.scrollHeight)")

    current = named(extract_elements(page), "Page 1, current page")

    assert current is not None
    assert current.text == "1"


@pytest.mark.fixture
def test_a_field_reports_its_contents_apart_from_its_name(page, fixture_site):
    """Assert a filled field shows both what it is and what is in it.

    This is how the agent tells that its own last ``type`` took effect. A
    field whose text replaced its name would lose the first; one that reported
    only its name would lose the second.

    Args:
        page: A blank page at the fixed viewport.
        fixture_site: The running static site.
    """
    page.goto(fixture_site.url_for("search.html"))
    page.fill("#query", "archive")

    field = named(extract_elements(page), "Search terms")

    assert field is not None
    assert field.text == "archive"


@pytest.mark.fixture
def test_a_consent_overlay_hides_the_page_beneath_it(page, fixture_site):
    """Assert occlusion is judged by what a click would hit, not by layout.

    The notices page lays out in full beneath its consent dialog: every link is
    positioned, sized, and visible to any measurement of the DOM alone. Every
    click on one goes to the overlay. Offering them would put the agent in a
    loop of clicks that change nothing — the single most common way a browser
    agent wastes a run.

    Args:
        page: A blank page at the fixed viewport.
        fixture_site: The running static site.
    """
    page.goto(fixture_site.url_for("consent.html"))

    behind_the_dialog = extract_elements(page)
    assert names(behind_the_dialog) == ["Accept", "Decline"]

    page.click("#accept-consent")
    after = names(extract_elements(page))

    assert "Home" in after
    assert "Accept" not in after


@pytest.mark.fixture
def test_iframes_and_shadow_roots_are_not_traversed(page):
    """Assert the v0 scope boundary holds, and holds quietly.

    Both are out of scope, and the way that has to fail is by the element
    simply not being listed — so the planner cannot name it, and the validator
    never has to reason about a reference it cannot resolve.

    The markup is built here rather than added to the fixture site: what is
    under test is a property of page structure, not of the site, and a reader
    checking the claim should not have to open another file to see the frame
    and the shadow root that the claim is about.

    Args:
        page: A blank page at the fixed viewport.
    """
    page.set_content(
        """
        <button id="top-level">In the page</button>
        <iframe srcdoc="<button>Inside a frame</button>"></iframe>
        <div id="host"></div>
        <script>
          document.getElementById("host")
            .attachShadow({ mode: "open" })
            .innerHTML = "<button>Inside a shadow root</button>";
        </script>
        """
    )

    assert names(extract_elements(page)) == ["In the page"]


@pytest.mark.fixture
def test_a_password_field_never_reports_its_value(page):
    """Assert a typed secret does not reach the observation.

    Every observation is written to the run's trace, and traces are read,
    shared, and committed. The field is still listed — the agent has to know it
    is there — but its contents are not read at all.

    The markup is inline because the fixture site has no password field and
    should not grow one: logging in is a project non-goal, and this is a
    guarantee about what is recorded rather than about anything the agent does.

    Args:
        page: A blank page at the fixed viewport.
    """
    page.set_content('<label for="pw">Passphrase</label><input id="pw" type="password">')
    page.fill("#pw", "hunter2")

    field = named(extract_elements(page), "Passphrase")

    assert field is not None
    assert field.role == "textbox"
    assert field.text == ""


@pytest.mark.fixture
def test_an_empty_page_observes_as_empty_rather_than_failing(page):
    """Assert a page with nothing to act on is an observation, not an error.

    It is also one of the two signals that gate a screenshot in, so it has to
    arrive as an ordinary empty list.

    Args:
        page: A blank page at the fixed viewport.
    """
    page.set_content("<p>Nothing here but this sentence.</p>")

    assert extract_elements(page) == ()


@pytest.mark.unit
def test_an_element_renders_as_one_line_for_the_planner():
    """Assert the rendering shows the number, the role, the name, and contents."""
    box = Box(x=10, y=20, width=100, height=40)
    link = Element(id=3, tag="a", role="link", name="Home", text="Home", box=box)
    field = Element(id=6, tag="input", role="searchbox", name="Search terms", text="rust", box=box)

    assert link.describe() == '[3] link "Home"'
    assert field.describe() == '[6] searchbox "Search terms" = "rust"'


@pytest.mark.unit
def test_an_empty_element_list_renders_as_a_sentence():
    """Assert an empty observation is said out loud in the prompt.

    An empty string here would read as a section the prompt failed to fill in,
    which is a different thing from a page that offers nothing.
    """
    assert describe_elements(()) != ""
    assert "no interactive elements" in describe_elements(())


@pytest.mark.unit
def test_a_box_reports_its_center():
    """Assert the center is where a mark is drawn and a click would land."""
    assert Box(x=10, y=20, width=100, height=40).center == (60, 40)
