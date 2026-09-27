"""Finding the same element again after somebody has redesigned the page.

The tests that matter here run against two real pages in a real browser:
``shelves.html`` and ``shelves-rebuilt.html``, which are the same page a
redesign apart. A skill is *described* against the first and *resolved* against
the second, which is exactly what happens to a stored skill over the weeks
between being learnt and being replayed.

That pairing is the point. A test that made up an element list could prove the
matching rules are implemented; only a real page can show that the accessible
name of a real button is a thing the extractor actually produces, that a
nameless icon button really is nameless, and that the descriptions survive a
round trip through a browser. Every tier of the fallback is reachable on those
two pages, and so is failing all of them.

Two things get as much attention as the successes.

**Refusing to guess.** Two buttons called "Reserve" mean the accessible name
does not know which was meant. Picking the first would be a wrong click that a
trace records as a success, which is the hardest kind of mistake to find later.
So an ambiguous tier resolves nothing and hands the problem down.

**Failing usefully.** When every description has been tried and nothing
matched, the answer says so and carries what each tier attempted — because a
site redesigned out from under a skill is ordinary, and the record of *which*
description stopped matching is what makes it diagnosable.
"""

from __future__ import annotations

from typing import Any

import pytest

from web_action_agent.observer import Box, Element, extract_elements
from web_action_agent.selectors import (
    AMBIGUOUS,
    FOUND,
    NO_MATCH,
    NOT_DESCRIBED,
    TIER_NAME,
    TIER_ROLE_TEXT,
    TIER_STRUCTURAL,
    TIERS,
    Resolution,
    normalise,
    resolve,
    selector_for,
    structural_path,
)
from web_action_agent.skill_schema import Selector


def element(number: int, role: str, name: str, text: str | None = None, tag: str = "a") -> Element:
    """Build one element, for the tests that need no browser.

    Args:
        number: Its number in the observation.
        role: What it is.
        name: Its accessible name.
        text: What it displays. Defaults to the name, as it does on most real
            elements.
        tag: The HTML tag, which only matters where the role is empty.

    Returns:
        Element: The element.
    """
    return Element(
        id=number,
        tag=tag,
        role=role,
        name=name,
        text=name if text is None else text,
        box=Box(x=0, y=number * 20, width=80, height=18),
    )


def named(resolution: Resolution) -> str:
    """Say which element a resolution found, for a readable assertion.

    Args:
        resolution: What came back.

    Returns:
        str: The element's name, its text when it has no name, or a note that
        nothing was found.
    """
    if resolution.element is None:
        return "<nothing>"
    return resolution.element.name or resolution.element.text or "<unnamed>"


def outcomes(resolution: Resolution) -> dict[str, str]:
    """Collect what each tier reported, keyed by tier.

    Args:
        resolution: What came back.

    Returns:
        dict[str, str]: The outcome of every tier that was tried. Tiers after
        the one that resolved are absent, because they were never reached.
    """
    return {attempt.tier: attempt.outcome for attempt in resolution.attempts}


# --- Against two real pages, a redesign apart -------------------------------


@pytest.fixture
def shelves(page: Any, fixture_site: Any) -> tuple[Element, ...]:
    """The elements of the shelf guide, before it was redesigned.

    Args:
        page: A real browser page.
        fixture_site: The practice site.

    Returns:
        tuple[Element, ...]: What the agent would see, with the marks written.
    """
    page.goto(fixture_site.url_for("shelves.html"))
    return extract_elements(page)


@pytest.fixture
def rebuilt(page: Any, fixture_site: Any) -> tuple[Element, ...]:
    """The elements of the same page after the redesign.

    Args:
        page: A real browser page.
        fixture_site: The practice site.

    Returns:
        tuple[Element, ...]: What the agent would see now.
    """
    page.goto(fixture_site.url_for("shelves-rebuilt.html"))
    return extract_elements(page)


def stored(elements: tuple[Element, ...], label: str) -> Selector:
    """Describe an element of the old page the way a skill would store it.

    This is what the inducer will do: take the element a successful run acted
    on and write down every way of recognising it again.

    Args:
        elements: The old page's elements.
        label: The element's accessible name, or its text when it has none.

    Returns:
        Selector: The stored description.

    Raises:
        AssertionError: If no element matches, which would make the rest of the
            test assert nothing.
    """
    for candidate in elements:
        if label in (candidate.name, candidate.text):
            return selector_for(candidate, elements)
    raise AssertionError(f"no element labelled {label!r} in {[e.label() for e in elements]}")


@pytest.mark.fixture
def test_a_label_the_redesign_left_alone_resolves_on_its_name(
    shelves: tuple[Element, ...], rebuilt: tuple[Element, ...]
) -> None:
    """The ordinary case, and the reason the name is tried first.

    Most of a redesign moves boxes and renames CSS classes. It very rarely
    renames the button, because the name is what the page promises to people
    who cannot see it.
    """
    selector = stored(shelves, "Request a reprint")

    resolution = resolve(selector, rebuilt)

    assert resolution.found
    assert resolution.tier == TIER_NAME
    assert named(resolution) == "Request a reprint"
    assert resolution.fell_back is False, "nothing weaker was needed"


@pytest.mark.fixture
def test_a_relabelled_element_is_found_by_what_it_still_says(
    shelves: tuple[Element, ...], rebuilt: tuple[Element, ...]
) -> None:
    """The name changed; the words on screen did not.

    "Ask a librarian" became "Contact the desk" in the page's label while the
    visible wording stayed as it was. The strongest description is now wrong,
    and the second one is exactly right — which is the whole argument for having
    more than one.
    """
    selector = stored(shelves, "Ask a librarian")

    resolution = resolve(selector, rebuilt)

    assert resolution.found
    assert resolution.tier == TIER_ROLE_TEXT
    assert named(resolution) == "Contact the desk"
    assert outcomes(resolution)[TIER_NAME] == NO_MATCH, "and it recorded why it had to"
    assert resolution.fell_back is True


@pytest.mark.fixture
def test_an_element_with_no_name_and_no_text_is_found_by_counting(
    shelves: tuple[Element, ...], rebuilt: tuple[Element, ...]
) -> None:
    """The case the last resort exists for, and it is not a contrivance.

    An icon button with no accessible name and no text is a real and common
    failure of real sites. Nothing about it can be described except where it
    sits, so a skill that needs to press it has nothing else to store.
    """
    icon = next(e for e in shelves if not e.name and not e.text)
    selector = selector_for(icon, shelves)

    resolution = resolve(selector, rebuilt)

    assert selector.structural == "button[4]", "the fourth button on the page"
    assert resolution.found
    assert resolution.tier == TIER_STRUCTURAL
    assert outcomes(resolution) == {
        TIER_NAME: NOT_DESCRIBED,
        TIER_ROLE_TEXT: NOT_DESCRIBED,
        TIER_STRUCTURAL: FOUND,
    }, "the two better descriptions were never available, which is not the same as failing"


@pytest.mark.fixture
def test_two_elements_with_one_name_are_settled_by_the_next_description(
    shelves: tuple[Element, ...], rebuilt: tuple[Element, ...]
) -> None:
    """The refusal to guess, on a page where guessing would look fine.

    Both reserve buttons are labelled "Reserve". A resolver that took the first
    match would click the wrong shelf roughly half the time, succeed as far as
    the browser is concerned, and leave a trace that says the step worked.
    """
    selector = stored(shelves, "Reserve reading room")

    resolution = resolve(selector, rebuilt)

    assert resolution.found
    assert resolution.tier == TIER_ROLE_TEXT
    # Checked on the text, because the *name* is the half that cannot tell them
    # apart -- asserting on it here would pass whichever button came back.
    assert resolution.element is not None
    assert resolution.element.text == "Reserve reading room"
    assert outcomes(resolution)[TIER_NAME] == AMBIGUOUS
    assert resolution.attempts[0].matches == 2, "and it said how many it could not choose between"


@pytest.mark.fixture
def test_an_element_that_is_simply_gone_resolves_to_nothing(
    shelves: tuple[Element, ...], rebuilt: tuple[Element, ...]
) -> None:
    """Every tier tried, nothing matched, and the answer says so.

    The second nameless button was removed in the redesign. It could only ever
    have been found by counting, and there is no fifth button to count to. The
    dangerous outcome would be landing on the fourth — the *other* icon button,
    which is still there and still nameless.
    """
    icons = [e for e in shelves if not e.name and not e.text]
    selector = selector_for(icons[-1], shelves)

    resolution = resolve(selector, rebuilt)

    assert selector.structural == "button[5]"
    assert resolution.found is False
    assert resolution.element is None, "and emphatically not the button that is still there"
    assert resolution.tier == ""
    assert [attempt.tier for attempt in resolution.attempts] == list(TIERS), "all three were tried"
    assert "no element matched" in resolution.summary()


@pytest.mark.fixture
def test_every_tier_is_reachable_on_one_page(
    shelves: tuple[Element, ...], rebuilt: tuple[Element, ...]
) -> None:
    """One page, three tiers, and one failure — so the fixture earns its place.

    Asserted together because the value of the pair of pages is that it covers
    the whole ladder. If a change to the fixture quietly removed a rung, the
    individual tests above would still pass on the rungs that remained.
    """
    icons = [e for e in shelves if not e.name and not e.text]

    reached = {
        resolve(stored(shelves, "Request a reprint"), rebuilt).tier,
        resolve(stored(shelves, "Ask a librarian"), rebuilt).tier,
        resolve(selector_for(icons[0], shelves), rebuilt).tier,
        resolve(selector_for(icons[-1], shelves), rebuilt).tier,
    }

    assert reached == {TIER_NAME, TIER_ROLE_TEXT, TIER_STRUCTURAL, ""}


@pytest.mark.fixture
def test_a_description_is_taken_from_a_real_element_whole(
    shelves: tuple[Element, ...],
) -> None:
    """All three descriptions are written down, because nobody knows which lasts.

    Storing only the best one available would mean a skill with no fallback
    exactly when the best one is what breaks.
    """
    reprint = next(e for e in shelves if e.name == "Request a reprint")

    selector = selector_for(reprint, shelves)

    assert selector.name == "Request a reprint"
    assert selector.role == "button"
    assert selector.text == "Request a reprint"
    assert selector.structural == "button[1]"


# --- The rules themselves, without a browser --------------------------------


@pytest.mark.unit
def test_counting_is_done_within_a_role_not_across_the_page() -> None:
    """So that adding a button does not move every link's description.

    It makes the weakest tier slightly less weak. It does not make it strong,
    and nothing here pretends otherwise.
    """
    page = (
        element(1, "link", "Home"),
        element(2, "button", "Search"),
        element(3, "link", "Browse"),
        element(4, "link", "Contact"),
    )

    assert structural_path(page[2], page) == "link[2]"
    assert structural_path(page[1], page) == "button[1]"


@pytest.mark.unit
def test_an_element_inserted_above_breaks_a_structural_description() -> None:
    """Why counting is the last resort, shown rather than asserted in prose.

    A link added to the top of a page is not a change anybody would describe as
    touching the third link. It silently makes the third link a different
    element — and this is the tier with no way to notice.
    """
    before = (element(1, "link", "Home"), element(2, "link", "Browse"), element(3, "link", "Help"))
    after = (
        element(1, "link", "New"),
        *[element(i + 2, "link", e.name) for i, e in enumerate(before)],
    )

    selector = Selector(structural=structural_path(before[2], before))
    resolution = resolve(selector, after)

    assert selector.structural == "link[3]"
    assert named(resolution) == "Browse", "which is the wrong element, and nothing here can tell"


@pytest.mark.unit
def test_the_name_tier_uses_the_role_only_to_break_a_tie() -> None:
    """A link called "Search" and a button called "Search" are different things.

    Narrowing by role costs nothing when the name leaves several candidates.
    It is not applied when the name already settles it, so a site that changed
    an element's role still resolves on its name alone.
    """
    page = (element(1, "link", "Search"), element(2, "button", "Search"))

    tie = resolve(Selector(name="Search", role="button"), page)
    changed_role = resolve(
        Selector(name="Request a reprint", role="link"),
        (element(9, "button", "Request a reprint"),),
    )

    assert tie.found and tie.element is not None and tie.element.role == "button"
    assert changed_role.found, "the name is the durable half; the role is only a tiebreak"


@pytest.mark.unit
def test_spacing_and_capitals_do_not_count_as_a_changed_label() -> None:
    """A page reindented by a build tool has not been relabelled.

    Two liberties, and only two: whitespace is collapsed and case is ignored.
    Wording is not touched, because a site that changed "Search" to "Search
    catalogue" changed the button.
    """
    page = (element(1, "link", "Next   Page"),)

    assert resolve(Selector(name="next page"), page).found
    assert resolve(Selector(name="Next"), page).found is False, "not a prefix match"
    assert normalise("  Next   Page ") == "next page"


@pytest.mark.unit
def test_a_tier_with_nothing_to_try_is_not_a_tier_that_failed() -> None:
    """Two things a trace must not confuse.

    "The skill never described it that way" and "the description no longer
    matches" look the same in a resolution that lumps them together — and one
    of them means the site changed while the other means nothing at all.
    """
    page = (element(1, "link", "Home"),)

    resolution = resolve(Selector(name="Gone", role="link"), page)

    assert outcomes(resolution)[TIER_NAME] == NO_MATCH, "described, and no longer there"
    assert outcomes(resolution)[TIER_ROLE_TEXT] == NOT_DESCRIBED, (
        "a role with no text is not a tier"
    )
    assert outcomes(resolution)[TIER_STRUCTURAL] == NOT_DESCRIBED


@pytest.mark.unit
def test_an_ambiguity_nothing_settles_finds_nothing() -> None:
    """Better to fail than to pick one of two.

    Nothing later in the ladder can help when the only description stored is
    the ambiguous one, and this is the case where returning a match would be
    most tempting and most wrong.
    """
    page = (element(1, "button", "Reserve"), element(2, "button", "Reserve"))

    resolution = resolve(Selector(name="Reserve"), page)

    assert resolution.found is False
    assert outcomes(resolution)[TIER_NAME] == AMBIGUOUS
    assert "settled nothing" in resolution.summary()


@pytest.mark.unit
def test_a_structural_description_nobody_wrote_here_matches_nothing() -> None:
    """A hand-edited skill file is not a licence to guess at what was meant.

    The description is generated by this module in one shape. Anything else is
    something a person typed, and inventing a click from it would be worse than
    saying it did not resolve.
    """
    page = (element(1, "link", "Home"), element(2, "link", "Browse"))

    for written_by_hand in ("the second link", "link[]", "link(2)", "2", ""):
        assert resolve(Selector(structural=written_by_hand, name="x"), page).found is False


@pytest.mark.unit
def test_a_position_past_the_end_of_the_page_finds_nothing() -> None:
    """The page is shorter than it was; the answer is not the last element."""
    page = (element(1, "link", "Home"), element(2, "link", "Browse"))

    resolution = resolve(Selector(structural="link[7]"), page)

    assert resolution.found is False
    assert outcomes(resolution)[TIER_STRUCTURAL] == NO_MATCH


@pytest.mark.unit
def test_a_resolution_is_recordable_for_a_trace() -> None:
    """What a replay writes down, so a fragile skill is visible before it breaks.

    A skill resolving on its accessible name is working as designed. One
    resolving structurally is one small edit away from finding the wrong thing,
    and its eventual failure should not be a surprise.
    """
    page = (element(1, "link", "Home"), element(2, "link", "Browse"))

    record = resolve(Selector(name="Browse", role="link"), page).to_dict()

    assert record["found"] is True
    assert record["element_id"] == 2
    assert record["element"] == '[2] link "Browse"', (
        "the label, because the number means nothing later"
    )
    assert record["tier"] == TIER_NAME
    assert record["attempts"][0]["tier"] == TIER_NAME
