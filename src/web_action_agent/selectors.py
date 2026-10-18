"""Finding the same element again, on a page that has changed since.

A run says ``click 4``. The number came from one look at one page and is
rebuilt from scratch on the next look, so it is meaningless by the time a
stored skill wants to replay that click -- possibly minutes later, probably
weeks later, quite likely after somebody has redesigned the site. This module
is what replaces the number: a *description* of the element, and the ordered
attempt to find something matching it again.

**Three descriptions, tried in order of how well each survives.**

1. *The accessible name.* What a screen reader would announce -- the "Search
   the catalogue" of a link. The most durable thing on a page, because it is
   the one part a site is least free to change: it is what the page promises to
   people who cannot see it. A redesign moves boxes and renames classes; it very
   rarely renames the button.

2. *The role plus the visible text.* For the many elements that have no
   accessible name at all, which is more of them than anybody would like.
   Weaker, because the text is styling-adjacent and gets edited, but it still
   describes the element in terms of what a person sees.

3. *Where it sits* -- the third link on the page. Genuinely last-resort, and
   named as such: it breaks on a change that means nothing to a reader, like
   adding a link above it. It is here because something has to be, and because
   a weak description that *says* it is weak is better than no description.

**A tier that is unsure does not guess.** The obvious way to write this is "find
the matches, take the first one". That is the single worst thing this module
could do. Two links called "Next" mean the first tier does not know which was
meant, and picking either is a wrong click that will be recorded in the trace as
a success. So an ambiguous tier resolves nothing, says why, and hands the
problem to the next tier -- which is often narrower and settles it. Only a
description matching exactly one element counts as having found it.

**Failing is a result, not an exception.** When every tier has been tried and
nothing matched, the answer is a :class:`Resolution` that says so and carries
what each tier tried. A site has been redesigned and the skill no longer fits
it: that is ordinary, it is the reason skills get retired, and it needs to be
reportable rather than thrown.

**Counting may overrule a name only when the name was the task's own.** A
skill that searched for "membership" stored the result it clicked as
"Membership application"; replayed for "privacy", the result in that place is
rightly called something else, and finding it by position is the point. But
a name with nothing task-specific in it -- a menu link called "Tax residency
and tax rates" -- does not change from one task to the next. If the element
found by counting is called something else, that is not a weak match but a
contradiction: the page has changed around the position, and the element
there is a different one. It is refused, and says what it found instead.
(Seen in a live replay: the IRAS menu link not yet on screen, and "the 23rd
link" was LOGIN.) Which names were the task's own is told by the caller, as
the values the skill was recorded with.

**Resolution answers with a number, not a click.** What comes back is one of the
elements from the current observation, which the caller turns into an ordinary
``click 4`` against *this* look at the page. So a replayed skill goes through
the same validator and the same executor as anything the planner proposes. A
stored skill is not a way around the rules a fresh action follows, and the way
to keep that true is to give it no separate path to the browser.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from .observer import Element
from .skill_schema import Selector

#: Matched on the element's accessible name. The first tier tried, and the one
#: that resolves almost every step on a page that has not been rebuilt.
TIER_NAME = "accessible_name"

#: Matched on the element's role together with its visible text. For elements
#: with no accessible name, and for pages where the name has been edited but
#: the wording a person reads has not.
TIER_ROLE_TEXT = "role_and_text"

#: Matched on position: the nth element of a role, in document order. The last
#: resort, and the one that a change nobody would notice can break.
TIER_STRUCTURAL = "structural"

#: The tiers, in the order they are tried. Ordered most durable first, which is
#: not the same as cheapest first -- all three cost nothing, so there is no
#: reason to try a weaker description before a stronger one.
TIERS = (TIER_NAME, TIER_ROLE_TEXT, TIER_STRUCTURAL)

#: This tier found exactly one element, and that is the answer.
FOUND = "found"

#: The stored selector has nothing for this tier, so there was nothing to try.
#: Not a failure: a selector is allowed to describe an element only some of the
#: ways, and usually does.
NOT_DESCRIBED = "not_described"

#: The tier had a description and nothing on the page matched it. The ordinary
#: shape of a site having changed.
NO_MATCH = "no_match"

#: The position tier found an element, but the element's own name or text says
#: it is not the one described -- the "LOGIN" link standing where "Tax
#: residency and tax rates" used to be. Refused rather than clicked: a wrong
#: click is recorded as a success, which is the mistake this module exists to
#: avoid.
CONTRADICTED = "contradicted"

#: The tier had a description and several elements matched it. Recorded
#: distinctly from :data:`NO_MATCH` because it means something different and
#: needs a different fix: the description is not specific enough, rather than
#: wrong.
AMBIGUOUS = "ambiguous"

#: How a structural description is written: a role, then the element's position
#: among the elements of that role, counting from 1 in document order. Short
#: enough to read in a stored skill -- ``link[3]`` is "the third link".
STRUCTURAL = re.compile(r"^(?P<role>[a-z-]+)\[(?P<index>\d+)\]$")

#: Collapses the whitespace inside an accessible name before comparing. A page
#: rebuilt with different indentation produces the same name with different
#: spacing, and that is not a change anybody meant to make.
WHITESPACE = re.compile(r"\s+")


def normalise(text: str) -> str:
    """Put a name or a piece of text into the form comparisons use.

    Two liberties are taken, and both are deliberate. Whitespace is collapsed,
    because a page reindented by a build tool yields the same words with
    different spacing. Case is ignored, because ``Next Page`` and ``Next page``
    are the same button to everybody except a string comparison.

    Nothing else is normalised. Punctuation, accents and wording are left
    exactly as they are: a site that changed "Search" to "Search catalogue"
    changed the button, and a match there would be a guess.

    Args:
        text: A stored description or a live element's name.

    Returns:
        str: The comparable form.
    """
    return WHITESPACE.sub(" ", text).strip().casefold()


@dataclass(frozen=True)
class Attempt:
    """What one tier tried, and how it went.

    Kept for every tier, including the ones that were never going to work.
    A resolution that fell through to the structural tier is a warning sign --
    the site has drifted -- and a record showing *which* description stopped
    matching is the difference between "this skill is getting fragile" and
    "this skill broke one day".

    Attributes:
        tier: Which description was tried, one of :data:`TIERS`.
        outcome: :data:`FOUND`, :data:`NOT_DESCRIBED`, :data:`NO_MATCH` or
            :data:`AMBIGUOUS`.
        matches: How many elements the description matched. Zero for a tier
            with nothing to try, and the interesting number when the outcome is
            ambiguous.
        detail: The same thing in a sentence, for a person reading a trace.
    """

    tier: str
    outcome: str
    matches: int = 0
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Render this attempt as plain data, for a run's trace.

        Returns:
            dict[str, Any]: Every field, all JSON-safe already.
        """
        return {
            "tier": self.tier,
            "outcome": self.outcome,
            "matches": self.matches,
            "detail": self.detail,
        }


@dataclass(frozen=True)
class Resolution:
    """The answer to "which element on this page is the stored one?".

    Attributes:
        element: The element found, or None when nothing matched. An element
            from the observation that was searched, so the caller can use its
            number to build an ordinary action.
        tier: Which description found it, one of :data:`TIERS`. Empty when
            nothing did.
        attempts: What every tier tried, in order, including the ones after the
            one that succeeded were never reached -- those are simply absent,
            because a tier that was not tried has nothing to report.
    """

    element: Element | None = None
    tier: str = ""
    attempts: tuple[Attempt, ...] = ()

    @property
    def found(self) -> bool:
        """bool: Whether exactly one element matched some description."""
        return self.element is not None

    @property
    def fell_back(self) -> bool:
        """bool: Whether something weaker than the accessible name found it.

        The signal worth watching. A skill resolving on its accessible name is
        working as designed; one resolving structurally is one small edit away
        from finding the wrong thing, and its next failure is not a surprise.
        """
        return self.found and self.tier != TIER_NAME

    def summary(self) -> str:
        """Render this resolution as the line a replay log shows.

        Returns:
            str: e.g. ``resolved [3] link "Next page" by role_and_text (the
            accessible name matched nothing)``, or, when nothing matched, what
            each tier tried.
        """
        if self.element is None:
            tried = "; ".join(attempt.detail for attempt in self.attempts if attempt.detail)
            return f"no element matched: {tried}" if tried else "no element matched"
        line = f"resolved {self.element.label()} by {self.tier}"
        # Only the failures before the winning tier are worth naming: they are
        # why a weaker description had to be used.
        earlier = [a.detail for a in self.attempts if a.outcome not in (FOUND, NOT_DESCRIBED)]
        return f"{line} ({'; '.join(earlier)})" if earlier else line

    def to_dict(self) -> dict[str, Any]:
        """Render this resolution as plain data, for a run's trace.

        Returns:
            dict[str, Any]: The element's number and label, the tier that found
            it, and every attempt. The label as well as the number because the
            number means nothing when the trace is read later, which is the
            whole reason this module exists.
        """
        return {
            "found": self.found,
            "element_id": self.element.id if self.element else None,
            "element": self.element.label() if self.element else "",
            "tier": self.tier,
            "attempts": [attempt.to_dict() for attempt in self.attempts],
        }


def resolve(
    selector: Selector, elements: tuple[Element, ...], *, task_values: tuple[str, ...] = ()
) -> Resolution:
    """Find the element a stored selector describes, in a fresh look at a page.

    Each description is tried in :data:`TIERS` order and the first that matches
    *exactly one* element wins. A tier matching several does not pick one: it
    records the ambiguity and lets a narrower description settle it, because a
    confident wrong click is recorded in a trace as a success and is the hardest
    kind of mistake to find afterwards.

    Args:
        selector: The stored description, from a skill file.
        elements: The elements of the current observation, in document order.
            Numbered for *this* look at the page, which is why the element
            itself is returned rather than any stored identifier.
        task_values: The values the skill was recorded with -- its
            parameters' examples, such as ``membership``. A stored name
            containing one was specific to that task, so counting may find an
            element called something else; a name containing none may not be
            overruled that way.

    Returns:
        Resolution: The element and the tier that found it, or a resolution
        saying nothing matched -- never an exception. A site redesigned out from
        under a skill is ordinary, and it is the reason skills get retired.
    """
    attempts: list[Attempt] = []

    for tier in TIERS:
        matches = _matches_for(tier, selector, elements)
        if matches is None:
            attempts.append(
                Attempt(tier, NOT_DESCRIBED, detail=f"{tier}: the skill does not describe it")
            )
            continue
        contradiction = (
            _contradicts(tier, selector, matches[0], task_values) if len(matches) == 1 else ""
        )
        if contradiction:
            attempts.append(Attempt(tier, CONTRADICTED, matches=1, detail=contradiction))
            continue
        if len(matches) == 1:
            attempts.append(Attempt(tier, FOUND, matches=1, detail=f"{tier}: matched"))
            return Resolution(element=matches[0], tier=tier, attempts=tuple(attempts))
        if not matches:
            attempts.append(Attempt(tier, NO_MATCH, detail=f"{tier} matched nothing"))
        else:
            attempts.append(
                Attempt(
                    tier,
                    AMBIGUOUS,
                    matches=len(matches),
                    detail=f"{tier} matched {len(matches)} elements, so it settled nothing",
                )
            )

    return Resolution(attempts=tuple(attempts))


def selector_for(element: Element, elements: tuple[Element, ...]) -> Selector:
    """Describe one element every way a stored skill can, for later.

    The other half of this module, used when a skill is being written rather
    than replayed: it turns "the element the run clicked" into the descriptions
    that will be tried when the skill is replayed. All three tiers are filled in
    where the element supports them, because which of them will still work is
    exactly the thing nobody can know in advance.

    Args:
        element: The element to describe.
        elements: Every element of the observation it came from, needed to work
            out its position for the structural description.

    Returns:
        Selector: The element described by name, by role and text, and by
        position.
    """
    return Selector(
        name=element.name,
        role=element.role,
        text=element.text,
        structural=structural_path(element, elements),
    )


def structural_path(element: Element, elements: tuple[Element, ...]) -> str:
    """Say where an element sits among the elements of its own kind.

    Counted among elements of the same role rather than among all of them, so
    that a button added to a page of links does not move every link's
    description. That makes the weakest tier slightly less weak; it does not
    make it strong, and nothing here pretends otherwise.

    Args:
        element: The element to place.
        elements: Every element of its observation, in document order.

    Returns:
        str: e.g. ``link[3]`` -- the third link. Empty when the element is not
        in the list it is being placed within, which would make any position
        written for it a fiction.
    """
    kind = element.role or element.tag
    position = 0
    for candidate in elements:
        if (candidate.role or candidate.tag) == kind:
            position += 1
            if candidate.id == element.id:
                return f"{kind}[{position}]"
    return ""


def _matches_for(
    tier: str, selector: Selector, elements: tuple[Element, ...]
) -> list[Element] | None:
    """Find every element one tier's description matches.

    Args:
        tier: Which description to use.
        selector: The stored descriptions.
        elements: The current observation's elements.

    Returns:
        list[Element] | None: The matches, or None when this selector says
        nothing for this tier. None rather than an empty list because the two
        mean different things -- "nothing to try" is not "tried and found
        nothing" -- and a trace that confused them would report a site as
        changed when the skill was simply never described that way.
    """
    if tier == TIER_NAME:
        return _by_name(selector, elements) if selector.name else None
    if tier == TIER_ROLE_TEXT:
        # Both halves are needed. A role alone matches every button on the
        # page, and text alone is what the name tier already covers better.
        if not (selector.role and selector.text):
            return None
        return [
            element
            for element in elements
            if normalise(element.role) == normalise(selector.role)
            and normalise(element.text) == normalise(selector.text)
        ]
    return _by_position(selector, elements) if selector.structural else None


def _contradicts(
    tier: str, selector: Selector, element: Element, task_values: tuple[str, ...]
) -> str:
    """Say why an element found by counting is not the one described, if it is not.

    Only the position tier can contradict itself: the other two match on the
    name or text, so whatever they find agrees with it by definition.

    Args:
        tier: The tier that found the element.
        selector: The stored descriptions.
        element: The one element the tier found.
        task_values: The values the skill was recorded with.

    Returns:
        str: Empty when there is no contradiction -- another tier, a stored
        description with no name or text to compare, an element with none of
        its own (an icon), or a stored name that held one of the task's own
        values and so was expected to differ. Otherwise one sentence naming
        both, for the trace.
    """
    if tier != TIER_STRUCTURAL:
        return ""
    wanted = {normalise(part) for part in (selector.name, selector.text) if part}
    found = {normalise(part) for part in (element.name, element.text) if part}
    if not wanted or not found or wanted & found:
        return ""
    values = [normalise(value) for value in task_values if value.strip()]
    if any(value in name for value in values for name in wanted):
        return ""
    said = element.name or element.text
    meant = selector.name or selector.text
    return (
        f"{tier}: the element at {selector.structural} is called {said!r}, not {meant!r}, "
        "so it is a different element"
    )


def _by_name(selector: Selector, elements: tuple[Element, ...]) -> list[Element]:
    """Match on the accessible name, using the role only to break a tie.

    The name is the durable signal, so it is what the tier matches on. The role
    is brought in only when the name alone leaves several candidates -- a link
    called "Search" and a button called "Search" are different things, and
    saying which was meant costs nothing here. Narrowing is skipped when it
    would leave nothing, so that a site which changed an element's role still
    resolves on its name.

    Args:
        selector: The stored descriptions.
        elements: The current observation's elements.

    Returns:
        list[Element]: The matches, after the tiebreak.
    """
    named = [element for element in elements if normalise(element.name) == normalise(selector.name)]
    if len(named) <= 1 or not selector.role:
        return named
    same_role = [
        element for element in named if normalise(element.role) == normalise(selector.role)
    ]
    return same_role or named


def _by_position(selector: Selector, elements: tuple[Element, ...]) -> list[Element]:
    """Match on a structural description like ``link[3]``.

    Args:
        selector: The stored descriptions.
        elements: The current observation's elements.

    Returns:
        list[Element]: The element at that position, or nothing -- either
        because the description is not one this module writes, or because the
        page no longer has that many elements of that kind. A malformed
        description matching nothing is right: it was not written here, so
        guessing at what it meant would be inventing a click.
    """
    match = STRUCTURAL.match(selector.structural.strip())
    if match is None:
        return []
    kind = match["role"]
    wanted = int(match["index"])
    position = 0
    for element in elements:
        if (element.role or element.tag) == kind:
            position += 1
            if position == wanted:
                return [element]
    return []
