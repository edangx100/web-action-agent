"""What the agent can see and act on, at one moment, on one screenful.

The planner is a language model. It cannot point at a page, so the page has to
be turned into something it can name — and the name has to be short, because
every element listed is tokens spent on every step for the rest of the run.
This module is the first half of that translation: the interactive elements of
the current viewport, numbered.

**Numbered, and only for now.** The elements are numbered from 1 in document
order, so the planner says ``click 4`` rather than describing a button in prose
and hoping the description resolves to one element. The numbers are assigned
fresh for every observation and are meaningless outside it: scroll the page and
the same button is a different number. That is not a flaw to be engineered
away — a page changes under the agent, and a number that survived across
changes would be a promise the page cannot keep. What survives instead is the
element's role and its accessible name, which is what history records so that
"clicked 4" still means something later.

**Viewport-only.** Only what is on screen is listed. The alternative — listing
the whole document — would offer the planner elements it cannot see in the
screenshot and cannot click without scrolling first, and would make the element
list and the annotated screenshot describe different pages. Scrolling is
therefore an ordinary action with an ordinary purpose: it changes what can be
seen, and the next observation says so.

**Where the work happens.** The filtering runs inside the page, in
``extract_elements.js``, because every question it answers is one only the
browser can answer: what is actually visible, what is covered by the consent
dialog, what a screen reader would call this. None of that is in the HTML
source — it is in the layout the browser just computed. This module is the
Python side of that script: it injects it, and turns the plain data that comes
back into :class:`Element` values the rest of the agent can hold.

**Not traversed:** iframes and shadow roots, out of scope for v0. An element
inside either is not listed, so the planner cannot refer to it.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from playwright.sync_api import Page

#: The script injected into the page. It lives beside this module rather than
#: inside it as a string so that it stays readable JavaScript — with its own
#: comments, and syntax highlighting in an editor.
SCRIPT_PATH = Path(__file__).resolve().parent / "extract_elements.js"

#: Longest accessible name or visible text kept for one element, in characters.
#: Some pages carry link text a paragraph long; a hundred or so characters is
#: enough to tell two elements apart, which is all the planner needs of them.
TEXT_LIMIT = 120


@functools.cache
def _script() -> str:
    """Read the injected script, once per process.

    Returns:
        str: The contents of :data:`SCRIPT_PATH`.

    Raises:
        FileNotFoundError: If the script is missing, which means the package
            was built without its data files rather than that anything went
            wrong at run time.
    """
    return SCRIPT_PATH.read_text(encoding="utf-8")


@dataclass(frozen=True)
class Box:
    """Where an element is on screen, in CSS pixels from the viewport's corner.

    The box is clipped to the viewport, so it describes the visible part of a
    half-scrolled element rather than the whole of it. That is the part the
    screenshot shows and the part a click can land in.

    Attributes:
        x: Distance from the left edge of the viewport.
        y: Distance from the top edge of the viewport.
        width: Width of the visible part.
        height: Height of the visible part.
    """

    x: int
    y: int
    width: int
    height: int

    @property
    def center(self) -> tuple[int, int]:
        """tuple[int, int]: The middle of the box, as ``(x, y)``.

        Where a numeric mark is drawn for the screenshot overlay, and where a
        click aimed at the element by coordinate would go.
        """
        return (self.x + self.width // 2, self.y + self.height // 2)


@dataclass(frozen=True)
class Element:
    """One thing on the page the agent could act on, as of one observation.

    Attributes:
        id: The number the planner uses to refer to this element, from 1, in
            document order. Valid only within the observation it came from.
        tag: The HTML tag name, lowercased.
        role: The ARIA role the element presents — ``link``, ``button``,
            ``searchbox`` — taken from an explicit ``role`` attribute where
            there is one and from the tag otherwise. Recorded alongside the
            name in history, because together they identify what was acted on
            after the numbering has moved on.
        name: The accessible name: what a screen reader would call it, and what
            the planner reasons about. Empty for the rare element that has
            none, which is shown as such rather than hidden.
        text: What the element currently displays — the caption of a button,
            the contents of a field. Distinct from the name, and the difference
            is how the agent tells whether its last ``type`` took effect. Empty
            for a password field, whose value is never read.
        box: The visible bounds, in viewport coordinates.
    """

    id: int
    tag: str
    role: str
    name: str
    text: str
    box: Box

    def describe(self) -> str:
        """Render this element as the single line the planner is shown.

        Returns:
            str: The number, the role, and the name — ``[3] link "Home"`` — with
            the element's current contents appended when they differ from the
            name, as ``[6] searchbox "Search terms" = "rust"``. Rendering lives
            here so that the prompt, the trace, and a debugging print cannot
            drift into describing the same element three ways.
        """
        line = f'[{self.id}] {self.role} "{self.name}"'
        if self.text and self.text != self.name:
            line += f' = "{self.text}"'
        return line


def _element_from(record: dict[str, Any]) -> Element:
    """Convert one record from the injected script into an :class:`Element`.

    Args:
        record: One entry of the script's return value.

    Returns:
        Element: The same values, typed.
    """
    box = record["box"]
    return Element(
        id=record["id"],
        tag=record["tag"],
        role=record["role"],
        name=record["name"],
        text=record["text"],
        box=Box(x=box["x"], y=box["y"], width=box["width"], height=box["height"]),
    )


def extract_elements(page: Page, *, text_limit: int = TEXT_LIMIT) -> tuple[Element, ...]:
    """List the interactive elements visible in the page's current viewport.

    The listing is a snapshot. It describes the page as it was laid out at the
    moment of the call, and any scroll, click, or script that runs afterwards
    can invalidate it — which is why the validator re-checks, against the
    current observation, that an element the planner named still exists.

    Args:
        page: The page to observe. Its viewport size is read from the browser,
            so the elements listed are the ones inside whatever viewport the
            run was configured with.
        text_limit: Longest accessible name or visible text to keep, in
            characters. Longer values are truncated with an ellipsis.

    Returns:
        tuple[Element, ...]: The visible, enabled, unobstructed interactive
        elements, numbered from 1 in document order. Empty for a page that
        genuinely offers nothing to act on — which is a legitimate observation,
        and one of the two signals that gate a screenshot in.

    Raises:
        playwright.sync_api.Error: If the page navigates or closes while the
            script is running. The controller categorizes that as a failed
            step and observes again; it is not this function's to recover.
    """
    records = page.evaluate(_script(), {"textLimit": text_limit})
    return tuple(_element_from(record) for record in records)


def describe_elements(elements: tuple[Element, ...]) -> str:
    """Render a whole element list as the block the planner is shown.

    Args:
        elements: The elements from one observation.

    Returns:
        str: One line per element, or a sentence saying there are none — never
        an empty string, which in a prompt reads as a missing section rather
        than as an empty page.
    """
    if not elements:
        return "(no interactive elements are visible in the viewport)"
    return "\n".join(element.describe() for element in elements)
