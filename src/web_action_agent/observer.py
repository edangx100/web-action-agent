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

#: The attribute the injected script writes onto every element it numbers,
#: holding that element's number. It is how the executor acts on the very node
#: the observer numbered, a moment later and after the page may have moved:
#: ``[data-waa-element-id="4"]`` finds element 4 or finds nothing at all, and
#: finding nothing is the honest report that it went away.
#:
#: Declared here rather than in the script so that the one spelling of the name
#: is shared by whoever writes it and whoever reads it. Prefixed with the
#: project's short name because it is written into somebody else's page, where
#: a generic ``data-id`` could collide with the site's own.
ELEMENT_MARK = "data-waa-element-id"


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

    def to_dict(self) -> dict[str, int]:
        """Render this box as plain data, for a trace file or a log line.

        Returns:
            dict[str, int]: The four numbers, under the same keys
            :meth:`from_dict` reads them back from.
        """
        return {"x": self.x, "y": self.y, "width": self.width, "height": self.height}

    @classmethod
    def from_dict(cls, data: dict[str, int]) -> Box:
        """Rebuild a box from plain data.

        Args:
            data: A mapping with ``x``, ``y``, ``width`` and ``height``.

        Returns:
            Box: The same values, typed.

        Raises:
            KeyError: If any of the four is missing. A box with a missing side
                is not a box, so there is no sensible default to fall back to.
        """
        return cls(
            x=data["x"],
            y=data["y"],
            width=data["width"],
            height=data["height"],
        )


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

    def to_dict(self) -> dict[str, Any]:
        """Render this element as plain data, for a trace file.

        Every field is written, not just the ones a prompt shows. A trace is
        read after the run, by someone asking why a step went the way it did,
        and the box is often the answer -- so it is kept even though the
        planner never sees it.

        Returns:
            dict[str, Any]: JSON-safe values, under the same keys
            :meth:`from_dict` reads them back from.
        """
        return {
            "id": self.id,
            "tag": self.tag,
            "role": self.role,
            "name": self.name,
            "text": self.text,
            "box": self.box.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Element:
        """Rebuild an element from plain data.

        This is also how a freshly extracted element is built, because the
        injected script emits exactly the shape :meth:`to_dict` writes. Keeping
        the two shapes identical is what stops a trace from describing a
        different element than the one the run acted on.

        Args:
            data: One record, either from the injected script or from a trace.

        Returns:
            Element: The same values, typed.

        Raises:
            KeyError: If a field is missing. Every field is required; an
                element with no role or no box cannot be reasoned about, and
                filling one in with a default would hide the real problem,
                which is that whatever produced the record is broken.
        """
        return cls(
            id=data["id"],
            tag=data["tag"],
            role=data["role"],
            name=data["name"],
            text=data["text"],
            box=Box.from_dict(data["box"]),
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

    Each listed element is also marked in the page with its number, under
    :data:`ELEMENT_MARK`. The listing is what the planner reads; the marks are
    what the executor acts through. Marks from a previous call on the same
    document are cleared first, so the marks always describe this listing and
    no earlier one.

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
    records = page.evaluate(_script(), {"textLimit": text_limit, "markAttribute": ELEMENT_MARK})
    return tuple(Element.from_dict(record) for record in records)


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
