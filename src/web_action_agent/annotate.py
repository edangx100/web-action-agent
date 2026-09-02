"""Drawing the element numbers onto the screenshot, so the picture can be named.

A vision model looking at a screenshot can see that there is a blue button near
the top right. What it cannot do is *refer* to it: any description it writes —
"the blue button", "the one next to the search field" — has to be matched back
to a real element by something, and that matching is guesswork that fails
silently. The fix, published under the name **Set-of-Marks**, is to stop asking
the model to describe and let it point instead. Every element the agent is
willing to act on gets a small number painted on the screenshot, at the element
it belongs to. The model then says ``click 7``, and 7 is not a description of
anything — it is an index into a list the agent already holds.

**The same numbers, in both channels.** The numbers drawn here are the ones
:mod:`web_action_agent.observer` already assigned to the element list. They are
not re-derived, re-sorted or re-numbered; this module is handed the elements and
paints their existing IDs. That matters more than it sounds: the planner is
shown the list *and* the image, and if the two disagreed about which element is
4, every step would be a coin flip that the trace would record as a reasoning
failure.

**Geometry first, pixels second.** The work splits in two, and the split is
deliberate.

* :func:`plan_marks` decides *where* every number goes. It is pure arithmetic
  over rectangles — no image is involved — so the placement rules can be tested
  by asserting on coordinates rather than by looking at a picture.
* :func:`draw_marks` paints the plan onto an actual image, and is the only part
  that knows about colour.

A test can therefore ask "did element 7 get exactly one label, and is that
label attached to element 7's box?" and get a real answer, which is not
something you can assert about a PNG.

**Two problems worth naming.**

*Contrast.* A dark number on a dark page is not a number, it is noise. Each
badge samples the brightness of the pixels it is about to cover and picks its
colours to oppose them, so a label on a white page and a label on a black hero
image are both readable.

*Collision.* Web pages put controls right next to each other — a row of nav
links, a toolbar of icons — and labels placed naively at the same corner of
each would land on top of one another, leaving a pile of overlapping digits
that names nothing. Placement therefore tries several positions per element and
takes the first that is still clear.

**A viewport picture, not a full-page one.** Element boxes are measured in
viewport coordinates, so the screenshot passed in must be a viewport
screenshot. Hand this a full-page capture of a scrolled page and every number
lands in the wrong place — the coordinates are fine, they just refer to a
different origin than the image does.
"""

from __future__ import annotations

import functools
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont, ImageStat

from .observer import Box, Element

#: Height of the digits, in pixels, on a screenshot captured at the configured
#: viewport size. Small enough not to bury the page under its own labels, large
#: enough to survive whatever downscaling a model applies before it looks.
FONT_SIZE = 15

#: Blank space between the digits and the edge of the badge behind them.
PADDING = 3

#: Thickness of the rectangle drawn around each element. The badge alone says
#: *where* an element is; the outline says how big it is, which is what tells a
#: reader whether the agent thought the button was the icon or the whole row.
OUTLINE_WIDTH = 2

#: Colour of that rectangle. One colour for every element, because the outline
#: answers "what is element 7?" and nothing else — the numbers do the naming.
OUTLINE_COLOR = (255, 64, 0)

#: Badge colours as ``(background, digits)``, for the two cases the contrast
#: check decides between: a label landing on a bright part of the page, and one
#: landing on a dark part. Near-black rather than pure black, and near-white
#: rather than pure white, because the extremes look harsher than they read.
BADGE_ON_LIGHT = ((17, 17, 17), (255, 255, 255))
BADGE_ON_DARK = ((245, 245, 245), (17, 17, 17))

#: Mean brightness, on the 0–255 scale, above which a region counts as light.
#: The midpoint: there is no cleverer number, and a wrong call here costs a
#: slightly less crisp label rather than an unreadable one.
LIGHT_THRESHOLD = 128


@functools.cache
def _font() -> ImageFont.FreeTypeFont:
    """Load the badge font, once per process.

    Pillow ships a font inside the library itself, which is used here rather
    than one found on the machine. A font looked up by name is whatever that
    machine happens to have installed, so the same page would produce
    differently sized labels on a laptop and on a continuous-integration
    runner — and the label sizes decide the placement, so the two would
    disagree about the picture, not just about its looks.

    Returns:
        ImageFont.FreeTypeFont: The bundled font at :data:`FONT_SIZE`.
    """
    return ImageFont.load_default(size=FONT_SIZE)


def _label_size(text: str) -> tuple[int, int]:
    """Measure the badge needed to hold a number.

    Args:
        text: The digits to be drawn, such as ``"7"`` or ``"12"``.

    Returns:
        tuple[int, int]: The badge's ``(width, height)`` in pixels, digits plus
        :data:`PADDING` on every side.
    """
    font = _font()
    left, _, right, _ = font.getbbox(text)
    ascent, descent = font.getmetrics()
    # Height comes from the font's own metrics rather than from this particular
    # string's bounding box, so that "1" and "8" get identically sized badges.
    return (right - left + 2 * PADDING, ascent + descent + 2 * PADDING)


def _clip(box: Box, width: int, height: int) -> Box:
    """Trim a box to the part of it that lies on the image.

    Args:
        box: The element's bounds, in viewport pixels.
        width: Image width in pixels.
        height: Image height in pixels.

    Returns:
        Box: The overlapping part, with a width and height of at least zero.
        A box entirely off the image collapses to an empty box rather than
        raising, because an element half off the bottom edge is an ordinary
        thing for a page to contain and not an error to report.
    """
    left = min(max(box.x, 0), width)
    top = min(max(box.y, 0), height)
    right = min(max(box.x + box.width, 0), width)
    bottom = min(max(box.y + box.height, 0), height)
    return Box(x=left, y=top, width=right - left, height=bottom - top)


def _overlaps(one: Box, other: Box) -> bool:
    """Say whether two rectangles share any pixel.

    Args:
        one: The first rectangle.
        other: The second.

    Returns:
        bool: True if they intersect. Rectangles that merely touch along an
        edge do not count, since neither hides any of the other.
    """
    return (
        one.x < other.x + other.width
        and other.x < one.x + one.width
        and one.y < other.y + other.height
        and other.y < one.y + one.height
    )


@dataclass(frozen=True)
class Mark:
    """One number, and everything needed to draw it and to check it later.

    Attributes:
        element_id: The number painted, which is the element's ID from the
            observation's element list — never a fresh number assigned here.
        outline: The element's own bounds, clipped to the image. This is the
            rectangle drawn around it.
        label: Where the badge goes. May sit outside :attr:`outline` when the
            element is too small or too crowded to hold its own label.
        anchor: The point on the element that the badge is attached to, as
            ``(x, y)``. Always inside :attr:`outline`, and that is the whole
            point of recording it: it is the machine-checkable statement that
            this label belongs to *that* element, which survives the badge
            being nudged elsewhere to avoid a neighbour.
    """

    element_id: int
    outline: Box
    label: Box
    anchor: tuple[int, int]

    def to_dict(self) -> dict[str, Any]:
        """Render this mark as plain data, for the step log.

        Worth recording because grounding failures are diagnosed backwards. The
        planner said ``click 4`` and the wrong thing happened; the first
        question is where 4 was actually drawn, and without this the only way
        to answer it is to open the image and squint.

        Returns:
            dict[str, Any]: JSON-safe values, one key per field.
        """
        return {
            "element_id": self.element_id,
            "outline": self.outline.to_dict(),
            "label": self.label.to_dict(),
            "anchor": list(self.anchor),
        }


def _candidates(box: Box, label: tuple[int, int], width: int, height: int) -> list[Mark]:
    """List the places one element's badge could go, best first.

    The order is the preference order, and it puts the badge *outside* the
    element first. That is not a cosmetic choice: the model reads the button's
    own caption to decide whether to press it, and a number painted over
    ``Search`` leaves it reading ``earch``. Tucking the badge into the corner
    just outside the box keeps the element intact and still reads as belonging
    to it, because the outline drawn around the element meets the badge there.

    The inside positions come next, for the cases where outside is unavailable:
    an element at the very top of the page has nothing above it, and a crowded
    toolbar may have a neighbour's badge in every gap. Covering part of an
    element is the lesser harm; having no number at all is not an option.

    Args:
        box: The element's bounds, already clipped to the image.
        label: The badge's ``(width, height)``.
        width: Image width in pixels.
        height: Image height in pixels.

    Returns:
        list[Mark]: One candidate per position, each already clamped so the
        badge stays on the image. The element ID is filled in by the caller,
        which is the only field a candidate cannot know.
    """
    label_width, label_height = label
    left, top = box.x, box.y
    right, bottom = box.x + box.width, box.y + box.height
    # The rightmost and bottommost pixels still inside the element. Used for
    # anchors rather than `right` and `bottom`, which are one past the edge.
    inner_right = box.x + max(box.width - 1, 0)
    inner_bottom = box.y + max(box.height - 1, 0)

    # Each entry is (badge x, badge y, anchor x, anchor y).
    positions = [
        # Outside the element, so its own text stays readable.
        (left - label_width, top - label_height, left, top),  # Tucked into the top-left corner.
        (left, top - label_height, left, top),  # Directly above, left edges aligned.
        (left - label_width, top, left, top),  # Directly left, top edges aligned.
        (left, bottom, left, inner_bottom),  # Directly below.
        (right, top, inner_right, top),  # Directly right.
        # Inside the element, when there is nowhere clear outside it.
        (left, top, left, top),  # Top-left corner.
        (right - label_width, top, inner_right, top),  # Top-right corner.
        (left, bottom - label_height, left, inner_bottom),  # Bottom-left corner.
        (right - label_width, bottom - label_height, inner_right, inner_bottom),  # Bottom-right.
    ]

    candidates = []
    for badge_x, badge_y, anchor_x, anchor_y in positions:
        placed = Box(
            # Clamp so no badge is drawn half off the image, where the part of
            # the number that carries the meaning may be the part that is cut.
            x=min(max(badge_x, 0), max(width - label_width, 0)),
            y=min(max(badge_y, 0), max(height - label_height, 0)),
            width=label_width,
            height=label_height,
        )
        candidates.append(
            Mark(element_id=0, outline=box, label=placed, anchor=(anchor_x, anchor_y))
        )
    return candidates


def plan_marks(elements: Sequence[Element], *, canvas: tuple[int, int]) -> tuple[Mark, ...]:
    """Decide where every element's number goes, without touching an image.

    The rule is greedy and deliberately simple: take the elements in ID order,
    and give each one the first of its candidate positions that does not
    overlap a badge already placed. If every candidate collides — a genuinely
    dense corner of a page — take the least bad one and accept the overlap.

    That last clause is what makes this function total. Every element gets
    exactly one badge, always, so the picture and the element list can never
    disagree about which numbers exist. The price is that placement is
    first-come-first-served rather than globally best: element 1 takes its
    favourite corner and element 2 works around it, which is not necessarily
    the tidiest arrangement of the two. That trade is worth making, because a
    slightly awkward label is a cosmetic problem and a missing number is a
    grounding failure.

    Args:
        elements: The observation's elements, in ID order.
        canvas: The screenshot's ``(width, height)`` in pixels.

    Returns:
        tuple[Mark, ...]: One mark per element, in the same order. Empty for a
        page offering nothing to act on, which is a legitimate observation.
    """
    width, height = canvas
    marks: list[Mark] = []
    taken: list[Box] = []

    for element in elements:
        box = _clip(element.box, width, height)
        candidates = _candidates(box, _label_size(str(element.id)), width, height)

        # Count how many placed badges each candidate would sit on, and stop at
        # the first that sits on none. `best` keeps the least-bad candidate in
        # case none is clear, so the loop always has an answer to give.
        chosen = candidates[0]
        fewest = len(taken) + 1
        for candidate in candidates:
            clashes = sum(1 for other in taken if _overlaps(candidate.label, other))
            if clashes == 0:
                chosen = candidate
                break
            if clashes < fewest:
                chosen, fewest = candidate, clashes

        marks.append(
            Mark(
                element_id=element.id,
                outline=chosen.outline,
                label=chosen.label,
                anchor=chosen.anchor,
            )
        )
        taken.append(chosen.label)

    return tuple(marks)


def _badge_colors(image: Image.Image, label: Box) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Choose badge colours that stand out from whatever is underneath.

    The average brightness of the pixels the badge will cover decides it: a
    dark badge over a light region, a light badge over a dark one. Averaging is
    enough here because the badge is small and opaque — it does not have to
    read against a busy background, only to be obviously different from it.

    Args:
        image: The unannotated screenshot, in RGB.
        label: Where the badge will be drawn.

    Returns:
        tuple: The badge's ``(background, digits)`` colours.
    """
    region = image.crop(
        (label.x, label.y, label.x + max(label.width, 1), label.y + max(label.height, 1))
    )
    brightness = ImageStat.Stat(region.convert("L")).mean[0]
    return BADGE_ON_LIGHT if brightness >= LIGHT_THRESHOLD else BADGE_ON_DARK


def draw_marks(image: Image.Image, marks: Iterable[Mark]) -> Image.Image:
    """Paint a planned set of marks onto a copy of a screenshot.

    Every colour decision reads from the original screenshot, not from the copy
    being drawn on, so a badge's contrast is judged against the page rather than
    against whatever was painted just before it.

    Args:
        image: The screenshot to annotate. Left untouched.
        marks: The marks from :func:`plan_marks`.

    Returns:
        Image.Image: A new RGB image, the same size as ``image``, with an
        outline and a numbered badge per mark.
    """
    source = image.convert("RGB")
    canvas = source.copy()
    draw = ImageDraw.Draw(canvas)
    font = _font()

    for mark in marks:
        outline = mark.outline
        if outline.width > 0 and outline.height > 0:
            # Pillow treats rectangle coordinates as inclusive, so the far edge
            # is one pixel back from where the box arithmetic ends.
            draw.rectangle(
                (
                    outline.x,
                    outline.y,
                    outline.x + outline.width - 1,
                    outline.y + outline.height - 1,
                ),
                outline=OUTLINE_COLOR,
                width=OUTLINE_WIDTH,
            )

        label = mark.label
        background, ink = _badge_colors(source, label)
        draw.rectangle(
            (label.x, label.y, label.x + label.width - 1, label.y + label.height - 1),
            fill=background,
            outline=ink,
        )
        # "mm" centres the digits on the point, horizontally and vertically,
        # which keeps a one-digit and a three-digit number equally well seated.
        draw.text(label.center, str(mark.element_id), font=font, fill=ink, anchor="mm")

    return canvas


def annotate(
    screenshot: Path | str,
    elements: Sequence[Element],
    destination: Path | str,
) -> tuple[Mark, ...]:
    """Write an annotated copy of a screenshot, numbered to match the elements.

    Args:
        screenshot: The viewport screenshot to annotate, as captured by the
            browser. Not modified.
        elements: The same observation's elements, in ID order.
        destination: Where to write the annotated image. Its parent directories
            are created if missing, and the file format follows the suffix —
            ``.png`` for a run's artifacts, which keeps text sharp where a JPEG
            would smear the digits.

    Returns:
        tuple[Mark, ...]: Where each number was drawn, for the step log.

    Raises:
        FileNotFoundError: If ``screenshot`` does not exist.
        PIL.UnidentifiedImageError: If it is not a readable image.
    """
    with Image.open(screenshot) as image:
        marks = plan_marks(elements, canvas=image.size)
        annotated = draw_marks(image, marks)

    target = Path(destination)
    target.parent.mkdir(parents=True, exist_ok=True)
    annotated.save(target)
    return marks
