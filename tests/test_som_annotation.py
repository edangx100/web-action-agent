"""Whether the numbers on the picture mean the same thing as the numbers in the list.

The annotated screenshot exists for one reason: to let a model that can see the
page name a part of it. That only works if a promise holds — the number drawn
over a button is the number the element list gives that button — and the
promise is invisible when it breaks. A label drawn over the wrong element, or
drawn twice, or quietly dropped, produces an image that looks perfectly
reasonable. What follows is a run where the agent clicks the wrong thing and a
trace in which nothing looks wrong.

So the properties are asserted on the *geometry*, not on the picture. Every
element has a number; no number appears twice; every label is anchored inside
the element it names. Those are checkable statements about coordinates, and
they are exactly the statements a person eyeballing a screenshot would be
guessing at.

The unit tests below run on made-up rectangles, because the awkward cases —
elements stacked on each other, an element hanging off the edge of the image, a
control smaller than its own label — are easier to construct than to find. The
fixture tests then run the same properties against a real browser screenshot of
a real page, which is what catches a mistake in the translation between the two.
"""

from dataclasses import fields

import pytest
from PIL import Image

from web_action_agent.annotate import Mark, annotate, draw_marks, plan_marks
from web_action_agent.observer import Box, Element, extract_elements

CANVAS = (1280, 800)
"""tuple[int, int]: Image size the unit tests plan against, matching the
viewport every browser test uses."""


def element(element_id, x, y, width, height):
    """Build an element that exists only as a rectangle.

    The fields the annotator never reads are filled with placeholders, which
    keeps each test's setup down to the four numbers it actually cares about.

    Args:
        element_id: The number to be drawn.
        x: Left edge, in viewport pixels.
        y: Top edge, in viewport pixels.
        width: Width in pixels.
        height: Height in pixels.

    Returns:
        Element: An element at those bounds.
    """
    return Element(
        id=element_id,
        tag="button",
        role="button",
        name=f"element {element_id}",
        text="",
        box=Box(x=x, y=y, width=width, height=height),
    )


def contains(box, point):
    """Say whether a point lies within a rectangle, edges included.

    Args:
        box: The rectangle.
        point: An ``(x, y)`` pair.

    Returns:
        bool: True if the point is inside or on the boundary.
    """
    px, py = point
    return box.x <= px <= box.x + box.width and box.y <= py <= box.y + box.height


def boxes_overlap(one, other):
    """Say whether two rectangles share a pixel.

    Written out here rather than imported from the module under test, so that
    the same mistake in both would not cancel itself out.

    Args:
        one: The first rectangle.
        other: The second.

    Returns:
        bool: True if they intersect.
    """
    return (
        one.x < other.x + other.width
        and other.x < one.x + one.width
        and one.y < other.y + other.height
        and other.y < one.y + one.height
    )


def overlapping_pairs(marks):
    """Find every pair of labels that share a pixel.

    Args:
        marks: The marks from one plan.

    Returns:
        list[tuple[int, int]]: The IDs of each colliding pair, for a failure
        message that names the elements rather than their coordinates.
    """
    return [
        (one.element_id, other.element_id)
        for index, one in enumerate(marks)
        for other in marks[index + 1 :]
        if boxes_overlap(one.label, other.label)
    ]


@pytest.mark.unit
def test_every_element_gets_exactly_one_number():
    """Assert the drawn numbers are the element list's numbers, all of them.

    This is the property the whole module exists to provide. A missing number
    is an element the vision model cannot refer to; a repeated one is a number
    that refers to two different things.
    """
    elements = [element(index, 40 * index, 60 * index, 120, 40) for index in range(1, 8)]

    marks = plan_marks(elements, canvas=CANVAS)

    drawn = [mark.element_id for mark in marks]
    assert drawn == [e.id for e in elements]
    assert len(set(drawn)) == len(drawn)


@pytest.mark.unit
def test_every_label_is_anchored_inside_its_element():
    """Assert each label is tied to a point on the element it names.

    A badge may be nudged off its element to dodge a neighbour, so the badge's
    own position cannot carry the association. The anchor does, and it stays
    inside the element no matter where the badge ends up.
    """
    elements = [element(1, 10, 10, 200, 50), element(2, 10, 12, 200, 50), element(3, 4, 4, 8, 8)]

    marks = plan_marks(elements, canvas=CANVAS)

    for mark, source in zip(marks, elements, strict=True):
        assert contains(source.box, mark.anchor), f"element {source.id} anchor escaped its box"


@pytest.mark.unit
def test_a_label_keeps_off_the_element_it_names():
    """Assert an uncrowded element's own text is not painted over.

    The model reads a button's caption to decide whether to press it, so a
    badge sitting on ``Search`` costs the very thing the screenshot was sent
    for. Where there is room, the badge goes just outside the corner instead.
    """
    source = element(1, 400, 300, 120, 40)

    mark = plan_marks([source], canvas=CANVAS)[0]

    assert not boxes_overlap(mark.label, source.box)


@pytest.mark.unit
def test_labels_of_stacked_elements_are_moved_apart():
    """Assert crowding is resolved rather than ignored.

    Three elements at nearly the same place is what a toolbar of icon buttons
    looks like to this module. Placed naively they would all take the same
    corner and the labels would become one unreadable smudge.
    """
    elements = [element(index, 100, 100 + index, 24, 24) for index in range(1, 4)]

    marks = plan_marks(elements, canvas=CANVAS)

    assert overlapping_pairs(marks) == []


@pytest.mark.unit
def test_labels_stay_on_the_image():
    """Assert no badge is drawn half off the edge.

    Half a number is not a smaller number, it is a different one — a clipped
    ``18`` can read as ``1``. Elements in the page's four corners are the case,
    and every one of them is an ordinary place to find a control.
    """
    width, height = CANVAS
    elements = [
        element(1, 0, 0, 30, 20),
        element(2, width - 30, 0, 30, 20),
        element(3, 0, height - 20, 30, 20),
        element(4, width - 30, height - 20, 30, 20),
    ]

    marks = plan_marks(elements, canvas=CANVAS)

    for mark in marks:
        label = mark.label
        assert label.x >= 0 and label.y >= 0
        assert label.x + label.width <= width
        assert label.y + label.height <= height


@pytest.mark.unit
def test_an_element_hanging_off_the_edge_still_gets_a_number():
    """Assert a partly visible element is numbered, not skipped.

    Elements are clipped to the viewport, so one at the bottom of the screen
    arrives here as a box that runs past the image. It is still clickable, so
    it still needs a number.
    """
    marks = plan_marks([element(1, 1200, 780, 200, 60)], canvas=CANVAS)

    assert [mark.element_id for mark in marks] == [1]
    assert marks[0].outline.width > 0
    assert marks[0].outline.height > 0


@pytest.mark.unit
def test_a_page_with_nothing_to_click_is_annotated_with_nothing():
    """Assert an empty element list is a legitimate input, not an error.

    A page offering nothing to act on is a real observation — and one of the
    signals that a screenshot might be worth sending in the first place.
    """
    assert plan_marks([], canvas=CANVAS) == ()


@pytest.mark.unit
def test_planning_the_same_elements_twice_gives_the_same_placement():
    """Assert placement is deterministic.

    Two runs over the same page must produce the same picture, or a trace can
    never be compared against a re-run of the step it recorded.
    """
    elements = [element(index, 20 * index, 20 * index, 60, 30) for index in range(1, 12)]

    assert plan_marks(elements, canvas=CANVAS) == plan_marks(elements, canvas=CANVAS)


@pytest.mark.unit
def test_a_badge_on_a_light_page_is_drawn_dark():
    """Assert contrast is chosen from what the badge covers, on a white page."""
    marks = plan_marks([element(1, 50, 50, 200, 60)], canvas=(400, 300))
    page = Image.new("RGB", (400, 300), (255, 255, 255))

    annotated = draw_marks(page, marks)

    label = marks[0].label
    red, green, blue = annotated.getpixel((label.x + 1, label.y + 1))
    assert red + green + blue < 3 * 128, "a light page should get a dark badge"


@pytest.mark.unit
def test_a_badge_on_a_dark_page_is_drawn_light():
    """Assert the same decision goes the other way on a dark page.

    Without this pair, a badge colour hard-coded for one kind of page would
    pass every geometric test in this file and be invisible on half the web.
    """
    marks = plan_marks([element(1, 50, 50, 200, 60)], canvas=(400, 300))
    page = Image.new("RGB", (400, 300), (10, 10, 10))

    annotated = draw_marks(page, marks)

    label = marks[0].label
    red, green, blue = annotated.getpixel((label.x + 1, label.y + 1))
    assert red + green + blue > 3 * 128, "a dark page should get a light badge"


@pytest.mark.unit
def test_drawing_leaves_the_original_screenshot_alone():
    """Assert annotation copies rather than paints over its input.

    The unannotated screenshot is what the contrast check reads and what a
    person may want to compare against; overwriting it in place would lose both.
    """
    page = Image.new("RGB", (400, 300), (255, 255, 255))
    marks = plan_marks([element(1, 50, 50, 200, 60)], canvas=(400, 300))

    annotated = draw_marks(page, marks)

    assert page.getpixel((51, 51)) == (255, 255, 255)
    assert annotated.getpixel((51, 51)) != (255, 255, 255)


@pytest.mark.unit
def test_a_mark_records_every_field_it_holds():
    """Assert the step log's mark record cannot silently lose a field.

    The keys are derived from the dataclass rather than listed here, so adding
    a field without extending :meth:`Mark.to_dict` fails this test instead of
    quietly writing incomplete records from then on.
    """
    mark = plan_marks([element(1, 10, 20, 100, 40)], canvas=CANVAS)[0]

    assert set(mark.to_dict()) == {field.name for field in fields(Mark)}


@pytest.mark.fixture
def test_every_element_of_a_real_page_is_numbered_exactly_once(page, fixture_site, tmp_path):
    """Assert the picture and the element list agree, on a browser screenshot.

    The unit tests above prove the placement rules; this proves they are being
    applied to the elements the observer actually found, at the coordinates the
    browser actually reported.

    Args:
        page: A blank page at the fixed viewport.
        fixture_site: The running static site.
        tmp_path: A directory for this test's images.
    """
    page.goto(fixture_site.url_for("search.html"))
    shot = tmp_path / "step.png"
    page.screenshot(path=str(shot))
    elements = extract_elements(page)

    marks = annotate(shot, elements, tmp_path / "step-annotated.png")

    assert [mark.element_id for mark in marks] == [e.id for e in elements]
    for mark, source in zip(marks, elements, strict=True):
        assert contains(source.box, mark.anchor), f"element {source.id} anchor escaped its box"


@pytest.mark.fixture
def test_labels_on_a_real_page_do_not_cover_each_other(page, fixture_site, tmp_path):
    """Assert a genuine page's labels come out separated.

    Args:
        page: A blank page at the fixed viewport.
        fixture_site: The running static site.
        tmp_path: A directory for this test's images.
    """
    page.goto(fixture_site.url_for("browse.html"))
    shot = tmp_path / "step.png"
    page.screenshot(path=str(shot))

    marks = annotate(shot, extract_elements(page), tmp_path / "step-annotated.png")

    assert overlapping_pairs(marks) == []


@pytest.mark.fixture
def test_the_annotated_image_is_written_beside_the_plain_one(page, fixture_site, tmp_path):
    """Assert a same-sized, visibly different file lands where it was asked to.

    Same size because the model is shown this instead of the plain screenshot,
    and different because an annotation that changed no pixels would pass every
    geometric assertion in this file.

    Args:
        page: A blank page at the fixed viewport.
        fixture_site: The running static site.
        tmp_path: A directory for this test's images.
    """
    page.goto(fixture_site.url_for("search.html"))
    shot = tmp_path / "step.png"
    page.screenshot(path=str(shot))
    destination = tmp_path / "annotated" / "step.png"

    annotate(shot, extract_elements(page), destination)

    with Image.open(shot) as plain, Image.open(destination) as annotated:
        assert annotated.size == plain.size
        assert annotated.convert("RGB").tobytes() != plain.convert("RGB").tobytes()


@pytest.mark.fixture
def test_annotating_the_same_screenshot_twice_gives_the_same_file(page, fixture_site, tmp_path):
    """Assert the whole path from page to PNG is reproducible.

    Args:
        page: A blank page at the fixed viewport.
        fixture_site: The running static site.
        tmp_path: A directory for this test's images.
    """
    page.goto(fixture_site.url_for("search.html"))
    shot = tmp_path / "step.png"
    page.screenshot(path=str(shot))
    elements = extract_elements(page)

    first = annotate(shot, elements, tmp_path / "first.png")
    second = annotate(shot, elements, tmp_path / "second.png")

    assert first == second
    assert (tmp_path / "first.png").read_bytes() == (tmp_path / "second.png").read_bytes()
