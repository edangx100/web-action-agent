"""That a step's evidence survives being written down and read back.

An observation is the record of what the agent was looking at when it made a
decision. It is written to the run's trace, and read back later by whoever is
asking why the run went the way it did — possibly long after the site has
changed and the run can no longer be reproduced. So the round trip is not a
convenience here, it is the whole value of keeping a trace at all, and it is
what most of this module asserts.

Two failure modes are worth naming, because neither announces itself.

**A field that stops being written.** Adding a field to a record and forgetting
to serialize it loses data silently: the round trip still passes, the trace
still parses, and the missing value is only noticed by the person who needed
it. So the field lists are checked structurally, against the dataclass itself,
rather than by listing the keys a test author happened to remember.

**A channel that quietly empties.** The element list and the page text are
supposed to be present on every observation, whatever the screenshot does. If
gating the image out ever took a text channel with it, the agent would keep
running and would simply get worse at reading pages, which looks like a bad
model rather than a bug.
"""

import json
from dataclasses import FrozenInstanceError, fields
from pathlib import Path

import pytest

from web_action_agent.observation import Observation
from web_action_agent.observer import Box, Element

SEARCH_BOX = Element(
    id=1,
    tag="input",
    role="searchbox",
    name="Search terms",
    text="rust",
    box=Box(x=40, y=180, width=420, height=36),
)
"""Element: A filled-in search field, standing in for a real extraction.

Built by hand rather than taken from a browser so these tests stay pure
Python. It carries text that differs from its name, which is the case that
distinguishes "what this control is called" from "what is currently in it".
"""

SUBMIT = Element(
    id=2,
    tag="button",
    role="button",
    name="Search",
    text="Search",
    box=Box(x=470, y=180, width=90, height=36),
)
"""Element: A submit button, whose visible text and accessible name agree."""

PAGE_TEXT = "Results for rust. Twelve titles matched your search."
"""str: Stand-in for distilled main-content text — the page's own words."""


def observation(**overrides) -> Observation:
    """Build a representative observation, with fields replaced as needed.

    Args:
        **overrides: Any field of :class:`Observation`, replacing the default.

    Returns:
        Observation: A search-results page carrying both text channels and no
        screenshot, which is the ordinary shape of a step.
    """
    defaults = {
        "url": "http://localhost:8000/results.html",
        "title": "Search results",
        "elements": (SEARCH_BOX, SUBMIT),
        "text": PAGE_TEXT,
    }
    return Observation(**{**defaults, **overrides})


@pytest.mark.unit
def test_round_trip_preserves_the_whole_record():
    """A written-out observation reads back equal to the original."""
    original = observation(text_truncated=True, screenshot=Path("runs/x/step-03.png"))

    restored = Observation.from_dict(original.to_dict())

    assert restored == original


@pytest.mark.unit
def test_round_trip_preserves_element_identity_and_geometry():
    """Elements come back as elements, not as the dicts they were stored as.

    Equality on the whole record would pass even if the elements came back in
    some other form that happened to compare equal, so the restored list is
    inspected directly for the two things every later step needs of it: the
    number the planner uses, and the box a click is aimed at.
    """
    restored = Observation.from_dict(observation().to_dict())

    assert [element.id for element in restored.elements] == [1, 2]
    assert restored.elements[0].box == SEARCH_BOX.box
    assert restored.elements[0].box.center == SEARCH_BOX.box.center


@pytest.mark.unit
def test_serialized_form_is_json():
    """The trace is written as JSON, so the record has to survive being JSON.

    A ``Path`` or a tuple in the wrong place would raise here rather than at
    the end of a long run, which is the point of asserting it directly.
    """
    written = json.dumps(observation(screenshot=Path("runs/x/step-03.png")).to_dict())

    assert Observation.from_dict(json.loads(written)) == observation(
        screenshot=Path("runs/x/step-03.png")
    )


@pytest.mark.unit
def test_screenshot_path_survives_as_a_path():
    """The image reference comes back as a path, not as the string it was stored as."""
    restored = Observation.from_dict(observation(screenshot=Path("runs/x/step-03.png")).to_dict())

    assert restored.screenshot == Path("runs/x/step-03.png")
    assert restored.has_screenshot


@pytest.mark.unit
def test_text_channels_survive_the_screenshot_being_gated_out():
    """With no image, the agent still has the full element list and page text.

    This is the case the whole design turns on: a model that cannot see, or a
    step the cost gate judged not worth an image, must still be given a page it
    can read.
    """
    gated_out = observation(screenshot=None)

    assert not gated_out.has_screenshot
    assert gated_out.to_dict()["screenshot"] is None
    assert len(gated_out.elements) == 2
    assert gated_out.text == PAGE_TEXT
    assert Observation.from_dict(gated_out.to_dict()) == gated_out


@pytest.mark.unit
def test_an_empty_page_is_a_valid_observation():
    """Nothing to click and nothing to read is a real page state, not an error.

    It is also the state that argues for spending money on a screenshot, so it
    has to be representable rather than rejected.
    """
    empty = observation(elements=(), text="")

    assert Observation.from_dict(empty.to_dict()) == empty
    assert "0 elements" in empty.summary()


@pytest.mark.unit
@pytest.mark.parametrize("record_type", [Box, Element, Observation])
def test_every_field_is_written_to_the_trace(record_type):
    """No field of any record may be silently left out of its serialized form.

    Checked against the dataclass definition rather than a hand-written list,
    so that adding a field to a record and forgetting to write it out fails
    here — at the moment it is added — instead of turning up as a hole in a
    trace nobody can fill in afterwards.
    """
    sample = {Box: SEARCH_BOX.box, Element: SEARCH_BOX, Observation: observation()}[record_type]

    assert set(sample.to_dict()) == {field.name for field in fields(record_type)}


@pytest.mark.unit
def test_an_extracted_element_and_a_stored_one_have_the_same_shape():
    """What the browser hands back and what the trace stores are one shape.

    Extraction builds its elements with the same :meth:`Element.from_dict` that
    reads a trace. If the two shapes ever drifted apart, a replayed trace would
    describe different elements than the live run did, and the drift would show
    up as a trace that reads plausibly and is wrong.
    """
    as_stored = SEARCH_BOX.to_dict()

    assert Element.from_dict(as_stored) == SEARCH_BOX
    assert set(as_stored) == {"id", "tag", "role", "name", "text", "box"}


@pytest.mark.unit
def test_missing_required_field_is_refused():
    """A corrupt trace fails loudly rather than reading as a blank page.

    Defaulting a missing element list to empty would produce an observation
    that looks like a page offering nothing — a real state — and so would be
    believed.
    """
    incomplete = observation().to_dict()
    del incomplete["elements"]

    with pytest.raises(KeyError):
        Observation.from_dict(incomplete)


@pytest.mark.unit
def test_optional_fields_default_when_a_trace_omits_them():
    """An older trace without the two capture flags still reads.

    Their defaults are the honest reading of a record that never mentioned
    them: nothing was cut short, and no image was carried.
    """
    minimal = observation().to_dict()
    del minimal["text_truncated"]
    del minimal["screenshot"]

    restored = Observation.from_dict(minimal)

    assert restored.text_truncated is False
    assert restored.screenshot is None


@pytest.mark.unit
def test_element_lookup_finds_the_element_the_planner_named():
    """``click 2`` resolves to the element numbered 2 in this observation."""
    assert observation().element_by_id(2) == SUBMIT


@pytest.mark.unit
def test_element_lookup_returns_nothing_for_a_number_that_is_not_there():
    """An invented or stale number is a miss, not an exception.

    The validator turns the miss into a correctable complaint back to the
    planner, so raising here would convert a routine mistake into a crash.
    """
    assert observation().element_by_id(99) is None


@pytest.mark.unit
def test_summary_reports_every_channel():
    """The one-line step log names the page and the size of all three channels."""
    line = observation(screenshot=Path("runs/x/step-03.png")).summary()

    assert "http://localhost:8000/results.html" in line
    assert '"Search results"' in line
    assert "2 elements" in line
    assert f"{len(PAGE_TEXT)} chars of text" in line
    assert "step-03.png" in line


@pytest.mark.unit
def test_summary_says_a_screenshot_was_not_taken():
    """Absence is stated, so a reader can tell it from a field nobody filled in."""
    assert "no screenshot" in observation().summary()


@pytest.mark.unit
def test_summary_flags_truncated_text():
    """Whether the page was read in full changes how a wrong answer reads."""
    assert "(truncated)" in observation(text_truncated=True).summary()
    assert "(truncated)" not in observation().summary()


@pytest.mark.unit
def test_an_observation_cannot_be_edited_after_the_fact():
    """Evidence is frozen: a changed page means a new observation, not an edit."""
    with pytest.raises(FrozenInstanceError):
        observation().url = "http://localhost:8000/elsewhere.html"
