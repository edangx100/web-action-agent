"""The agent's vocabulary is exactly seven actions, and stays that way.

Everything the agent can do to a web page has to be one of seven typed shapes.
Two failures matter here and they fail differently, so they are tested
separately.

The first is a model that says something we do not support -- an invented
action, a missing parameter, a parameter of the wrong type. That has to be
*rejected*, loudly and with a specific complaint, because the alternative is an
agent that reads ``{"action": "tap", "element": 4}`` and does something close
enough to look right in a demo and wrong in a benchmark.

The second is quieter and is the reason half of this file exists. The
vocabulary is written down twice: once as the models that check a reply, and
once as the JSON Schema sent to the endpoint to constrain what it may generate
in the first place. Those two are consumed by entirely different code paths, so
if they ever disagree nothing crashes -- the endpoint is simply constrained to a
different vocabulary than the one we accept, and the symptom is a run that
mysteriously cannot use one action. The correspondence tests at the end of this
file exist to make that impossible to introduce.

All pure Python: no browser, no endpoint, no key.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from pydantic import ValidationError

from web_action_agent.actions import (
    ACTION_ADAPTER,
    ACTION_NAMES,
    DONE_STATUSES,
    SCROLL_DIRECTIONS,
    ActionError,
    Click,
    Done,
    Fill,
    action_schema,
    parse_action,
)

#: One valid instance of each of the seven actions, by name. Written out in
#: full rather than built from the models, because a fixture derived from the
#: thing under test cannot catch the thing under test being wrong. Optional
#: fields are spelled explicitly, the way a schema-constrained endpoint emits
#: them.
VALID_ACTIONS: dict[str, dict[str, Any]] = {
    "click": {"action": "click", "element_id": 4},
    "fill": {"action": "fill", "element_id": 6, "text": "opening hours"},
    "press": {"action": "press", "key": "Enter"},
    "scroll": {"action": "scroll", "direction": "down"},
    "navigate": {"action": "navigate", "url": "https://example.org/contact"},
    "use_skill": {"action": "use_skill", "skill": "site_search", "params": {"query": "rust"}},
    "done": {"action": "done", "status": "success", "answer": "09:00 to 17:30", "reason": None},
}


def example_for(schema: dict[str, Any], defs: dict[str, Any]) -> Any:
    """Invent the smallest value one schema fragment would accept.

    Used only by the correspondence tests, which need *some* instance of every
    branch of the wire schema in order to feed it back through the parser. It
    understands only the keywords the wire schema is allowed to contain, which
    is the point: a keyword this cannot handle is a keyword that should not
    have reached the wire, and the ``KeyError`` says so.

    Args:
        schema: The fragment to satisfy.
        defs: The document's ``$defs``, for resolving references.

    Returns:
        Any: A value conforming to the fragment.
    """
    if "$ref" in schema:
        return example_for(defs[schema["$ref"].rsplit("/", 1)[-1]], defs)
    if "const" in schema:
        return schema["const"]
    if "enum" in schema:
        return schema["enum"][0]
    if "anyOf" in schema:
        return example_for(schema["anyOf"][0], defs)

    kind = schema["type"]
    if kind == "object":
        properties = schema.get("properties")
        if properties is None:
            # A free-form map, as use_skill's params is. Empty satisfies it.
            return {}
        return {name: example_for(child, defs) for name, child in properties.items()}
    return {"string": "x", "integer": 1, "number": 1.0, "boolean": True, "null": None}[kind]


# --- The seven actions ------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize("name", ACTION_NAMES)
def test_each_of_the_seven_actions_is_accepted(name: str) -> None:
    """A valid instance of every action in the vocabulary parses to a typed value.

    Parametrized by :data:`ACTION_NAMES` rather than by the fixture's own keys,
    so that adding an action to the vocabulary without adding an example here
    fails immediately with a ``KeyError`` instead of leaving the new action
    untested.
    """
    action = parse_action(VALID_ACTIONS[name])

    assert action.action == name
    # Round-tripping is what the trace relies on: an action written to a trace
    # file has to read back as the same action, or the trace is decoration.
    assert parse_action(action.to_dict()) == action


@pytest.mark.unit
def test_the_vocabulary_is_exactly_seven() -> None:
    """The closed list is closed, and this is the test that says so out loud.

    Adding an action type is a deliberate design change, not something that
    should arrive as a side effect. If this fails, the question is not
    "update the test" but "was this meant to happen".
    """
    assert ACTION_NAMES == ("click", "fill", "press", "scroll", "navigate", "use_skill", "done")


@pytest.mark.unit
def test_an_invented_action_is_rejected_and_the_real_ones_named() -> None:
    """An eighth action type does not exist, and the error says which seven do.

    The message is written to be quoted back to the model, so it carries the
    list rather than just saying no -- a model told only that it was wrong will
    often be wrong again the same way.
    """
    with pytest.raises(ActionError) as raised:
        parse_action({"action": "teleport", "url": "https://example.org/"})

    message = str(raised.value)
    assert "teleport" in message
    for name in ACTION_NAMES:
        assert name in message


@pytest.mark.unit
def test_a_missing_parameter_is_rejected_and_named() -> None:
    """A click with nothing to click on is not a click.

    The detail names the field, because "invalid action" tells a model nothing
    it can act on.
    """
    with pytest.raises(ActionError) as raised:
        parse_action({"action": "click"})

    assert any("element_id" in line for line in raised.value.detail)


@pytest.mark.unit
def test_a_wrong_typed_parameter_is_rejected() -> None:
    """An element ID has to be a number, not the word for one.

    Left to coerce itself, ``"four"`` would reach the executor and fail there,
    one component further from the mistake and with a worse message.
    """
    with pytest.raises(ActionError) as raised:
        parse_action({"action": "click", "element_id": "four"})

    assert any("element_id" in line for line in raised.value.detail)


@pytest.mark.unit
def test_a_parameter_the_action_does_not_take_is_rejected() -> None:
    """A click carrying text is a model that thinks it is filling something in.

    Ignoring the extra field would execute a different action from the one
    intended, and the run's trace would record the click as if that had been
    the whole request.
    """
    with pytest.raises(ActionError) as raised:
        parse_action({"action": "click", "element_id": 4, "text": "rust"})

    assert any("text" in line for line in raised.value.detail)


@pytest.mark.unit
@pytest.mark.parametrize(
    "reply",
    [
        {},
        {"element_id": 4},
        {"action": None, "element_id": 4},
    ],
    ids=["empty", "no-action-field", "null-action"],
)
def test_a_reply_with_no_action_name_is_rejected(reply: dict[str, Any]) -> None:
    """Without a name there is nothing to check the rest against.

    These are the shapes a model produces when it answers the question in prose
    and lets the JSON mode wrap whatever was lying around.
    """
    with pytest.raises(ActionError):
        parse_action(reply)


@pytest.mark.unit
@pytest.mark.parametrize("direction", ["sideways", "downwards", "DOWN", ""])
def test_scroll_takes_one_of_two_directions(direction: str) -> None:
    """Near-misses are rejected rather than guessed at.

    ``"downwards"`` is exactly the kind of value a pattern match would happily
    read as ``down``, and exactly the kind of leniency that turns into a wrong
    move somewhere it matters.
    """
    with pytest.raises(ActionError):
        parse_action({"action": "scroll", "direction": direction})

    assert SCROLL_DIRECTIONS == ("up", "down")


@pytest.mark.unit
@pytest.mark.parametrize("element_id", [0, -1])
def test_element_numbers_start_at_one(element_id: int) -> None:
    """Elements are numbered from 1, so 0 is not a low number but a wrong one.

    A zero here usually means a model counting from zero, and it would
    otherwise be reported by the validator as "element 0 does not exist",
    which describes the symptom rather than the mistake.
    """
    with pytest.raises(ActionError):
        parse_action({"action": "click", "element_id": element_id})


@pytest.mark.unit
def test_a_skill_may_take_no_parameters() -> None:
    """Not every skill has parameters, and omitting them is not an error."""
    action = parse_action({"action": "use_skill", "skill": "accept_cookies"})

    assert action.params == {}


@pytest.mark.unit
@pytest.mark.parametrize("status", DONE_STATUSES)
def test_a_run_can_end_either_way(status: str) -> None:
    """Both terminal statuses parse, and there is no third one."""
    action = parse_action({"action": "done", "status": status})

    assert action.status == status
    assert DONE_STATUSES == ("success", "failed")


@pytest.mark.unit
def test_an_action_cannot_be_edited_after_it_is_chosen() -> None:
    """Actions are frozen, like the observations they were decided from.

    A trace whose recorded decisions could be rewritten after the fact is worth
    much less as an account of why a run went the way it did.
    """
    action = Click(action="click", element_id=4)

    with pytest.raises(ValidationError):
        action.element_id = 5  # type: ignore[misc]


# --- What a step log and a trace show ---------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("click", "click 4"),
        ("fill", 'fill 6 "opening hours"'),
        ("press", "press Enter"),
        ("scroll", "scroll down"),
        ("navigate", "navigate https://example.org/contact"),
        ("use_skill", "use_skill site_search(query=rust)"),
        ("done", "done success: 09:00 to 17:30"),
    ],
)
def test_every_action_renders_one_readable_line(name: str, expected: str) -> None:
    """A trace is read by a person, and this is the line they read.

    Pinned exactly, because these lines are also what history shows the planner
    on later steps: if the rendering drifts, the model's account of what it has
    already done drifts with it.
    """
    assert parse_action(VALID_ACTIONS[name]).describe() == expected


@pytest.mark.unit
def test_a_failure_is_described_by_its_reason_not_its_answer() -> None:
    """The interesting half of a ``done`` is whichever half is filled in."""
    failed = Done(action="done", status="failed", reason="no result matched")

    assert failed.describe() == "done failed: no result matched"


@pytest.mark.unit
def test_an_action_serializes_to_plain_json() -> None:
    """Traces are JSON files, so every action has to survive being one."""
    action = Fill(action="fill", element_id=6, text="rust")

    assert json.loads(json.dumps(action.to_dict())) == {
        "action": "fill",
        "element_id": 6,
        "text": "rust",
    }


# --- The schema the endpoint is given ---------------------------------------


@pytest.mark.unit
def test_the_wire_schema_carries_only_keywords_a_decoder_implements() -> None:
    """Guided decoding is not a validator, and rejects what it does not know.

    ``oneOf`` in particular: pydantic emits it for a discriminated union, and
    several decoding engines implement ``anyOf`` alone. A schema carrying one
    fails the *first call of every run* against those endpoints, for a reason
    that has nothing to do with the page -- which is the kind of breakage worth
    a test that does not need an endpoint to catch it.
    """
    rendered = json.dumps(action_schema())

    for unsupported in ["oneOf", "title", "default", "minimum", "discriminator", "description"]:
        assert unsupported not in rendered


@pytest.mark.unit
def test_the_wire_schema_leaves_the_model_no_optional_fields() -> None:
    """A decoder told a field is optional will leave out the answer.

    Every property of every branch is required on the wire. The parser stays
    lenient about omissions -- the previous tests rely on that -- and the
    asymmetry is deliberate: tighten what the model may emit without tightening
    what we accept back from a hand-written retry or an older trace.
    """
    schema = action_schema()

    for name, branch in schema["$defs"].items():
        assert set(branch["required"]) == set(branch["properties"]), name
        assert branch["additionalProperties"] is False, name


@pytest.mark.unit
def test_the_schema_stays_small_enough_to_send_every_step() -> None:
    """It goes on every call of every step, so its size is a running cost.

    The number is a ceiling with room in it, not a target. It exists to catch
    the schema quietly regaining the class docstrings, which took it from
    1.9 KB to 5.6 KB the first time and would have been invisible.
    """
    assert len(json.dumps(action_schema())) < 3000


# --- The two definitions cannot drift apart ---------------------------------


@pytest.mark.unit
def test_the_schema_offers_exactly_the_actions_the_parser_accepts() -> None:
    """The endpoint is constrained to the same seven the parser knows.

    If these disagree nothing raises: the model is simply steered towards an
    action we reject, or away from one we would have accepted, and the symptom
    is a run that inexplicably never scrolls.
    """
    branches = action_schema()["$defs"].values()
    offered = {branch["properties"]["action"]["const"] for branch in branches}

    assert offered == set(ACTION_NAMES)


@pytest.mark.unit
def test_every_branch_of_the_schema_parses_as_the_action_it_describes() -> None:
    """An instance built from the schema alone is accepted by the parser.

    Stronger than comparing the action names: this walks each branch's fields,
    invents the minimum value each one demands, and feeds the result back
    through the parser. A field added to a model but rendered differently on
    the wire -- or a field the wire requires that the parser will not take --
    fails here, without an endpoint to discover it against.
    """
    schema = action_schema()
    defs = schema["$defs"]

    for name, branch in defs.items():
        instance = example_for(branch, defs)
        parsed = parse_action(instance)

        assert type(parsed).__name__ == name
        assert parsed.action == branch["properties"]["action"]["const"]


@pytest.mark.unit
def test_the_union_lists_every_definition() -> None:
    """No branch is defined but left out of the choice offered to the model."""
    schema = action_schema()
    referenced = {branch["$ref"].rsplit("/", 1)[-1] for branch in schema["anyOf"]}

    assert referenced == set(schema["$defs"])


@pytest.mark.unit
def test_the_adapter_and_the_helper_agree() -> None:
    """``parse_action`` is the whole of parsing; nothing bypasses its errors.

    The adapter is exported for the schema, and it would be easy for a caller
    to reach past the helper and get a raw ``ValidationError`` that no retry
    prompt can quote. This pins that the two accept the same things, so the
    helper is a wrapper rather than a second opinion.
    """
    for reply in VALID_ACTIONS.values():
        assert ACTION_ADAPTER.validate_python(reply) == parse_action(reply)
