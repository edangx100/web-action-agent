"""The complete vocabulary of moves the agent is allowed to make.

Seven actions, and no eighth. Click a numbered element, type into one, press a
key, scroll, go to a URL, replay a library skill, or stop. Everything the agent
can do on a web page is one of those, and adding to the list is a deliberate
change to this file rather than something a model can do by inventing a word.

**Why the list is closed.** The alternative is an agent that reads a model's
prose and guesses -- "it said *tap the login button*, that probably means
click" -- and guessing is exactly where agents do the wrong thing while looking
like they are working. A closed list turns "the model asked for something we do
not support" from a silent misinterpretation into a loud, early rejection with
a name.

**Two audiences, one definition.** The same seven shapes have to reach two very
different readers, and this module serves both from one source:

* *The endpoint*, before the reply exists. :func:`action_schema` renders the
  vocabulary as JSON Schema, which the model client hands to the endpoint so
  that decoding is constrained to a conforming object where the endpoint
  supports it. A malformed action then cannot be produced in the first place.
* *This process*, after the reply arrives. :func:`parse_action` checks the
  object against the same models and returns a typed value. This runs whether
  or not the endpoint constrained anything, because not every route can -- one
  of the shipped vision routes cannot -- and a guarantee that holds only on
  some routes is not a guarantee.

**Strict on the way out, forgiving on the way in.** The wire schema requires
every field of the chosen action, including the optional ones, because a
constrained decoder needs to know exactly what it may emit; the parser accepts
an object that omits an optional field, because a retry prompt written by hand
or a trace recorded by an older version is not wrong for leaving out a null.
The two are deliberately not the same strictness, and :func:`_wire_schema`
explains the seam.

**What is not decided here.** Whether an action is *sensible* -- whether element
4 exists on the page in front of the planner, whether that URL is on the
allowlist, whether that skill is one we have -- is the validator's business.
This module answers only "is this one of the seven, spelled correctly". Both
checks are needed, and running them in one place would make it impossible to
tell a model that cannot follow a schema from a model that is confused about
the page.
"""

from __future__ import annotations

import functools
from collections.abc import Mapping
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError

#: The seven action names, in a fixed order. Used for error
#: messages that tell a model what it could have said instead, and by tests
#: that assert the vocabulary has not quietly grown.
ACTION_NAMES = ("click", "fill", "press", "scroll", "navigate", "use_skill", "done")

#: Scroll takes a direction, not a pixel count. The planner does not know how
#: tall the page is, and a viewport-at-a-time step is the unit that matches how
#: observations are taken: scroll once, observe, see what is there now.
SCROLL_DIRECTIONS = ("up", "down")

#: How a run can end. Anything that is not success is a categorized failure --
#: there is no third state where the agent stops without saying which happened.
DONE_STATUSES = ("success", "failed")

#: The two keywords whose contents are named schemas rather than more schema
#: keywords: ``properties`` maps a field name to its schema, ``$defs`` maps a
#: definition name to its schema. Their keys survive the filter untouched.
NAMED_SCHEMA_MAPS = frozenset({"properties", "$defs"})

#: JSON Schema keywords kept when the vocabulary is rendered for an endpoint.
#: Everything outside this set is dropped by :func:`_wire_schema`; see there for
#: why a whitelist rather than a list of things to remove.
WIRE_KEYWORDS = frozenset(
    {
        "type",
        "properties",
        "required",
        "additionalProperties",
        "items",
        "anyOf",
        "enum",
        "const",
        "$ref",
        "$defs",
    }
)


class ActionError(Exception):
    """A reply was not one of the seven actions, correctly spelled.

    Expected rather than exceptional. A model that returns an invented action
    type or leaves out a required parameter is having an ordinary bad step, and
    the designed response is to hand it back a specific complaint and let it
    try again -- which is why the failure carries the field-level detail rather
    than just a sentence.

    Attributes:
        detail: One line per problem, naming the field and what was wrong with
            it, e.g. ``element_id: field required``. Short enough to quote back
            into a retry prompt, specific enough to be actionable.
    """

    def __init__(self, message: str, *, detail: tuple[str, ...] = ()) -> None:
        """Build the error.

        Args:
            message: The summary line.
            detail: Field-level problems, already rendered.
        """
        super().__init__(message)
        self.detail = detail


class BaseAction(BaseModel):
    """What every action has in common: it is closed, frozen, and printable.

    ``extra="forbid"`` is the half of the closed vocabulary that catches a model
    adding a parameter we do not honour -- a ``click`` carrying a ``text`` field
    is a model that thinks it is filling something in, and letting that through
    would execute a different action than the one intended.

    Frozen for the same reason observations are: a decision that can be edited
    after the fact is worth much less as evidence of why a step went the way it
    did.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    def describe(self) -> str:
        """Render this action as the one line a log or trace records.

        Returns:
            str: A short human-readable form, e.g. ``click 4``. Subclasses
            override it; the base form names the action alone, which is right
            for an action with no parameters and a safe fallback otherwise.
        """
        return str(self.action)  # type: ignore[attr-defined]

    def to_dict(self) -> dict[str, Any]:
        """Render this action as plain data, for the run's trace.

        Returns:
            dict[str, Any]: JSON-safe values, in the same shape
            :func:`parse_action` reads back.
        """
        return self.model_dump(mode="json")


class Click(BaseAction):
    """Click a numbered interactive element.

    Attributes:
        element_id: The element's number *in the observation the planner was
            shown*. Numbers are assigned per observation, so this one means
            nothing anywhere else -- which is why the validator re-checks it
            against the current observation before the browser sees it.
    """

    action: Literal["click"]
    element_id: int = Field(ge=1, description="Number of the element to click.")

    def describe(self) -> str:
        """Returns: str: e.g. ``click 4``."""
        return f"click {self.element_id}"


class Fill(BaseAction):
    """Focus a numbered element and enter text into it.

    Attributes:
        element_id: The field's number in the current observation.
        text: What to type. May be empty, which is how a field is cleared.
    """

    action: Literal["fill"]
    element_id: int = Field(ge=1, description="Number of the field to fill.")
    text: str = Field(description="Text to enter; empty clears the field.")

    def describe(self) -> str:
        """Returns: str: e.g. ``fill 6 "rust"``."""
        return f'fill {self.element_id} "{self.text}"'


class Press(BaseAction):
    """Press a keyboard key in whatever currently has focus.

    Separate from :class:`Fill` because submitting a form is a different intent
    from typing into it, and on many pages Enter is the only way to submit --
    there is no button to click, so no element to name.

    Attributes:
        key: The key name as the browser spells it, e.g. ``Enter``,
            ``Escape``, ``ArrowDown``. Not validated against a list here: the
            set of legal names belongs to the browser driver, and duplicating
            it would give us two lists to keep in step.
    """

    action: Literal["press"]
    key: str = Field(min_length=1, description="Key name, e.g. Enter.")

    def describe(self) -> str:
        """Returns: str: e.g. ``press Enter``."""
        return f"press {self.key}"


class Scroll(BaseAction):
    """Move the viewport up or down.

    Attributes:
        direction: ``up`` or ``down``. A closed pair rather than free text,
            because ``"downwards"`` and ``"scroll down a bit"`` are the kind of
            near-miss that pattern matching turns into a wrong move.
    """

    action: Literal["scroll"]
    direction: Literal["up", "down"] = Field(description="Which way to scroll the viewport.")

    def describe(self) -> str:
        """Returns: str: e.g. ``scroll down``."""
        return f"scroll {self.direction}"


class Navigate(BaseAction):
    """Go to a URL.

    Attributes:
        url: Where to go. Checked here only for being a non-empty string; that
            it is a URL the run is *allowed* to visit is the validator's
            allowlist check, and it is deliberately not duplicated here -- a
            schema that also enforced policy would report a blocked domain as a
            malformed action.
    """

    action: Literal["navigate"]
    url: str = Field(min_length=1, description="Absolute URL to open.")

    def describe(self) -> str:
        """Returns: str: e.g. ``navigate https://example.org/help``."""
        return f"navigate {self.url}"


class UseSkill(BaseAction):
    """Replay a stored skill: a sequence of steps the agent has done before.

    A skill is an action, not a program -- it is available at every step, and
    control comes back to the planner when it finishes. That is why it lives in
    this list alongside ``click`` rather than in a mode of its own.

    Attributes:
        skill: The skill's name in the library. That a skill by this name
            exists, is active, and belongs to this site is the validator's
            check, not this one's.
        params: Values for the skill's parameters, by name. Strings only:
            everything a skill fills into a page is text by the time it gets
            there, and admitting other types would mean deciding here how a
            number is spelled into a form field.
    """

    action: Literal["use_skill"]
    skill: str = Field(min_length=1, description="Name of the skill to replay.")
    params: dict[str, str] = Field(
        default_factory=dict,
        description="Parameter values by name; empty for a skill that takes none.",
    )

    def describe(self) -> str:
        """Returns: str: e.g. ``use_skill search(query=rust)``."""
        arguments = ", ".join(f"{name}={value}" for name, value in sorted(self.params.items()))
        return f"use_skill {self.skill}({arguments})"


class Done(BaseAction):
    """Stop, and say how it went.

    Every run ends with one of these. There is no implicit ending: reaching the
    step cap or running out of retries also produces a ``done``, constructed by
    the controller rather than the planner, so that a trace always has one
    explicit terminal record instead of simply stopping.

    Attributes:
        status: ``success`` or ``failed``.
        answer: The answer, for a task that asked a question. None for a task
            that asked for something to be done rather than found out.
        reason: Why it failed, in a few words. None on success.
    """

    action: Literal["done"]
    status: Literal["success", "failed"] = Field(description="How the run ended.")
    answer: str | None = Field(
        default=None,
        description="The answer, when the task asked for one. Null otherwise.",
    )
    reason: str | None = Field(
        default=None,
        description="Why the run failed. Null on success.",
    )

    def describe(self) -> str:
        """Returns: str: e.g. ``done success: 1,024`` or ``done failed: no results``."""
        note = self.answer if self.status == "success" else self.reason
        return f"done {self.status}: {note}" if note else f"done {self.status}"


#: The action a planner reply must be, as one type. The discriminator makes
#: ``action`` the field that selects which of the seven is meant, so a reply is
#: checked against one model rather than tried against all seven -- which is
#: also what makes the error message name the right fields instead of listing
#: everything that could have been wrong under any reading.
Action = Annotated[
    Click | Fill | Press | Scroll | Navigate | UseSkill | Done,
    Field(discriminator="action"),
]

#: The validator behind :func:`parse_action`. Built once at import: constructing
#: a TypeAdapter compiles the schema, and the planner parses one of these per
#: step for the length of a run.
ACTION_ADAPTER: TypeAdapter[Action] = TypeAdapter(Action)


def parse_action(data: Mapping[str, Any]) -> Action:
    """Turn a decoded model reply into one typed action, or say why not.

    Args:
        data: The object parsed out of the model's reply.

    Returns:
        Action: One of the seven, with its parameters typed and checked.

    Raises:
        ActionError: If the object is not one of the seven -- an unknown action
            name, a missing parameter, a parameter of the wrong type, or an
            extra parameter that action does not take. The message and
            ``detail`` are written to be quoted back to the model.
    """
    try:
        return ACTION_ADAPTER.validate_python(dict(data))
    except ValidationError as error:
        detail = tuple(_render_problem(problem) for problem in error.errors())
        name = data.get("action")
        summary = (
            f"{name!r} is not one of the {len(ACTION_NAMES)} actions: {', '.join(ACTION_NAMES)}"
            if name not in ACTION_NAMES
            else f"{name} action does not conform to its schema"
        )
        raise ActionError(summary, detail=detail) from error


def _render_problem(problem: Mapping[str, Any]) -> str:
    """Render one pydantic error as a line a model can act on.

    Pydantic's own rendering names the union member a field belongs to --
    ``use_skill.params.query`` -- which is accurate and useless to a model that
    is not reading our class names. This keeps the field path and the message
    and drops the rest.

    Args:
        problem: One entry from ``ValidationError.errors()``.

    Returns:
        str: e.g. ``element_id: Input should be a valid integer``, or just the
        message where the problem is with the object as a whole.
    """
    # The first element of `loc` is the discriminator tag ('click', 'done', ...)
    # whenever the union resolved to a member, so it is dropped: the model knows
    # which action it asked for, and repeating it reads as a second field name.
    location = [str(part) for part in problem.get("loc", ())]
    if location and location[0] in ACTION_NAMES:
        location = location[1:]
    message = str(problem.get("msg", "invalid"))
    return f"{'.'.join(location)}: {message}" if location else message


@functools.cache
def action_schema() -> dict[str, Any]:
    """The vocabulary as JSON Schema, ready to constrain an endpoint's decoding.

    Cached because it never changes within a process and is sent on every step
    of every run.

    Returns:
        dict[str, Any]: A schema describing exactly the seven actions, with the
        keywords a decoder honours and nothing else. Handed to
        ``ModelClient.complete(schema=...)``, which forwards it to routes that
        support schema-constrained decoding and omits it on routes that do not.
    """
    return _wire_schema(ACTION_ADAPTER.json_schema())


def _wire_schema(schema: Any) -> Any:
    """Rewrite a pydantic schema into the subset endpoints actually implement.

    Guided decoding is not a JSON Schema validator. Engines implement a subset,
    and they disagree about which one, so a schema that carries keywords beyond
    the common core is rejected by some endpoints outright -- a run failing at
    the first call for a reason that has nothing to do with the page. Three
    rewrites keep it inside that core:

    * ``oneOf`` becomes ``anyOf``. They mean different things to a validator
      (exactly one branch versus at least one) and the same thing to a decoder,
      and ``anyOf`` is the one every engine implements. Nothing is lost: the
      seven branches are distinguished by a constant ``action``, so no object
      can match two of them anyway.
    * Every property is required. A decoder is being told what it may emit, and
      "you may leave this out" is an invitation to leave out the answer. The
      parser stays lenient about omissions -- see the module docstring -- so
      this tightens the model's output without tightening what we accept.
    * Every other keyword is dropped, including ``title``, ``default`` and the
      numeric and string bounds. They are rules about *values*, which pydantic
      enforces on the reply anyway, and the strictest engines reject them.
      ``description`` goes too, which is worth a word: pydantic fills it from
      the docstrings above, and those are written for whoever maintains this
      file. Sending them would put several thousand characters of prose on
      every call for the length of every run. What the *model* needs to know
      about the vocabulary is in the planner prompt, written for that reader,
      and said once.

    A whitelist rather than a blocklist, because a pydantic upgrade that starts
    emitting a new keyword should quietly not reach the wire, rather than reach
    it and break a run against one endpoint out of three.

    Args:
        schema: A schema fragment -- the whole document on the outermost call,
            then each nested object as the walk recurses.

    Returns:
        Any: The same fragment, rewritten. Values that are not schemas (a list
        of required names, a string) come back untouched.
    """
    if isinstance(schema, list):
        return [_wire_schema(item) for item in schema]
    if not isinstance(schema, Mapping):
        return schema

    rewritten: dict[str, Any] = {}
    for keyword, value in schema.items():
        if keyword == "oneOf":
            rewritten["anyOf"] = _wire_schema(value)
        elif keyword in NAMED_SCHEMA_MAPS and isinstance(value, Mapping):
            # The keys here are field and model names, not schema keywords, so
            # they are kept as they are and only the values are rewritten.
            # Filtering them the way keywords are filtered would delete the
            # whole vocabulary, which is precisely what it did once.
            rewritten[keyword] = {name: _wire_schema(item) for name, item in value.items()}
        elif keyword in WIRE_KEYWORDS:
            rewritten[keyword] = _wire_schema(value)

    properties = rewritten.get("properties")
    if isinstance(properties, Mapping):
        rewritten["required"] = list(properties)
        # A model that may add fields we do not read is a model that can
        # smuggle an unsupported parameter past a decoder and into the parser.
        rewritten.setdefault("additionalProperties", False)
    return rewritten
