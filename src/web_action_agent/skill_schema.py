"""What a learned skill is, written down so a person can read one.

A skill is the project's whole claim in one file: the agent did a task once, the
useful part of that run was lifted out, and next time the same job on the same
site is a replay rather than a fresh conversation with a model. This module is
the *shape* of that file, and nothing else -- it does not induce skills, replay
them, or decide when one has stopped working.

**A folder of JSON files, not a database.** That is a spec-level decision and it
is worth stating as one: the point of the library is that somebody can open a
file and see what the agent learned. A row in a database is a thing you need a
tool to look at; a file is a thing you can read, diff, review, delete, and put
in a pull request. The library is small -- tens of skills, not millions -- so
the usual reasons to reach for a database do not apply, and the usual reason to
avoid one does.

**Two of the seven actions may not be a skill step.** A skill is a recorded
route through a site, so its steps come from the action space -- but not all of
it. ``use_skill`` is out because skills do not invoke skills: composition would
turn a flat, readable recording into a call graph nobody can audit, and the
project excludes it by name. ``done`` is out because a skill is an action, not a
program: it hands control back to the planner when it finishes, and a skill that
could end the run would be deciding something only the controller may decide.
What is left -- click, fill, press, scroll, navigate -- is
:data:`SKILL_STEP_ACTIONS`, and a file naming anything else is refused.

**Elements are described, never numbered.** The number the planner used --
``click 4`` -- was assigned for one look at one page and means nothing an hour
later, let alone next week. So a stored step carries a *description* of the
element: what it is called, what it is, what it says, where it sits. Which of
those actually finds the element at replay time, and in what order they are
tried, is a separate problem with its own module; this one says what has to be
written down for that to be possible at all.

**Specifics are lifted out into named parameters.** A skill recorded from a run
that searched for "membership" is not a skill for searching for membership, it
is a skill for searching -- so the text that was typed becomes ``{query}``, and
the file declares ``query`` as a parameter. The rule that makes this trustworthy
is checked here: every placeholder a step uses must be a declared parameter, and
every declared parameter must be used by some step. A skill with a placeholder
nobody declared breaks on replay; a skill declaring a parameter nothing uses is
lying about what it needs.

**No skill is stored without the means of checking it.** The ``verification``
block -- a second, different set of parameter values plus a one-line description
of what should happen -- is required, not optional. A skill induced from one run
and stored on the strength of that one run is a guess; the block is what lets a
candidate be replayed with *different* values before it is allowed into the
library, which is the difference between "this worked once" and "this works".
"""

from __future__ import annotations

import re
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .actions import ACTION_NAMES

#: Actions that may never be a skill step, and why. ``use_skill`` because
#: skills do not invoke skills -- composition would turn a flat, readable
#: recording into a call graph nobody can audit, and the project excludes it by
#: name. ``done`` because a skill is an action rather than a program: it hands
#: control back to the planner, and a skill that could end the run would be
#: deciding something only the controller may decide.
NOT_SKILL_STEPS = ("use_skill", "done")

#: The actions a stored skill step may be: the action space minus the two
#: above. Subtracted from :data:`~web_action_agent.actions.ACTION_NAMES` rather
#: than written out again, so that adding an eighth action to the agent makes
#: it appear here -- and being silently excluded is then impossible, which is
#: the failure this would otherwise have.
SKILL_STEP_ACTIONS = tuple(name for name in ACTION_NAMES if name not in NOT_SKILL_STEPS)

#: How a step refers to one of the skill's parameters: ``{query}``. Chosen
#: because it is the shape every reader already knows from string formatting,
#: and because a name in braces is visibly not a literal value -- a file full of
#: ``{query}`` cannot be mistaken for a file full of somebody's search terms.
PLACEHOLDER = re.compile(r"\{([a-z][a-z0-9_]*)\}")

#: The characters a skill's name may use. Restrictive on purpose: the name is
#: also the filename, so anything a filesystem treats specially -- a slash, a
#: dot, a space -- would make the library's shape depend on which machine it was
#: written on.
SKILL_NAME = re.compile(r"^[a-z][a-z0-9_]*$")


def placeholders_in(text: str) -> tuple[str, ...]:
    """Find every parameter a piece of stored text refers to.

    Args:
        text: A stored step's text or URL, which may contain ``{name}``
            references.

    Returns:
        tuple[str, ...]: The parameter names, in the order they appear, with
        duplicates kept -- a caller counting references wants them all, and a
        caller checking names does not care.
    """
    return tuple(PLACEHOLDER.findall(text))


class Selector(BaseModel):
    """How a stored step finds an element after the numbers are gone.

    Every field here is a *way of describing* the same element, ordered by how
    well each tends to survive a site being rebuilt. None of them is resolution:
    trying them in order, and recording which one worked, is replay's job. What
    this class guarantees is that there is something to try.

    At least one description is required. A selector with nothing in it would be
    a step that can never be replayed, stored as though it could -- which is
    worse than refusing to store the skill, because it fails later and somewhere
    else.

    Attributes:
        name: The element's accessible name -- what a screen reader would call
            it. The most durable description there is, because it is the thing
            a site is least free to change: it is what the page promises to
            people using it.
        role: What the element is, as the page declares it: ``link``,
            ``button``, ``searchbox``. Too broad alone -- a page has many
            buttons -- which is why it is paired with the text.
        text: What the element displayed. Pairs with ``role`` for the elements
            that have no accessible name, which is more of them than anybody
            would like.
        structural: Where the element sits, when nothing else identifies it.
            The last resort, and named as one: it is the description most
            easily broken by a change that means nothing to a person looking at
            the page.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(default="", description="The element's accessible name.")
    role: str = Field(default="", description="What the element is: link, button, searchbox.")
    text: str = Field(default="", description="What the element displayed.")
    structural: str = Field(default="", description="Where it sits; the last resort.")

    @model_validator(mode="after")
    def _describes_something(self) -> Selector:
        """Refuse a selector that describes no element at all.

        Returns:
            Selector: This selector, when it names at least one way to find the
            element.

        Raises:
            ValueError: If every field is empty. Checked at the point the skill
                is written rather than at replay, because a step that can never
                resolve is a fault in the recording and a person can still fix
                the recording.
        """
        if not (self.name or self.role or self.text or self.structural):
            raise ValueError(
                "a selector must describe the element somehow: give an accessible name, "
                "a role and text, or a structural description"
            )
        return self

    def describe(self) -> str:
        """Render this selector as the fragment a log or a listing shows.

        Returns:
            str: e.g. ``link "Browse the catalogue"``, falling back through the
            descriptions it actually has. Built from the same fields replay
            will try, in the same order, so what a person reads matches what
            the machine will do.
        """
        if self.name:
            return f'{self.role} "{self.name}"'.strip()
        if self.text:
            return f'{self.role} showing "{self.text}"'.strip()
        return self.structural or self.role


class BaseStep(BaseModel):
    """What every stored step has, whatever it does.

    Attributes:
        note: One short line saying why this step is here, written for a person
            reading the file. Optional, and worth having: a list of five clicks
            is a recording, and a list of five clicks each saying what it was
            for is an explanation.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    note: str = Field(default="", description="One line on why this step exists.")

    def describe(self) -> str:
        """Render this step as the one line a listing shows.

        Returns:
            str: The action and its target. Subclasses override it; the base
            form names the action alone, which is right for a step with no
            parameters and a safe fallback otherwise.
        """
        return str(self.action)  # type: ignore[attr-defined]

    def parameters_used(self) -> tuple[str, ...]:
        """Name every parameter this step refers to.

        Returns:
            tuple[str, ...]: Parameter names from any ``{name}`` placeholder in
            this step's text. Empty for the steps that carry no text, which is
            most of them -- only what gets typed and where the run goes are ever
            task-specific.
        """
        return ()


class ClickStep(BaseStep):
    """Click the element this step describes.

    Attributes:
        selector: How to find the element.
    """

    action: Literal["click"]
    selector: Selector

    def describe(self) -> str:
        """Returns: str: e.g. ``click link "Browse the catalogue"``."""
        return f"click {self.selector.describe()}"


class FillStep(BaseStep):
    """Type into the element this step describes.

    The step where a skill usually earns its parameters: what was typed during
    the run it was learnt from is exactly the part that must not be kept.

    Attributes:
        selector: How to find the field.
        text: What to type, with ``{name}`` where a parameter goes. A literal
            string is allowed and occasionally right -- a filter that must
            always be set to the same value is not a parameter -- but a skill
            whose every fill is literal has learnt one task rather than a skill.
    """

    action: Literal["fill"]
    selector: Selector
    text: str = Field(description='What to type; "{query}" for a parameter.')

    def describe(self) -> str:
        """Returns: str: e.g. ``fill searchbox "Search terms" with "{query}"``."""
        return f'fill {self.selector.describe()} with "{self.text}"'

    def parameters_used(self) -> tuple[str, ...]:
        """Returns: tuple[str, ...]: Parameters referred to by the typed text."""
        return placeholders_in(self.text)


class PressStep(BaseStep):
    """Press a key in whatever has focus.

    Attributes:
        key: The key name, e.g. ``Enter``. Not parameterised: which key submits
            a form is a fact about the page, not about the task.
    """

    action: Literal["press"]
    key: str = Field(min_length=1, description="The key to press, e.g. Enter.")

    def describe(self) -> str:
        """Returns: str: e.g. ``press Enter``."""
        return f"press {self.key}"


class ScrollStep(BaseStep):
    """Move the viewport.

    Attributes:
        direction: ``up`` or ``down``.
    """

    action: Literal["scroll"]
    direction: Literal["up", "down"] = Field(description="Which way to scroll.")

    def describe(self) -> str:
        """Returns: str: e.g. ``scroll down``."""
        return f"scroll {self.direction}"


class NavigateStep(BaseStep):
    """Go to an address.

    Attributes:
        url: Where to go, with ``{name}`` where a parameter goes. Checked
            against the run's allowlist at replay like any other navigation --
            a stored skill is not a way around the rules a fresh action follows.
    """

    action: Literal["navigate"]
    url: str = Field(min_length=1, description="Where to go; may contain parameters.")

    def describe(self) -> str:
        """Returns: str: e.g. ``navigate https://example.org/search``."""
        return f"navigate {self.url}"

    def parameters_used(self) -> tuple[str, ...]:
        """Returns: tuple[str, ...]: Parameters referred to by the address."""
        return placeholders_in(self.url)


#: One stored step, as one type. The discriminator makes ``action`` the field
#: that selects which of the five is meant, so a step is checked against one
#: model rather than tried against all five -- which is also what makes the
#: error name the right fields instead of listing every field of every step.
SkillStep = Annotated[
    ClickStep | FillStep | PressStep | ScrollStep | NavigateStep,
    Field(discriminator="action"),
]


class Parameter(BaseModel):
    """One value a skill needs supplied before it can replay.

    Attributes:
        name: What steps refer to it as, inside ``{}``.
        description: One line saying what belongs here, for the planner deciding
            whether this skill fits the task in front of it and for the person
            reading the file.
        required: Whether a replay without it is impossible. Optional
            parameters exist for the parts of a route that are genuinely
            optional -- a filter that can be left unset -- and default to empty
            when not supplied.
        example: A value that works, taken from the run the skill was learnt
            from. Not used by anything at replay; it is there because a
            parameter called ``query`` described as "the search term" is still
            ambiguous, and one example settles it.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(pattern=r"^[a-z][a-z0-9_]*$", description="What steps refer to it as.")
    description: str = Field(default="", description="One line on what belongs here.")
    required: bool = Field(default=True, description="Whether a replay needs it.")
    example: str = Field(default="", description="A value that worked.")


class Verification(BaseModel):
    """How to tell whether a candidate skill actually works.

    Required on every skill, which is the point. A skill induced from one run
    and stored on the strength of that run has only been shown to repeat
    itself; replaying it with *different* values is what distinguishes a skill
    from a recording of one afternoon.

    Attributes:
        params: A second set of parameter values -- deliberately not the ones
            the skill was learnt from. Every required parameter must appear
            here, or the check could not be run at all.
        expected_outcome: One line describing what should be true afterwards,
            for the judge to compare the finished page against. Written in
            plain words because that is what a judge reads.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    params: dict[str, str] = Field(description="Alternate values to replay with.")
    expected_outcome: str = Field(
        min_length=1, description="One line on what should be true afterwards."
    )


class SkillStats(BaseModel):
    """What has happened to a skill since it entered the library.

    The evidence behind retirement, and the reason a retired skill is kept
    rather than deleted: the numbers are why it was taken out of service, and
    throwing them away would leave the library unable to say why it is the
    shape it is.

    Attributes:
        uses: How many times the skill has been replayed.
        successes: How many of those replays worked.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    uses: int = Field(default=0, ge=0, description="Replays attempted.")
    successes: int = Field(default=0, ge=0, description="Replays that worked.")

    @model_validator(mode="after")
    def _successes_fit_uses(self) -> SkillStats:
        """Refuse a record claiming more successes than attempts.

        Returns:
            SkillStats: The statistics, when they are arithmetically possible.

        Raises:
            ValueError: If successes exceed uses. Not a situation any code
                path produces, which is exactly why it is worth refusing: a
                file saying so has been edited by hand or written by something
                broken, and a success rate above 100% would quietly keep a
                failing skill in service forever.
        """
        if self.successes > self.uses:
            raise ValueError(f"{self.successes} successes out of {self.uses} uses is impossible")
        return self

    @property
    def success_rate(self) -> float | None:
        """float | None: The share of replays that worked, or None if never used.

        None rather than 0.0 for a skill nobody has replayed yet, for the same
        reason an unpriced model call records None: a skill that has never been
        tried has not failed, and a fresh skill reading as a 0% success rate
        would be retired the moment retirement is switched on.
        """
        return self.successes / self.uses if self.uses else None


class Skill(BaseModel):
    """One learned route through one site, as a file on disk.

    Carries the five attributes the validator's gate asks of a skill --
    ``name``, ``domain``, ``active``, ``required_params``, ``optional_params``
    -- so the library satisfies that gate's structural type without either
    component importing the other.

    Attributes:
        name: What ``use_skill`` calls it, and what its file is called.
            Restricted to lowercase words and underscores so a library is the
            same shape on every machine.
        description: One line on what the skill does, shown to the planner when
            the skill is offered. The planner chooses between skills on this
            sentence alone, so it says what the skill *achieves* rather than
            how.
        domain: The registrable domain the skill was learnt on. A skill is a
            route through one site's pages; replaying it elsewhere is not a
            long shot, it is meaningless.
        params: Every value the skill needs supplied.
        steps: The route, in order.
        verification: How to check the skill works. Required.
        stats: What has happened to it since.
        active: False once retirement has taken it out of service. Retired
            skills stay in the library because their statistics are the
            evidence for why they were retired.
        retired_reason: Why it was retired, for a person reading the library.
            Empty while active.
        retired_at: When, RFC 3339 in UTC. Empty while active.
        induced_from: The id of the run this was learnt from, so a skill can be
            traced back to the evidence it came out of. Empty for a
            hand-written skill.
        created_at: When the skill was written, RFC 3339 in UTC.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(pattern=SKILL_NAME.pattern, description="What use_skill calls it.")
    description: str = Field(min_length=1, description="One line on what it achieves.")
    domain: str = Field(min_length=1, description="The site it was learnt on.")
    params: tuple[Parameter, ...] = Field(default=(), description="Values it needs supplied.")
    steps: tuple[SkillStep, ...] = Field(min_length=1, description="The route, in order.")
    verification: Verification = Field(description="How to check it works.")
    stats: SkillStats = Field(default_factory=SkillStats, description="How it has done.")
    active: bool = Field(default=True, description="False once retired.")
    retired_reason: str = Field(default="", description="Why it was retired.")
    retired_at: str = Field(default="", description="When it was retired.")
    induced_from: str = Field(default="", description="The run it was learnt from.")
    created_at: str = Field(default="", description="When it was written.")

    @property
    def required_params(self) -> tuple[str, ...]:
        """tuple[str, ...]: Names of the parameters a replay cannot go without."""
        return tuple(param.name for param in self.params if param.required)

    @property
    def optional_params(self) -> tuple[str, ...]:
        """tuple[str, ...]: Names of the parameters a replay may leave out."""
        return tuple(param.name for param in self.params if not param.required)

    def parameters_used(self) -> tuple[str, ...]:
        """Name every parameter the steps actually refer to.

        Returns:
            tuple[str, ...]: The distinct names, in the order they first appear
            across the steps.
        """
        seen: dict[str, None] = {}
        for step in self.steps:
            for name in step.parameters_used():
                seen.setdefault(name, None)
        return tuple(seen)

    @model_validator(mode="after")
    def _parameters_and_steps_agree(self) -> Skill:
        """Refuse a skill whose parameters and steps disagree about what it needs.

        Two ways for that to happen, and each breaks the skill differently.

        A step referring to a parameter nobody declared cannot be replayed: at
        replay time there is no value to put there, and the skill fails on a
        page rather than on the way into the library.

        A declared parameter no step uses is a skill lying about what it needs.
        The planner reads the parameter list to decide whether it can supply
        what the skill wants, so a phantom parameter makes a usable skill look
        inapplicable -- and if it is marked required, the gate will refuse
        every invocation that leaves it out.

        Returns:
            Skill: This skill, when its two halves agree.

        Raises:
            ValueError: Naming the parameters that are used but not declared,
                or declared but not used.
        """
        declared = {param.name for param in self.params}
        used = set(self.parameters_used())

        if undeclared := sorted(used - declared):
            raise ValueError(
                f"{self.name}: steps use {', '.join(undeclared)}, which the skill does not declare"
            )
        if unused := sorted(declared - used):
            raise ValueError(f"{self.name}: declares {', '.join(unused)}, which no step uses")
        return self

    @model_validator(mode="after")
    def _duplicate_parameters_are_refused(self) -> Skill:
        """Refuse a skill declaring the same parameter twice.

        Two entries for one name mean two descriptions, two examples, and
        possibly two answers to whether it is required -- and whichever one a
        reader happens to look at is the one they will believe.

        Returns:
            Skill: This skill, when every parameter is declared once.

        Raises:
            ValueError: Naming the parameters declared more than once.
        """
        names = [param.name for param in self.params]
        if repeated := sorted({name for name in names if names.count(name) > 1}):
            raise ValueError(f"{self.name}: declares {', '.join(repeated)} more than once")
        return self

    @model_validator(mode="after")
    def _verification_can_actually_run(self) -> Skill:
        """Refuse a skill whose verification could never be replayed.

        The block is required so that no skill enters the library unchecked, and
        a block missing the values the check would need is the same thing as not
        having one -- except that it looks like having one, which is worse.

        Returns:
            Skill: This skill, when its verification names every required
            parameter and invents none.

        Raises:
            ValueError: Naming the required parameters the block leaves out, or
                the values it supplies for parameters that do not exist.
        """
        supplied = set(self.verification.params)
        if missing := sorted(set(self.required_params) - supplied):
            raise ValueError(
                f"{self.name}: verification has no value for {', '.join(missing)}, "
                "so the check could not be run"
            )
        known = set(self.required_params) | set(self.optional_params)
        if unknown := sorted(supplied - known):
            raise ValueError(
                f"{self.name}: verification supplies {', '.join(unknown)}, "
                "which is not a parameter of this skill"
            )
        return self

    def to_dict(self) -> dict[str, Any]:
        """Render this skill as the plain data its file holds.

        Returns:
            dict[str, Any]: JSON-safe values, in the shape :meth:`from_dict`
            reads back.
        """
        return self.model_dump(mode="json")

    @classmethod
    def from_dict(cls, data: Any) -> Skill:
        """Rebuild a skill from the plain data of its file.

        Args:
            data: One skill, as written by :meth:`to_dict`.

        Returns:
            Skill: The same value the file was written from.

        Raises:
            pydantic.ValidationError: If the data is not a valid skill. Raised
                rather than repaired: a skill file is something a person may
                have edited, and quietly fixing one would mean replaying
                something nobody wrote.
        """
        return cls.model_validate(data)

    def summary(self) -> str:
        """Render this skill as the line a library listing shows.

        Returns:
            str: e.g. ``search_catalogue (example.org) - 7/9 uses - search the
            catalogue and open a record``, with the record reading ``unused``
            where there is none and ``retired`` where it applies.
        """
        rate = self.stats.success_rate
        record = f"{self.stats.successes}/{self.stats.uses} uses" if rate is not None else "unused"
        state = "" if self.active else " - retired"
        return f"{self.name} ({self.domain}) - {record}{state} - {self.description}"
