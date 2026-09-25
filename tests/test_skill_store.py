"""What a learned skill is allowed to be, and what the library does with one.

A skill file is the most interesting thing this project produces: it is what the
agent learned, written down where a person can read it. Two kinds of test
follow from that.

**The format's refusals.** Most of this file is about skills that must *not* be
storable. A format that accepts anything is not a format, and every rule here
exists because breaking it would fail somewhere worse: on a page, mid-replay,
during somebody's benchmark, with an error about an element rather than about
the file. So a step naming an action the agent does not have is refused, a
placeholder nobody declared is refused, and a skill with no way of being checked
is refused before it can reach the library at all.

**The library's promises.** The rest is the folder: that a skill written is the
skill read back, that a half-finished write cannot replace a good file with a
broken one, and that a corrupt file is reported rather than skipped — because a
skill that silently vanishes from a library looks exactly like one that was
never learnt, and the run that should have used it just quietly costs more.

One test here is about a seam rather than a rule: the validator's gate declared
what it needs from a skill library weeks before a library existed, as a
structural type neither component imports. That the store satisfies it is worth
asserting, since nothing else would notice if it stopped.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from web_action_agent.actions import ACTION_NAMES, parse_action
from web_action_agent.allowlist import Allowlist
from web_action_agent.observation import Observation
from web_action_agent.observer import Box, Element
from web_action_agent.skill_schema import (
    NOT_SKILL_STEPS,
    SKILL_STEP_ACTIONS,
    Skill,
    SkillStats,
    placeholders_in,
)
from web_action_agent.skill_store import SkillStore, SkillStoreError
from web_action_agent.validator import SKILL_NOT_FOUND, UNUSABLE_SKILL, Validator

SITE = "example.org"
"""str: The site every skill here was learnt on."""


def a_skill(**overrides: Any) -> dict[str, Any]:
    """Build the plain data of one complete, valid skill.

    A realistic one rather than a minimal one: two parameters, one of them
    optional, and steps that use both. Tests that break exactly one thing about
    it are then breaking it against a skill that would otherwise work.

    Args:
        **overrides: Fields to replace.

    Returns:
        dict[str, Any]: The skill, as its file would hold it.
    """
    skill = {
        "name": "search_catalogue",
        "description": "search the catalogue and open the first matching record",
        "domain": SITE,
        "params": [
            {"name": "query", "description": "what to search for", "example": "membership"},
            {
                "name": "subject",
                "description": "an optional subject filter",
                "required": False,
                "example": "archive",
            },
        ],
        "steps": [
            {
                "action": "fill",
                "selector": {"name": "Search terms", "role": "searchbox"},
                "text": "{query}",
                "note": "the search box on the catalogue page",
            },
            {
                "action": "fill",
                "selector": {"name": "Subject", "role": "combobox"},
                "text": "{subject}",
            },
            {"action": "press", "key": "Enter"},
            {"action": "click", "selector": {"name": "First result", "role": "link"}},
        ],
        "verification": {
            "params": {"query": "reading room", "subject": "privacy"},
            "expected_outcome": "a record page whose subject tags include privacy",
        },
    }
    return {**skill, **overrides}


def a_store(tmp_path: Path) -> SkillStore:
    """Build a library in a temporary folder.

    Args:
        tmp_path: The test's own directory.

    Returns:
        SkillStore: An empty library.
    """
    return SkillStore(tmp_path / "skills")


# --- The format round-trips -------------------------------------------------


@pytest.mark.unit
def test_a_skill_survives_being_written_and_read_back(tmp_path: Path) -> None:
    """The first thing a file format has to do, checked against the real folder.

    Not a round trip in memory: written to a real file, read back through the
    real store, and compared. A format that only round-trips in memory has not
    been through JSON, which is where tuples, paths and enums go wrong.
    """
    store = a_store(tmp_path)
    original = Skill.from_dict(a_skill())

    store.save(original)

    assert store.find("search_catalogue") == original


@pytest.mark.unit
def test_the_file_is_something_a_person_can_read(tmp_path: Path) -> None:
    """The whole reason the library is files rather than a database.

    Indented, in a predictable order, with the parameters and steps spelled out
    — so a skill can be reviewed in a diff and disagreed with by hand.
    """
    store = a_store(tmp_path)

    path = store.save(Skill.from_dict(a_skill()))

    text = path.read_text(encoding="utf-8")
    assert path.name == "search_catalogue.json", "the name is the filename"
    assert "\n" in text and '  "name": "search_catalogue"' in text, "indented, not one long line"
    assert json.loads(text)["steps"][0]["note"] == "the search box on the catalogue page"


@pytest.mark.unit
def test_an_unused_skill_has_no_success_rate_rather_than_a_bad_one() -> None:
    """A fresh skill has not failed; it has not been tried.

    Zero would be a number, and it would be the number that retires every skill
    the moment it enters the library.
    """
    assert SkillStats().success_rate is None
    assert SkillStats(uses=4, successes=3).success_rate == 0.75


# --- Refusals: the steps ----------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize("action", ["evaluate_js", "screenshot", "wait", ""])
def test_a_step_outside_the_action_space_is_refused(action: str) -> None:
    """A skill may only do what the agent can do.

    The agent's vocabulary is closed, and a stored skill is not a way around
    that: a file asking for an action nobody implemented would fail during a
    replay, on a page, with an error about the page rather than about the file.

    Args:
        action: An action name the agent does not have.
    """
    broken = a_skill(steps=[{"action": action, "key": "Enter"}])

    with pytest.raises(ValidationError, match="action"):
        Skill.from_dict(broken)


@pytest.mark.unit
@pytest.mark.parametrize("action", NOT_SKILL_STEPS)
def test_the_two_actions_a_skill_may_not_take_are_refused(action: str) -> None:
    """``use_skill`` and ``done`` are real actions and still not skill steps.

    ``use_skill`` because skills do not invoke skills — composition would turn a
    flat recording into a call graph nobody can audit. ``done`` because a skill
    is an action rather than a program: it hands control back to the planner,
    and a skill able to end the run would be deciding something only the
    controller may decide.

    Args:
        action: One of the two.
    """
    assert action in ACTION_NAMES, "both really are actions the agent has"
    broken = a_skill(steps=[{"action": action, "status": "success"}])

    with pytest.raises(ValidationError, match="action"):
        Skill.from_dict(broken)


@pytest.mark.unit
def test_the_allowed_steps_are_the_action_space_minus_those_two() -> None:
    """Subtracted rather than listed, so an eighth action cannot be forgotten.

    If the agent ever gains an action, it appears here and a decision has to be
    made about it. Writing the list out independently would mean the new action
    was silently excluded from every skill, which is the kind of omission nobody
    goes looking for.
    """
    assert set(SKILL_STEP_ACTIONS) == set(ACTION_NAMES) - set(NOT_SKILL_STEPS)
    assert SKILL_STEP_ACTIONS == ("click", "fill", "press", "scroll", "navigate")


@pytest.mark.unit
def test_a_skill_with_no_steps_is_refused() -> None:
    """A route that goes nowhere. Storable only as a thing that wastes a replay."""
    with pytest.raises(ValidationError, match="steps"):
        Skill.from_dict(a_skill(steps=[]))


@pytest.mark.unit
def test_an_element_nothing_describes_is_refused() -> None:
    """A selector with nothing in it is a step that can never be replayed.

    Caught when the skill is written, because that is when somebody can still
    fix the recording. Caught at replay, it is a failure on a page with no way
    back to what went wrong.
    """
    broken = a_skill(
        steps=[{"action": "click", "selector": {}}],
        params=[],
        verification={"params": {}, "expected_outcome": "something happened"},
    )

    with pytest.raises(ValidationError, match="selector must describe"):
        Skill.from_dict(broken)


# --- Refusals: the parameters -----------------------------------------------


@pytest.mark.unit
def test_a_step_using_a_parameter_nobody_declared_is_refused() -> None:
    """The failure this rule prevents happens on a page, not in a file.

    At replay there would be no value to substitute, so the skill types the
    literal text ``{branch}`` into a search box and reports that it found
    nothing.
    """
    broken = a_skill(
        steps=[
            {"action": "fill", "selector": {"name": "Search terms"}, "text": "{branch}"},
            {"action": "fill", "selector": {"name": "Subject"}, "text": "{subject}"},
            {"action": "press", "key": "Enter"},
        ],
    )

    with pytest.raises(ValidationError, match="branch"):
        Skill.from_dict(broken)


@pytest.mark.unit
def test_a_parameter_no_step_uses_is_refused() -> None:
    """A skill lying about what it needs, which costs it work it could do.

    The planner reads the parameter list to decide whether it can supply what a
    skill wants. A phantom required parameter makes a perfectly usable skill
    look inapplicable, and the gate refuses every invocation that leaves it out.
    """
    broken = a_skill(
        params=[
            {"name": "query", "description": "what to search for"},
            {"name": "subject", "description": "an optional filter", "required": False},
            {"name": "sort_order", "description": "never used by any step"},
        ],
        verification={
            "params": {"query": "reading room", "subject": "privacy", "sort_order": "newest"},
            "expected_outcome": "a record page",
        },
    )

    with pytest.raises(ValidationError, match="sort_order"):
        Skill.from_dict(broken)


@pytest.mark.unit
def test_the_same_parameter_declared_twice_is_refused() -> None:
    """Two descriptions, two examples, and possibly two answers about required.

    Whichever entry a reader happens to look at is the one they will believe.
    """
    broken = a_skill(
        params=[
            {"name": "query", "description": "what to search for"},
            {"name": "query", "description": "no, what to search for", "required": False},
            {"name": "subject", "description": "an optional filter", "required": False},
        ]
    )

    with pytest.raises(ValidationError, match="query"):
        Skill.from_dict(broken)


@pytest.mark.unit
def test_a_parameter_can_be_used_in_an_address_as_well_as_a_field() -> None:
    """Navigation is the other place a run's specifics hide.

    A skill that opens ``/search?q=membership`` has learnt one search, not
    searching.
    """
    skill = Skill.from_dict(
        a_skill(
            params=[{"name": "query", "description": "what to search for"}],
            steps=[{"action": "navigate", "url": "https://example.org/search?q={query}"}],
            verification={
                "params": {"query": "archive"},
                "expected_outcome": "results for archive",
            },
        )
    )

    assert skill.parameters_used() == ("query",)
    assert placeholders_in("https://example.org/search?q={query}&sort={order}") == (
        "query",
        "order",
    )


# --- Refusals: the verification block ---------------------------------------


@pytest.mark.unit
def test_a_skill_with_no_verification_block_cannot_be_built_or_stored(tmp_path: Path) -> None:
    """The rule that separates "this worked once" from "this works".

    A skill induced from one run and stored on the strength of that run has
    only been shown to repeat itself. The block is what lets a candidate be
    replayed with *different* values before it is allowed in — so it is
    required, and a skill without one cannot be constructed at all. Which is
    what makes "nothing unverifiable is in the library" a property of the
    library rather than a check somebody has to remember to run.
    """
    without = a_skill()
    del without["verification"]

    with pytest.raises(ValidationError, match="verification"):
        Skill.from_dict(without)

    # And the same file, hand-written straight into the folder, is refused on
    # the way back out — so there is no way in that skips the check.
    store = a_store(tmp_path)
    store.root.mkdir(parents=True)
    store.path_for("search_catalogue").write_text(json.dumps(without), encoding="utf-8")

    with pytest.raises(SkillStoreError, match="not a valid skill"):
        store.find("search_catalogue")


@pytest.mark.unit
def test_a_verification_that_could_not_be_run_is_refused() -> None:
    """A block missing the values the check needs is not having one.

    Except that it looks like having one, which is worse: the skill would reach
    verification and fail there for a reason that has nothing to do with whether
    the skill works.
    """
    broken = a_skill(verification={"params": {"subject": "privacy"}, "expected_outcome": "a page"})

    with pytest.raises(ValidationError, match="query"):
        Skill.from_dict(broken)


@pytest.mark.unit
def test_a_verification_supplying_an_unknown_parameter_is_refused() -> None:
    """A value for a parameter the skill does not have is a mistake worth catching.

    Either the block was written for a different skill, or a parameter was
    renamed and the block was not. Both are silent at replay — the extra value
    is simply ignored — and both mean the check is not testing what it claims.
    """
    broken = a_skill(
        verification={
            "params": {"query": "reading room", "subject": "privacy", "branch": "Hollow Lane"},
            "expected_outcome": "a record page",
        }
    )

    with pytest.raises(ValidationError, match="branch"):
        Skill.from_dict(broken)


@pytest.mark.unit
def test_an_empty_expected_outcome_is_refused() -> None:
    """The judge is given this line and nothing else to compare the page against."""
    broken = a_skill(
        verification={"params": {"query": "x", "subject": "y"}, "expected_outcome": ""}
    )

    with pytest.raises(ValidationError, match="expected_outcome"):
        Skill.from_dict(broken)


@pytest.mark.unit
def test_more_successes_than_uses_is_refused() -> None:
    """Arithmetic no code path produces, which is exactly why it is worth refusing.

    A file saying so has been edited by hand or written by something broken, and
    a success rate above 100% would keep a failing skill in service forever.
    """
    with pytest.raises(ValidationError, match="impossible"):
        Skill.from_dict(a_skill(stats={"uses": 2, "successes": 3}))


# --- The library ------------------------------------------------------------


@pytest.mark.unit
def test_a_library_nobody_has_written_to_is_empty_rather_than_missing(tmp_path: Path) -> None:
    """What an agent that has learnt nothing looks like, which is the usual state.

    Asking an empty library a question is ordinary, so it answers rather than
    complaining — and does not leave a folder behind for having been asked.
    """
    store = a_store(tmp_path)

    assert len(store) == 0
    assert store.names() == ()
    assert store.load_all() == ()
    assert store.find("anything") is None
    assert not store.root.exists(), "asking is not writing"


@pytest.mark.unit
def test_skills_come_back_in_a_stable_order(tmp_path: Path) -> None:
    """So two machines listing one library agree, and a diff does not reshuffle."""
    store = a_store(tmp_path)
    for name in ("open_record", "search_catalogue", "browse_subjects"):
        store.save(Skill.from_dict(a_skill(name=name)))

    assert store.names() == ("browse_subjects", "open_record", "search_catalogue")
    assert [skill.name for skill in store] == list(store.names())


@pytest.mark.unit
def test_saving_a_skill_twice_replaces_it(tmp_path: Path) -> None:
    """The inducer rewrites the library; a second pass must not double it up."""
    store = a_store(tmp_path)
    store.save(Skill.from_dict(a_skill()))

    store.save(Skill.from_dict(a_skill(description="a better description")))

    assert len(store) == 1
    found = store.find("search_catalogue")
    assert found is not None and found.description == "a better description"


@pytest.mark.unit
def test_a_retired_skill_is_still_in_the_library(tmp_path: Path) -> None:
    """Retired is not deleted, and the library is not the thing that decides.

    A skill that stopped working because a site changed is evidence about that
    site. Its statistics are why it was taken out of service, so they are kept —
    and what to do about a retired skill is the gate's judgment, not storage's.
    """
    store = a_store(tmp_path)
    retired = Skill.from_dict(
        a_skill(
            active=False,
            retired_reason="success rate fell to 40% over 10 uses",
            retired_at="2026-09-25T09:00:00.000Z",
            stats={"uses": 10, "successes": 4},
        )
    )

    store.save(retired)

    found = store.find("search_catalogue")
    assert found is not None
    assert found.active is False
    assert found.retired_reason.startswith("success rate")
    assert "retired" in found.summary()


@pytest.mark.unit
def test_deleting_is_separate_from_retiring_and_forgiving_of_absence(tmp_path: Path) -> None:
    """Deleting is for a skill that should never have been written at all."""
    store = a_store(tmp_path)
    store.save(Skill.from_dict(a_skill()))

    assert store.delete("search_catalogue") is True
    assert store.find("search_catalogue") is None
    assert store.delete("search_catalogue") is False, "already gone is not an error"


# --- The library's failures -------------------------------------------------


@pytest.mark.unit
def test_a_corrupt_skill_file_is_reported_rather_than_skipped(tmp_path: Path) -> None:
    """Skipping is the dangerous option, and it is dangerous invisibly.

    A skill that quietly vanishes from a library looks exactly like one that was
    never induced. The run that should have used it simply thinks its way
    through from scratch, costs more, and nothing anywhere says why.
    """
    store = a_store(tmp_path)
    store.save(Skill.from_dict(a_skill()))
    store.path_for("search_catalogue").write_text("{not json", encoding="utf-8")

    with pytest.raises(SkillStoreError, match="not valid JSON"):
        store.find("search_catalogue")
    with pytest.raises(SkillStoreError, match="search_catalogue.json"):
        store.load_all()


@pytest.mark.unit
def test_a_file_whose_name_is_not_the_skills_name_is_refused(tmp_path: Path) -> None:
    """The filename is how a skill is found, so the two cannot disagree.

    A file called ``old_search.json`` holding a skill called ``search_catalogue``
    is unreachable by the name it answers to, and would be listed under a name
    it does not.
    """
    store = a_store(tmp_path)
    store.root.mkdir(parents=True)
    store.path_for("old_search").write_text(json.dumps(a_skill()), encoding="utf-8")

    with pytest.raises(SkillStoreError, match="must be its name"):
        store.find("old_search")


@pytest.mark.unit
def test_a_failed_write_leaves_the_previous_skill_intact(tmp_path: Path) -> None:
    """The inducer rewrites the library; a crash mid-pass must not corrupt it.

    Writing in place would leave a file that is valid JSON right up to the point
    the writer stopped — the worst possible outcome, because it parses. A
    temporary file moved into place gives either the old skill or the new one.
    """
    store = a_store(tmp_path)
    store.save(Skill.from_dict(a_skill(description="the original")))

    # The folder is made read-only, so the write of the replacement fails after
    # the original is already on disk.
    store.root.chmod(0o500)
    try:
        with pytest.raises(SkillStoreError, match="cannot write"):
            store.save(Skill.from_dict(a_skill(description="the replacement")))
    finally:
        store.root.chmod(0o700)

    survivor = store.find("search_catalogue")
    assert survivor is not None
    assert survivor.description == "the original", "the good file is still the good file"
    assert store.names() == ("search_catalogue",), "and no half-written leftovers"


# --- The seam the gate was built against ------------------------------------


@pytest.mark.unit
def test_the_store_answers_the_questions_the_validator_asks(tmp_path: Path) -> None:
    """The gate declared what it needed from a library before one existed.

    It did that as a structural type — five attributes and one method — so
    neither component has to import the other. Nothing would notice if the store
    drifted out of that shape until a ``use_skill`` reached the gate in a real
    run, so it is asserted here by putting a real store behind a real validator.
    """
    store = a_store(tmp_path)
    store.save(Skill.from_dict(a_skill()))
    validator = Validator(Allowlist.derive(f"https://{SITE}/search"), skills=store)
    page = Observation(
        url=f"https://{SITE}/search",
        title="Search",
        elements=(
            Element(
                id=1,
                tag="input",
                role="searchbox",
                name="Search terms",
                text="",
                box=Box(x=0, y=0, width=10, height=10),
            ),
        ),
        text="Search the catalogue.",
    )

    approved = validator.check(
        parse_action(
            {
                "action": "use_skill",
                "skill": "search_catalogue",
                "params": {"query": "membership"},
            }
        ),
        page,
    )
    unknown = validator.check(
        parse_action({"action": "use_skill", "skill": "no_such_skill", "params": {}}),
        page,
    )

    assert approved.action is not None, approved.rejection
    assert unknown.rejection is not None
    assert unknown.rejection.code == UNUSABLE_SKILL
    assert unknown.rejection.cause == SKILL_NOT_FOUND
