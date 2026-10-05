"""Counting what each skill achieves, and dropping the ones that stop working.

Websites change. A recipe learned in March and verified in March will one day
stop finding the button it was built around, and nobody is going to be watching
when that happens. This is how the library notices by itself.

The tests are about where the line sits, because both sides of it are mistakes
with real costs.

**Retiring too eagerly** throws away a working recipe because of a bad run —
a site that was briefly slow, a page that happened to load oddly. The recipe is
gone, the saving goes with it, and the only way back is for the agent to work
the whole route out again and be lucky enough to have that run induced from.

**Retiring too reluctantly** leaves a recipe in service that fails most of the
time. Every run that picks it pays an AI call to choose it, wastes a replay, and
then does the job by hand anyway — slower than having no library at all.

So the rule has two numbers: a success rate to clear, and a minimum number of
attempts before the rate means anything. The second exists entirely because a
brand-new recipe that fails once is sitting at 0%.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pytest

from web_action_agent.config import Config
from web_action_agent.retrieval import for_domain
from web_action_agent.skill_schema import Skill, SkillStats
from web_action_agent.skill_store import RetirementPolicy, SkillStore, SkillStoreError

SITE = "library.org"
"""str: The site every skill here belongs to."""

RFC_3339 = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d+)?Z$")
"""re.Pattern[str]: The shape a recorded moment has to have."""


def a_skill(name: str = "search_catalogue", **overrides: Any) -> Skill:
    """Build one verified skill.

    Args:
        name: What it is called.
        **overrides: Fields to replace, for the tests that start from a record
            already part-way through its life.

    Returns:
        Skill: The skill.
    """
    skill = {
        "name": name,
        "description": "search the catalogue and open the matching record",
        "domain": SITE,
        "params": [{"name": "query", "description": "what to search for"}],
        "steps": [
            {"action": "fill", "selector": {"name": "Search terms"}, "text": "{query}"},
            {"action": "press", "key": "Enter"},
        ],
        "verification": {"params": {"query": "privacy"}, "expected_outcome": "a record page"},
    }
    return Skill.from_dict({**skill, **overrides})


def a_library(tmp_path: Path, *, min_sample: int = 3, rate: float = 0.70) -> SkillStore:
    """Build a library curated by a stated rule.

    The defaults here are not the shipped ones. A minimum sample of three keeps
    the tests short enough to read; the shipped five is a judgment about real
    sites, and a separate test checks the store picks it up from the run's
    settings rather than from this file.

    Args:
        tmp_path: The test's own directory.
        min_sample: Replays before retirement may trigger.
        rate: The success rate a skill must keep.

    Returns:
        SkillStore: An empty library with that rule.
    """
    return SkillStore(
        tmp_path / "skills", policy=RetirementPolicy(rate=rate, min_sample=min_sample)
    )


def record(store: SkillStore, name: str, outcomes: str) -> Skill:
    """Replay a skill several times, reading the outcomes off a string.

    Args:
        store: The library.
        name: The skill replayed.
        outcomes: One character per replay -- ``.`` for a success, ``x`` for a
            failure. Written this way so a test can show a whole history at a
            glance: ``"..x"`` is two that worked and one that did not.

    Returns:
        Skill: The skill as it stands after the last replay.
    """
    skill = store.find(name)
    assert skill is not None, f"no skill named {name!r}"
    for outcome in outcomes:
        skill = store.record_use(name, succeeded=outcome == ".")
    return skill


# --- Where the line sits ----------------------------------------------------


@pytest.mark.unit
def test_a_skill_failing_more_often_than_the_bar_allows_is_retired(tmp_path: Path) -> None:
    """Two of three replays worked, the bar is seven in ten, so it goes.

    And the reason carries the numbers. "Retired for a low success rate" is
    not something anybody can check months later; "2 of 3 replays worked (67%)"
    is.
    """
    store = a_library(tmp_path)
    store.save(a_skill())

    retired = record(store, "search_catalogue", "..x")

    assert retired.active is False
    assert retired.stats == SkillStats(uses=3, successes=2)
    assert "2 of 3 replays worked" in retired.retired_reason
    assert "67%" in retired.retired_reason and "70%" in retired.retired_reason


@pytest.mark.unit
def test_a_skill_that_keeps_working_stays(tmp_path: Path) -> None:
    """Three of three, over the same sample that retired the one above."""
    store = a_library(tmp_path)
    store.save(a_skill())

    kept = record(store, "search_catalogue", "...")

    assert kept.active is True
    assert kept.stats == SkillStats(uses=3, successes=3)
    assert kept.retired_reason == "" and kept.retired_at == ""


@pytest.mark.unit
def test_one_unlucky_replay_cannot_retire_a_brand_new_skill(tmp_path: Path) -> None:
    """The entire reason the minimum sample exists.

    A recipe that fails its very first replay is sitting at 0%, which is below
    every imaginable bar. Without a floor on the sample, one slow page would
    throw away a recipe that works -- and the only way back is for the agent to
    work the whole route out again and be lucky enough to have that run learnt
    from.
    """
    store = a_library(tmp_path, min_sample=3)
    store.save(a_skill())

    after_one = record(store, "search_catalogue", "x")

    assert after_one.stats.success_rate == 0.0, "as bad a rate as there is"
    assert after_one.active is True, "and still in service, because one replay proves nothing"


@pytest.mark.unit
def test_exactly_meeting_the_bar_is_not_falling_below_it(tmp_path: Path) -> None:
    """The boundary, pinned because it is the one somebody changes by accident.

    Seven in ten is the bar. A skill at seven in ten has met it.
    """
    store = a_library(tmp_path, min_sample=5, rate=0.70)
    store.save(a_skill())

    assert record(store, "search_catalogue", ".......xxx").active is True, "7/10 is not below 70%"

    store.save(a_skill("other"))
    assert record(store, "other", "......xxxx").active is False, "6/10 is"


@pytest.mark.unit
def test_the_bar_comes_from_the_run_settings_not_from_this_code(tmp_path: Path) -> None:
    """The right numbers depend on how flaky the sites are, so they are settings.

    An experiment should be able to move them without a code change, which is
    only true if the store reads them rather than holding its own.
    """
    patient = SkillStore.from_config(
        Config(skill_retirement_rate=0.40, skill_retirement_min_sample=2), tmp_path / "patient"
    )

    assert patient.policy == RetirementPolicy(rate=0.40, min_sample=2)
    assert SkillStore(tmp_path / "default").policy == RetirementPolicy(), "the shipped rule"


# --- What a retired skill looks like afterwards -----------------------------


@pytest.mark.unit
def test_a_retired_skill_is_off_the_menu_but_still_on_disk(tmp_path: Path) -> None:
    """Both halves matter, and for different reasons.

    Out of the listing, because offering a recipe that fails most of the time
    costs a call to choose it and a wasted replay. Still on disk, because its
    record is the evidence for why it went -- and a library that deleted its
    failures could never answer "has this site changed, or did we learn it
    badly?".
    """
    store = a_library(tmp_path)
    store.save(a_skill())
    store.save(a_skill("browse_shelf"))

    record(store, "search_catalogue", "..x")

    offered = for_domain(store.load_all(), SITE)
    assert [skill.name for skill in offered.skills] == ["browse_shelf"]
    assert offered.held == 2, "the library still holds both"

    on_disk = json.loads(store.path_for("search_catalogue").read_text(encoding="utf-8"))
    assert on_disk["active"] is False
    assert on_disk["retired_reason"], "and says why"
    assert RFC_3339.match(on_disk["retired_at"]), "and when, in the form everything else uses"
    assert on_disk["stats"] == {"uses": 3, "successes": 2}, "with the record that decided it"


@pytest.mark.unit
def test_retirement_does_not_reverse_itself(tmp_path: Path) -> None:
    """The library can forget. It cannot spontaneously remember.

    A retired skill is left out of what the planner is offered, so it is never
    replayed, so its rate can never recover -- the one-way door is a
    consequence of that rather than a rule on top of it. What replaces a
    retired recipe is the agent working the route out by hand on its next run
    and that run being learnt from, which gives a site that changed a *new*
    recipe rather than a rehabilitated old one.
    """
    store = a_library(tmp_path)
    store.save(a_skill())
    retired = record(store, "search_catalogue", "..x")

    # Replayed again anyway, successfully, several times over.
    later = record(store, "search_catalogue", "......")

    assert later.active is False, "still retired, however well it does afterwards"
    assert later.retired_at == retired.retired_at, "and the day it went is a fact, not a total"
    assert later.retired_reason == retired.retired_reason
    assert later.stats == SkillStats(uses=9, successes=8), "though the counting carries on"


@pytest.mark.unit
def test_a_person_can_retire_a_skill_the_arithmetic_would_keep(tmp_path: Path) -> None:
    """Some reasons to retire a recipe are invisible to a success rate.

    A recipe that works perfectly and does the wrong thing has a flawless
    record. So retiring by hand is its own route, and the reason is recorded
    exactly as given -- a human explanation is almost always better than
    anything this code could write.
    """
    store = a_library(tmp_path)
    store.save(a_skill())
    record(store, "search_catalogue", "....")

    retired = store.retire("search_catalogue", "it opens the wrong record for two-word searches")

    assert retired.active is False
    assert retired.retired_reason == "it opens the wrong record for two-word searches"
    assert RFC_3339.match(retired.retired_at)
    assert retired.stats.success_rate == 1.0, "a flawless record, and gone anyway"


# --- Counting ---------------------------------------------------------------


@pytest.mark.unit
def test_every_replay_is_counted_whether_it_worked_or_not(tmp_path: Path) -> None:
    """A rate is only meaningful if the denominator is honest.

    Counting only the successes would give every skill in the library a
    perfect record and make retirement impossible.
    """
    store = a_library(tmp_path, min_sample=99)
    store.save(a_skill())

    counted = record(store, "search_catalogue", ".x.xx")

    assert counted.stats == SkillStats(uses=5, successes=2)
    assert counted.stats.success_rate == 0.4


@pytest.mark.unit
def test_a_replay_that_handed_control_back_counts_as_a_failure(tmp_path: Path) -> None:
    """What is being counted is whether the *recipe* worked, not the task.

    A replay that got two steps in and handed over may well be part of a run
    that went on to succeed by hand. That is a good outcome for the run and
    still a failure of the recipe, and recording it as a success would keep a
    half-broken recipe in service forever.
    """
    store = a_library(tmp_path, min_sample=99)
    store.save(a_skill())

    # The caller passes what the replay reported, which for a handover is False.
    partly = store.record_use("search_catalogue", succeeded=False)

    assert partly.stats == SkillStats(uses=1, successes=0)


@pytest.mark.unit
def test_recording_reads_the_skill_back_rather_than_trusting_a_held_copy(
    tmp_path: Path,
) -> None:
    """A run may have been holding its copy since before the replay.

    Writing a stale copy back would undo whatever else had been recorded in
    between -- and in a benchmark, "in between" is another worker.
    """
    store = a_library(tmp_path, min_sample=99)
    store.save(a_skill())
    stale = store.find("search_catalogue")
    assert stale is not None

    record(store, "search_catalogue", "...")  # three replays this copy knows nothing about
    after = store.record_use(stale.name, succeeded=True)

    assert after.stats.uses == 4, "the three it missed are still there"


@pytest.mark.unit
def test_recording_a_use_of_a_skill_that_is_not_there_is_an_error(tmp_path: Path) -> None:
    """Something upstream is inconsistent, and creating one would hide it."""
    store = a_library(tmp_path)

    with pytest.raises(SkillStoreError, match="no such skill"):
        store.record_use("never_existed", succeeded=True)

    with pytest.raises(SkillStoreError, match="no such skill"):
        store.retire("never_existed", "because")


# --- The rule on its own ----------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize(
    ("uses", "successes", "retires"),
    [
        (0, 0, False),
        (2, 0, False),
        (3, 0, True),
        (3, 2, True),
        (3, 3, False),
        (10, 7, False),
        (10, 6, True),
    ],
)
def test_the_rule_is_arithmetic_and_can_be_checked_without_a_library(
    uses: int, successes: int, retires: bool
) -> None:
    """Every edge in one table, with no folder or browser anywhere near it.

    Args:
        uses: Replays attempted.
        successes: Replays that worked.
        retires: Whether that record should end the skill's service.
    """
    policy = RetirementPolicy(rate=0.70, min_sample=3)

    verdict = policy.verdict_on(SkillStats(uses=uses, successes=successes))

    assert (verdict is not None) is retires, verdict


@pytest.mark.unit
def test_a_skill_nobody_has_replayed_has_no_rate_to_judge(tmp_path: Path) -> None:
    """None rather than zero, for the same reason an unpriced call records None.

    A recipe that has never been tried has not failed. Reading it as 0% would
    retire every new skill the moment the rule was switched on.
    """
    assert SkillStats().success_rate is None
    assert RetirementPolicy(min_sample=0).verdict_on(SkillStats()) is None
