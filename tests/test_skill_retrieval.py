"""Which learned skills a task gets to see, and how they read in the prompt.

This is the step that closes the loop. Everything before it taught the agent to
write skills down; a library the planner never sees saves nothing, so this is
where a stored route becomes something a run can actually use.

The tests are about two kinds of mistake, and they pull in opposite directions.

**Offering a skill that cannot work.** A skill is a route through one site's
pages, so one from another site names elements that do not exist. A retired one
was taken out of service for failing. Both would be invitations to waste a step,
and both are checked by asserting on the *prompt the planner is actually given*
rather than on an intermediate list — the prompt being the only place the
mistake would do any harm.

**Withholding a skill that would have worked.** The quieter failure, and the one
with no symptom: the agent simply works the route out again, costs more, and
nothing anywhere says why. So there are tests that nothing clever happens — the
task's wording is not consulted, a skill sharing no words with the task is still
offered, and the one thing that can drop a skill silently reports that it did.
"""

from __future__ import annotations

from typing import Any

import pytest

from web_action_agent.observation import Observation
from web_action_agent.observer import Box, Element
from web_action_agent.planner import Planner
from web_action_agent.retrieval import MAX_LISTED, Retrieval, describe, for_domain, for_url
from web_action_agent.skill_schema import Skill

LIBRARY_SITE = "library.org"
"""str: The site the task in these tests runs on."""


def a_skill(
    name: str,
    domain: str = LIBRARY_SITE,
    *,
    description: str = "search the catalogue and open the matching record",
    active: bool = True,
    optional: bool = False,
) -> Skill:
    """Build one verified skill.

    Args:
        name: What it is called.
        domain: The site it was learnt on.
        description: What it does, in a line.
        active: False for a skill retirement has taken out of service.
        optional: True to give it a second, optional parameter, for the tests
            about how a skill is written down.

    Returns:
        Skill: The skill.
    """
    params: list[dict[str, Any]] = [{"name": "query", "description": "what to search for"}]
    steps: list[dict[str, Any]] = [
        {"action": "fill", "selector": {"name": "Search terms"}, "text": "{query}"}
    ]
    values = {"query": "privacy"}
    if optional:
        params.append({"name": "sort", "description": "how to order them", "required": False})
        steps.append({"action": "navigate", "url": f"https://{domain}/results?sort=" + "{sort}"})
        values["sort"] = "newest"
    return Skill.from_dict(
        {
            "name": name,
            "description": description,
            "domain": domain,
            "params": params,
            "steps": steps,
            "verification": {"params": values, "expected_outcome": "a record page"},
            "active": active,
        }
    )


def a_library() -> list[Skill]:
    """Build a library spread across three sites, with one retired skill.

    Returns:
        list[Skill]: Five skills — two usable on the library site, one retired
        there, and one on each of two other sites.
    """
    return [
        a_skill("search_catalogue"),
        a_skill("browse_shelf", description="browse one shelf of the catalogue"),
        a_skill("old_search", description="the previous search route", active=False),
        a_skill("check_departure", domain="trains.org", description="look up a departure"),
        a_skill("order_groceries", domain="market.co.uk", description="fill a basket"),
    ]


def the_prompt(skills: Retrieval, task: str = "find the privacy notice") -> str:
    """Render the prompt a planner would be given with these skills.

    Asserting on the prompt rather than on the retrieval is the point: the
    retrieval is a list, and a list cannot mislead anybody. The prompt is where
    a wrongly offered skill would actually be offered.

    Args:
        skills: What retrieval produced.
        task: The task to put at the top.

    Returns:
        str: The whole user message.
    """
    page = Observation(
        url=f"https://{LIBRARY_SITE}/search",
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
    # The planner stores its client without touching it, and rendering a prompt
    # never calls one -- so there is nothing to stand in for here, and giving it
    # a stub would make these tests depend on something they are not about.
    planner = Planner(client=None)  # type: ignore[arg-type]
    return planner.render_prompt(task=task, observation=page, skills=skills.lines)


# --- Only this site's skills, and only the live ones ------------------------


@pytest.mark.unit
def test_only_this_sites_skills_reach_the_prompt() -> None:
    """A skill from another site names elements that do not exist here.

    Offering one is not a long shot, it is an invitation to waste a step on a
    route through a page the run will never be on.
    """
    retrieved = for_domain(a_library(), LIBRARY_SITE)
    prompt = the_prompt(retrieved)

    assert "search_catalogue" in prompt
    assert "browse_shelf" in prompt
    assert "check_departure" not in prompt, "another site's skill"
    assert "order_groceries" not in prompt, "nor a third site's"


@pytest.mark.unit
def test_a_retired_skill_is_never_offered() -> None:
    """Retired is kept, not used.

    It stays in the library because its statistics are the evidence for why it
    stopped working — which is worth keeping and worth nobody replaying.
    """
    retrieved = for_domain(a_library(), LIBRARY_SITE)

    assert "old_search" not in the_prompt(retrieved)
    assert [skill.name for skill in retrieved.skills] == ["browse_shelf", "search_catalogue"]
    assert retrieved.held == 5, "the library still holds it"


@pytest.mark.unit
def test_an_empty_library_leaves_the_section_out_entirely() -> None:
    """A heading with nothing under it invites the model to wonder what it missed.

    And a cold library is the ordinary state of a fresh install, not an error,
    so the prompt on such a run should look as though skills were never a
    feature.
    """
    retrieved = for_domain([], LIBRARY_SITE)
    prompt = the_prompt(retrieved)

    assert retrieved.lines == ()
    assert "Skills available" not in prompt
    assert "use_skill" not in prompt, "nor the instruction that goes with the list"
    assert "# Task" in prompt, "the rest of the prompt is unaffected"


@pytest.mark.unit
def test_a_site_with_skills_but_none_of_its_own_gets_no_section() -> None:
    """A populated library is not the same as a useful one, for this task."""
    retrieved = for_domain(a_library(), "nowhere.org")

    assert retrieved.lines == ()
    assert retrieved.held == 5 and retrieved.matched == 0
    assert "Skills available" not in the_prompt(retrieved)


# --- Matching the site, as the rest of the agent understands a site ---------


@pytest.mark.unit
def test_a_subdomain_is_the_same_site() -> None:
    """``www.library.org`` and ``library.org`` are one site to everybody but a string.

    Both sides go through the same function the allowlist and the inducer use,
    which is the real guarantee: a skill learnt on a run that started at
    ``shop.example.org`` is stored under ``example.org``, and a later run from
    ``www.example.org`` looks for exactly that.
    """
    library = a_library()

    assert for_url(library, f"https://www.{LIBRARY_SITE}/search").matched == 2
    assert for_url(library, f"https://{LIBRARY_SITE}/search").matched == 2
    assert for_url(library, "https://shop.market.co.uk/basket").matched == 1, "a real two-part TLD"


@pytest.mark.unit
def test_a_url_that_names_no_site_is_refused_as_the_allowlist_would() -> None:
    """The same refusal about the same URL, so a run that could not start cannot
    get as far as being offered skills."""
    from web_action_agent.allowlist import AllowlistError

    with pytest.raises(AllowlistError):
        for_url(a_library(), "not-a-url")


# --- Nothing clever happens -------------------------------------------------


@pytest.mark.unit
def test_the_task_is_never_consulted() -> None:
    """Choosing between skills is the planner's job, not a keyword match's.

    This is the quiet failure: a skill filtered out here never reaches the
    model, the agent works the route out again, the run costs more, and nothing
    anywhere says why. Listing is cheap; a wrongly withheld skill costs the
    saving the library exists for.
    """
    library = [a_skill("join_the_library", description="fill in the membership form")]

    offered = the_prompt(for_domain(library, LIBRARY_SITE), task="what are the opening hours?")

    assert "join_the_library" in offered, "no word in common with the task, and still offered"


@pytest.mark.unit
def test_the_same_library_produces_the_same_prompt_twice() -> None:
    """A stable order is what makes two runs of one task comparable.

    Sorted by name rather than by any measure of quality — deciding between
    skills is the planner's, and a listing ordered by success rate would be
    this module quietly voting.
    """
    shuffled = list(reversed(a_library()))

    assert for_domain(a_library(), LIBRARY_SITE).lines == for_domain(shuffled, LIBRARY_SITE).lines


# --- Outgrowing the prompt --------------------------------------------------


@pytest.mark.unit
def test_a_library_too_big_to_list_says_so_rather_than_quietly_dropping_skills() -> None:
    """The one thing that can withhold a skill, and it reports itself.

    How skills *should* be ranked once a library outgrows the prompt is a
    decision this project has deliberately not fixed in advance. Until it is,
    crossing the line has to be visible — because the alternative way to find
    out is noticing that the agent has stopped using a skill it has.
    """
    many = [a_skill(f"route_{number:02d}") for number in range(MAX_LISTED + 5)]

    retrieved = for_domain(many, LIBRARY_SITE)

    assert retrieved.truncated is True
    assert len(retrieved.skills) == MAX_LISTED
    assert retrieved.matched == MAX_LISTED + 5, "and it says how many there really were"
    assert "outgrown" in retrieved.summary()


@pytest.mark.unit
def test_a_library_that_fits_is_listed_whole() -> None:
    """At this size, listing everything is the policy rather than a shortcut."""
    retrieved = for_domain(a_library(), LIBRARY_SITE)

    assert retrieved.truncated is False
    assert "2 of 5 skills apply to this site" == retrieved.summary()


# --- How a skill reads in a prompt ------------------------------------------


@pytest.mark.unit
def test_a_skill_is_written_as_what_to_call_it_and_what_to_supply() -> None:
    """Three things and no more, because the planner needs exactly three.

    The domain is left out — every skill in the list shares it. So are the
    usage statistics: a skill that is offered has already been verified, and
    showing a success rate would invite the model to re-litigate that on worse
    evidence than the decision was made with.
    """
    line = describe(a_skill("search_catalogue"))

    assert line == "search_catalogue(query) - search the catalogue and open the matching record"
    assert "library.org" not in line
    assert "uses" not in line


@pytest.mark.unit
def test_optional_values_are_marked_as_optional() -> None:
    """Otherwise the model supplies them all, or refuses to invoke a skill it could."""
    line = describe(a_skill("browse_shelf", optional=True, description="browse one shelf"))

    assert line == "browse_shelf(query, [sort]) - browse one shelf"


@pytest.mark.unit
def test_the_prompt_says_how_to_invoke_one() -> None:
    """A list with no instruction is a list the model has to guess the use of.

    The instruction sits beside the list rather than in the system prompt
    because it only makes sense next to one: told in advance on a run where no
    skills were retrieved, a model has been invited to invent one.
    """
    prompt = the_prompt(for_domain(a_library(), LIBRARY_SITE))

    assert "# Skills available for this site" in prompt
    assert "use_skill" in prompt, "the action to invoke them with"
    assert "square brackets are optional" in prompt, "how to read the list"
    assert "Prefer one over working the same route out" in prompt, "and why to bother"


@pytest.mark.unit
def test_a_retrieval_is_recordable() -> None:
    """So a run that was never offered a skill reads differently from one that was.

    Without it, "the agent did not use a skill" and "the agent had none to use"
    are the same line in a trace, and only one of them is a problem.
    """
    record = for_domain(a_library(), LIBRARY_SITE).to_dict()

    assert record == {
        "offered": ["browse_shelf", "search_catalogue"],
        "held": 5,
        "matched": 2,
        "truncated": False,
    }
