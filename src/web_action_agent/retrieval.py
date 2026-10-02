"""Choosing which learned skills to put in front of the planner.

A library the planner never sees is a library that never saves anything. This is
the step that closes the loop: at the start of a task, the skills belonging to
the site the task is on are listed in the prompt, and the planner may invoke one
instead of working the route out again.

**Matching is by site, and that is the whole of the matching.** A skill is a
recorded route through one particular site's pages -- "search this box, click
that result". Offering it on a different site is not a long shot, it is
meaningless: none of the elements it names exist there. So the only question
asked of a skill is whether its domain is this task's domain, compared at the
registrable domain so that ``www.example.org`` and ``example.org`` are the one
site they obviously are.

Nothing here reads the task. It is tempting -- the planner is about to be shown
five skills when one of them is clearly the relevant one -- but picking the
relevant one *is the planner's job*, and doing it here would mean a keyword
match deciding, invisibly, that the right skill was not worth mentioning. The
listing is cheap; a wrongly filtered-out skill costs the saving the library
exists for.

**Retired skills are not offered.** They stay in the library because their
statistics are the evidence for why they stopped being used, which is worth
keeping and worth nobody replaying.

**Listing everything is the deliberate policy at this size.** A library of tens
of skills fits in a prompt, and listing them all is both the simplest thing and
the most faithful -- the planner sees exactly what it could use. Ranking only
becomes a question when the library outgrows the prompt, and the ordering that
should then apply is a decision this project has deliberately not fixed in
advance. So there is a cap, it is generous, and crossing it is *reported* rather
than quietly absorbed: a run whose retrieval was truncated is a run that has
outgrown this policy, and the way to find that out should not be noticing the
agent has stopped using a skill it has.

**How a skill is written down for the planner lives here.** The planner takes
already-rendered lines on purpose: it would be guessing at the library's format
otherwise. What a skill looks like in a prompt is different from what it looks
like in a library listing -- the planner needs the name to call it by and the
values it must supply, and has no use for how many times it has worked.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from .allowlist import derive_domain, host_of
from .skill_schema import Skill

#: How many skills may be listed in one prompt. Generous on purpose: the point
#: of the cap is not to keep the prompt small but to make it impossible for a
#: library nobody noticed growing to quietly fill it. A benchmark site with more
#: than twenty verified routes through it is a situation worth being told about.
MAX_LISTED = 20


@dataclass(frozen=True)
class Retrieval:
    """Which skills this task gets to see, and what was left out.

    Attributes:
        skills: The skills to offer, in name order. Empty is the ordinary state
            of a cold library and not a failure.
        held: How many skills the library holds in total, retired included.
        matched: How many belonged to this site and were usable. Equal to
            ``len(skills)`` unless the cap cut the list.
        truncated: Whether the cap dropped any. True means the library has
            outgrown listing everything, which is a finding rather than a
            detail -- it is the point at which somebody has to decide how
            skills should be ranked.
    """

    skills: tuple[Skill, ...] = ()
    held: int = 0
    matched: int = 0
    truncated: bool = False

    @property
    def lines(self) -> tuple[str, ...]:
        """tuple[str, ...]: One prompt line per skill, ready for the planner.

        Empty when nothing matched, which is what makes the planner leave the
        whole section out rather than writing a heading with nothing under it.
        """
        return tuple(describe(skill) for skill in self.skills)

    def summary(self) -> str:
        """Render the retrieval as the line a run's log records.

        Returns:
            str: e.g. ``2 of 7 skills apply to this site``, with the truncation
            spelled out where it happened, because that is the case somebody
            needs to see rather than count.
        """
        if self.truncated:
            return (
                f"{len(self.skills)} of {self.matched} matching skills listed "
                f"(the library has outgrown listing them all)"
            )
        return f"{len(self.skills)} of {self.held} skills apply to this site"

    def to_dict(self) -> dict[str, Any]:
        """Render the retrieval as plain data, for a run's trace.

        Returns:
            dict[str, Any]: What was offered and what was held back, so a run
            that did not use a skill can be told apart from a run that was
            never shown one.
        """
        return {
            "offered": [skill.name for skill in self.skills],
            "held": self.held,
            "matched": self.matched,
            "truncated": self.truncated,
        }


def describe(skill: Skill) -> str:
    """Write one skill the way the planner is shown it.

    Three things and no more: what to call it, what it needs supplied, and what
    it does. The domain is left out because every skill in the list has the same
    one, and the usage statistics because they are evidence for *retiring* a
    skill rather than grounds for the planner preferring one -- a skill that is
    offered at all has been verified, and inviting the model to second-guess
    that with a success rate would be asking it to re-litigate a decision made
    on better evidence.

    Args:
        skill: The skill to write down.

    Returns:
        str: e.g. ``search_catalogue(query) - search the catalogue and open the
        matching record``, with optional values in square brackets as
        ``browse_shelf(shelf, [sort])``.
    """
    wanted = list(skill.required_params) + [f"[{name}]" for name in skill.optional_params]
    return f"{skill.name}({', '.join(wanted)}) - {skill.description}"


def for_url(skills: Iterable[Skill], url: str, *, limit: int = MAX_LISTED) -> Retrieval:
    """Pick the skills that belong to the site a task is about to run on.

    Args:
        skills: The library's skills, in any order.
        url: Where the run starts. A URL rather than a domain because that is
            what every caller has, and working the domain out in one place is
            better than every caller remembering to.
        limit: How many may be listed.

    Returns:
        Retrieval: What to offer, and what was held back.

    Raises:
        AllowlistError: If the URL has no host, or has one that cannot be
            reduced to a site. The same refusal the run's own allowlist makes
            about the same URL, so a task that could not have started cannot
            get as far as being offered skills.
    """
    return for_domain(skills, derive_domain(host_of(url)), limit=limit)


def for_domain(skills: Iterable[Skill], domain: str, *, limit: int = MAX_LISTED) -> Retrieval:
    """Pick the skills belonging to one site.

    Args:
        skills: The library's skills, in any order.
        domain: The registrable domain to match, lowercase.
        limit: How many may be listed.

    Returns:
        Retrieval: What to offer, and what was held back. Sorted by name --
        not by any measure of quality, because choosing between skills is the
        planner's decision, and because a stable order means the same library
        produces the same prompt twice, which is what makes a run reproducible.
    """
    held = list(skills)
    matching = sorted(
        (skill for skill in held if skill.active and skill.domain == domain),
        key=lambda skill: skill.name,
    )
    return Retrieval(
        skills=tuple(matching[:limit]),
        held=len(held),
        matched=len(matching),
        truncated=len(matching) > limit,
    )
