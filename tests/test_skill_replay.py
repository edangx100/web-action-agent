"""Running a learned route for free, and handing back cleanly when it breaks.

This is where the project's claim is cashed, so the tests are mostly about two
numbers and one behaviour.

**The number that is the point: zero.** A replayed skill costs no planner calls.
Not "few" — none, and the tests count them rather than take the word for it. The
guarantee is structural: nothing reachable from a replay has a model client, so
the call cannot be made. A test that merely checked the count would pass on a
version that made calls and happened not to on that page.

**The number that makes a fallback worth having: how far it got.** A replay that
completed two of four steps leaves those two done. The planner resumes from the
page in front of it rather than from the beginning — which is the difference
between a skill being a gamble and a skill being a head start.

**And the behaviour: what the planner is handed afterwards.** A failed replay
has left the browser somewhere nobody planned for, so the page it sees has to be
read *fresh*. Handing it the page from before the replay would be asking it
about a page that no longer exists, and it would answer confidently.

Everything runs against a real browser and the practice site, because a replay
is exactly the thing a stubbed page cannot test: the whole question is whether a
route survives contact with a page it was not recorded on.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from web_action_agent import replay
from web_action_agent.actions import UseSkill, parse_action
from web_action_agent.allowlist import Allowlist
from web_action_agent.config import Config
from web_action_agent.executor import TIMEOUT, Executor
from web_action_agent.observation import Observation, capture
from web_action_agent.observer import Box, Element
from web_action_agent.replay import Replay, ReplayError, SkillReplayer, fill_in
from web_action_agent.selectors import TIER_NAME, TIER_STRUCTURAL
from web_action_agent.skill_schema import ClickStep, Selector, Skill
from web_action_agent.skill_store import SkillStore
from web_action_agent.validator import Validator


def a_skill(name: str = "search_and_open", domain: str = "127.0.0.1", **overrides: Any) -> Skill:
    """Build the route the practice site's search supports.

    Three steps, which is enough to have a middle one to break.

    Args:
        name: What the skill is called.
        domain: The site it belongs to, which the gate checks against the page.
        **overrides: Fields to replace.

    Returns:
        Skill: A stored route.
    """
    skill = {
        "name": name,
        "description": "search the catalogue and open the matching record",
        "domain": domain,
        "params": [{"name": "query", "description": "what to search for", "example": "membership"}],
        "steps": [
            {
                "action": "fill",
                "selector": {"name": "Search terms", "role": "searchbox"},
                "text": "{query}",
            },
            {"action": "press", "key": "Enter"},
            {
                "action": "click",
                "selector": {
                    "name": "Membership application",
                    "role": "link",
                    "structural": "link[7]",
                },
            },
        ],
        "verification": {"params": {"query": "privacy"}, "expected_outcome": "a record page"},
    }
    return Skill.from_dict({**skill, **overrides})


def a_replayer(task_session: Any, site: Any, library: Any = None, **settings: Any) -> SkillReplayer:
    """Assemble a replayer over a real browser and the practice site.

    Args:
        task_session: The test's browser session.
        site: The practice site.
        library: A skill store, for the tests that replay by name.
        **settings: Config overrides, for the timeout test.

    Returns:
        SkillReplayer: Ready to run a route.
    """
    start = site.url_for("search.html")
    config = Config(start_url=start, **settings)
    return SkillReplayer(
        executor=Executor(task_session, action_timeout_s=config.action_timeout_s),
        validator=Validator(Allowlist.derive(start)),
        config=config,
        library=library,
    )


# --- The point: it costs nothing --------------------------------------------


@pytest.mark.fixture
def test_replaying_a_route_makes_no_model_calls_at_all(
    task_session: Any, fixture_site: Any
) -> None:
    """Zero, counted rather than taken on trust.

    Three browser steps, and the only thing that would normally decide each of
    them -- a planner call -- does not happen. That saving is the whole reason
    a library exists, and it is the headline number the benchmark measures.
    """
    task_session.page.goto(fixture_site.url_for("search.html"))
    replayer = a_replayer(task_session, fixture_site)

    replay = replayer.run(a_skill(), {"query": "membership"})

    assert replay.ok, replay.summary()
    assert replay.completed == 3 and replay.total == 3
    assert "results.html" in task_session.page.url or "form-step-1" in task_session.page.url


@pytest.mark.fixture
def test_nothing_a_replay_can_reach_is_able_to_call_a_model(
    task_session: Any, fixture_site: Any
) -> None:
    """The guarantee is structural, not a matter of counting afterwards.

    A test that only counted calls would pass on a version that *could* make
    one and happened not to on that page. This asserts the stronger thing: a
    replayer is assembled from an executor, a gate and a page, and there is no
    model client anywhere in it to call.
    """
    replayer = a_replayer(task_session, fixture_site)

    held = [type(value).__name__ for value in vars(replayer).values()]

    assert not any("ModelClient" in name or "Planner" in name for name in held), held


# --- Through the same gate and the same executor ----------------------------


@pytest.mark.fixture
def test_a_replayed_step_trips_the_same_timeout_a_fresh_one_would(
    task_session: Any, fixture_site: Any
) -> None:
    """A stored route gets no private path to the browser, and no private clocks.

    The practice site has a button that never stops moving, which automation
    will not click. A replay of a step aiming at it has to end with a stated
    timeout rather than hanging the run -- which it does because the step goes
    through the run's own executor, not around it.
    """
    task_session.page.goto(fixture_site.url_for("hang.html"))
    replayer = a_replayer(task_session, fixture_site, action_timeout_s=1.0)
    drifting = a_skill(
        steps=[
            {
                "action": "click",
                "selector": {"name": "Reserve from trolley", "role": "button"},
            }
        ],
        params=[],
        verification={"params": {}, "expected_outcome": "a reservation"},
    )

    replay = replayer.run(drifting, {})

    assert replay.ok is False
    assert TIMEOUT in replay.detail, replay.detail
    assert replay.completed == 0


@pytest.mark.fixture
def test_a_route_that_wanders_off_the_allowed_site_is_refused(
    task_session: Any, fixture_site: Any
) -> None:
    """Replaying is not a reason to suspend the rules a fresh action follows.

    A stored step is still a proposal, and a proposal to leave the task's site
    is refused for a replayed step exactly as it would be for one the planner
    had just invented. Otherwise a skill would be a way of writing an exception
    to the allowlist down in a file.
    """
    task_session.page.goto(fixture_site.url_for("search.html"))
    replayer = a_replayer(task_session, fixture_site)
    wandering = a_skill(
        steps=[{"action": "navigate", "url": "https://elsewhere.example/page"}],
        params=[],
        verification={"params": {}, "expected_outcome": "somewhere else"},
    )

    replay = replayer.run(wandering, {})

    assert replay.ok is False
    assert "refused" in replay.detail
    assert "elsewhere.example" in replay.detail or "blocked" in replay.detail.lower()


# --- Stopping, and keeping what was done ------------------------------------


@pytest.mark.fixture
def test_a_route_that_breaks_partway_keeps_the_progress_it_made(
    task_session: Any, fixture_site: Any
) -> None:
    """The difference between a skill being a gamble and a skill being a head start.

    The third step is given a description matching nothing, so it stops there.
    The first two really happened -- the search was typed and submitted -- and
    the page is where they left it. A replay that rolled back, or that reported
    only "failed", would throw away work that was done.
    """
    task_session.page.goto(fixture_site.url_for("search.html"))
    replayer = a_replayer(task_session, fixture_site)
    broken = a_skill(
        steps=[
            *(step.model_dump(mode="json") for step in a_skill().steps[:2]),
            {
                "action": "click",
                "selector": {"name": "No such link", "role": "link", "structural": "link[99]"},
            },
        ]
    )

    replay = replayer.run(broken, {"query": "membership"})

    assert replay.ok is False
    assert replay.completed == 2, "the search really was typed and submitted"
    assert replay.handover == 3, "and the planner picks up at the third step"
    assert "q=membership" in task_session.page.url, "the page is where the replay left it"
    assert "step 3" in replay.detail and "no element matched" in replay.detail


@pytest.mark.fixture
def test_it_stops_at_the_first_failure_rather_than_running_on(
    task_session: Any, fixture_site: Any
) -> None:
    """Later steps would be attempted from a page the route did not expect.

    Carrying on would produce a cascade of failures that say only that the run
    went off the rails several steps ago -- which is already known -- while
    clicking at things on pages nobody planned for.
    """
    task_session.page.goto(fixture_site.url_for("search.html"))
    replayer = a_replayer(task_session, fixture_site)
    broken_early = a_skill(
        steps=[
            {"action": "click", "selector": {"name": "No such thing", "structural": "link[99]"}},
            {"action": "press", "key": "Enter"},
            {"action": "press", "key": "Enter"},
        ],
        params=[],
        verification={"params": {}, "expected_outcome": "nothing"},
    )

    replay = replayer.run(broken_early, {})

    assert len(replay.steps) == 1, "the two steps after the failure were never attempted"
    assert replay.total == 3, "and the record still says how long the route was"
    assert replay.handover == 1


# --- Replaying by name, the way a run does ----------------------------------


@pytest.mark.fixture
def test_a_use_skill_action_replays_the_named_route(
    task_session: Any, fixture_site: Any, tmp_path: Any
) -> None:
    """The shape the loop hands in: an approved ``use_skill``, nothing else.

    The gate has already checked the skill exists, is active, belongs to this
    site and has every value it needs -- so by the time it arrives here there
    is nothing left to validate about the invocation itself.
    """
    task_session.page.goto(fixture_site.url_for("search.html"))
    store = SkillStore(tmp_path / "skills")
    store.save(a_skill())
    replayer = a_replayer(task_session, fixture_site, library=store)

    action = parse_action(
        {"action": "use_skill", "skill": "search_and_open", "params": {"query": "membership"}}
    )
    assert isinstance(action, UseSkill)
    replay = replayer(action, capture(task_session.page, config=Config())[0])

    assert replay.ok, replay.summary()
    assert replay.skill == "search_and_open"


@pytest.mark.fixture
def test_a_skill_the_library_no_longer_holds_is_an_error_not_a_failed_replay(
    task_session: Any, fixture_site: Any, tmp_path: Any
) -> None:
    """The gate approved it, so the library changed underneath the run.

    A route that stopped working is ordinary and gets reported. A route that
    was never there is a different thing: something upstream is inconsistent,
    and reporting it as a failed replay would blame the skill for it.
    """
    task_session.page.goto(fixture_site.url_for("search.html"))
    replayer = a_replayer(task_session, fixture_site, library=SkillStore(tmp_path / "empty"))

    action = parse_action({"action": "use_skill", "skill": "vanished", "params": {}})
    with pytest.raises(ReplayError, match="no skill named"):
        replayer(action, capture(task_session.page, config=Config())[0])


# --- Filling the blanks in ---------------------------------------------------


@pytest.mark.unit
def test_a_step_with_no_value_for_its_blank_is_refused() -> None:
    """Otherwise the agent types the characters "{query}" into a search box.

    And reports the step as having worked, which is the quiet kind of wrong.
    """
    with pytest.raises(ReplayError, match="no value was given"):
        fill_in(a_skill().steps[0], {})


@pytest.mark.unit
def test_steps_that_carry_no_text_pass_through_untouched() -> None:
    """Most steps have no blank in them, and substitution must not invent one."""
    skill = a_skill()

    assert fill_in(skill.steps[1], {"query": "privacy"}) == skill.steps[1], "press Enter"
    assert fill_in(skill.steps[0], {"query": "privacy"}).text == "privacy"


# --- What the trace records --------------------------------------------------


@pytest.mark.unit
def test_a_replay_says_how_far_it_got_in_words_and_in_fields() -> None:
    """Both, because a trace is read by people and by programs.

    The sentence is what the planner is told and what a person reads first; the
    fields are what a report counts. Deriving one from the other at read time
    would mean a reader working out where the planner took over by counting,
    which is the first question a failed replay raises.
    """
    stopped = Replay(
        skill="search_and_open",
        steps=(),
        total=3,
        ok=False,
        detail="step 1: no element matched",
    )
    finished = Replay(skill="search_and_open", total=3, ok=True, detail="all 3 steps ran")

    assert stopped.handover == 1
    assert "stopped at step 1 of 3" in stopped.summary()
    assert finished.handover is None
    assert "3 of 3 steps" in finished.summary()


@pytest.mark.unit
def test_a_replay_survives_the_trace_round_trip() -> None:
    """A run folder has to be able to say where the planner took over.

    Written and read by the same class, so the two cannot come to disagree
    about what a handover looks like on disk.
    """
    original = Replay(skill="search_and_open", total=4, ok=False, detail="step 3: refused")

    rebuilt = Replay.from_dict(json.loads(json.dumps(original.to_dict())))

    assert rebuilt == original
    assert original.to_dict()["handover"] == 1, "written out, not left to be counted"


@pytest.mark.unit
def test_the_derived_counts_are_not_read_back_from_the_file() -> None:
    """One answer to "how far did it get", not two that could disagree.

    ``completed`` and ``handover`` are written for a reader's convenience and
    derived from the steps on the way back in, so a hand-edited file cannot
    claim a count its own steps contradict.
    """
    record = Replay(skill="x", total=2, ok=False).to_dict()
    record["completed"] = 99
    record["handover"] = 99

    assert Replay.from_dict(record).completed == 0
    assert Replay.from_dict(record).handover == 1


@pytest.mark.unit
def test_a_selector_describing_nothing_cannot_be_stored_so_cannot_be_replayed() -> None:
    """The format refuses it, which is why replay never has to handle it.

    Worth asserting from here: it is the reason this module has no code for a
    step that describes no element.
    """
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        ClickStep(action="click", selector=Selector())


# --- The whole loop: a run that uses a skill ---------------------------------


class CountingPlanner:
    """A planner that answers from a script and counts how often it is asked.

    The count is the measurement this whole module exists for: a three-step
    skill replayed must cost one planner call -- the one that chose it -- and
    not four.

    Attributes:
        asked: How many times the loop asked for an action.
        prompts: What it was shown each time, so a test can check the page it
            was handed after a failed replay is the current one.
    """

    def __init__(self, *replies: dict[str, Any]) -> None:
        """Set up the script.

        Args:
            *replies: Actions to return, as plain data. The last repeats.
        """
        self._replies = list(replies)
        self.asked = 0
        self.prompts: list[str] = []

    def decide(self, **kwargs: Any) -> Any:
        """Answer one step, and record what it was shown.

        Args:
            **kwargs: What the controller passed, including the observation.

        Returns:
            PlannerDecision: The next scripted answer.
        """
        from web_action_agent.model_client import CallRecord
        from web_action_agent.planner import PlannerDecision
        from web_action_agent.trace import new_id, utc_now

        self.asked += 1
        self.prompts.append(kwargs["observation"].url)
        reply = self._replies[min(self.asked - 1, len(self._replies) - 1)]
        return PlannerDecision(
            action=parse_action(reply),
            failure=None,
            text="{}",
            record=CallRecord(
                call_id=new_id("call"),
                started_at=utc_now(),
                ended_at=utc_now(),
                profile="local",
                model="stub",
                vision=False,
                prompt_tokens=100,
                completion_tokens=10,
                latency_s=0.01,
                finish_reason="stop",
            ),
        )


def a_run(task_session: Any, site: Any, store: SkillStore, *replies: dict[str, Any]) -> Any:
    """Drive one real run that has a skill library.

    Args:
        task_session: The test's browser session.
        site: The practice site.
        store: The library the run may replay from.
        *replies: What the planner answers, in order.

    Returns:
        tuple: The run's report and the planner, so a test can read the count.
    """
    from web_action_agent.controller import Controller
    from web_action_agent.memory import Memory

    start = site.url_for("search.html")
    config = Config(start_url=start)
    executor = Executor(task_session, action_timeout_s=config.action_timeout_s)
    validator = Validator(Allowlist.derive(start), skills=store)
    planner = CountingPlanner(*replies)
    controller = Controller(
        planner=planner,
        validator=validator,
        executor=executor,
        memory=Memory.from_config(config),
        max_steps=6,
        retry_budget=1,
        config=config,
        replay=SkillReplayer(executor=executor, validator=validator, config=config, library=store),
    )
    return controller.run("find the membership record", start_url=start), planner


@pytest.mark.fixture
def test_a_three_step_skill_costs_one_planner_call_not_four(
    task_session: Any, fixture_site: Any, tmp_path: Any
) -> None:
    """The headline number, measured through a real run rather than asserted.

    One call to choose the skill, three browser steps replayed for free, one
    more call to finish. Working the same route out step by step would have
    cost four. That difference, multiplied across a benchmark, is the entire
    claim the library exists to make.
    """
    store = SkillStore(tmp_path / "skills")
    store.save(a_skill())

    report, planner = a_run(
        task_session,
        fixture_site,
        store,
        {"action": "use_skill", "skill": "search_and_open", "params": {"query": "membership"}},
        {"action": "done", "status": "success", "answer": "Membership application"},
    )

    assert report.outcome.ok, report.summary()
    assert planner.asked == 2, "one call chose the skill, one finished the run"
    replayed = report.steps[0].replay
    assert replayed is not None and replayed.ok
    assert replayed.completed == 3, "and three browser steps happened inside that one call"
    assert report.cost.calls == 2, "which is what the run is billed for"


@pytest.mark.fixture
def test_a_broken_replay_hands_back_the_page_it_actually_left(
    task_session: Any, fixture_site: Any, tmp_path: Any
) -> None:
    """The planner must not be asked about a page that no longer exists.

    A replay that stopped at step three has moved the browser twice. Handing
    the planner the page from before the replay would get a confident answer
    about a page the run is not on -- and the elements it named would be
    numbered for a different page entirely.
    """
    store = SkillStore(tmp_path / "skills")
    store.save(
        a_skill(
            steps=[
                *(step.model_dump(mode="json") for step in a_skill().steps[:2]),
                {"action": "click", "selector": {"name": "No such link", "structural": "link[99]"}},
            ]
        )
    )

    report, planner = a_run(
        task_session,
        fixture_site,
        store,
        {"action": "use_skill", "skill": "search_and_open", "params": {"query": "membership"}},
        {"action": "done", "status": "success", "answer": "found it by hand"},
    )

    before, after = planner.prompts
    assert "search.html" in before, "the page the skill was chosen from"
    assert "q=membership" in after, "and the page the replay actually left the run on"
    assert report.outcome.ok, "the run carried on from there and finished"

    handed_over = report.steps[0].replay
    assert handed_over is not None
    assert (handed_over.completed, handed_over.handover) == (2, 3)
    # And it is all in the trace, so the handover is a thing a reader looks up.
    record = report.steps[0].to_dict()["replay"]
    assert record["handover"] == 3 and record["completed"] == 2


# --- Looking again before giving up ---------------------------------------------


def a_page(*elements: Any, url: str = "https://www.iras.gov.sg/") -> Observation:
    """Build an observation holding the given elements, for tests without a browser.

    Args:
        *elements: The page's elements.
        url: Where the page is.

    Returns:
        Observation: The page.
    """
    return Observation(url=url, title="IRAS", elements=tuple(elements), text="")


def a_link(number: int, name: str) -> Element:
    """Build one link."""
    return Element(
        id=number, tag="a", role="link", name=name, text=name, box=Box(x=0, y=0, width=1, height=1)
    )


def a_patient_replayer(*looks: Observation) -> tuple[SkillReplayer, list[float]]:
    """Build a replayer whose page changes from one look to the next.

    Args:
        *looks: What each successive look sees; the last repeats.

    Returns:
        tuple: The replayer, and the pauses it took.
    """
    waits: list[float] = []
    replayer = SkillReplayer(
        executor=None,
        validator=None,
        config=Config(),
        sleep=waits.append,  # type: ignore[arg-type]
    )
    seen = list(looks)
    replayer.look = lambda: seen.pop(0) if len(seen) > 1 else seen[0]  # type: ignore[method-assign]
    return replayer, waits


MENU_STEP = ClickStep(
    action="click",
    selector=Selector(
        name="Tax residency and tax rates",
        role="link",
        text="Tax residency and tax rates",
        structural="link[2]",
    ),
)


@pytest.mark.unit
def test_a_link_that_appears_a_moment_late_is_found_by_its_name(monkeypatch: Any) -> None:
    monkeypatch.setattr(replay, "RELOOK_WAIT_S", 1.0)
    menu_closed = a_page(a_link(1, "Taxes"), a_link(2, "LOGIN"))
    menu_open = a_page(
        a_link(1, "Taxes"), a_link(2, "LOGIN"), a_link(3, "Tax residency and tax rates")
    )
    # Any later look numbers the page afresh; the action must not belong to it.
    renumbered = a_page(a_link(1, "Tax residency and tax rates"), a_link(2, "LOGIN"))
    replayer, waits = a_patient_replayer(menu_closed, menu_open, renumbered)

    action, resolution, observation = replayer._find(MENU_STEP, ())

    assert action is not None and action.element_id == 3, "the menu link, not LOGIN"
    assert resolution is not None and resolution.tier == TIER_NAME
    assert observation is menu_open, "and the action belongs to the look it was found in"
    assert waits == [1.0], "after one pause"


@pytest.mark.unit
def test_a_link_that_never_appears_is_given_up_on_after_the_extra_looks() -> None:
    menu_closed = a_page(a_link(1, "Taxes"), a_link(2, "LOGIN"))
    replayer, waits = a_patient_replayer(menu_closed)

    action, resolution, _ = replayer._find(MENU_STEP, ())

    assert action is None, "LOGIN is never clicked"
    assert resolution is not None and resolution.found is False
    assert len(waits) == replay.RELOOKS


@pytest.mark.unit
def test_a_link_found_at_once_costs_no_pause() -> None:
    replayer, waits = a_patient_replayer(a_page(a_link(1, "Tax residency and tax rates")))

    action, _, _ = replayer._find(MENU_STEP, ())

    assert action is not None and waits == []


@pytest.mark.unit
def test_a_match_by_position_is_only_settled_for_after_looking_again() -> None:
    # A search result recorded for "membership": by position is the right
    # answer for another search, but only once its name has had time to show.
    result = ClickStep(
        action="click",
        selector=Selector(name="Membership application", role="link", structural="link[2]"),
    )
    page = a_page(a_link(1, "Home"), a_link(2, "Data and privacy notice"))
    replayer, waits = a_patient_replayer(page)

    action, resolution, _ = replayer._find(result, ("membership",))

    assert action is not None and action.element_id == 2
    assert resolution is not None and resolution.tier == TIER_STRUCTURAL
    assert len(waits) == replay.RELOOKS
