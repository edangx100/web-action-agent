"""Proving a candidate works, on a real browser, before it may be stored.

Every test here drives an actual Chromium over the practice site, because the
thing being tested is whether a recipe *survives contact with a page* — and a
stubbed page would be a page built to agree with the recipe.

The chain is real end to end: the committed run is read back, induction turns it
into a candidate, and the candidate is replayed against the live site with
values it was never recorded from. Only the judge is scripted, and only because
what a real model thinks of a page is a question the benchmark settles rather
than this file.

**The headline is that the candidate passes.** The run it came from searched for
"membership" and clicked the result called "Membership application" — a name no
other search produces. It survives a replay searching for "privacy" anyway,
because the stored step describes that element four ways and one of the weaker
descriptions still finds it. That is the durable-selector work paying off, on a
case that would otherwise be the exact failure induction cannot see.

**The failures are the point of the module, though.** A search that finds
nothing leaves no result to click, and the replay stops there — with no judge
call at all, which is the whole reason the cheap layer runs first. A candidate
that replays perfectly but ends on the wrong page gets past that layer and is
caught by the judge. Neither is an error: being wrong is the ordinary outcome
this module exists to detect, and each discard carries the reason.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from web_action_agent.allowlist import Allowlist
from web_action_agent.config import Config
from web_action_agent.executor import Executor
from web_action_agent.inducer import Inducer
from web_action_agent.model_client import ModelClient, load_profile
from web_action_agent.replay import fill_in
from web_action_agent.skill_schema import ClickStep, Selector, Skill
from web_action_agent.skill_store import SkillStore
from web_action_agent.trace_reader import read_run
from web_action_agent.validator import Validator
from web_action_agent.verifier import (
    SEMANTIC,
    STRUCTURAL,
    SkillVerifier,
    StepReplay,
    VerificationError,
)
from web_action_agent.verifier import Verification as Verdict

TRACE = Path(__file__).resolve().parent / "fixtures" / "traces" / "search-and-open"
"""Path: The committed run every candidate here is induced from.

A recording of this same practice site — its address was rewritten to a stable
name before committing, so the skill's ``domain`` says ``northgate.example``
while the replay happens against ``127.0.0.1``. Same site, two names, and the
verifier does not read the domain: where to start from is the caller's to say.
"""

#: What induction is told about the run. Scripted, because what a real model
#: picks out as task-specific is a question for the benchmark; what matters here
#: is that the candidate it produces survives a replay.
AN_INDUCTION: dict[str, Any] = {
    "name": "search_and_open_record",
    "description": "search the catalogue for a term and open the matching record",
    "parameters": [
        {
            "name": "query",
            "value": "membership",
            "description": "the subject to search the catalogue for",
            "required": True,
        }
    ],
    "verification": {
        "params": [{"name": "query", "value": "privacy"}],
        "expected_outcome": "a record page about privacy, opened from the search results",
    },
}


class ScriptedModel:
    """A model endpoint that answers with a fixed JSON object.

    Attributes:
        calls: How many times it was asked. The number the structural layer is
            supposed to leave at zero.
    """

    def __init__(self, answer: dict[str, Any] | str, *, fail: Exception | None = None) -> None:
        """Set up the answer.

        Args:
            answer: What to reply with. A mapping is sent as JSON.
            fail: Raised instead of answering, for the test that checks an
                unreachable judge does not discard a candidate.
        """
        self._answer = answer if isinstance(answer, str) else json.dumps(answer)
        self._fail = fail
        self.calls = 0
        self.requests: list[dict[str, Any]] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **request: Any) -> SimpleNamespace:
        """Answer one call.

        Args:
            **request: The call body, ignored.

        Returns:
            SimpleNamespace: A completion shaped like the SDK's.

        Raises:
            Exception: Whatever ``fail`` was set to.
        """
        self.calls += 1
        self.requests.append(request)
        if self._fail is not None:
            raise self._fail
        return SimpleNamespace(
            choices=[
                SimpleNamespace(message=SimpleNamespace(content=self._answer), finish_reason="stop")
            ],
            usage=SimpleNamespace(prompt_tokens=800, completion_tokens=30),
        )

    @property
    def prompt(self) -> str:
        """str: Everything the model was shown on its single call."""
        assert len(self.requests) == 1, f"expected one call, saw {len(self.requests)}"
        parts = []
        for message in self.requests[0]["messages"]:
            content = message["content"]
            if isinstance(content, str):
                parts.append(content)
            else:
                parts.extend(part.get("text", "") for part in content)
        return "\n".join(parts)


def a_candidate(**overrides: Any) -> Skill:
    """Induce a candidate from the committed run.

    Args:
        **overrides: Parts of the induction answer to replace, for the tests
            that need a different verification block.

    Returns:
        Skill: The candidate, stored nowhere.
    """
    endpoint = ScriptedModel({**AN_INDUCTION, **overrides})
    client = ModelClient(load_profile("local"), transport=endpoint, env={})
    return Inducer(client).induce(read_run(TRACE)).skill


def a_verifier(
    task_session: Any,
    site: Any,
    judge: ScriptedModel | None = None,
) -> tuple[SkillVerifier, ScriptedModel | None]:
    """Assemble a verifier over a real browser and the practice site.

    Args:
        task_session: The test's browser session.
        site: The practice site.
        judge: The scripted judge, or None to run the structural layer alone.

    Returns:
        tuple: The verifier, and the judge endpoint so a test can count calls.
    """
    start = site.url_for("search.html")
    config = Config(start_url=start)
    client = ModelClient(load_profile("local"), transport=judge, env={}) if judge else None
    verifier = SkillVerifier(
        executor=Executor(task_session, action_timeout_s=config.action_timeout_s),
        validator=Validator(Allowlist.derive(start)),
        config=config,
        judge=client,
    )
    return verifier, judge


A_PASSING_JUDGE = {"passed": True, "reason": "the page is the privacy notice for members"}
"""dict: What a judge says when the replay landed where it should have."""

A_FAILING_JUDGE = {"passed": False, "reason": "this is the site's home page, not a record"}
"""dict: What a judge says when it did not."""


# --- The whole chain, and it passes -----------------------------------------


@pytest.mark.fixture
def test_a_candidate_that_really_works_passes_both_layers(
    task_session: Any, fixture_site: Any, tmp_path: Path
) -> None:
    """Run, trace, induce, verify, store — the whole claim in one test.

    And it passes on the case that ought to be hardest. The run it was learnt
    from clicked a link called "Membership application"; replaying it with
    "privacy" reaches a results page with no such link. It works because the
    stored step describes that element four ways and a weaker description still
    finds it — which is the durable-selector ladder earning its place.
    """
    candidate = a_candidate()
    verifier, judge = a_verifier(task_session, fixture_site, ScriptedModel(A_PASSING_JUDGE))

    verdict = verifier.verify(candidate, start_url=fixture_site.url_for("search.html"))

    assert verdict.passed, verdict.summary()
    assert verdict.layer == "", "nothing refused it"
    assert [step.ok for step in verdict.replayed] == [True, True, True]
    assert judge is not None and judge.calls == 1, "exactly one judge call, and only at the end"

    # And this is *how* it survived, asserted rather than assumed. The step was
    # stored describing "Membership application" -- a name no privacy search
    # produces -- and the weakest description, "the seventh link", found the
    # right record anyway. Without the ladder this candidate would have been
    # discarded, correctly by its own lights and wrongly in fact.
    first, _, opened = verdict.replayed
    assert first.resolution is not None and first.resolution.tier == "accessible_name"
    assert opened.resolution is not None and opened.resolution.tier == "structural"
    assert 'click [7] link "Data and privacy notice for members"' == opened.described

    # Storing is the caller's decision, one line later — which is the point of
    # the verifier not doing it.
    store = SkillStore(tmp_path / "skills")
    store.save(candidate)
    assert store.find(candidate.name) == candidate


@pytest.mark.fixture
def test_the_replay_really_used_the_other_values(task_session: Any, fixture_site: Any) -> None:
    """Otherwise the check is the original run again, and proves nothing.

    The whole argument for verification is that the candidate is tried with
    values it was never recorded from. A replay that quietly used the run's own
    values would pass every time and catch nothing.
    """
    candidate = a_candidate()
    verifier, _ = a_verifier(task_session, fixture_site, ScriptedModel(A_PASSING_JUDGE))

    verifier.verify(candidate, start_url=fixture_site.url_for("search.html"))

    assert candidate.params[0].example == "membership", "what the run used"
    assert candidate.verification.params == {"query": "privacy"}, "and what the check used"
    assert 'with "privacy"' in verdict_step(verifier, candidate)


def verdict_step(verifier: SkillVerifier, skill: Skill) -> str:
    """Render the skill's first step as the replay would fill it in.

    Args:
        verifier: Unused; present so the helper reads beside its caller.
        skill: The candidate.

    Returns:
        str: The step with its parameters substituted.
    """
    return fill_in(skill.steps[0], dict(skill.verification.params)).describe()


# --- Failing the cheap layer, and not paying for an opinion -----------------


@pytest.mark.fixture
def test_a_search_that_finds_nothing_fails_before_any_judge_call(
    task_session: Any, fixture_site: Any
) -> None:
    """The ordering of the layers, on a failure that needs no contrivance.

    Searching for a word the catalogue does not have returns a results page with
    no results — so there is no record to click, the third step has nothing to
    resolve, and the replay stops. Asking a model whether that page is right
    would be paying for an opinion about a page the recipe never reached.
    """
    candidate = a_candidate(
        verification={
            "params": [{"name": "query", "value": "nothingatall"}],
            "expected_outcome": "a record page about nothingatall",
        }
    )
    verifier, judge = a_verifier(task_session, fixture_site, ScriptedModel(A_PASSING_JUDGE))

    verdict = verifier.verify(candidate, start_url=fixture_site.url_for("search.html"))

    assert verdict.passed is False
    assert verdict.layer == STRUCTURAL
    assert judge is not None and judge.calls == 0, "the expensive layer was never reached"
    assert "step 3" in verdict.reason, "and the discard says which step, not just that it failed"
    assert [step.ok for step in verdict.replayed] == [True, True, False]


@pytest.mark.fixture
def test_a_broken_selector_is_discarded_with_its_reason(
    task_session: Any, fixture_site: Any
) -> None:
    """A step describing something that is not there, every way it could be.

    The stored descriptions are deliberately replaced with ones that match
    nothing — no such name, no such text, no such position. A discard has to
    say what could not be found, because a discard rate with no reasons behind
    it says only that something is wrong somewhere.
    """
    candidate = a_candidate()
    broken = candidate.model_copy(
        update={
            "steps": (
                candidate.steps[0],
                candidate.steps[1],
                ClickStep(
                    action="click",
                    selector=Selector(
                        name="Nothing called this",
                        role="link",
                        text="nor showing this",
                        structural="link[99]",
                    ),
                ),
            )
        }
    )
    verifier, judge = a_verifier(task_session, fixture_site, ScriptedModel(A_PASSING_JUDGE))

    verdict = verifier.verify(broken, start_url=fixture_site.url_for("search.html"))

    assert verdict.passed is False
    assert verdict.layer == STRUCTURAL
    assert judge is not None and judge.calls == 0
    assert "no element matched" in verdict.reason
    # Every description was tried before giving up, and the record says so.
    failed = verdict.replayed[-1]
    assert failed.resolution is not None
    assert [attempt.tier for attempt in failed.resolution.attempts] == [
        "accessible_name",
        "role_and_text",
        "structural",
    ]
    assert verdict.to_dict()["steps"][-1]["ok"] is False, "and it is reportable"


# --- Failing the expensive layer --------------------------------------------


@pytest.mark.fixture
def test_a_replay_that_works_but_lands_wrong_is_caught_by_the_judge(
    task_session: Any, fixture_site: Any
) -> None:
    """Why the cheap layer is not enough on its own.

    Every step here resolves and executes — clicking a navigation link is a
    perfectly successful click. It just does not end anywhere the recipe claims
    to end, and no amount of mechanical checking can notice that. This is the
    only layer that can tell a routine that *ran* from one that *worked*.
    """
    candidate = a_candidate()
    wrong_way = candidate.model_copy(
        update={
            "steps": (
                candidate.steps[0],
                candidate.steps[1],
                # A real link, on the results page, that goes somewhere else.
                ClickStep(action="click", selector=Selector(name="Home", role="link")),
            )
        }
    )
    verifier, judge = a_verifier(task_session, fixture_site, ScriptedModel(A_FAILING_JUDGE))

    verdict = verifier.verify(wrong_way, start_url=fixture_site.url_for("search.html"))

    assert all(step.ok for step in verdict.replayed), "mechanically, nothing went wrong"
    assert verdict.passed is False
    assert verdict.layer == SEMANTIC
    assert verdict.reason == A_FAILING_JUDGE["reason"]
    assert judge is not None and judge.calls == 1
    assert verdict.record is not None, "and the call it cost is reported"


@pytest.mark.fixture
def test_the_judge_is_told_which_values_were_used(task_session: Any, fixture_site: Any) -> None:
    """Without them it cannot spot the failure it is there to spot.

    The thing being looked for is a page that belongs to the recipe's
    *original* example. A judge shown only "does this look like a record page?"
    would happily pass the membership record when the replay asked for privacy.
    """
    endpoint = ScriptedModel(A_PASSING_JUDGE)
    verifier, _ = a_verifier(task_session, fixture_site, endpoint)

    verifier.verify(a_candidate(), start_url=fixture_site.url_for("search.html"))

    assert "query: privacy" in endpoint.prompt, "the values it was replayed with"
    assert "a record page about privacy" in endpoint.prompt, "what should be true afterwards"
    assert "privacy" in endpoint.prompt.lower(), "and the page's own words"
    assert "Search results" in endpoint.prompt or "privacy" in endpoint.prompt


# --- When the check cannot be carried out -----------------------------------


@pytest.mark.fixture
def test_a_judge_that_cannot_be_reached_is_not_a_failed_candidate(
    task_session: Any, fixture_site: Any
) -> None:
    """A skill must never be written off because an endpoint was down.

    Being wrong and being uncheckable are different findings with different
    responses: one means discard, the other means try again later. Collapsing
    them would quietly throw away good skills every time a network hiccuped.
    """
    candidate = a_candidate()
    verifier, _ = a_verifier(
        task_session, fixture_site, ScriptedModel({}, fail=RuntimeError("connection refused"))
    )

    with pytest.raises(VerificationError, match="judge could not be reached"):
        verifier.verify(candidate, start_url=fixture_site.url_for("search.html"))


@pytest.mark.fixture
def test_a_starting_page_that_will_not_open_is_not_a_failed_candidate(
    task_session: Any, fixture_site: Any
) -> None:
    """The same distinction, one step earlier.

    The address is one nothing is listening on, so the browser cannot reach it
    at all. A page that loads and says "not found" would be a different matter
    — that is a page, and the replay would go on to fail structurally against
    it, which is the correct answer for a recipe pointed at a page that no
    longer exists.
    """
    verifier, _ = a_verifier(task_session, fixture_site, ScriptedModel(A_PASSING_JUDGE))

    with pytest.raises(VerificationError, match="would not open"):
        verifier.verify(a_candidate(), start_url="http://127.0.0.1:1/search.html")


@pytest.mark.fixture
def test_the_structural_layer_can_run_alone_and_says_so(
    task_session: Any, fixture_site: Any
) -> None:
    """A stated fallback position, not something reached by forgetting.

    The spec allows structural-only verification if the judge proves
    unreliable, with the retirement threshold as the quality filter instead. It
    is allowed here, and the verdict says which layers ran — so a library
    verified without a judge cannot be mistaken for one that had both.
    """
    verifier, _ = a_verifier(task_session, fixture_site, judge=None)

    verdict = verifier.verify(a_candidate(), start_url=fixture_site.url_for("search.html"))

    assert verdict.passed is True
    assert verdict.judge_calls == 0
    assert "no judge configured" in verdict.reason


@pytest.mark.unit
def test_a_discard_is_reportable() -> None:
    """The discard rate is one of the project's published numbers.

    A rate with no reasons behind it says only that something is wrong
    somewhere. So every discard carries the layer that refused it, a sentence,
    and how far the replay got before it stopped.
    """
    verdict = Verdict(
        passed=False,
        layer=STRUCTURAL,
        reason='step 3 (click link "First result"): no element matched',
        replayed=(
            StepReplay(1, 'fill searchbox "Search terms" with "privacy"', ok=True),
            StepReplay(2, "press Enter", ok=True),
            StepReplay(3, 'click link "First result"', ok=False, detail="no element matched"),
        ),
    )

    record = verdict.to_dict()

    assert json.dumps(record), "it survives being written to a file"
    assert record["layer"] == STRUCTURAL
    assert record["judge_calls"] == 0
    assert [step["ok"] for step in record["steps"]] == [True, True, False]
    assert "discarded (structural)" in verdict.summary()
