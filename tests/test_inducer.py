"""Turning one successful run into a recipe, and refusing to turn anything else.

Induction is the project's actual claim, so the tests are less about the code
working than about *where the judgment is allowed to be*. The model is asked
four questions — which values were specific to this task, what to call the
skill, what it does, how to check it — and everything else is copied from the
run. That division is what most of this file pins.

**The run is real and committed.** ``tests/fixtures/traces/search-and-open`` was
produced by the agent against the practice site, and is checked in so that
induction is asked about the same route every time. It was chosen because the
word "membership" appears in it twice, once typed and once carried into the
address of the next page — so a skill that blanked out only the typed one would
still look completely correct, and be wrong for every other search.

**The model is scripted.** What a real model would answer is a question the
benchmark gets to settle; what matters here is what the code does with an
answer. So the endpoint is a stub, and the interesting tests hand it answers
that are subtly wrong on purpose — a parameter for a value that is not in the
run, a verification block reusing the run's own values — to check the format
catches what the model cannot be relied on to get right.

**Refusals get their own section.** Only successful runs may be learnt from, and
"it only failed at the last step" is exactly the case where nobody can say which
earlier step doomed it. A recipe made from a failure is a recipe for failing.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import ValidationError

from web_action_agent.inducer import Inducer, InductionError, induction_schema
from web_action_agent.model_client import ModelClient, load_profile
from web_action_agent.skill_schema import ClickStep, FillStep, PressStep
from web_action_agent.trace_reader import ReplayedRun, read_run

TRACE = Path(__file__).resolve().parent / "fixtures" / "traces" / "search-and-open"
"""Path: The committed successful run every test here induces from."""

#: What a good answer to the four questions looks like for that run. The two
#: occurrences of "membership" are the point: one value, named once, replaced
#: wherever it appears.
A_GOOD_ANSWER: dict[str, Any] = {
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


class ScriptedEndpoint:
    """A model endpoint that answers with a fixed JSON object.

    Attributes:
        requests: Every request the client built, so a test can read what the
            model was actually shown.
    """

    def __init__(self, answer: dict[str, Any] | str) -> None:
        """Set up the answer.

        Args:
            answer: What to reply with. A mapping is sent as JSON; a string is
                sent as-is, for the test that checks prose is refused.
        """
        self._answer = answer if isinstance(answer, str) else json.dumps(answer)
        self.requests: list[dict[str, Any]] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **request: Any) -> SimpleNamespace:
        """Record a request and return the scripted reply.

        Args:
            **request: The call body.

        Returns:
            SimpleNamespace: A completion shaped like the SDK's.
        """
        self.requests.append(request)
        return SimpleNamespace(
            choices=[
                SimpleNamespace(message=SimpleNamespace(content=self._answer), finish_reason="stop")
            ],
            usage=SimpleNamespace(prompt_tokens=1200, completion_tokens=90),
        )

    @property
    def prompt(self) -> str:
        """str: Everything the model was shown, as one string."""
        assert len(self.requests) == 1, f"expected one call, saw {len(self.requests)}"
        return "\n".join(
            message["content"]
            for message in self.requests[0]["messages"]
            if isinstance(message.get("content"), str)
        )


def an_inducer(answer: dict[str, Any] | str | None = None) -> tuple[Inducer, ScriptedEndpoint]:
    """Build an inducer over a scripted endpoint.

    Args:
        answer: What the model should reply. Defaults to a good answer.

    Returns:
        tuple[Inducer, ScriptedEndpoint]: The inducer, and the endpoint so a
        test can read what was sent.
    """
    endpoint = ScriptedEndpoint(A_GOOD_ANSWER if answer is None else answer)
    client = ModelClient(load_profile("local"), transport=endpoint, env={})
    return Inducer(client), endpoint


def the_run() -> ReplayedRun:
    """Read the committed successful run.

    Returns:
        ReplayedRun: The run every test induces from.
    """
    return read_run(TRACE)


def a_run_that(tmp_path: Path, **outcome: Any) -> ReplayedRun:
    """Copy the committed run, changing how it ended.

    Args:
        tmp_path: Somewhere to put the copy.
        **outcome: Fields of the verdict to replace. An empty mapping removes
            the verdict file entirely, which is what an interrupted run leaves.

    Returns:
        ReplayedRun: The altered run.
    """
    folder = tmp_path / "run"
    folder.mkdir()
    for path in TRACE.glob("*.json*"):
        (folder / path.name).write_text(path.read_text(encoding="utf-8"), encoding="utf-8")

    verdict_file = folder / "outcome.json"
    if not outcome:
        verdict_file.unlink()
        return read_run(folder)
    verdict = json.loads(verdict_file.read_text(encoding="utf-8"))
    verdict["outcome"] = {**verdict["outcome"], **outcome}
    verdict_file.write_text(json.dumps(verdict), encoding="utf-8")
    return read_run(folder)


# --- What a good induction produces -----------------------------------------


@pytest.mark.mock_llm
def test_a_successful_run_becomes_a_valid_skill() -> None:
    """The whole claim, end to end: a run in, a storable recipe out.

    Schema-valid is the bar, and it is a real one — the skill format refuses a
    parameter no step uses, a placeholder nothing declares, and a verification
    block that could not be run. Getting a valid skill out means all of those
    line up.
    """
    inducer, _ = an_inducer()

    induced = inducer.induce(the_run())

    assert induced.skill.name == "search_and_open_record"
    assert induced.skill.domain == "northgate.example", "taken from where the run started"
    assert induced.skill.required_params == ("query",)
    assert induced.skill.induced_from == the_run().run_id, "traceable to the run it came from"
    assert induced.notes == (), "nothing was dropped from this route"


@pytest.mark.mock_llm
def test_no_numbered_element_survives_into_the_skill() -> None:
    """The reason durable selectors exist, checked on a real induction.

    ``click 7`` is meaningless by the time a skill replays it, so every step
    that named an element has to come out describing one instead. A single
    number left behind would be a step that silently acts on whatever happens
    to be seventh next month.
    """
    inducer, _ = an_inducer()

    steps = inducer.induce(the_run()).skill.steps

    assert [step.action for step in steps] == ["fill", "press", "click"]
    for step in steps:
        assert not hasattr(step, "element_id")
    fill, _, click = steps
    assert isinstance(fill, FillStep) and isinstance(click, ClickStep)
    assert fill.selector.name == "Search terms", "described by what the page calls it"
    assert click.selector.name == "Membership application"
    assert click.selector.structural == "link[7]", "and by every weaker description too"


@pytest.mark.mock_llm
def test_the_value_that_was_typed_becomes_a_blank() -> None:
    """What the model names as task-specific is what the code blanks out.

    The whole field was the search term, so the whole field becomes the
    placeholder and nothing of the original survives in it. A skill that still
    typed "membership" would be a skill for one search wearing the costume of a
    skill for searching.
    """
    inducer, _ = an_inducer()

    skill = inducer.induce(the_run()).skill

    fill = skill.steps[0]
    assert isinstance(fill, FillStep)
    assert fill.text == "{query}", "the whole field is the parameter, so it is only the parameter"
    assert skill.parameters_used() == ("query",)
    assert skill.params[0].example == "membership", "kept as the example, which is what it is"


@pytest.mark.mock_llm
def test_a_step_can_still_name_something_the_run_only_found_by_chance() -> None:
    """A real limit of induction, pinned so it is a known thing rather than a surprise.

    The third step clicks the link called "Membership application" — which is
    not a value anybody typed, it is *what the search returned*. So the induced
    skill reads as though it searches for anything and then opens whatever it
    finds, while actually looking for that one link by name.

    Nothing in induction can see this. The step is a faithful record of what
    the run did, the model was never shown a second search to compare against,
    and the result is schema-valid.

    **This is exactly what the verification block is for.** Replayed with
    ``query = "privacy"``, the results page has no "Membership application" to
    click, the step resolves to nothing, and the candidate is discarded before
    it can enter the library. Induction proposes; verification disposes. The
    test is here to say that the gap is known and is somebody else's job,
    rather than to pretend it does not exist.
    """
    inducer, _ = an_inducer()

    click = inducer.induce(the_run()).skill.steps[2]

    assert isinstance(click, ClickStep)
    assert click.selector.name == "Membership application", "the run's own result, by name"
    assert "{" not in click.selector.name, "selectors carry no placeholders; only typed text does"
    # And the weaker descriptions are stored too, which is what gives a replay
    # any chance at all: "the seventh link" survives a different search where
    # the name does not.
    assert click.selector.structural == "link[7]"


@pytest.mark.mock_llm
def test_a_step_that_names_no_element_is_copied_across_untouched() -> None:
    """Pressing a key needs no description and gets none.

    Which key submits a form is a fact about the page, not about the task, so
    it is neither parameterised nor described — it is simply the same key
    again.
    """
    inducer, _ = an_inducer()

    press = inducer.induce(the_run()).skill.steps[1]

    assert isinstance(press, PressStep)
    assert press.key == "Enter"


@pytest.mark.mock_llm
def test_the_verification_block_comes_from_the_model_and_is_required() -> None:
    """A candidate arrives with the means of checking it, or it does not arrive.

    The block is what will let the next component replay the skill with values
    it was never recorded from — the difference between "this worked once" and
    "this works".
    """
    inducer, _ = an_inducer()

    skill = inducer.induce(the_run()).skill

    assert skill.verification.params == {"query": "privacy"}
    assert "privacy" in skill.verification.expected_outcome
    assert skill.verification.params["query"] != "membership", "different from the run's own value"


@pytest.mark.mock_llm
def test_the_candidate_is_not_stored_anywhere() -> None:
    """Induction proposes; it does not admit anything to the library.

    A skill that went straight into the library on the strength of the run it
    came from would never be checked, and the verification block would be
    decoration.
    """
    inducer, _ = an_inducer()

    induced = inducer.induce(the_run())

    assert induced.skill.stats.uses == 0
    assert induced.skill.active is True, "a candidate, pending a check — not a retired skill"
    assert induced.record.prompt_tokens == 1200, "and the call it cost is reported to the caller"


# --- What the model is shown ------------------------------------------------


@pytest.mark.mock_llm
def test_the_model_sees_the_route_with_elements_named_not_numbered() -> None:
    """Showing numbers would invite an answer that refers to one.

    The numbering is rebuilt every time the agent looks at a page, so a number
    in the prompt is a fact with a shelf life of one observation. What the model
    needs is what a person watching would have seen.
    """
    inducer, endpoint = an_inducer()

    inducer.induce(the_run())

    assert 'fill [7] searchbox "Search terms" with "membership"' in endpoint.prompt
    assert 'click [7] link "Membership application"' in endpoint.prompt
    assert "find the record about membership and open it" in endpoint.prompt, "the task"
    assert "northgate.example" in endpoint.prompt, "and which site it happened on"


@pytest.mark.mock_llm
def test_the_model_is_asked_for_judgment_and_nothing_it_could_invent() -> None:
    """The division this module exists to enforce.

    A model asked for a skill file would be asked to reproduce accessible names
    it has read once — and would occasionally produce a plausible name for a
    button that says something else, which nothing downstream could catch. So
    the answer it is allowed to give has room for four things and no more.
    """
    schema = induction_schema()

    assert set(schema["required"]) == {"name", "description", "parameters", "verification"}
    assert "steps" not in schema["properties"], "the route is known, not asked for"
    assert "selector" not in json.dumps(schema), "and so is every element in it"


@pytest.mark.mock_llm
def test_a_reply_that_is_not_a_skill_description_is_refused() -> None:
    """Asked once, got prose. There is nothing partial worth keeping."""
    inducer, _ = an_inducer("I think this run was about searching, probably.")

    with pytest.raises(InductionError, match="not a skill description"):
        inducer.induce(the_run())


# --- What the format catches that the model gets wrong ----------------------


@pytest.mark.mock_llm
def test_a_parameter_for_a_value_not_in_the_run_is_dropped_and_said() -> None:
    """The likeliest model mistake, and it is corrected out loud, not quietly.

    Naming a blank for a value that was never typed leaves a parameter no step
    uses. Kept, it would make the skill claim to need something it never uses
    -- the planner reads that list to decide whether it can supply what the
    skill needs. So it is dropped, with its verification value, and the notes
    say which and why.
    """
    inducer, _ = an_inducer(
        {
            **A_GOOD_ANSWER,
            "parameters": [
                *A_GOOD_ANSWER["parameters"],
                {
                    "name": "branch",
                    "value": "Hollow Lane",
                    "description": "never typed during this run",
                    "required": True,
                },
            ],
            "verification": {
                "params": [
                    {"name": "query", "value": "privacy"},
                    {"name": "branch", "value": "Northgate"},
                ],
                "expected_outcome": "a record page about privacy",
            },
        }
    )

    induction = inducer.induce(the_run())

    assert [p.name for p in induction.skill.params] == ["query"]
    assert induction.skill.verification.params == {"query": "privacy"}
    assert any("'branch'" in note and "'Hollow Lane'" in note for note in induction.notes)


@pytest.mark.mock_llm
def test_a_verification_missing_a_value_is_caught() -> None:
    """A block that could not be run is the same as not having one.

    Except that it looks like having one, which is worse: the candidate would
    reach verification and fail there for a reason that has nothing to do with
    whether the skill works.
    """
    inducer, _ = an_inducer(
        {**A_GOOD_ANSWER, "verification": {"params": [], "expected_outcome": "a record page"}}
    )

    with pytest.raises(ValidationError, match="query"):
        inducer.induce(the_run())


@pytest.mark.mock_llm
def test_a_parameter_whose_value_contains_another_is_replaced_whole() -> None:
    """Two blanks where one value is inside the other, which a naive pass mangles.

    Replacing "reading" first would turn "reading room" into "{shelf} room" and
    leave a placeholder buried in a literal. Longest first is the fix, and it
    is worth a test because the bug it prevents produces a skill that is valid,
    storable, and subtly wrong.
    """
    inducer, _ = an_inducer(
        {
            **A_GOOD_ANSWER,
            "parameters": [
                {
                    "name": "query",
                    "value": "membership",
                    "description": "the subject",
                    "required": True,
                },
                {
                    "name": "letter",
                    "value": "member",
                    "description": "a prefix inside the other value",
                    "required": False,
                },
            ],
        }
    )

    # `letter` is never used once `query` has taken the whole field, so it is
    # dropped -- which proves the longer value won: had `member` been
    # replaced first, the field would read `{letter}ship`.
    induction = inducer.induce(the_run())

    assert [p.name for p in induction.skill.params] == ["query"]
    assert any("'letter'" in note for note in induction.notes)
    assert "{letter}" not in json.dumps(induction.skill.to_dict())


@pytest.mark.mock_llm
def test_a_run_that_only_clicked_makes_a_skill_with_no_parameters() -> None:
    """As in a live rehearsal: a one-click run, and a model that invents inputs.

    The run typed nothing, so nothing in it is task-specific. The model read
    facts off the page anyway and proposed them as parameters -- an age group
    and a wage threshold, in the run that found this. None can be used, so the
    candidate keeps the route and no parameters, and is still a valid skill.
    """
    run = the_run()
    # Keep only the click and the finish: no typing anywhere in the run.
    clicked = dataclasses.replace(
        run,
        steps=tuple(
            s for s in run.steps if s.action is not None and s.action.action in {"click", "done"}
        ),
    )
    inducer, _ = an_inducer(
        {
            "name": "open_membership_record",
            "description": "open the membership record",
            "parameters": [
                {
                    "name": "age_group",
                    "value": "55 and below",
                    "description": "x",
                    "required": True,
                },
                {"name": "wage_threshold", "value": "$750", "description": "y", "required": False},
            ],
            "verification": {
                "params": [{"name": "age_group", "value": "above 55 to 60"}],
                "expected_outcome": "the membership record opens",
            },
        }
    )

    induction = inducer.induce(clicked)

    assert induction.skill.params == ()
    assert [step.action for step in induction.skill.steps] == ["click"]
    assert induction.skill.verification.params == {}
    assert len([n for n in induction.notes if "was dropped" in n]) == 2


@pytest.mark.mock_llm
def test_the_model_is_shown_the_only_values_that_can_become_parameters() -> None:
    inducer, endpoint = an_inducer()

    inducer.induce(the_run())

    assert "## Values the run typed or visited\n\n- 'membership'" in endpoint.prompt


@pytest.mark.mock_llm
def test_a_run_that_typed_nothing_is_said_to_have_no_parameters() -> None:
    run = the_run()
    clicked = dataclasses.replace(
        run,
        steps=tuple(
            s for s in run.steps if s.action is not None and s.action.action in {"click", "done"}
        ),
    )
    inducer, endpoint = an_inducer(
        {
            **A_GOOD_ANSWER,
            "parameters": [],
            "verification": {"params": [], "expected_outcome": "the membership record page"},
        }
    )

    inducer.induce(clicked)

    assert "(none: this run only clicked, scrolled or pressed keys" in endpoint.prompt


@pytest.mark.unit
def test_the_instructions_say_where_parameters_come_from() -> None:
    instructions = an_inducer()[0].system_prompt

    assert 'Parameters come only from "Values the run typed or visited"' in instructions
    assert "With no parameters," in instructions


# --- Runs that may not be learnt from ---------------------------------------


@pytest.mark.unit
def test_a_failed_run_is_refused(tmp_path: Path) -> None:
    """Only successful runs are eligible, and this is the whole rule.

    A recipe made from a failure is a recipe for failing. The tempting
    exception — "it only failed at the last step, the earlier ones were fine" —
    is exactly the case where nobody can say which earlier step doomed it.
    """
    inducer, endpoint = an_inducer()

    with pytest.raises(InductionError, match="only successful runs"):
        inducer.induce(a_run_that(tmp_path, category="step_cap", ok=False, answer=None))

    assert endpoint.requests == [], "and it was refused before anything was spent"


@pytest.mark.unit
@pytest.mark.parametrize("category", ["gave_up", "validation_exhausted", "run_timeout"])
def test_every_way_of_not_succeeding_is_refused(tmp_path: Path, category: str) -> None:
    """Not just the obvious failure. Eleven ways a run can end, one is eligible.

    Args:
        tmp_path: Somewhere for the altered copy.
        category: A verdict that is not success.
    """
    inducer, _ = an_inducer()

    with pytest.raises(InductionError, match=category):
        inducer.induce(a_run_that(tmp_path, category=category, ok=False, answer=None))


@pytest.mark.unit
def test_an_interrupted_run_is_refused(tmp_path: Path) -> None:
    """No verdict is not the same as a bad verdict, and neither may be learnt from.

    A run that was killed partway leaves a readable folder on purpose — that is
    what flushing each step is for — but nothing in it says the job was done.
    """
    inducer, _ = an_inducer()

    with pytest.raises(InductionError, match="never reached a verdict"):
        inducer.induce(a_run_that(tmp_path))
