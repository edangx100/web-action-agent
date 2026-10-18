"""One step of thinking: the page in, one typed action out, or a named failure.

The planner is the only place a language model gets a say in what happens, so
this file is mostly about what it does when the model is unhelpful. A model
that answers in prose, or with an action that does not exist, is not an
exceptional event -- one of the shipped vision routes cannot constrain decoding
at all, so on those steps a stray sentence of preamble is an ordinary Tuesday.

What must never happen is a *partial* reading. There is no state in which the
planner returns "an action, sort of": either the reply was one of the seven, or
the decision carries a failure and no action at all. Half of the tests below
are that one property, approached from different directions.

The rest cover the prompt. It is assembled fresh each step from five pieces, and
a prompt that is subtly wrong -- a stale element list, history that grows without
bound, a truncated page that does not say it was truncated -- produces plausible
actions that are wrong for reasons no amount of staring at the planner code
would reveal. So the prompt is asserted on directly, rather than inferred from
the actions it happens to produce.

Nothing here reaches a network. The transport is a stub that returns whatever
reply the test wants to see handled.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from test_model_client import without_vision_schemas

from web_action_agent.model_client import ModelClient, load_profile
from web_action_agent.observation import Observation
from web_action_agent.observer import Box, Element
from web_action_agent.planner import (
    FAILURE_NOT_AN_ACTION,
    FAILURE_NOT_JSON,
    MAX_HISTORY_STEPS,
    Planner,
    render_history,
)

#: A profile whose text route honours schema-constrained decoding, which is the
#: ordinary case and the one most tests want.
CONSTRAINED_PROFILE = "openrouter"

#: The profile with no vision route, for the half of the screenshot tests that
#: check a text-only endpoint degrades rather than fails.
TEXT_ONLY_PROFILE = "local"

#: Stand-in for an annotated screenshot. Nothing decodes it.
A_PNG = b"\x89PNG\r\n\x1a\n fake bytes"

#: A well-formed reply, for tests whose subject is something other than parsing.
A_CLICK = '{"action": "click", "element_id": 2}'


class StubTransport:
    """An ``openai.OpenAI`` lookalike returning scripted replies.

    Attributes:
        requests: Every request the client built, in order.
    """

    def __init__(self, *replies: str, usage: tuple[int, int] | None = (900, 20)) -> None:
        """Set up a transport that answers with the given replies in turn.

        Args:
            *replies: Message contents to return, one per call. The last is
                repeated if more calls are made than replies given.
            usage: Prompt and completion token counts to report, or None to
                report no usage at all, as a local server does.
        """
        self._replies = list(replies) or [A_CLICK]
        self._usage = usage
        self.requests: list[dict[str, Any]] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **request: Any) -> SimpleNamespace:
        """Record a request and return the next scripted reply.

        Args:
            **request: The keyword arguments the client passed.

        Returns:
            SimpleNamespace: A response shaped like the SDK's.
        """
        self.requests.append(request)
        index = min(len(self.requests) - 1, len(self._replies) - 1)
        usage = (
            None
            if self._usage is None
            else SimpleNamespace(prompt_tokens=self._usage[0], completion_tokens=self._usage[1])
        )
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=self._replies[index]),
                    finish_reason="stop",
                )
            ],
            usage=usage,
        )

    @property
    def request(self) -> dict[str, Any]:
        """The single request made, for tests that make exactly one.

        Returns:
            dict[str, Any]: The recorded request.
        """
        assert len(self.requests) == 1, f"expected one call, saw {len(self.requests)}"
        return self.requests[0]


def an_observation(
    *,
    elements: int = 3,
    text: str = "We are open 09:00 to 17:30, Monday to Friday.",
    truncated: bool = False,
) -> Observation:
    """Build an observation to plan against.

    Args:
        elements: How many interactive elements the page offers, numbered from 1.
        text: The distilled page text.
        truncated: Whether that text was cut to fit its budget.

    Returns:
        Observation: A page with plausible contents.
    """
    return Observation(
        url="http://localhost:8000/contact.html",
        title="Contact us",
        elements=tuple(
            Element(
                id=number,
                tag="a",
                role="link",
                name=f"Link {number}",
                text=f"Link {number}",
                box=Box(x=0, y=20 * number, width=80, height=18),
            )
            for number in range(1, elements + 1)
        ),
        text=text,
        text_truncated=truncated,
    )


def planner_for(
    profile: str = CONSTRAINED_PROFILE,
    transport: StubTransport | None = None,
    **kwargs: Any,
) -> Planner:
    """Build a planner on a shipped profile and a stub transport.

    Args:
        profile: Profile name, which decides schema support and vision.
        transport: The stub to record into. A fresh one by default.
        **kwargs: Passed to :class:`Planner`.

    Returns:
        Planner: The planner. The environment is empty on purpose -- with a stub
        transport no key is needed, and none is read.
    """
    client = ModelClient(load_profile(profile), transport=transport or StubTransport(), env={})
    return Planner(client, **kwargs)


def user_text(request: dict[str, Any]) -> str:
    """Pull the text of the user message out of a recorded request.

    Args:
        request: One recorded request.

    Returns:
        str: The user message's text, whether it was sent as a bare string or as
        the text part of a multi-part message carrying an image.
    """
    content = request["messages"][-1]["content"]
    if isinstance(content, str):
        return content
    return "\n".join(part["text"] for part in content if part["type"] == "text")


# --- One good step ----------------------------------------------------------


@pytest.mark.mock_llm
def test_a_well_formed_reply_becomes_a_typed_action() -> None:
    """The ordinary case: the model answers with an action and gets one back."""
    decision = planner_for().decide(task="find the opening hours", observation=an_observation())

    assert decision.ok
    assert decision.action is not None
    assert decision.action.describe() == "click 2"
    assert decision.failure is None
    assert decision.summary() == "click 2"


@pytest.mark.mock_llm
def test_the_schema_constrains_decoding_where_the_route_supports_it() -> None:
    """The vocabulary reaches the endpoint, so a bad action cannot be generated.

    This is the first of the two defences. The second -- parsing the reply
    anyway -- is tested below, and both are needed because not every route can
    do this one.
    """
    transport = StubTransport()
    planner_for(transport=transport).decide(task="t", observation=an_observation())

    response_format = transport.request["response_format"]
    assert response_format["type"] == "json_schema"
    offered = response_format["json_schema"]["schema"]["$defs"]
    assert {branch["properties"]["action"]["const"] for branch in offered.values()} == {
        "click",
        "fill",
        "press",
        "scroll",
        "navigate",
        "use_skill",
        "done",
    }


@pytest.mark.mock_llm
def test_the_trace_records_whether_decoding_was_actually_constrained() -> None:
    """Whether the schema was enforced or merely hoped for is worth recording.

    A run with an unusual number of parse failures is explained by these being
    False. Without the flag, the same run looks like a model that cannot follow
    instructions, and the fix would be sought in the prompt rather than in the
    routing.
    """
    constrained = planner_for().decide(task="t", observation=an_observation())
    assert constrained.record is not None
    assert constrained.record.schema_enforced is True

    # A vision route that cannot honour a schema (made up: every shipped route
    # can), so a step carrying a screenshot is decoded unconstrained and says so.
    profile = without_vision_schemas(load_profile(CONSTRAINED_PROFILE))
    client = ModelClient(profile, transport=StubTransport(), env={})
    unconstrained = Planner(client).decide(
        task="t", observation=an_observation(), screenshot_png=A_PNG
    )
    assert unconstrained.record is not None
    assert unconstrained.record.schema_enforced is False


@pytest.mark.mock_llm
def test_calls_stay_deterministic() -> None:
    """A planner that answers differently on identical input cannot be evaluated."""
    transport = StubTransport()
    planner_for(transport=transport).decide(task="t", observation=an_observation())

    assert transport.request["temperature"] == 0.0
    assert transport.request["seed"] == 0


# --- Bad replies, and the promise that none of them half-succeeds ------------


@pytest.mark.mock_llm
@pytest.mark.parametrize(
    "reply",
    [
        "I think you should click the second link.",
        "",
        "{",
        '{"action": "click", "element_id": ',
    ],
    ids=["prose", "empty", "brace", "truncated"],
)
def test_a_reply_that_is_not_json_is_a_named_failure_not_an_action(reply: str) -> None:
    """No action comes back, and the reason is a code a retry loop can read.

    The truncated case matters most: it is a reply that was *going* to be a
    valid click, and anything willing to salvage it would have to guess the
    element number.
    """
    decision = planner_for(transport=StubTransport(reply)).decide(
        task="t", observation=an_observation()
    )

    assert not decision.ok
    assert decision.action is None
    assert decision.failure is not None
    assert decision.failure.code == FAILURE_NOT_JSON
    # The reply is kept verbatim: a person reading the trace has to be able to
    # see whether the model wrote an essay or missed a brace.
    assert decision.failure.text == reply


@pytest.mark.mock_llm
def test_a_json_object_that_is_not_an_action_is_a_different_named_failure() -> None:
    """ "Not JSON" and "not one of the seven" have different fixes.

    The first is usually a route that cannot constrain decoding; the second is
    a model that has understood the format and not the vocabulary. Counting
    them together would hide which one a run actually suffered from.
    """
    decision = planner_for(transport=StubTransport('{"action": "teleport", "to": "home"}')).decide(
        task="t", observation=an_observation()
    )

    assert decision.action is None
    assert decision.failure is not None
    assert decision.failure.code == FAILURE_NOT_AN_ACTION
    assert "teleport" in decision.failure.detail


@pytest.mark.mock_llm
def test_a_missing_parameter_comes_back_naming_the_field() -> None:
    """The complaint is specific enough to be quoted straight into the retry."""
    decision = planner_for(transport=StubTransport('{"action": "click"}')).decide(
        task="t", observation=an_observation()
    )

    assert decision.failure is not None
    assert "element_id" in decision.failure.detail


@pytest.mark.mock_llm
def test_a_wasted_call_is_still_billed() -> None:
    """A reply that could not be used was still paid for.

    Dropping the record of a failed call would understate cost per task by
    exactly the calls the retry loop had to make -- which is to say, most in the
    runs where cost is worth looking at.
    """
    decision = planner_for(transport=StubTransport("no JSON here")).decide(
        task="t", observation=an_observation()
    )

    assert decision.record is not None
    assert decision.record.prompt_tokens == 900
    assert decision.record.completion_tokens == 20
    assert decision.record.latency_s >= 0.0


@pytest.mark.mock_llm
def test_json_wrapped_in_prose_is_still_read() -> None:
    """The one leniency, and the boundary of it.

    A route that cannot constrain decoding often returns the object inside a
    fenced block. Recovering the span between the outer braces is worth doing
    and is where it stops -- there is no attempt to interpret the prose itself.
    """
    reply = 'Sure! Here is the action:\n```json\n{"action": "scroll", "direction": "down"}\n```'
    decision = planner_for(transport=StubTransport(reply)).decide(
        task="t", observation=an_observation()
    )

    assert decision.action is not None
    assert decision.action.describe() == "scroll down"
    # The raw reply survives, so the trace shows this step needed salvaging.
    assert decision.text == reply


@pytest.mark.mock_llm
def test_a_failed_step_serializes_with_everything_needed_to_explain_it() -> None:
    """The trace record of a bad step carries the failure, the reply, the cost."""
    decision = planner_for(transport=StubTransport("nope")).decide(
        task="t", observation=an_observation()
    )
    record = decision.to_dict()

    assert record["action"] is None
    assert record["failure"]["code"] == FAILURE_NOT_JSON
    assert record["text"] == "nope"
    assert record["record"]["prompt_tokens"] == 900


# --- What the model is shown ------------------------------------------------


@pytest.mark.mock_llm
def test_the_instructions_go_in_a_system_message_and_the_page_in_a_user_one() -> None:
    """The standing rules are sent once per call, separately from the page."""
    transport = StubTransport()
    planner = planner_for(transport=transport)
    planner.decide(task="find the opening hours", observation=an_observation())

    messages = transport.request["messages"]
    assert [message["role"] for message in messages] == ["system", "user"]
    assert messages[0]["content"] == planner.system_prompt
    assert "find the opening hours" in user_text(transport.request)


@pytest.mark.mock_llm
def test_the_page_reaches_the_model_through_all_the_channels_it_has() -> None:
    """Elements and page text both go, because either alone answers less."""
    transport = StubTransport()
    planner_for(transport=transport).decide(
        task="t", observation=an_observation(elements=2, text="Open 09:00 to 17:30.")
    )

    prompt = user_text(transport.request)
    assert "http://localhost:8000/contact.html" in prompt
    assert '[1] link "Link 1"' in prompt
    assert '[2] link "Link 2"' in prompt
    assert "Open 09:00 to 17:30." in prompt


@pytest.mark.mock_llm
def test_a_screenshot_is_carried_where_the_profile_can_read_one() -> None:
    """The optional third channel travels when it was gated in."""
    transport = StubTransport()
    planner_for(transport=transport).decide(
        task="t", observation=an_observation(), screenshot_png=A_PNG
    )

    content = transport.request["messages"][-1]["content"]
    assert any(part["type"] == "image_url" for part in content)


@pytest.mark.mock_llm
def test_a_screenshot_on_a_text_only_endpoint_degrades_rather_than_fails() -> None:
    """The step goes ahead without the image, because text has to be enough.

    A caller that went around the screenshot gate must not be able to end the
    run; it must only be able to waste the effort of having read the file.
    """
    transport = StubTransport()
    planner = planner_for(TEXT_ONLY_PROFILE, transport=transport)

    decision = planner.decide(task="t", observation=an_observation(), screenshot_png=A_PNG)

    assert decision.ok
    assert isinstance(transport.request["messages"][-1]["content"], str)


# --- Assembling the prompt --------------------------------------------------


@pytest.mark.unit
def test_a_first_step_carries_no_empty_headings() -> None:
    """With no history, notes or skills yet, those sections are absent.

    A heading with nothing under it invites the model to wonder what it was
    supposed to find there, and costs tokens to say nothing.
    """
    prompt = planner_for().render_prompt(task="find the hours", observation=an_observation())

    assert "# Task" in prompt
    assert "# Current page" in prompt
    for absent in ["# Notes so far", "# Recent steps", "# Skills available", "rejected"]:
        assert absent not in prompt


@pytest.mark.unit
def test_history_is_bounded_and_keeps_the_most_recent_steps() -> None:
    """A prompt that grows every step gets more expensive and less focused.

    Memory owns the real window; this is the backstop, so a memory bug shows up
    as a slightly short prompt rather than as a run whose cost climbs to the
    step cap.
    """
    steps = [f"{number}. click {number} -> ok" for number in range(1, 26)]

    prompt = planner_for().render_prompt(task="t", observation=an_observation(), history=steps)

    assert steps[-1] in prompt
    assert steps[-MAX_HISTORY_STEPS] in prompt
    assert steps[-MAX_HISTORY_STEPS - 1] not in prompt


@pytest.mark.unit
def test_a_shorter_window_can_be_asked_for() -> None:
    """The bound is a setting, not a constant baked into the rendering."""
    steps = [f"{number}. click {number} -> ok" for number in range(1, 6)]

    prompt = planner_for(max_history=2).render_prompt(
        task="t", observation=an_observation(), history=steps
    )

    assert steps[-1] in prompt and steps[-2] in prompt
    assert steps[-3] not in prompt


@pytest.mark.unit
def test_truncated_page_text_says_so() -> None:
    """A model that cannot tell a short page from a cut-off one answers from half of it.

    This is the difference between "the page does not mention a phone number"
    and "the part of the page I was shown does not".
    """
    prompt = planner_for().render_prompt(
        task="t", observation=an_observation(text="Long page...", truncated=True)
    )

    assert "cut short" in prompt


@pytest.mark.unit
def test_a_page_with_nothing_on_it_says_that_too() -> None:
    """Empty channels are stated, never left blank.

    A blank section reads as a missing section, and a model that thinks the
    page text was withheld will behave differently from one that knows there is
    none.
    """
    prompt = planner_for().render_prompt(task="t", observation=an_observation(elements=0, text=""))

    assert "(no interactive elements are visible in the viewport)" in prompt
    assert "(this page has no main text)" in prompt


@pytest.mark.unit
def test_a_rejection_is_shown_last_and_as_an_instruction() -> None:
    """On a retry, the one thing that has to change is the last thing read."""
    prompt = planner_for().render_prompt(
        task="t",
        observation=an_observation(),
        feedback=["element 9 does not exist in this observation"],
    )

    assert "element 9 does not exist in this observation" in prompt
    assert prompt.rstrip().endswith("Answer again, fixing that.")


@pytest.mark.unit
def test_retrieved_skills_are_offered_when_there_are_any() -> None:
    """Skills are an option at every step, so they are listed at every step."""
    prompt = planner_for().render_prompt(
        task="t",
        observation=an_observation(),
        skills=["site_search(query) - searches this site and opens the first result"],
    )

    assert "# Skills available for this site" in prompt
    assert "site_search(query)" in prompt


@pytest.mark.unit
def test_history_lines_are_rendered_one_way() -> None:
    """The trace and the prompt describe a step identically, or one of them lies."""
    lines = render_history(
        [
            {"action": "click 4", "result": "navigated to /contact"},
            {"action": "press Enter", "result": "no change"},
        ]
    )

    assert lines == (
        "1. click 4 -> navigated to /contact",
        "2. press Enter -> no change",
    )


@pytest.mark.unit
def test_a_step_with_no_recorded_result_is_shown_as_such() -> None:
    """An unrecorded result is said out loud rather than rendered as success."""
    assert render_history([{"action": "click 4"}]) == ("1. click 4 -> no result recorded",)


# --- The prompt file --------------------------------------------------------


@pytest.mark.unit
def test_the_shipped_prompt_names_every_action() -> None:
    """The vocabulary is explained to the model somewhere it can read it.

    The wire schema deliberately carries no descriptions -- they would cost
    several thousand characters on every call -- so this file is the only place
    the model is told what the seven actions *mean*. An action missing from it
    is an action the model will not use.
    """
    prompt = planner_for().system_prompt

    for name in ["click", "fill", "press", "scroll", "navigate", "use_skill", "done"]:
        assert f"`{name}`" in prompt


@pytest.mark.unit
def test_a_different_prompt_can_be_supplied(tmp_path: Path) -> None:
    """Trying a prompt is editing a file, not editing the package."""
    prompt_file = tmp_path / "planner.md"
    prompt_file.write_text("Answer with one action.", encoding="utf-8")

    assert planner_for(prompt_path=prompt_file).system_prompt == "Answer with one action."
