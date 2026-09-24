"""The guarantee that nothing in the agent hard-depends on being able to see.

A vision-language model is an option this project takes when it is worth
paying for, and never a requirement. Point the agent at an endpoint that serves
no vision model — a laptop running one set of open weights, which is the normal
case for the ``local`` profile — and the whole thing still has to work: no image
is asked for, none is sent, and the run finishes on DOM text alone.

That promise is easy to state and easy to break, because breaking it does not
look like a bug. It looks like a feature that works on the developer's profile.
So it is defended here in one file, at every layer that could break it:

* the **capability report**, which has to say so at startup rather than let a
  user discover it from a failed run;
* the **screenshot gate**, which must answer "no" for every step, including the
  steps whose cost heuristics are screaming for an image;
* the **message builder**, which must leave an image behind rather than refuse
  the call, so a caller that went around the gate degrades instead of stopping;
* the **wire**, where the finished request must contain no image part and must
  have gone to the text route.

* and finally the **whole task**, run end to end against a real browser and the
  practice site under a profile that cannot see, checking both halves of the
  promise at once: no image left the process, and the run folder still holds a
  picture of every step.

That last one is the half most easily lost, and it is worth stating plainly
because the obvious implementation gets it backwards. "Send no screenshots"
sounds like it means "take no screenshots", and a gate that decides both at once
would make a text-only run leave no visual record at all — so the mode that most
needs a human to look at what happened would be the one mode that saved nothing
for them to look at. Photographing and sending are two decisions, and
:func:`~web_action_agent.observation.capture` keeps them apart.

Everything but the end-to-end test runs offline against a stub transport. A real
call would need a key, a network and money, and would tell us nothing about our
own code that the stub does not.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from web_action_agent.allowlist import Allowlist
from web_action_agent.browser import BrowserProcess, BrowserSettings
from web_action_agent.capabilities import AGENT, VISION, report_capabilities
from web_action_agent.config import Config
from web_action_agent.controller import Controller
from web_action_agent.executor import Executor
from web_action_agent.memory import Memory
from web_action_agent.model_client import (
    ModelClient,
    ModelClientError,
    Profile,
    has_image,
    load_profile,
    system_message,
    user_message,
)
from web_action_agent.observation import Observation, capture
from web_action_agent.observer import Box, Element
from web_action_agent.planner import Planner
from web_action_agent.screenshot_gate import TRIGGER_NO_VISION_ROUTE, ScreenshotGate
from web_action_agent.trace import RUN_FILE, TraceWriter
from web_action_agent.validator import Validator

#: The shipped profile that serves no vision model. Using the real one rather
#: than a hand-built stand-in is the point: if someone ever gives ``local`` a
#: vision route, these tests should start failing and be re-pointed on purpose.
TEXT_ONLY_PROFILE = "local"

#: A shipped profile that does serve one, for the contrast half of each pair.
VISION_PROFILE = "openrouter"

#: Stand-in for an annotated screenshot. Nothing decodes it, so its only job is
#: to be bytes that are recognisably not text.
A_PNG = b"\x89PNG\r\n\x1a\n fake bytes"

#: The task the end-to-end run is given, answerable from the practice site's
#: text alone -- which is the point, since the model cannot see the page.
TASK = "how many records mention membership"


class StubTransport:
    """An ``openai.OpenAI`` lookalike that records requests instead of sending them.

    Only the one method the client uses is implemented. What every test here
    asserts on is the recorded request: it is the boundary between our code and
    someone else's service, and so the last place the guarantee can be checked.

    Attributes:
        requests: Every request the client built, in order.
    """

    def __init__(self) -> None:
        """Set up a transport that answers every call with the same object."""
        self.requests: list[dict[str, Any]] = []
        # The client calls transport.chat.completions.create(...), so the same
        # three-deep attribute path has to exist here.
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **request: Any) -> SimpleNamespace:
        """Record a request and return a canned completion.

        Args:
            **request: The keyword arguments the client passed.

        Returns:
            SimpleNamespace: A response shaped like the SDK's.
        """
        self.requests.append(request)
        message = SimpleNamespace(content='{"action": "done"}')
        return SimpleNamespace(
            choices=[SimpleNamespace(message=message, finish_reason="stop")],
            usage=None,
        )

    @property
    def request(self) -> dict[str, Any]:
        """The single request made, for tests that make exactly one.

        Returns:
            dict[str, Any]: The recorded request.
        """
        assert len(self.requests) == 1, f"expected one call, saw {len(self.requests)}"
        return self.requests[0]


class ScriptedEndpoint(StubTransport):
    """A recording transport that answers with real actions instead of one.

    :class:`StubTransport` always replies ``done``, which is right for the tests
    that make a single call and read the request back. A whole run needs the
    planner to be led through several steps, so this adds the script while
    keeping the recording -- the requests are still what the guarantee is
    checked against.
    """

    def __init__(self, *replies: str) -> None:
        """Set up the script.

        Args:
            *replies: Message contents to return, one per call, the last
                repeating if the run asks more often than the script provides.
        """
        super().__init__()
        self._replies = list(replies)

    def _create(self, **request: Any) -> SimpleNamespace:
        """Record a request and answer with the next scripted action.

        Args:
            **request: The keyword arguments the client passed.

        Returns:
            SimpleNamespace: A response shaped like the SDK's.
        """
        self.requests.append(request)
        reply = self._replies[min(len(self.requests) - 1, len(self._replies) - 1)]
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=reply), finish_reason="stop")],
            usage=SimpleNamespace(prompt_tokens=900, completion_tokens=20),
        )


def profile(name: str) -> Profile:
    """Load one of the shipped profiles.

    Args:
        name: The profile name, as it appears in ``profiles.yaml``.

    Returns:
        Profile: The profile as shipped.
    """
    return load_profile(name)


def client_for(name: str, transport: StubTransport | None = None) -> ModelClient:
    """Build a client on a shipped profile and a stub transport.

    Args:
        name: The profile name.
        transport: The stub to record into. A fresh one by default.

    Returns:
        ModelClient: The client. The environment is empty on purpose — with a
        stub transport no key is needed, and none is read.
    """
    return ModelClient(profile(name), transport=transport or StubTransport(), env={})


def gate_for(name: str, **overrides: int) -> ScreenshotGate:
    """Build a screenshot gate wired to a shipped profile's vision availability.

    Args:
        name: The profile name, which decides ``supports_vision``.
        **overrides: Threshold overrides for :class:`ScreenshotGate`. The
            defaults are the ones a run would use.

    Returns:
        ScreenshotGate: A gate for one run, with no failures recorded yet.
    """
    config = Config()
    settings = {
        "min_elements": config.screenshot_min_elements,
        "max_dom_failures": config.screenshot_max_dom_failures,
    }
    settings.update(overrides)
    return ScreenshotGate(supports_vision=profile(name).supports_vision, **settings)


def an_observation(*, screenshot: Path | None) -> Observation:
    """Build an observation with both text channels populated.

    The channels are non-empty on purpose: the claim under test is that they
    carry the step on their own, which says nothing if they are blank.

    Args:
        screenshot: What the observation's screenshot field holds.

    Returns:
        Observation: A small but complete record.
    """
    box = Box(x=10, y=20, width=100, height=30)
    return Observation(
        url="http://localhost:8000/search.html",
        title="Search",
        elements=(
            Element(id=1, tag="input", role="searchbox", name="Search", text="", box=box),
            Element(id=2, tag="button", role="button", name="Go", text="Go", box=box),
        ),
        text="Results for widgets: three matches, priced from 4.50.",
        screenshot=screenshot,
    )


# --- The profile says so, and says so out loud ------------------------------


@pytest.mark.unit
def test_the_text_only_profile_really_has_no_vision_route() -> None:
    """Everything below rests on this, so it is asserted rather than assumed."""
    assert profile(TEXT_ONLY_PROFILE).vision is None
    assert profile(TEXT_ONLY_PROFILE).supports_vision is False
    assert profile(VISION_PROFILE).supports_vision is True


@pytest.mark.unit
def test_startup_reports_the_missing_vision_model_as_a_mode_not_a_fault() -> None:
    """A user learns this from the banner, not from a run that behaved oddly.

    The wording matters as much as the flag. An unavailable capability that
    reads like a broken installation invites someone to "fix" it; this one has
    to read as a description of how the run will go.
    """
    report = report_capabilities(
        Config(profile=TEXT_ONLY_PROFILE),
        env={},
        profiles_path=None,
    )

    vision = report.get(VISION)
    assert vision.available is False
    assert "DOM text alone" in vision.detail
    # The agent itself is unaffected: a missing vision model is not a missing
    # endpoint, and nothing about the run is blocked by it.
    assert report.can(AGENT) is True


# --- The gate never asks for an image ---------------------------------------


@pytest.mark.unit
def test_the_gate_knows_the_whole_run_is_text_only() -> None:
    """One boolean to consult, so no caller has to re-derive it per step."""
    assert gate_for(TEXT_ONLY_PROFILE).text_only is True
    assert gate_for(VISION_PROFILE).text_only is False


@pytest.mark.unit
@pytest.mark.parametrize("element_count", [0, 2, 30])
def test_no_element_count_gates_a_screenshot_in(element_count: int) -> None:
    """Not the sparse page, not the empty one, not the busy one.

    A page yielding nothing at all is the strongest possible case for an image,
    which is exactly why it is worth pinning: the override has to hold where it
    is most tempting to make an exception.
    """
    decision = gate_for(TEXT_ONLY_PROFILE).decide(element_count=element_count)

    assert decision.send_screenshot is False
    assert decision.trigger == TRIGGER_NO_VISION_ROUTE


@pytest.mark.unit
def test_a_long_failure_streak_still_gates_nothing_in() -> None:
    """The escalation path has nowhere to escalate to, and must not pretend.

    On a vision profile this run would have switched to images several steps
    ago. Here the streak is recorded honestly — the gate is not lying about what
    happened — and the answer stays "no" anyway.
    """
    gate = gate_for(TEXT_ONLY_PROFILE)
    for _ in range(5):
        gate.record_planner_outcome(succeeded=False, had_screenshot=False)

    decision = gate.decide(element_count=1)

    assert gate.dom_failures == 5
    assert decision.send_screenshot is False
    assert decision.trigger == TRIGGER_NO_VISION_ROUTE
    assert decision.dom_failures == 5


@pytest.mark.unit
@pytest.mark.parametrize(
    ("setting", "value"),
    [
        # min_elements high enough that every page is "sparse".
        ("min_elements", 10_000),
        # max_dom_failures at zero, which on a vision profile turns images on
        # from the very first step.
        ("max_dom_failures", 0),
    ],
)
def test_no_threshold_setting_can_turn_the_image_channel_on(setting: str, value: int) -> None:
    """Tuning is for the heuristics; it may not reach the guarantee.

    This is the failure mode the ordering inside ``decide`` exists to prevent —
    a guarantee sitting below a preference stops being a guarantee the moment
    someone adjusts a number in a config file.
    """
    decision = gate_for(TEXT_ONLY_PROFILE, **{setting: value}).decide(element_count=3)

    assert decision.send_screenshot is False
    assert decision.trigger == TRIGGER_NO_VISION_ROUTE


# --- The message builder never sends one ------------------------------------


@pytest.mark.unit
def test_an_image_handed_to_a_text_only_client_is_dropped_not_refused() -> None:
    """The second, independent defence: degrade, do not stop.

    A caller reaching here with PNG bytes has already gone around the gate. The
    run still has to finish, so the image is left behind rather than turned into
    an exception — and the drop is counted, because a silent one would be
    invisible to whoever later asks why the run had no images in it.
    """
    client = client_for(TEXT_ONLY_PROFILE)

    message = client.observation_message("what now?", screenshot_png=A_PNG)

    assert has_image([message]) is False
    assert message["content"] == "what now?"
    assert client.dropped_screenshots == 1


@pytest.mark.unit
def test_an_image_handed_to_a_vision_client_is_carried_through() -> None:
    """The contrast case, so the test above is not passing for the wrong reason."""
    client = client_for(VISION_PROFILE)

    message = client.observation_message("what now?", screenshot_png=A_PNG)

    assert has_image([message]) is True
    assert client.dropped_screenshots == 0


@pytest.mark.unit
def test_a_text_only_step_counts_no_drop() -> None:
    """Nothing was discarded, so nothing is reported as discarded.

    The counter has to mean "a caller went around the gate". If ordinary
    text-only steps incremented it, a non-zero value would say nothing at all.
    """
    client = client_for(TEXT_ONLY_PROFILE)

    message = client.observation_message("what now?", screenshot_png=None)

    assert has_image([message]) is False
    assert client.dropped_screenshots == 0


@pytest.mark.unit
def test_going_around_the_builder_as_well_is_still_refused() -> None:
    """The last line of defence has not been softened by the one above it.

    ``observation_message`` keeps a run going; it does not make it acceptable to
    send an image to a model that cannot read one. A caller that skips both the
    gate and the builder gets the refusal, because at that point the alternative
    is a request the endpoint will answer nonsensically.
    """
    client = client_for(TEXT_ONLY_PROFILE)

    with pytest.raises(ModelClientError, match="no vision route"):
        client.complete([user_message("what now?", screenshot_png=A_PNG)])


# --- Nothing image-shaped reaches the wire ----------------------------------


@pytest.mark.mock_llm
def test_a_step_whose_observation_has_a_screenshot_still_sends_text_alone() -> None:
    """The end of the chain, asserted on the finished request.

    This is the shape of a real step under a text-only profile: the observation
    has a screenshot on disk — it was captured for the trace, as it always is —
    and the gate says not to send it, so the payload that leaves the process
    carries no image part and goes to the text route.
    """
    transport = StubTransport()
    client = client_for(TEXT_ONLY_PROFILE, transport)
    gate = ScreenshotGate.from_config(Config(), supports_vision=client.supports_vision)
    observation = an_observation(screenshot=Path("runs/example/step-01.png"))

    decision = gate.decide(element_count=len(observation.elements))
    # Exactly what a planner will do: read the image only if it is going to send
    # one, which under this profile is never.
    png = A_PNG if decision.send_screenshot else None
    client.complete(
        [
            system_message("be brief"),
            client.observation_message(observation.summary(), screenshot_png=png),
        ]
    )

    assert decision.send_screenshot is False
    assert has_image(transport.request["messages"]) is False
    assert transport.request["model"] == profile(TEXT_ONLY_PROFILE).text.model
    assert client.dropped_screenshots == 0


@pytest.mark.mock_llm
def test_every_step_of_a_run_stays_free_of_images_however_badly_it_goes() -> None:
    """A run where the planner fails repeatedly is the one that would escalate.

    Five steps, every one of them a failure on a page that also trips the
    element-count rule: on a vision profile this run would be sending images
    long before the end. Here the requests are checked as a set, because the
    guarantee is about the run and not about any one step of it.
    """
    transport = StubTransport()
    client = client_for(TEXT_ONLY_PROFILE, transport)
    gate = ScreenshotGate.from_config(Config(), supports_vision=client.supports_vision)

    for _ in range(5):
        decision = gate.decide(element_count=1)
        png = A_PNG if decision.send_screenshot else None
        client.complete([client.observation_message("stuck", screenshot_png=png)])
        gate.record_planner_outcome(succeeded=False, had_screenshot=decision.send_screenshot)

    assert len(transport.requests) == 5
    text_model = profile(TEXT_ONLY_PROFILE).text.model
    for request in transport.requests:
        assert has_image(request["messages"]) is False
        assert request["model"] == text_model


# --- The two text channels carry the step on their own ----------------------


@pytest.mark.unit
def test_the_observation_keeps_its_text_channels_with_no_image() -> None:
    """The reason the guarantee is affordable at all.

    Elements and page text are always present. If they were not, "runs without
    vision" would mean "runs blind", and no ordering of checks would save it.
    """
    observation = an_observation(screenshot=None)

    assert observation.has_screenshot is False
    assert len(observation.elements) == 2
    assert observation.text
    assert "no screenshot" in observation.summary()


@pytest.mark.unit
def test_a_screenshot_on_disk_is_a_different_fact_from_one_in_the_prompt() -> None:
    """Two meanings of "no screenshot", kept apart by the record itself.

    Under a text-only profile the screenshot channel of the *observation* is
    empty, and that is what the trace records. It is not a claim that no image
    was captured — the run folder is expected to hold one per step for a human
    to review — only that this observation did not carry one to the model.
    """
    captured = an_observation(screenshot=Path("runs/example/step-01.png"))
    sent = an_observation(screenshot=None)

    # The path survives the trace round trip, so a reader can still find the
    # image that was taken.
    assert Observation.from_dict(captured.to_dict()).screenshot == captured.screenshot
    assert Observation.from_dict(sent.to_dict()).screenshot is None
    # The channels the planner actually read are identical either way.
    assert captured.elements == sent.elements
    assert captured.text == sent.text


# --- The whole task, on an endpoint that cannot see -------------------------


@pytest.mark.mock_llm
def test_a_whole_run_on_a_blind_endpoint_finishes_and_still_leaves_its_pictures(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    playwright: Any,
    fixture_site: Any,
) -> None:
    """Both halves of the promise, checked on one real run.

    Everything above tests a layer. This tests the *system*: a real Chromium on
    the practice site, a real form filled in and submitted, the real loop, the
    real trace writer — with the one profile that serves no vision model.

    Three things have to be true at once, and the third is the one that the
    obvious implementation loses:

    1. the run finishes the task, on the two text channels alone;
    2. nothing image-shaped left the process, on any step;
    3. the run folder still holds an annotated screenshot of every step.

    A run that satisfied the first two and not the third would look entirely
    correct from the outside, and would have quietly made the mode that most
    needs review the one mode with nothing to review.
    """
    transport = ScriptedEndpoint(
        '{"action": "fill", "element_id": 7, "text": "membership"}',
        '{"action": "press", "key": "Enter"}',
        '{"action": "done", "status": "success", "answer": "17 records"}',
    )
    start = fixture_site.url_for("search.html")
    config = Config(start_url=start)
    client = ModelClient(profile(TEXT_ONLY_PROFILE), transport=transport, env={})
    # The premise of the whole test. Asserted rather than assumed, because if
    # `local` ever gained a vision route this test would keep passing while
    # testing nothing.
    assert client.supports_vision is False

    trace = TraceWriter.open(
        tmp_path / "run",
        task=TASK,
        start_url=start,
        config=config,
        text_only=not client.supports_vision,
    )
    starts_its_own = BrowserProcess
    with (
        starts_its_own(BrowserSettings.from_config(config), playwright=playwright) as browser,
        browser.task_session() as session,
    ):
        controller = Controller(
            planner=Planner(client),
            validator=Validator(Allowlist.derive(start)),
            executor=Executor(session, action_timeout_s=config.action_timeout_s),
            memory=Memory.from_config(config),
            max_steps=config.max_steps,
            retry_budget=config.retry_budget,
            gate=ScreenshotGate.from_config(config, supports_vision=client.supports_vision),
            screenshot_dir=trace.screenshot_dir,
            config=config,
            trace=trace,
        )
        report = controller.run(TASK, start_url=start)

    # 1. It finished.
    assert report.outcome.ok, report.summary()
    assert report.outcome.answer == "17 records"

    # 2. Nothing image-shaped reached the wire, and nothing was even offered:
    # `dropped_screenshots` counts images the client had to throw away, so zero
    # means the gate did its job rather than the client covering for it.
    assert transport.requests, "the planner really was asked"
    text_model = profile(TEXT_ONLY_PROFILE).text.model
    for request in transport.requests:
        assert has_image(request["messages"]) is False
        assert request["model"] == text_model
    assert client.dropped_screenshots == 0
    assert all(step.observation.screenshot is None for step in report.steps)
    assert all(
        step.gate is not None and step.gate.trigger == TRIGGER_NO_VISION_ROUTE
        for step in report.steps
    )

    # 3. And the pictures are there anyway, one per step, each a real PNG.
    for step in report.steps:
        image = step.observation.screenshot_on_disk
        assert image is not None, f"step {step.number} left no picture to review"
        assert image.is_file() and image.read_bytes().startswith(b"\x89PNG")
        assert image.parent == trace.screenshot_dir

    # Which is exactly why the folder has to say so in words: its screenshots
    # look the same either way, so they are no evidence about what the model was
    # shown, and a reader would otherwise have to deduce the mode from every
    # step's trigger.
    described = json.loads((trace.folder / RUN_FILE).read_text(encoding="utf-8"))
    assert described["text_only"] is True


@pytest.mark.fixture
def test_the_same_page_is_photographed_either_way_and_sent_only_one_way(
    page: Any,
    fixture_site: Any,
    tmp_path: Path,
) -> None:
    """The split itself, on one real page looked at twice.

    Same page, same capture, two gates: one wired to an endpoint that can see
    and one to an endpoint that cannot. The file on disk is written both times
    — that is the record, and it does not depend on who is reading it — while
    only the seeing endpoint's observation carries it into the prompt.

    Asserted as a contrast rather than two separate tests because the claim is
    about the *difference* between the two modes, and a pair of tests each
    checking one mode could both pass while the difference had collapsed.
    """
    page.goto(fixture_site.url_for("search.html"))
    config = Config()

    blind, blind_decision = capture(
        page,
        config=config,
        gate=gate_for(TEXT_ONLY_PROFILE),
        screenshot_dir=tmp_path / "blind",
        label="step-01",
    )
    seeing, seeing_decision = capture(
        page,
        config=config,
        # Forced open by asking for more elements than the page has, so the
        # contrast is about vision rather than about the cost heuristics.
        gate=gate_for(VISION_PROFILE, min_elements=1000),
        screenshot_dir=tmp_path / "seeing",
        label="step-01",
    )

    # The record: both runs leave a real picture for a person to look at.
    for observation in (blind, seeing):
        assert observation.screenshot_on_disk is not None
        assert observation.screenshot_on_disk.is_file()
        assert observation.screenshot_on_disk.read_bytes().startswith(b"\x89PNG")

    # The prompt: only the endpoint that can read one is given one.
    assert blind_decision.send_screenshot is False
    assert blind_decision.trigger == TRIGGER_NO_VISION_ROUTE
    assert blind.screenshot is None
    assert seeing_decision.send_screenshot is True
    assert seeing.screenshot == seeing.screenshot_on_disk

    # And the text channels are the same either way, which is what makes the
    # blind run a complete run rather than a crippled one.
    assert blind.elements == seeing.elements
    assert blind.text == seeing.text


@pytest.mark.unit
def test_the_picture_in_a_prompt_is_never_a_different_one_from_the_record() -> None:
    """An observation carries at most one image, under two names.

    ``screenshot`` is either empty or the very same file as
    ``screenshot_on_disk``. If the two could name different images, the trace
    would say the model saw a page it did not, and every later reading of that
    run would be wrong in a way nothing could detect.
    """
    on_disk = Path("runs/example/screenshots/step-01.png")

    sent = Observation(
        url="u", title="t", elements=(), text="x", screenshot=on_disk, screenshot_on_disk=on_disk
    )
    kept = Observation(url="u", title="t", elements=(), text="x", screenshot_on_disk=on_disk)

    for observation in (sent, kept):
        assert observation.screenshot in (None, observation.screenshot_on_disk)
    assert sent.has_screenshot is True
    assert kept.has_screenshot is False, "on disk is not the same as in the prompt"
