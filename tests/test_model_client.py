"""What the model client and its profiles promise, asserted without a network.

Two claims are being defended here, and they are the claims the project makes
about itself.

The first is that the endpoint is genuinely a setting. It is not enough that
the three shipped profiles work; nothing in the client may *know* them. So the
shipped three are checked against the file that declares them, and then a
profile invented inside a test — a name the code has never seen — is checked to
get identical treatment. If someone ever special-cases a profile by name, the
second test is the one that fails.

The second is that a run's numbers can be reproduced. That rests on the pinned
host reaching the wire and on temperature staying at zero, so the request built
for each call is inspected directly rather than assumed.

Every test drives a stub transport. Real calls would need a key, a network, and
money, and would assert nothing about our code that the stub does not.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from web_action_agent.config import Config, load_config
from web_action_agent.model_client import (
    DEFAULT_PROFILES_PATH,
    UNUSED_API_KEY,
    CallRecord,
    MissingAPIKeyError,
    ModelClient,
    ModelClientError,
    ModelResponseError,
    Profile,
    ProfileError,
    has_image,
    load_profile,
    load_profiles,
    parse_json_object,
    resolve_api_key,
    resolve_profile,
    system_message,
    user_message,
)
from web_action_agent.trace import new_id

ACTION_SCHEMA = {
    "type": "object",
    "properties": {"action": {"type": "string"}, "element_id": {"type": "integer"}},
    "required": ["action"],
}

A_PNG = b"\x89PNG\r\n\x1a\n fake bytes"


class StubTransport:
    """An ``openai.OpenAI`` lookalike that records calls instead of making them.

    Only the one method the client uses is implemented. The recorded request is
    what most tests here assert on: it is the boundary between our code and
    someone else's service, and therefore the last thing we can check.

    Attributes:
        requests: Every request the client built, in order.
        closed: Whether the client closed the transport.
    """

    def __init__(self, content: str = '{"action": "click"}', **usage: Any) -> None:
        """Set up a transport that replies with fixed content.

        Args:
            content: The message content every call returns.
            **usage: Token counts to report, e.g. ``prompt_tokens=10``. Omitted
                entirely if none are given, which is how some endpoints behave.
        """
        self._content = content
        self._usage = SimpleNamespace(**usage) if usage else None
        self.requests: list[dict[str, Any]] = []
        self.closed = False
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
        message = SimpleNamespace(content=self._content)
        choice = SimpleNamespace(message=message, finish_reason="stop")
        return SimpleNamespace(choices=[choice], usage=self._usage)

    @property
    def request(self) -> dict[str, Any]:
        """The single request made, for tests that make exactly one.

        Returns:
            dict[str, Any]: The recorded request.
        """
        assert len(self.requests) == 1, f"expected one call, saw {len(self.requests)}"
        return self.requests[0]

    def close(self) -> None:
        """Record that the client closed us."""
        self.closed = True


def without_vision_schemas(profile: Profile) -> Profile:
    """Copy a profile, with its vision route unable to honour a JSON schema.

    Args:
        profile: A profile with a vision route.

    Returns:
        Profile: The same profile, but the vision route says it has no schema
        support -- as a free or older endpoint might.
    """
    assert profile.vision is not None
    vision = profile.vision.model_copy(update={"supports_json_schema": False})
    return profile.model_copy(update={"vision": vision})


def client_for(
    profile: Profile,
    *,
    transport: StubTransport | None = None,
    env: dict[str, str] | None = None,
) -> ModelClient:
    """Build a client on a stub transport.

    Args:
        profile: The profile under test.
        transport: The stub to record into. A fresh one by default.
        env: Environment for key resolution. Empty by default — with a stub
            transport no key is needed, and none is read.

    Returns:
        ModelClient: The client.
    """
    return ModelClient(profile, transport=transport or StubTransport(), env=env or {})


def call(client: ModelClient, **kwargs: Any) -> Any:
    """Make one ordinary text call.

    Args:
        client: The client to call.
        **kwargs: Passed through to ``complete``.

    Returns:
        ModelResponse: Whatever the client returned.
    """
    return client.complete([system_message("be brief"), user_message("what now?")], **kwargs)


# --- The shipped profiles ---------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize("name", ["openrouter", "nvidia", "local"])
def test_each_shipped_profile_builds_a_client_matching_its_declaration(name: str) -> None:
    """The three profiles the project ships each construct a working client.

    Resolved base URL, model, and vision flag are compared against the file
    that declares them — the point being that the client carries none of these
    values itself, so the file is the only place they can come from.
    """
    declared = load_profiles(DEFAULT_PROFILES_PATH)[name]
    config = load_config(
        config_path=Path("absent.yaml"),
        dotenv_path=Path("absent.env"),
        env={},
        cli={"profile": name},
    )

    client = client_for(resolve_profile(config))

    assert client.profile.base_url == declared.base_url
    assert client.profile.model == declared.text.model
    assert client.supports_vision is declared.supports_vision


@pytest.mark.unit
def test_shipped_profiles_cover_the_three_documented_run_modes() -> None:
    """Hosted aggregator, first-party endpoint, and self-hosted server all ship.

    The abstraction is only demonstrated if all three modes are actually
    configured; one of them missing would make it a claim rather than a demo.
    """
    profiles = load_profiles(DEFAULT_PROFILES_PATH)
    # The three ways to run the agent, the two comparison models, and the
    # evaluation judge, which is a profile like any other but never drives a run.
    assert set(profiles) == {
        "openrouter",
        "nvidia",
        "local",
        "openrouter-qwen",
        "openrouter-mistral",
        "openrouter-judge",
    }
    assert profiles["local"].api_key_required is False
    assert all(p.api_key_env.startswith("WAA_") for p in profiles.values())
    # Committed file, published repository: it holds variable names, not keys.
    assert "sk-" not in DEFAULT_PROFILES_PATH.read_text()


@pytest.mark.unit
def test_a_profile_the_code_has_never_seen_behaves_identically(tmp_path: Path) -> None:
    """Switching endpoints is data, not code — proven with an unknown name.

    This is the test the whole module rests on. A profile defined in a
    file written seconds ago, under a name nothing has been taught, must build
    a client exactly as the shipped ones do. Any ``if profile.name ==`` added
    outside the loader shows up here first.
    """
    profiles_file = tmp_path / "profiles.yaml"
    profiles_file.write_text(
        "invented:\n"
        "  base_url: https://example.invalid/v1\n"
        "  api_key_env: WAA_INVENTED_API_KEY\n"
        "  text:\n"
        "    model: some/model\n"
        "    provider_order: [SomeHost]\n"
        "  vision:\n"
        "    model: some/vision-model\n"
    )
    config = load_config(
        config_path=Path("absent.yaml"),
        dotenv_path=Path("absent.env"),
        env={},
        cli={"profile": "invented"},
    )

    profile = resolve_profile(config, path=profiles_file)
    transport = StubTransport()
    client = ModelClient(profile, transport=transport, env={"WAA_INVENTED_API_KEY": "k"})
    call(client, schema=ACTION_SCHEMA)

    assert client.profile.base_url == "https://example.invalid/v1"
    assert client.supports_vision is True
    assert transport.request["model"] == "some/model"
    assert transport.request["extra_body"] == {
        "provider": {"order": ["SomeHost"], "allow_fallbacks": False}
    }


@pytest.mark.unit
def test_unknown_profile_names_the_ones_that_exist() -> None:
    """A misspelled profile fails at startup, listing the valid names."""
    with pytest.raises(ProfileError) as caught:
        load_profile("openroute")
    message = str(caught.value)
    assert "openroute" in message and "openrouter" in message and "local" in message


@pytest.mark.unit
def test_malformed_profile_raises_rather_than_being_ignored(tmp_path: Path) -> None:
    """An unknown key in a profile is an error, as it is in the config file.

    A misspelled ``supports_json_schema`` that was quietly ignored would turn
    structured decoding off for a whole evaluation without anyone noticing.
    """
    profiles_file = tmp_path / "profiles.yaml"
    profiles_file.write_text(
        "broken:\n"
        "  base_url: https://example.invalid/v1\n"
        "  api_key_env: WAA_BROKEN_API_KEY\n"
        "  text:\n"
        "    model: some/model\n"
        "    supports_json_schemas: true\n"
    )
    with pytest.raises(ProfileError) as caught:
        load_profiles(profiles_file)
    assert "broken" in str(caught.value)


# --- Configuration overrides ------------------------------------------------


@pytest.mark.unit
def test_run_overrides_redirect_the_profile(tmp_path: Path) -> None:
    """``base_url`` and ``model`` override the profile for one run.

    Pointing the local profile at a server on another box, or trying one model,
    must not require editing a committed file.
    """
    config = load_config(
        config_path=Path("absent.yaml"),
        dotenv_path=Path("absent.env"),
        env={
            "WAA_PROFILE": "local",
            "WAA_BASE_URL": "http://gpu-box:8000/v1",
            "WAA_MODEL": "other/model",
        },
    )

    profile = resolve_profile(config)

    assert (profile.name, profile.base_url, profile.model) == (
        "local",
        "http://gpu-box:8000/v1",
        "other/model",
    )


@pytest.mark.unit
def test_client_from_config_uses_the_configured_call_timeout() -> None:
    """The model-call timeout is a configured tunable, not a constant in here."""
    config = load_config(
        config_path=Path("absent.yaml"),
        dotenv_path=Path("absent.env"),
        env={"WAA_LLM_TIMEOUT_S": "12"},
    )

    client = ModelClient.from_config(config, env={}, transport=StubTransport())

    assert client.timeout_s == 12.0


# --- Keys -------------------------------------------------------------------


@pytest.mark.unit
def test_key_comes_from_the_profiles_named_variable() -> None:
    """Each profile reads its own variable, so several can be configured at once."""
    profiles = load_profiles(DEFAULT_PROFILES_PATH)
    env = {"WAA_OPENROUTER_API_KEY": "or-key", "WAA_NVIDIA_API_KEY": "nv-key"}

    assert resolve_api_key(profiles["openrouter"], env) == "or-key"
    assert resolve_api_key(profiles["nvidia"], env) == "nv-key"


@pytest.mark.unit
def test_one_off_override_beats_the_profiles_variable() -> None:
    """``WAA_API_KEY`` overrides, for a throwaway key or a second account."""
    profile = load_profile("openrouter")
    env = {"WAA_OPENROUTER_API_KEY": "or-key", "WAA_API_KEY": "just-this-once"}

    assert resolve_api_key(profile, env) == "just-this-once"


@pytest.mark.unit
def test_missing_key_fails_at_startup_naming_the_variable() -> None:
    """A required key that is unset stops the run before the first call.

    Failing here means "set this variable", instead of an HTTP 401 arriving
    mid-run and being recorded as a task the agent could not do.
    """
    with pytest.raises(MissingAPIKeyError) as caught:
        ModelClient(load_profile("openrouter"), env={})
    assert "WAA_OPENROUTER_API_KEY" in str(caught.value)


@pytest.mark.unit
def test_local_endpoint_needs_no_key() -> None:
    """A self-hosted server authenticates nobody, so no key is demanded."""
    assert resolve_api_key(load_profile("local"), {}) == UNUSED_API_KEY


@pytest.mark.unit
def test_repr_does_not_disclose_the_key() -> None:
    """Reprs reach logs and traces, which are committed artifacts."""
    client = client_for(
        load_profile("openrouter"), env={"WAA_OPENROUTER_API_KEY": "sk-secret-value"}
    )

    assert "sk-secret-value" not in repr(client)
    assert "openrouter" in repr(client)


# --- Requests: routing, determinism, structured output ----------------------


@pytest.mark.unit
def test_pinned_host_reaches_the_wire_with_fallbacks_off() -> None:
    """The shipped aggregator profile pins its host on every call.

    Reproducibility is the reason: the same model served by a different host
    can quantise differently and honour different parameters. A pin that allows
    fallback is not a pin, so both halves are asserted.
    """
    transport = StubTransport()
    call(client_for(load_profile("openrouter"), transport=transport))

    assert transport.request["extra_body"] == {
        "provider": {"order": ["Nebius"], "allow_fallbacks": False}
    }


@pytest.mark.unit
def test_no_routing_block_where_the_endpoint_has_one_host() -> None:
    """A first-party endpoint gets no vendor routing block it cannot read."""
    transport = StubTransport()
    call(client_for(load_profile("nvidia"), transport=transport))

    assert "extra_body" not in transport.request


@pytest.mark.unit
def test_calls_are_deterministic_by_default() -> None:
    """Temperature zero and a fixed seed, unless a caller says otherwise.

    A planner that answers differently on identical input turns every
    cold-versus-warm comparison into noise.
    """
    transport = StubTransport()
    call(client_for(load_profile("nvidia"), transport=transport))

    assert transport.request["temperature"] == 0.0
    assert transport.request["seed"] == 0


@pytest.mark.unit
def test_schema_request_constrains_decoding_where_the_route_supports_it() -> None:
    """Where the host honours schemas, the schema goes on the request."""
    transport = StubTransport()
    call(client_for(load_profile("openrouter"), transport=transport), schema=ACTION_SCHEMA)

    response_format = transport.request["response_format"]
    assert response_format["type"] == "json_schema"
    assert response_format["json_schema"]["schema"] == ACTION_SCHEMA


@pytest.mark.unit
def test_schema_omitted_where_the_route_cannot_honour_it() -> None:
    """A route without schema support is not sent a parameter it will reject.

    Every shipped route supports schemas today, so the case is made up here: a
    vision route without that support. Sending the parameter anyway would fail
    the call outright; omitting it leaves the validator's schema check and
    bounded retry as the safety net, which is the design.
    """
    transport = StubTransport()
    client = client_for(without_vision_schemas(load_profile("openrouter")), transport=transport)

    client.complete([user_message("what now?", screenshot_png=A_PNG)], schema=ACTION_SCHEMA)

    assert "response_format" not in transport.request


@pytest.mark.unit
def test_parsed_object_is_returned_alongside_the_raw_text() -> None:
    """Callers get the object; the raw text stays available for the trace."""
    client = client_for(
        load_profile("nvidia"), transport=StubTransport('{"action": "click", "element_id": 4}')
    )

    response = call(client, schema=ACTION_SCHEMA)

    assert response.data == {"action": "click", "element_id": 4}
    assert response.text.startswith("{")


@pytest.mark.unit
def test_json_wrapped_in_prose_is_recovered() -> None:
    """An unconstrained route fencing its object is salvaged, not failed.

    Narrow on purpose: the span from the first brace to the last, and nothing
    cleverer. Parsing model prose is what schemas exist to avoid.
    """
    fenced = 'Sure!\n```json\n{"action": "scroll"}\n```\n'
    assert parse_json_object(fenced) == {"action": "scroll"}


@pytest.mark.unit
def test_unparseable_reply_raises_for_the_caller_to_retry() -> None:
    """Malformed JSON is a retryable event carrying the text that failed."""
    client = client_for(load_profile("nvidia"), transport=StubTransport("I could not decide."))

    with pytest.raises(ModelResponseError) as caught:
        call(client, schema=ACTION_SCHEMA)
    assert caught.value.text == "I could not decide."


@pytest.mark.unit
def test_no_parsing_when_no_schema_was_asked_for() -> None:
    """Prose callers — the judge's rationale, say — get their text unmangled."""
    client = client_for(load_profile("nvidia"), transport=StubTransport("looks right to me"))

    response = call(client)

    assert (response.text, response.data) == ("looks right to me", None)


# --- Vision -----------------------------------------------------------------


@pytest.mark.unit
def test_a_screenshot_selects_the_vision_route() -> None:
    """Attaching an image switches the model, without the caller asking.

    The call is a vision call because of what it contains, so the two cannot
    fall out of step.
    """
    transport = StubTransport()
    client = client_for(load_profile("openrouter"), transport=transport)

    response = client.complete([user_message("what now?", screenshot_png=A_PNG)])

    assert transport.request["model"] == load_profile("openrouter").vision.model
    assert response.record.vision is True


@pytest.mark.unit
def test_text_only_calls_stay_on_the_text_route() -> None:
    """No image, no vision model — screenshots are gated on cost, so this is the common path."""
    transport = StubTransport()
    client = client_for(load_profile("openrouter"), transport=transport)

    response = call(client)

    assert transport.request["model"] == load_profile("openrouter").model
    assert response.record.vision is False


@pytest.mark.unit
def test_screenshot_to_a_text_only_profile_is_refused() -> None:
    """Sending an image where none can be read fails loudly.

    Dropping it silently would leave the planner answering about a page it was
    never shown, and the trace would not say so.
    """
    client = client_for(load_profile("local"))

    with pytest.raises(ModelClientError) as caught:
        client.complete([user_message("what now?", screenshot_png=A_PNG)])
    assert "vision" in str(caught.value)


@pytest.mark.unit
def test_screenshots_travel_as_inline_data() -> None:
    """PNG bytes are encoded here, so no other component learns the wire format."""
    message = user_message("look", screenshot_png=A_PNG)

    assert has_image([message]) is True
    assert message["content"][1]["image_url"]["url"].startswith("data:image/png;base64,")
    assert has_image([user_message("no picture")]) is False


# --- The cost record --------------------------------------------------------


@pytest.mark.unit
def test_every_call_records_tokens_and_wall_clock() -> None:
    """Each call yields the record the trace needs, including for local runs.

    Wall-clock is recorded always because it is the only cost a self-hosted
    endpoint has: there, cost per task is seconds, not dollars.
    """
    client = client_for(
        load_profile("openrouter"),
        transport=StubTransport(prompt_tokens=1200, completion_tokens=48),
    )

    record = call(client).record

    assert (record.profile, record.model) == ("openrouter", load_profile("openrouter").model)
    assert (record.prompt_tokens, record.completion_tokens, record.total_tokens) == (1200, 48, 1248)
    assert record.latency_s >= 0.0
    assert record.finish_reason == "stop"
    # Priced from the checked-in table: $0.06 and $0.24 per million tokens on
    # the pinned host, so 1,200 in and 48 out come to $0.00008352.
    assert record.cost_usd == pytest.approx(1200 * 0.06e-6 + 48 * 0.24e-6)


@pytest.mark.unit
def test_unreported_usage_is_unknown_rather_than_zero() -> None:
    """An endpoint that reports no usage must not look like a free call."""
    record = call(client_for(load_profile("local"))).record

    assert (record.prompt_tokens, record.total_tokens) == (None, None)
    assert (
        CallRecord(
            call_id=new_id("call"),
            started_at="2026-09-17T08:00:00.000Z",
            ended_at="2026-09-17T08:00:00.100Z",
            profile="x",
            model="m",
            vision=False,
            prompt_tokens=5,
            completion_tokens=None,
            latency_s=0.1,
            finish_reason=None,
        ).total_tokens
        is None
    )


# --- Failures at the boundary -----------------------------------------------


@pytest.mark.unit
def test_transport_failures_arrive_as_one_error_type() -> None:
    """Whatever the SDK raises, the agent loop sees a model-layer failure.

    The message keeps the profile and model, because "which endpoint was this"
    is the first question asked of a failed run.
    """
    transport = StubTransport()

    def explode(**_: Any) -> None:
        raise RuntimeError("connection reset")

    transport.chat.completions.create = explode
    client = client_for(load_profile("nvidia"), transport=transport)

    with pytest.raises(ModelClientError) as caught:
        call(client)
    assert "nvidia" in str(caught.value) and "connection reset" in str(caught.value)


@pytest.mark.unit
def test_a_reply_with_no_choices_is_a_failure_not_an_empty_action() -> None:
    """Some endpoints answer 200 with nothing in it; that is not a plan."""
    transport = StubTransport()
    transport.chat.completions.create = lambda **_: SimpleNamespace(choices=[], usage=None)
    # Waited out before giving up (see test_backoff.py); no real waiting here.
    client = ModelClient(load_profile("nvidia"), transport=transport, env={}, sleep=lambda s: None)

    with pytest.raises(ModelClientError):
        call(client)


@pytest.mark.unit
def test_closing_the_client_closes_the_transport() -> None:
    """Used as a context manager, the client releases its connections."""
    transport = StubTransport()

    with client_for(load_profile("nvidia"), transport=transport) as client:
        call(client)

    assert transport.closed is True


# --- Consistency with the documentation -------------------------------------


@pytest.mark.unit
def test_env_template_documents_every_key_variable() -> None:
    """Every key a profile can read is named in ``.env.example``.

    A variable the template omits is one nobody knows they may set, and the
    failure it produces — "profile needs an API key" — would look like a bug.
    """
    template = (Path(__file__).resolve().parent.parent / ".env.example").read_text()
    missing = [p.api_key_env for p in load_profiles().values() if p.api_key_env not in template]

    assert not missing, f".env.example does not mention: {', '.join(missing)}"


@pytest.mark.unit
def test_profile_named_in_config_defaults_exists() -> None:
    """The committed default profile is one that is actually defined.

    Two committed files have to agree; nothing but a test makes them.
    """
    config = Config()
    assert load_profile(config.profile).name == config.profile
