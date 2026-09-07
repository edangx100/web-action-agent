"""The one place the agent talks to a language model.

Planner, judge, and inducer are three different jobs, but they are one kind of
call: some messages go out, some JSON comes back, and the tokens it cost are
written down. Giving all three a single transport is what makes the endpoint a
setting rather than a fact of the code — swap ``--profile openrouter`` for
``--profile local`` and every call in the system moves, with nothing recompiled
and nothing else edited.

**Profiles.** An endpoint is described entirely by data, in ``profiles.yaml``:
a base URL, the name of the environment variable holding its key, and one or
two *routes* — the text model, and the vision-language model used when an
observation carries a screenshot. Nothing in this module knows what
``openrouter`` means; a profile invented in a file five minutes ago is handled
exactly like the three that ship. That property is what the abstraction claim
rests on, so there is a test for it.

**Pinned routing.** An aggregator serves one model from several hosts, which
differ in quantisation, in which parameters they honour, and in what they cost.
A benchmark that does not say which host answered is not reproducible, so each
route pins its host and forbids silent fallback. This has to be set before the
evaluation runs, not after: numbers already collected cannot be pinned
retroactively.

**Structured output, and its safety net.** Every call that expects JSON asks
for schema-constrained decoding, so the endpoint itself refuses to emit
anything but a conforming object. Support is uneven — one shipped route is a
host that does not offer it — so a route that cannot constrain decoding still
gets the schema in the prompt, and unparseable output raises
:class:`ModelResponseError` for the caller's retry loop to catch. Malformed
JSON is a normal event to be retried, never a crash.

**Keys, and where they are not.** A profile names the *variable* that holds
its key, never the key. The value is read from the environment — ``.env``
merged under the exported variables — at construction, so a missing key is
"you have not set one" before the browser opens rather than an HTTP 401 in the
middle of step seven. Once read, a key lives in the underlying HTTP client and
nowhere else: the repr shows the profile and model, and every error this
module raises is scrubbed of key values on its way out, because those errors
quote provider responses into logs and committed traces.

**What is deliberately not here yet.** Rate-limit backoff and per-call dollar
costs are separate pieces of work that plug into the seams this module already
provides: ``max_retries=0`` on the underlying client leaves one owner for retry
policy rather than two uncoordinated ones, and :class:`CallRecord` carries the
``cost_usd`` field that the pricing table will fill in.
"""

from __future__ import annotations

import base64
import json
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Self

import yaml
from openai import OpenAI
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from web_action_agent.config import Config, redact, register_secret, resolve_env

#: The committed endpoint descriptions. Packaged next to the module so the file
#: is found identically from a checkout and from an installed wheel.
DEFAULT_PROFILES_PATH = Path(__file__).resolve().parent / "profiles.yaml"

#: Environment variable that overrides the selected profile's key. Useful for a
#: throwaway key or a second account without editing anything.
API_KEY_OVERRIDE_ENV = "WAA_API_KEY"

#: Sent where a server requires the OpenAI client to present *some* key but
#: authenticates nobody — the usual case for a local vLLM server.
UNUSED_API_KEY = "not-needed"

#: Planner and judge both decode greedily, and pass a fixed seed to endpoints
#: that honour one. Two runs of one task should differ because the web changed,
#: not because the sampler rolled differently.
DEFAULT_TEMPERATURE = 0.0
DEFAULT_SEED = 0


class ModelClientError(Exception):
    """Any failure of this module: bad profile, bad key, bad call, bad reply.

    One exception type per component keeps a caller's error handling honest —
    "the model layer failed" is the distinction that matters to the agent loop,
    not which library raised underneath.

    Every message is redacted on the way in. This is the module that holds the
    keys, and it is also the module whose errors quote provider responses back
    verbatim — a rate-limit body, an auth failure, an SDK error that echoes the
    request it made. Scrubbing in the base constructor means the guarantee
    holds for errors this module has not grown yet, rather than depending on
    each new ``raise`` site remembering.
    """

    def __init__(self, message: str) -> None:
        """Build the error with any key value removed from its message.

        Args:
            message: The diagnostic, which may quote a provider's response.
        """
        super().__init__(redact(message))


class ProfileError(ModelClientError):
    """A profile is missing, misspelled, or malformed."""


class MissingAPIKeyError(ModelClientError):
    """A profile requires a key and its environment variable is unset or empty.

    The message names the variable to set. It never contains a key value: this
    exception text reaches logs and traces, which are committed artifacts.
    """


class ModelResponseError(ModelClientError):
    """The endpoint replied, but not with the JSON that was asked for.

    Expected rather than exceptional. Where a route cannot constrain decoding,
    a model wrapping its object in prose is an ordinary event, and the caller's
    bounded retry is the designed response.

    Attributes:
        text: The raw reply, kept so a retry prompt can quote what went wrong.
        record: What the failed call cost, when the failure happened after the
            endpoint answered. A reply that could not be parsed was still
            billed, and a run whose reported cost quietly omitted its wasted
            calls would flatter itself -- so the record travels out on the
            exception rather than being lost with it. None where the call never
            got far enough to have a cost.
    """

    def __init__(self, message: str, *, text: str = "", record: CallRecord | None = None) -> None:
        super().__init__(message)
        self.text = text
        self.record = record


class Route(BaseModel):
    """One model on one endpoint, plus how it has to be called.

    A profile has a text route and, where the endpoint serves a vision-language
    model, a vision route. They are separate because the two models are often
    reached through different hosts with different capabilities — which is the
    case for one of the shipped profiles today.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    model: str = Field(description="Model identifier as the endpoint spells it.")
    provider_order: tuple[str, ...] = Field(
        default=(),
        description=(
            "Hosts to route to, in order, on an aggregator that has several. "
            "Empty where the endpoint does its own routing or has one host."
        ),
    )
    supports_json_schema: bool = Field(
        default=True,
        description=(
            "Whether this route honours schema-constrained decoding. False "
            "leaves the validator's schema check and retry as the only net."
        ),
    )


class Profile(BaseModel):
    """A named endpoint: where to send calls, how to authenticate, what to call.

    This is the whole of what the agent knows about a model provider. Switching
    provider means selecting a different one of these.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(description="The name selected with --profile.")
    base_url: str = Field(description="OpenAI-compatible API root.")
    api_key_env: str = Field(description="Environment variable holding the key — never the key.")
    api_key_required: bool = Field(
        default=True,
        description="False only where the server authenticates nobody, e.g. a local vLLM.",
    )
    text: Route = Field(description="Route used when the observation is DOM text alone.")
    vision: Route | None = Field(
        default=None,
        description=(
            "Route used when a screenshot is attached; null where the endpoint "
            "serves no vision model, which disables the screenshot channel."
        ),
    )

    @property
    def supports_vision(self) -> bool:
        """Whether observations may include a screenshot under this profile.

        Derived rather than declared, so a profile cannot advertise vision it
        has no model to serve.

        Returns:
            bool: True if a vision route is configured.
        """
        return self.vision is not None

    @property
    def model(self) -> str:
        """The text model — what "which model is this run using" ordinarily means.

        Returns:
            str: The text route's model identifier.
        """
        return self.text.model

    def route_for(self, *, vision: bool) -> Route:
        """Pick the route a call needs.

        Args:
            vision: Whether the call carries an image.

        Returns:
            Route: The vision route if one was asked for and exists, else the
            text route.

        Raises:
            ModelClientError: If an image is sent under a profile with no vision
                route. Refusing beats silently dropping the image: the planner
                would then be answering about a page it was never shown.
        """
        if not vision:
            return self.text
        if self.vision is None:
            raise ModelClientError(
                f"profile {self.name!r} has no vision route: it cannot accept screenshots. "
                f"Use a profile that defines one, or disable the screenshot channel."
            )
        return self.vision


def load_profiles(path: Path | None = None) -> dict[str, Profile]:
    """Read every profile from the committed profiles file.

    Args:
        path: File to read. Defaults to the packaged ``profiles.yaml``.

    Returns:
        dict[str, Profile]: Profiles keyed by name.

    Raises:
        ProfileError: If the file is missing, is not a mapping of mappings, or
            any entry fails validation — an unknown key included, since a
            misspelled ``supports_json_schema`` that was ignored would mean a
            silent loss of structured decoding.
    """
    path = path if path is not None else DEFAULT_PROFILES_PATH
    if not path.exists():
        raise ProfileError(f"{path}: no profiles file")

    try:
        parsed = yaml.safe_load(path.read_text())
    except yaml.YAMLError as error:
        raise ProfileError(f"{path}: not valid YAML: {error}") from error

    if not isinstance(parsed, dict) or not parsed:
        raise ProfileError(f"{path}: expected a non-empty mapping of profile name to settings")

    profiles: dict[str, Profile] = {}
    for name, body in parsed.items():
        if not isinstance(body, dict):
            raise ProfileError(f"{path}: profile {name!r} is not a mapping")
        try:
            # The name is the file's key rather than a field inside the entry,
            # so the two cannot disagree about what a profile is called.
            profiles[name] = Profile(name=name, **body)
        except ValidationError as error:
            raise ProfileError(f"{path}: profile {name!r} is invalid: {error}") from error
    return profiles


def load_profile(name: str, *, path: Path | None = None) -> Profile:
    """Look one profile up by name.

    Args:
        name: The profile name, as passed to ``--profile``.
        path: File to read. Defaults to the packaged ``profiles.yaml``.

    Returns:
        Profile: The named profile.

    Raises:
        ProfileError: If no profile has that name. The message lists the names
            that do exist, since the usual cause is a typo.
    """
    profiles = load_profiles(path)
    try:
        return profiles[name]
    except KeyError:
        known = ", ".join(sorted(profiles))
        raise ProfileError(f"unknown profile {name!r}. Known profiles: {known}") from None


def resolve_profile(config: Config, *, path: Path | None = None) -> Profile:
    """Select the profile for a run and apply the run's overrides to it.

    The overrides exist for the case the profiles file cannot anticipate:
    pointing the local profile at a vLLM server on another host, or trying one
    model for a single run. They change the *text* route only — a vision model
    is chosen together with the host that serves it, so swapping it is an edit
    to ``profiles.yaml``, where both can change together.

    Args:
        config: The resolved run configuration, supplying ``profile`` and the
            optional ``base_url`` and ``model`` overrides.
        path: File to read. Defaults to the packaged ``profiles.yaml``.

    Returns:
        Profile: The selected profile, with any overrides applied.

    Raises:
        ProfileError: If the named profile does not exist.
    """
    profile = load_profile(config.profile, path=path)

    if config.base_url:
        profile = profile.model_copy(update={"base_url": config.base_url})
    if config.model:
        profile = profile.model_copy(
            update={"text": profile.text.model_copy(update={"model": config.model})}
        )
    return profile


def resolve_api_key(profile: Profile, env: Mapping[str, str]) -> str:
    """Find the key for a profile, without ever putting one in a config file.

    Precedence mirrors configuration generally: the ad-hoc override wins over
    the profile's own variable. Keys are read from the environment alone — the
    committed files hold variable *names*, so a key cannot ride into a public
    commit through them.

    Args:
        profile: The profile whose endpoint is being called.
        env: The environment to read, ``.env`` merged under the real one.

    Returns:
        str: The key, or a placeholder where the endpoint requires none. The
        OpenAI client insists on a non-empty string even when the server does
        not look at it.

    Raises:
        MissingAPIKeyError: If a key is required and the variable is unset or
            blank. Raised at construction, before any call is attempted, so the
            failure is "you have not set a key" rather than an HTTP 401 in the
            middle of step seven.
    """
    for variable in (API_KEY_OVERRIDE_ENV, profile.api_key_env):
        value = (env.get(variable) or "").strip()
        if value:
            # Remembered so that any later message quoting a provider response
            # has this value scrubbed out of it, wherever it is formatted.
            register_secret(value)
            return value

    if not profile.api_key_required:
        return UNUSED_API_KEY
    raise MissingAPIKeyError(
        f"profile {profile.name!r} needs an API key: set {profile.api_key_env} "
        f"in .env (or {API_KEY_OVERRIDE_ENV} to override it for one run)"
    )


@dataclass(frozen=True)
class CallRecord:
    """What one model call cost, written to the run's trace.

    Every call produces one of these, whether it succeeded in returning usable
    JSON or not: a run's reported cost has to include the calls that were
    wasted, or the cost-per-task figure flatters itself.

    Attributes:
        profile: Which profile served the call.
        model: The exact model identifier used — the text or vision route's,
            resolved, not the one configured.
        vision: Whether an image was part of the request.
        prompt_tokens: Input tokens billed, or None if the endpoint reports none.
        completion_tokens: Output tokens billed, or None if not reported.
        latency_s: Wall-clock seconds for the call. The only cost measure a
            self-hosted endpoint has, which is why it is always recorded.
        finish_reason: Why generation stopped; ``"length"`` here explains a
            truncated action better than the parse failure it causes.
        cost_usd: Dollars, filled in later from the checked-in pricing table.
            None where the route is free, local, or unpriced — reported as
            unknown rather than guessed at as zero.
        schema_enforced: Whether the endpoint was asked to constrain decoding
            to the schema, rather than merely being shown one in the prompt.
            False covers both "no schema was asked for" and "this route cannot
            honour one", which is what makes the field worth recording: a run
            with a high rate of unparseable replies is explained by these being
            False, and that is a routing problem rather than a model one.
    """

    profile: str
    model: str
    vision: bool
    prompt_tokens: int | None
    completion_tokens: int | None
    latency_s: float
    finish_reason: str | None
    cost_usd: float | None = None
    schema_enforced: bool = False

    @property
    def total_tokens(self) -> int | None:
        """Tokens billed for the call, input plus output.

        Returns:
            int | None: The sum, or None if either half went unreported.
        """
        if self.prompt_tokens is None or self.completion_tokens is None:
            return None
        return self.prompt_tokens + self.completion_tokens


@dataclass(frozen=True)
class ModelResponse:
    """One reply, in both the forms callers need, plus its cost record.

    Attributes:
        text: The raw reply text.
        data: The parsed object, when the call asked for JSON. None otherwise.
        record: The call's token/latency record, for the trace.
    """

    text: str
    data: dict[str, Any] | None
    record: CallRecord


def system_message(text: str) -> dict[str, Any]:
    """Build a system message.

    Args:
        text: The system prompt.

    Returns:
        dict[str, Any]: A chat message in the wire format.
    """
    return {"role": "system", "content": text}


def user_message(text: str, *, screenshot_png: bytes | None = None) -> dict[str, Any]:
    """Build a user message, optionally carrying a screenshot.

    Keeping the image encoding here means the observer hands over PNG bytes and
    never learns how a chat API spells an attachment; if that spelling changes,
    it changes once.

    Args:
        text: The message text — for the planner, the observation and the task.
        screenshot_png: PNG bytes of the annotated screenshot, or None for a
            text-only observation. Present bytes make this a vision call, which
            in turn selects the profile's vision route.

    Returns:
        dict[str, Any]: A chat message in the wire format.
    """
    if screenshot_png is None:
        return {"role": "user", "content": text}

    encoded = base64.b64encode(screenshot_png).decode("ascii")
    return {
        "role": "user",
        "content": [
            {"type": "text", "text": text},
            # Inline data URL rather than a hosted URL: the screenshots are
            # local, transient, and not worth uploading anywhere.
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded}"}},
        ],
    }


def has_image(messages: Sequence[Mapping[str, Any]]) -> bool:
    """Report whether any message carries an image part.

    The caller does not declare "this is a vision call"; it is inferred from
    what is being sent, so the two cannot disagree.

    Args:
        messages: The chat messages about to be sent.

    Returns:
        bool: True if at least one message has an ``image_url`` content part.
    """
    for message in messages:
        content = message.get("content")
        if isinstance(content, list) and any(
            isinstance(part, Mapping) and part.get("type") == "image_url" for part in content
        ):
            return True
    return False


def parse_json_object(text: str) -> dict[str, Any]:
    """Parse a JSON object out of a model reply.

    A route that constrains decoding returns bare JSON and the first attempt
    succeeds. A route that cannot often returns the object inside a fenced
    block or a sentence of preamble, so one narrow salvage is attempted: the
    span from the first brace to the last. That is the whole of the leniency —
    anything further would be the regex-parsing of model prose this design
    exists to avoid.

    Args:
        text: The raw reply.

    Returns:
        dict[str, Any]: The parsed object.

    Raises:
        ModelResponseError: If no JSON object can be read out of the reply.
    """
    candidates = [text.strip()]
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        candidates.append(text[start : end + 1])

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed

    raise ModelResponseError("model reply was not a JSON object", text=text)


class ModelClient:
    """The transport every model call in the project goes through.

    Construct one per profile and reuse it; the underlying HTTP client pools
    connections, and the agent loop makes one call per step.

    The client holds the key only inside that HTTP client, and its repr shows
    the profile and model alone — traces and crash dumps quote reprs.
    """

    def __init__(
        self,
        profile: Profile,
        *,
        api_key: str | None = None,
        timeout_s: float = 90.0,
        env: Mapping[str, str] | None = None,
        transport: Any | None = None,
    ) -> None:
        """Build a client for one endpoint.

        Args:
            profile: The endpoint to call.
            api_key: The key, if already resolved. Normally left to be read
                from ``env``.
            timeout_s: Ceiling on one call. Expiry is a categorized failure the
                run reports, not an exception that ends it without a verdict.
            env: Environment to read the key from. Defaults to ``.env``
                merged under the process environment.
            transport: An object shaped like ``openai.OpenAI``. Injected by
                tests so the whole request-building path can be asserted on
                without a network, a key, or a bill.

        Raises:
            MissingAPIKeyError: If the profile needs a key and none is set.
        """
        self.profile = profile
        self.timeout_s = timeout_s
        # Screenshots discarded by observation_message for want of a vision
        # route. Expected to stay at zero; see the property for why it is
        # counted rather than raised on.
        self._dropped_screenshots = 0
        if api_key:
            register_secret(api_key)
        self._transport = transport or OpenAI(
            api_key=api_key or resolve_api_key(profile, env if env is not None else resolve_env()),
            base_url=profile.base_url,
            timeout=timeout_s,
            # Retries are switched off here on purpose. Bounded backoff for
            # rate-limited endpoints is a policy this project owns and records
            # in the trace; two retry layers stacked would multiply the wait
            # and hide the 429s from the cost log.
            max_retries=0,
        )

    @classmethod
    def from_config(
        cls,
        config: Config,
        *,
        env: Mapping[str, str] | None = None,
        profiles_path: Path | None = None,
        transport: Any | None = None,
    ) -> Self:
        """Build the client a run's configuration asks for.

        The single place profile selection happens, so no component downstream
        has a reason to know profile names at all.

        Args:
            config: The resolved run configuration.
            env: Environment to read the key from. Defaults to ``.env``
                merged under the process environment.
            profiles_path: Profiles file. Defaults to the packaged one.
            transport: Injected client, as in :meth:`__init__`.

        Returns:
            ModelClient: A client for the configured profile, using the run's
            model-call timeout.
        """
        return cls(
            resolve_profile(config, path=profiles_path),
            timeout_s=config.llm_timeout_s,
            env=env,
            transport=transport,
        )

    @property
    def supports_vision(self) -> bool:
        """Whether this client can accept screenshots.

        Returns:
            bool: True if the profile has a vision route.
        """
        return self.profile.supports_vision

    @property
    def dropped_screenshots(self) -> int:
        """int: Screenshots discarded because this profile cannot receive one.

        Expected to be zero on a correctly wired run. The screenshot gate
        already answers "no screenshot" for every step of a profile without a
        vision route, so an image should never be read off disk in the first
        place; a non-zero count here means some caller went around the gate.

        It is a count rather than an exception because the guarantee this
        backstop exists to serve is that the run *keeps going* on a text-only
        endpoint. But a drop nobody can see afterwards is its own kind of bug,
        so the number is kept for the trace to report.
        """
        return self._dropped_screenshots

    def observation_message(
        self, text: str, *, screenshot_png: bytes | None = None
    ) -> dict[str, Any]:
        """Build the planner's message for one observation, with or without vision.

        This is what makes "no vision model" a degraded mode instead of a dead
        run. :func:`user_message` attaches an image to anything it is given, and
        :meth:`complete` then refuses the call outright when the profile has no
        vision route. That refusal is right as a last line of defence -- sending
        an image to a model that cannot read one is worse than stopping -- but
        it is useless as a way to finish a task. Built here instead, the image
        is left behind and the step proceeds on its text channels, which are the
        channels that have to be sufficient anyway.

        The two defences are deliberately independent. The screenshot gate
        decides not to ask for an image; this decides not to send one it was
        handed regardless. Either alone would hold the guarantee, so a bug in
        one does not breach it.

        Args:
            text: The observation and the task, already rendered for the prompt.
            screenshot_png: PNG bytes of the annotated screenshot, or None for a
                text-only step. Bytes given under a profile with no vision route
                are dropped, and the drop counted in :attr:`dropped_screenshots`.

        Returns:
            dict[str, Any]: A user message carrying the image only when this
            client can actually deliver it, and text alone otherwise.
        """
        if screenshot_png is not None and not self.supports_vision:
            self._dropped_screenshots += 1
            screenshot_png = None
        return user_message(text, screenshot_png=screenshot_png)

    def complete(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        schema: Mapping[str, Any] | None = None,
        schema_name: str = "response",
        temperature: float = DEFAULT_TEMPERATURE,
        seed: int | None = DEFAULT_SEED,
        max_tokens: int | None = None,
    ) -> ModelResponse:
        """Make one call and return the reply, parsed, with its cost record.

        Args:
            messages: Chat messages, built with :func:`system_message` and
                :meth:`observation_message`. An image part in any of them makes
                this a vision call and selects the profile's vision route -- or,
                where the profile has none, fails the call. Building the user
                message through :meth:`observation_message` is what keeps a
                text-only run from ever reaching that failure.
            schema: JSON Schema the reply must satisfy. Supplying one asks the
                endpoint to constrain decoding where the route supports it, and
                makes the reply be parsed either way.
            schema_name: Name for the schema in the request; some endpoints
                report it back in errors.
            temperature: Sampling temperature. Zero by default, because a
                planner that answers differently on identical input makes an
                evaluation impossible to interpret.
            seed: Seed for endpoints that honour one. None omits it.
            max_tokens: Output ceiling, or None for the endpoint's default.

        Returns:
            ModelResponse: The reply text, the parsed object when a schema was
            asked for, and the record of what the call cost.

        Raises:
            ModelClientError: If the call itself failed — network, auth, rate
                limit, or an image sent to a profile with no vision route.
            ModelResponseError: If a schema was asked for and the reply is not
                a JSON object. The caller's bounded retry is meant to catch it,
                and the exception carries the call's cost record so the retry
                can still bill the attempt.
        """
        vision = has_image(messages)
        route = self.profile.route_for(vision=vision)

        request: dict[str, Any] = {
            "model": route.model,
            "messages": list(messages),
            "temperature": temperature,
        }
        if seed is not None:
            request["seed"] = seed
        if max_tokens is not None:
            request["max_tokens"] = max_tokens
        constrained = schema is not None and route.supports_json_schema
        if constrained:
            # Schema-constrained decoding: the endpoint may emit nothing but a
            # conforming object, so there is no parse to fail. Where the route
            # does not support it, the schema still reaches the model through
            # the prompt, and parse_json_object plus the caller's retry cover.
            request["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": schema_name, "schema": dict(schema), "strict": True},
            }

        extra_body = self._routing_body(route)
        if extra_body:
            request["extra_body"] = extra_body

        started = time.perf_counter()
        try:
            completion = self._transport.chat.completions.create(**request)
        except Exception as error:
            # Everything from DNS failure to HTTP 429 arrives here as some
            # provider-specific type. Callers care that the model layer failed
            # and why in words; the retry policy reads the message.
            raise ModelClientError(
                f"call to {self.profile.name} ({route.model}) failed: {error}"
            ) from error
        latency_s = time.perf_counter() - started

        text, finish_reason = self._read_choice(completion)
        record = CallRecord(
            profile=self.profile.name,
            model=route.model,
            vision=vision,
            prompt_tokens=self._usage(completion, "prompt_tokens"),
            completion_tokens=self._usage(completion, "completion_tokens"),
            latency_s=latency_s,
            finish_reason=finish_reason,
            schema_enforced=constrained,
        )
        if schema is None:
            return ModelResponse(text=text, data=None, record=record)
        try:
            data = parse_json_object(text)
        except ModelResponseError as error:
            # Re-raised rather than propagated so the caller's retry can bill
            # the call it is retrying. Same message, same reply text, now with
            # the cost attached.
            raise ModelResponseError(str(error), text=error.text, record=record) from error
        return ModelResponse(text=text, data=data, record=record)

    @staticmethod
    def _routing_body(route: Route) -> dict[str, Any]:
        """Build the vendor-specific routing block for a route, if it pins one.

        Args:
            route: The route about to be called.

        Returns:
            dict[str, Any]: The extra request body, or empty where the route
            pins no host. ``allow_fallbacks`` is off whenever an order is
            given: a pin that may be silently departed from is not a pin, and
            the run would no longer be reproducible without saying so.
        """
        if not route.provider_order:
            return {}
        return {"provider": {"order": list(route.provider_order), "allow_fallbacks": False}}

    @staticmethod
    def _read_choice(completion: Any) -> tuple[str, str | None]:
        """Pull the reply text and finish reason out of a completion.

        Args:
            completion: The endpoint's response object.

        Returns:
            tuple[str, str | None]: The message content (empty string if the
            model returned none) and the finish reason.

        Raises:
            ModelClientError: If the response carries no choices at all, which
                some endpoints do instead of returning an error.
        """
        choices = getattr(completion, "choices", None) or []
        if not choices:
            raise ModelClientError("endpoint returned no choices")
        choice = choices[0]
        return getattr(choice.message, "content", None) or "", getattr(
            choice, "finish_reason", None
        )

    @staticmethod
    def _usage(completion: Any, field: str) -> int | None:
        """Read one token count, tolerating endpoints that report none.

        Args:
            completion: The endpoint's response object.
            field: ``"prompt_tokens"`` or ``"completion_tokens"``.

        Returns:
            int | None: The count, or None where the endpoint omits usage —
            recorded as unknown rather than as zero, so a missing number cannot
            masquerade as a free call.
        """
        usage = getattr(completion, "usage", None)
        value = getattr(usage, field, None) if usage is not None else None
        return int(value) if isinstance(value, int) else None

    def close(self) -> None:
        """Close the underlying HTTP connections, if the transport has any."""
        close = getattr(self._transport, "close", None)
        if callable(close):
            close()

    def __enter__(self) -> Self:
        """Return self, so a client can be used as a context manager."""
        return self

    def __exit__(self, *exc_info: object) -> None:
        """Close the client on leaving the block."""
        self.close()

    def __repr__(self) -> str:
        """Describe the client without disclosing its key.

        Returns:
            str: The profile name, text model, and vision availability.
        """
        return (
            f"ModelClient(profile={self.profile.name!r}, model={self.profile.model!r}, "
            f"vision={self.supports_vision})"
        )
