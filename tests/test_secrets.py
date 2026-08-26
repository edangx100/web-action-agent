"""The no-account guarantee, and the rule that a key is never written down.

Two promises are asserted here, and they pull in opposite directions, which is
why they share a file.

The first is that **nothing about this project requires an account**. Reading
it, installing it, and running the offline suite need no key at all; a task
needs one endpoint; reproducing the published evaluation needs a judge as well.
A user should learn which of those they have before a command does anything,
not after a benchmark has silently skipped grading — so the startup report is
tested for what it says is *missing* at least as hard as for what it says is
present.

The second is that **a key that is supplied never comes back out**. Keys enter
through the environment and ``.env`` alone; the committed configuration file
refuses them by name. Once read, no repr, error message, or startup report may
contain the value — and provider errors are the dangerous case, because they
quote a request back at you and land in logs and committed traces.

The tests below therefore lean on values, not names: a fake key is set, the
program is made to talk, and the output is searched for the key.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from web_action_agent import cli
from web_action_agent.capabilities import (
    AGENT,
    JUDGE,
    JUDGE_API_KEY_ENV,
    JUDGE_BASE_URL_ENV,
    JUDGE_MODEL_ENV,
    JUDGE_PROFILE_ENV,
    VISION,
    report_capabilities,
)
from web_action_agent.config import (
    REDACTED,
    Config,
    ConfigError,
    forget_secrets,
    is_secret_name,
    load_config,
    read_dotenv,
    redact,
    register_secret,
    resolve_env,
    secret_values,
)
from web_action_agent.model_client import (
    MissingAPIKeyError,
    ModelClient,
    ModelClientError,
    load_profile,
    resolve_api_key,
    system_message,
    user_message,
)

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parent.parent
ENV_TEMPLATE = REPO_ROOT / ".env.example"

#: Long enough to be treated as a real key, and unmistakable in output.
A_KEY = "sk-test-000000000000000000000001"
ANOTHER_KEY = "nv-test-000000000000000000000002"


@pytest.fixture(autouse=True)
def _clean_registry() -> Any:
    """Empty the redaction registry around every test in this module.

    The registry is process-wide by design, so without this a key registered by
    one test would still be scrubbed in the next, and a test asserting that
    redaction *happened* would pass whether or not its own code path worked.

    Yields:
        None: after clearing, and clears again on the way out.
    """
    forget_secrets()
    yield
    forget_secrets()


class RaisingTransport:
    """A transport whose every call fails with a message quoting the key.

    Real SDKs do this: an authentication failure or a rate-limit response is
    formatted with the request that caused it, headers included. That string
    then reaches an exception message, a log line, and a run's trace.
    """

    def __init__(self, message: str) -> None:
        """Set up a transport that always raises.

        Args:
            message: The failure text, which the tests seed with a key.
        """
        self._message = message
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **request: Any) -> Any:
        """Fail, the way a provider client fails.

        Args:
            **request: The request the client built. Unused.

        Raises:
            RuntimeError: Always, with the seeded message.
        """
        raise RuntimeError(self._message)


def env_with(**values: str) -> dict[str, str]:
    """Build a standalone environment mapping.

    Args:
        **values: Variable names and values.

    Returns:
        dict[str, str]: The mapping, with nothing inherited from the real
        environment — so a developer's own exported key cannot make a test that
        should see no keys pass for the wrong reason.
    """
    return dict(values)


def report_for(env: dict[str, str], *, profile: str = "openrouter") -> Any:
    """Produce a capability report for one environment.

    Args:
        env: The environment the report is derived from.
        profile: Which endpoint profile the run selects.

    Returns:
        CapabilityReport: The report.
    """
    return report_capabilities(Config(profile=profile), env=env)


# --- What counts as a secret ------------------------------------------------


@pytest.mark.parametrize(
    "name",
    [
        "WAA_OPENROUTER_API_KEY",
        "WAA_API_KEY",
        "WAA_JUDGE_API_KEY",
        "OPENAI_API_KEY",
        "api_key",
        "openrouter_api_key",
        "GITHUB_TOKEN",
        "SOME_SECRET",
        "DB_PASSWORD",
        "TOKEN",
    ],
)
def test_a_name_that_says_it_holds_a_secret_is_treated_as_one(name: str) -> None:
    """The rule reads names, in either the file convention or the shell one."""
    assert is_secret_name(name)


@pytest.mark.parametrize(
    "name",
    ["max_steps", "WAA_PROFILE", "WAA_BASE_URL", "WAA_MODEL", "monkey", "keyboard", "keys"],
)
def test_an_ordinary_name_is_not_mistaken_for_a_secret(name: str) -> None:
    """Nothing that merely contains the letters is caught.

    ``monkey`` ends in ``key`` and ``keyboard`` starts with it; the rule
    matches whole underscore-delimited words, so neither is a secret.
    """
    assert not is_secret_name(name)


def test_short_and_blank_values_are_not_collected() -> None:
    """A one-character value is a flag, not a key, and redacting it would blank
    out unrelated text everywhere that character appeared."""
    collected = secret_values(env_with(A_API_KEY=A_KEY, B_API_KEY="1", C_API_KEY="  "))
    assert collected == {A_KEY}


# --- Redaction --------------------------------------------------------------


def test_every_secret_in_the_environment_is_scrubbed() -> None:
    """Both keys go, and the text around them survives."""
    env = env_with(WAA_OPENROUTER_API_KEY=A_KEY, WAA_NVIDIA_API_KEY=ANOTHER_KEY)
    scrubbed = redact(f"auth failed for {A_KEY} and {ANOTHER_KEY}", env=env)
    assert A_KEY not in scrubbed
    assert ANOTHER_KEY not in scrubbed
    assert scrubbed == f"auth failed for {REDACTED} and {REDACTED}"


def test_a_key_containing_another_is_replaced_whole() -> None:
    """Longest first, or the longer key would be left half-substituted."""
    short, long = "abcdefghij", "abcdefghij-plus-more"
    env = env_with(A_API_KEY=short, B_API_KEY=long)
    assert redact(long, env=env) == REDACTED


def test_ordinary_text_passes_through_untouched() -> None:
    """Redaction is a substitution of known values, not a guess at what a key
    looks like; text with no secret in it is returned as it was."""
    message = "call to openrouter (nemotron-3-nano) failed: connection reset"
    assert redact(message, env=env_with(WAA_API_KEY=A_KEY)) == message


def test_a_key_read_earlier_is_scrubbed_with_no_environment_to_hand() -> None:
    """The registry is what makes redaction work at a call site that has no
    environment — an exception constructor deep in the model client."""
    register_secret(A_KEY)
    assert redact(f"leaked {A_KEY}", env={}) == f"leaked {REDACTED}"


def test_forgetting_the_registry_stops_the_scrubbing() -> None:
    """Pins that the previous test's result came from the registry, not from
    somewhere else scrubbing the value anyway."""
    register_secret(A_KEY)
    forget_secrets()
    assert redact(f"leaked {A_KEY}", env={}) == f"leaked {A_KEY}"


# --- Keys reach the agent through .env, and only through .env ---------------


def test_a_secret_written_into_the_committed_file_is_refused_by_name(tmp_path: Path) -> None:
    """The message names the setting and says where the value belongs.

    The generic unknown-key rule would also catch this, but with a message that
    does not say the one thing that matters: this file is committed, and a key
    in it is a key on its way to a public history.
    """
    config_path = tmp_path / "config.yaml"
    config_path.write_text("openrouter_api_key: sk-not-a-real-key\n")

    with pytest.raises(ConfigError) as caught:
        load_config(config_path=config_path, dotenv_path=tmp_path / ".env", env={})

    message = str(caught.value)
    assert "openrouter_api_key" in message
    assert ".env" in message
    assert "unknown setting" not in message


def test_no_setting_in_the_schema_could_hold_a_key() -> None:
    """The refusal above is a backstop. The structural guarantee is that there
    is no field a key could legitimately be written into in the first place."""
    named = [field for field in Config.model_fields if is_secret_name(field)]
    assert not named, f"configuration fields naming a secret: {', '.join(named)}"


def test_a_key_in_dotenv_reaches_the_endpoint(tmp_path: Path) -> None:
    """``.env`` is not merely read by the configuration loader: it is the
    environment the model client resolves its key from, which is what the
    documented setup — copy the template, fill in a key — depends on."""
    dotenv = tmp_path / ".env"
    dotenv.write_text(f"WAA_OPENROUTER_API_KEY={A_KEY}\n")

    env = resolve_env(dotenv_path=dotenv, env={})
    assert resolve_api_key(load_profile("openrouter"), env) == A_KEY


def test_an_exported_variable_beats_the_same_name_in_the_file(tmp_path: Path) -> None:
    """Both are machine-local; the one typed for a single invocation wins."""
    dotenv = tmp_path / ".env"
    dotenv.write_text(f"WAA_OPENROUTER_API_KEY={A_KEY}\n")

    env = resolve_env(dotenv_path=dotenv, env={"WAA_OPENROUTER_API_KEY": ANOTHER_KEY})
    assert env["WAA_OPENROUTER_API_KEY"] == ANOTHER_KEY


def test_reading_the_environment_registers_its_secrets(tmp_path: Path) -> None:
    """So a key set before the program started is redactable afterwards, even
    though nothing has had a reason to resolve it yet."""
    dotenv = tmp_path / ".env"
    dotenv.write_text(f"WAA_OPENROUTER_API_KEY={A_KEY}\n")

    resolve_env(dotenv_path=dotenv, env={})
    assert redact(f"leaked {A_KEY}", env={}) == f"leaked {REDACTED}"


def test_a_trailing_comment_is_a_comment_not_a_value(tmp_path: Path) -> None:
    """The committed template annotates its variables with trailing comments,
    and the documented setup is to copy it. Reading those comments as values
    hands a numeric setting unparseable text and, worse, hands the endpoint
    that needs no key a key it never had."""
    dotenv = tmp_path / ".env"
    dotenv.write_text(
        "WAA_PROFILE=local           # which endpoint\n"
        "WAA_LOCAL_API_KEY=          # usually none\n"
        f'WAA_API_KEY="{A_KEY}"      # quoted, with a comment after\n'
        "WAA_MODEL=some#model\n"
    )

    values = read_dotenv(dotenv)
    assert values["WAA_PROFILE"] == "local"
    assert values["WAA_LOCAL_API_KEY"] == ""
    assert values["WAA_API_KEY"] == A_KEY
    # A '#' inside a value, with no space before it, is part of the value: keys
    # and model identifiers are allowed to contain one.
    assert values["WAA_MODEL"] == "some#model"


def test_the_committed_template_can_be_copied_verbatim(tmp_path: Path) -> None:
    """Copying ``.env.example`` to ``.env`` is what the README tells a new user
    to do. It must yield a working configuration and no secrets at all."""
    dotenv = tmp_path / ".env"
    dotenv.write_text(ENV_TEMPLATE.read_text())

    assert not secret_values(read_dotenv(dotenv))
    config = load_config(dotenv_path=dotenv, env={})
    assert config.max_steps > 0


def test_the_committed_template_carries_no_key_values() -> None:
    """The template is committed; every secret-named line in it must be empty."""
    filled = {
        name: value
        for name, value in read_dotenv(ENV_TEMPLATE).items()
        if is_secret_name(name) and value
    }
    assert not filled, f".env.example has values for: {', '.join(filled)}"


# --- A key that was supplied never comes back out ---------------------------


def test_a_provider_error_quoting_the_key_is_scrubbed_before_it_is_raised() -> None:
    """The dangerous path: an SDK error formats the request it made, key and
    all, and the agent puts that message into a log and a committed trace."""
    register_secret(A_KEY)
    client = ModelClient(
        load_profile("openrouter"),
        transport=RaisingTransport(f"401 Unauthorized: Authorization=Bearer {A_KEY}"),
        env={},
    )

    with pytest.raises(ModelClientError) as caught:
        client.complete([system_message("be brief"), user_message("what now?")])

    message = str(caught.value)
    assert A_KEY not in message
    assert REDACTED in message
    # Still diagnostic: the endpoint and the status survive.
    assert "openrouter" in message
    assert "401" in message


def test_the_client_describes_itself_without_its_key() -> None:
    """Reprs are quoted into tracebacks and debug logs."""
    client = ModelClient(load_profile("openrouter"), api_key=A_KEY, transport=object(), env={})
    assert A_KEY not in repr(client)
    assert "openrouter" in repr(client)


def test_a_missing_key_names_the_variable_and_nothing_else() -> None:
    """The one error that is *about* a key must still not contain one."""
    with pytest.raises(MissingAPIKeyError) as caught:
        resolve_api_key(load_profile("openrouter"), env_with(WAA_NVIDIA_API_KEY=A_KEY))

    message = str(caught.value)
    assert "WAA_OPENROUTER_API_KEY" in message
    assert A_KEY not in message


def test_resolving_a_key_registers_it_for_redaction() -> None:
    """After the model client has read a key, anything formatted anywhere in
    the program has that value scrubbed out of it."""
    resolve_api_key(load_profile("openrouter"), env_with(WAA_OPENROUTER_API_KEY=A_KEY))
    assert redact(f"trace line: {A_KEY}", env={}) == f"trace line: {REDACTED}"


def test_the_call_record_written_to_the_trace_holds_no_key() -> None:
    """Every call appends one of these to the run's cost log, so it is the
    record most certain to be written to disk."""

    class Transport:
        def __init__(self) -> None:
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

        def _create(self, **request: Any) -> Any:
            message = SimpleNamespace(content="{}")
            return SimpleNamespace(
                choices=[SimpleNamespace(message=message, finish_reason="stop")],
                usage=SimpleNamespace(prompt_tokens=7, completion_tokens=3),
            )

    client = ModelClient(load_profile("openrouter"), api_key=A_KEY, transport=Transport(), env={})
    response = client.complete([user_message("what now?")])
    assert A_KEY not in repr(response.record)
    assert response.record.prompt_tokens == 7


# --- The startup report -----------------------------------------------------


def test_with_no_keys_the_report_names_the_judge_unavailable() -> None:
    """The headline case: a checkout with nothing configured. Nothing raises,
    and every capability says plainly that it is not available."""
    report = report_for(env_with())

    assert not report.can(AGENT)
    assert not report.can(VISION)
    assert not report.can(JUDGE)
    assert JUDGE_PROFILE_ENV in report.get(JUDGE).detail


def test_configuration_loads_with_no_keys_at_all(tmp_path: Path) -> None:
    """Loading configuration must never need a key: the offline suite, and
    ``waa --help``, run on a checkout with no account behind it."""
    config = load_config(dotenv_path=tmp_path / "absent.env", env={})
    assert config.profile


def test_help_prints_with_no_keys_and_no_configuration(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Asking what the commands are is the first thing a new reader does, and
    it must work before anything has been set up."""
    monkeypatch.setattr("sys.argv", ["waa", "--help"])
    assert cli.main() == 0

    printed = capsys.readouterr().out
    assert "usage: waa" in printed
    for command in cli.COMMANDS:
        assert command in printed


def test_a_key_makes_the_agent_available_and_names_the_model() -> None:
    """One key is the whole of what running tasks requires."""
    report = report_for(env_with(WAA_OPENROUTER_API_KEY=A_KEY))

    assert report.can(AGENT)
    assert "openrouter" in report.get(AGENT).detail
    assert not report.can(JUDGE)


def test_an_endpoint_that_needs_no_key_is_available_without_one() -> None:
    """The no-account claim at its strongest: a self-hosted server, no key
    anywhere, and the agent still reports itself ready."""
    report = report_for(env_with(), profile="local")
    assert report.can(AGENT)


def test_vision_is_unavailable_where_the_profile_serves_no_vision_model() -> None:
    """Said out loud rather than discovered when a screenshot is dropped."""
    report = report_for(env_with(), profile="local")
    assert not report.can(VISION)
    assert "DOM text" in report.get(VISION).detail


def test_a_judge_profile_with_a_key_is_available() -> None:
    """One of the two ways to name a judge: a profile from the packaged file."""
    report = report_for(
        env_with(
            WAA_OPENROUTER_API_KEY=A_KEY,
            **{JUDGE_PROFILE_ENV: "nvidia", "WAA_NVIDIA_API_KEY": ANOTHER_KEY},
        )
    )
    assert report.can(JUDGE)
    assert "nvidia" in report.get(JUDGE).detail


def test_an_ad_hoc_judge_endpoint_is_available_when_fully_described() -> None:
    """The other way: an endpoint with no profile of its own."""
    report = report_for(
        env_with(
            WAA_OPENROUTER_API_KEY=A_KEY,
            **{
                JUDGE_BASE_URL_ENV: "https://example.test/v1",
                JUDGE_API_KEY_ENV: ANOTHER_KEY,
                JUDGE_MODEL_ENV: "a-stronger-model",
            },
        )
    )
    assert report.can(JUDGE)
    assert "a-stronger-model" in report.get(JUDGE).detail


@pytest.mark.parametrize("omitted", [JUDGE_API_KEY_ENV, JUDGE_MODEL_ENV])
def test_a_half_described_judge_is_unavailable_and_says_which_half(omitted: str) -> None:
    """A partially configured judge is the failure this report exists to catch:
    it looks configured, and would grade nothing."""
    env = env_with(
        **{
            JUDGE_BASE_URL_ENV: "https://example.test/v1",
            JUDGE_API_KEY_ENV: ANOTHER_KEY,
            JUDGE_MODEL_ENV: "a-stronger-model",
        }
    )
    del env[omitted]

    report = report_for(env)
    assert not report.can(JUDGE)
    assert omitted in report.get(JUDGE).detail


def test_a_misspelled_judge_profile_is_reported_not_raised() -> None:
    """Nothing about the judge may stop a run that never needed one."""
    report = report_capabilities(
        Config(profile="openrouter"),
        env=env_with(WAA_OPENROUTER_API_KEY=A_KEY, **{JUDGE_PROFILE_ENV: "nvidai"}),
    )
    assert report.can(AGENT)
    assert not report.can(JUDGE)
    assert "nvidai" in report.get(JUDGE).detail


def test_a_misspelled_run_profile_is_reported_not_raised() -> None:
    """Likewise for the agent's own profile: the report says what is wrong
    rather than the program ending with a traceback before it prints anything."""
    report = report_for(env_with(WAA_OPENROUTER_API_KEY=A_KEY), profile="openrooter")
    assert not report.can(AGENT)
    assert not report.can(VISION)
    assert "openrooter" in report.get(AGENT).detail


def test_the_rendered_report_never_contains_a_key() -> None:
    """The report is the one string guaranteed to be printed on every run,
    which makes it the worst possible place for a key to appear."""
    env = env_with(
        WAA_OPENROUTER_API_KEY=A_KEY,
        **{JUDGE_BASE_URL_ENV: "https://example.test/v1", JUDGE_API_KEY_ENV: ANOTHER_KEY},
    )
    rendered = report_capabilities(Config(profile="openrouter"), env=env).render()

    assert A_KEY not in rendered
    assert ANOTHER_KEY not in rendered
    assert "WAA_OPENROUTER_API_KEY" not in rendered  # available: no need to name it


def test_the_report_has_one_line_for_each_capability() -> None:
    """Three capabilities under a heading, so a reader sees all three whether
    or not they are available."""
    lines = report_for(env_with()).render().splitlines()
    assert lines[0] == "capabilities:"
    assert len(lines) == 4
    assert [line.split()[1] for line in lines[1:]] == [AGENT, VISION, JUDGE]


def test_asking_for_a_capability_that_does_not_exist_is_an_error() -> None:
    """A typo in a caller must not read as "unavailable"."""
    with pytest.raises(KeyError):
        report_for(env_with()).can("judgement")
