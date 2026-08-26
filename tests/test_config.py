"""What the three configuration layers promise, asserted layer by layer.

Precedence is the kind of rule that is easy to state and easy to get subtly
wrong: an override that silently loses to a default produces a run that looks
fine and measures the wrong thing. So each pairing is pinned here directly —
all three layers set, then two, then one — rather than inferred from the
merge order in the code.

The other half of these tests is about what the loader *refuses*. A
configuration system that ignores what it does not recognise turns a typo into
a run at a value nobody chose, and turns an API key written into the committed
file into a secret in a public commit. Both are asserted to raise.
"""

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from web_action_agent.config import (
    DEFAULT_CONFIG_PATH,
    Config,
    ConfigError,
    env_var_for,
    flag_for,
    load_config,
    read_dotenv,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
ENV_TEMPLATE = REPO_ROOT / ".env.example"


def write(path: Path, text: str) -> Path:
    """Write ``text`` to ``path`` and return the path, for use inline in a call.

    Args:
        path: File to write. Its parent must exist.
        text: Contents to write.

    Returns:
        Path: The path just written.
    """
    path.write_text(text)
    return path


def yaml_keys(path: Path) -> list[str]:
    """Return the top-level keys of a YAML mapping file.

    Args:
        path: The file to read.

    Returns:
        list[str]: Its top-level keys, in file order.
    """
    return list(yaml.safe_load(path.read_text()))


def load(
    tmp_path: Path,
    *,
    yaml_text: str = "",
    dotenv_text: str = "",
    env: dict[str, str] | None = None,
    cli: dict[str, object] | None = None,
) -> Config:
    """Resolve a configuration from layers supplied as text, in a temp directory.

    Keeps every test isolated from the real ``config.yaml``, the real ``.env``,
    and the real environment: each layer here is only what the test names.

    Args:
        tmp_path: pytest's per-test directory, used to hold the two files.
        yaml_text: Contents of the committed-defaults layer.
        dotenv_text: Contents of the ``.env`` layer.
        env: Exported environment variables.
        cli: Per-run flag overrides, keyed by field name.

    Returns:
        Config: The resolved configuration.
    """
    return load_config(
        config_path=write(tmp_path / "config.yaml", yaml_text),
        dotenv_path=write(tmp_path / ".env", dotenv_text),
        env=env or {},
        cli=cli or {},
    )


@pytest.mark.unit
def test_flag_beats_env_beats_file(tmp_path: Path) -> None:
    """With one setting in all three layers, the flag is the one that lands."""
    config = load(
        tmp_path,
        yaml_text="max_steps: 10\n",
        dotenv_text="WAA_MAX_STEPS=20\n",
        cli={"max_steps": 30},
    )
    assert config.max_steps == 30


@pytest.mark.unit
def test_env_beats_file(tmp_path: Path) -> None:
    """With the flag absent, the environment beats the committed default."""
    config = load(tmp_path, yaml_text="max_steps: 10\n", dotenv_text="WAA_MAX_STEPS=20\n")
    assert config.max_steps == 20


@pytest.mark.unit
def test_file_beats_schema_default(tmp_path: Path) -> None:
    """With both other layers absent, the committed default is what stands."""
    config = load(tmp_path, yaml_text="max_steps: 10\n")
    assert config.max_steps == 10


@pytest.mark.unit
def test_layers_merge_per_setting(tmp_path: Path) -> None:
    """Overriding one setting leaves every other layer's values in place.

    The layers are merged key by key, not swapped whole: a ``.env`` that sets a
    timeout must not discard the committed viewport.
    """
    config = load(
        tmp_path,
        yaml_text="max_steps: 10\nviewport_width: 1024\n",
        dotenv_text="WAA_ACTION_TIMEOUT_S=5\n",
        cli={"workers": 1},
    )
    assert (config.max_steps, config.viewport_width) == (10, 1024)
    assert config.action_timeout_s == 5.0
    assert config.workers == 1
    assert config.viewport_height == Config.model_fields["viewport_height"].default


@pytest.mark.unit
def test_exported_variable_beats_dotenv_file(tmp_path: Path) -> None:
    """Inside the middle layer, an exported variable wins over the file.

    Both are machine-local, but one was typed for this invocation.
    """
    config = load(tmp_path, dotenv_text="WAA_WORKERS=2\n", env={"WAA_WORKERS": "8"})
    assert config.workers == 8


@pytest.mark.unit
def test_unset_flag_does_not_override(tmp_path: Path) -> None:
    """A ``None`` override means "flag not passed", not "set it to nothing"."""
    config = load(tmp_path, yaml_text="max_steps: 10\n", cli={"max_steps": None})
    assert config.max_steps == 10


@pytest.mark.unit
def test_blank_variable_does_not_override(tmp_path: Path) -> None:
    """A variable listed with no value is unset.

    ``.env.example`` lists every variable with a blank value so it can be
    copied verbatim; that copy must not override the committed defaults with
    empty strings.
    """
    config = load(tmp_path, yaml_text="max_steps: 10\n", dotenv_text="WAA_MAX_STEPS=\n")
    assert config.max_steps == 10


@pytest.mark.unit
def test_unknown_file_key_raises(tmp_path: Path) -> None:
    """A misspelled setting in the file is an error naming the file and key."""
    with pytest.raises(ConfigError) as caught:
        load(tmp_path, yaml_text="max_step: 40\n")
    assert "max_step" in str(caught.value)
    assert "config.yaml" in str(caught.value)


@pytest.mark.unit
def test_api_key_in_committed_file_raises(tmp_path: Path) -> None:
    """A secret written into the committed file is rejected, not read.

    No field of the schema holds a key, so the unknown-key rule would catch
    this on its own; the loader recognises the *name* first so the message can
    say where the value belongs instead. What that message says is asserted in
    ``test_secrets.py``; what matters here is that the loader refuses rather
    than ignores.
    """
    with pytest.raises(ConfigError):
        load(tmp_path, yaml_text="openrouter_api_key: sk-not-a-real-key\n")


@pytest.mark.unit
def test_unknown_flag_override_raises(tmp_path: Path) -> None:
    """An override keyed by a name that is not a setting is a caller bug."""
    with pytest.raises(ConfigError) as caught:
        load(tmp_path, cli={"max_stpes": 5})
    assert "max_stpes" in str(caught.value)


@pytest.mark.unit
def test_unrelated_environment_variables_are_ignored(tmp_path: Path) -> None:
    """Non-setting variables in ``.env`` are left alone.

    The unknown-key rule is deliberately not applied to the environment: that
    is where the API keys live, alongside whatever else the machine exports.
    """
    config = load(
        tmp_path,
        dotenv_text="WAA_OPENROUTER_API_KEY=sk-not-a-real-key\nWAA_MAX_STEPS=12\n",
        env={"PATH": "/usr/bin", "WAA_PROFILE": "nvidia"},
    )
    assert config.max_steps == 12


@pytest.mark.unit
def test_out_of_range_value_raises(tmp_path: Path) -> None:
    """A step cap of zero fails at startup rather than producing an empty run."""
    with pytest.raises(ConfigError):
        load(tmp_path, yaml_text="max_steps: 0\n")


@pytest.mark.unit
def test_non_numeric_value_raises(tmp_path: Path) -> None:
    """Strings from the environment are parsed, and failures to parse raise."""
    with pytest.raises(ConfigError):
        load(tmp_path, dotenv_text="WAA_MAX_STEPS=lots\n")


@pytest.mark.unit
def test_malformed_yaml_raises(tmp_path: Path) -> None:
    """A file that is not a mapping of settings is reported as such."""
    with pytest.raises(ConfigError):
        load(tmp_path, yaml_text="- max_steps\n- 10\n")


@pytest.mark.unit
def test_allowlist_accepts_both_spellings(tmp_path: Path) -> None:
    """The allowlist is a YAML list in the file and a comma string in ``.env``.

    An environment variable can only ever be one string, so the field has to
    accept both spellings for the layers to be interchangeable.
    """
    from_file = load(tmp_path, yaml_text='allowlist_extra: ["a.example", "b.example"]\n')
    from_env = load(tmp_path, dotenv_text="WAA_ALLOWLIST_EXTRA=a.example, b.example\n")
    assert from_file.allowlist_extra == from_env.allowlist_extra == ("a.example", "b.example")


@pytest.mark.unit
def test_resolved_config_is_frozen(tmp_path: Path) -> None:
    """Nothing may retune the agent mid-run; the trace has to describe the run."""
    config = load(tmp_path)
    with pytest.raises(ValidationError):
        config.max_steps = 99


@pytest.mark.unit
def test_dotenv_parsing_tolerates_export_and_quotes(tmp_path: Path) -> None:
    """Shell and editor habits are read, and a non-assignment line is an error.

    A silently skipped line in this file would mean a silently missing API key.
    """
    parsed = read_dotenv(
        write(
            tmp_path / ".env",
            '# a comment\n\nexport WAA_MAX_STEPS=7\nWAA_MODEL="a/model:free"\n',
        )
    )
    assert parsed == {"WAA_MAX_STEPS": "7", "WAA_MODEL": "a/model:free"}

    with pytest.raises(ConfigError):
        read_dotenv(write(tmp_path / "bad.env", "WAA_MAX_STEPS 7\n"))


@pytest.mark.unit
def test_missing_files_fall_back_to_schema_defaults(tmp_path: Path) -> None:
    """Neither file is required; an installation with neither still runs."""
    config = load_config(
        config_path=tmp_path / "absent.yaml",
        dotenv_path=tmp_path / "absent.env",
        env={},
    )
    assert config.max_steps == Config.model_fields["max_steps"].default


@pytest.mark.unit
def test_shipped_file_matches_the_schema() -> None:
    """The committed file states every setting, at the schema's own default.

    Two sources of a default would drift apart, and the disagreement would only
    surface as a run behaving unlike the file that documents it. Local tuning
    is meant to go in ``.env``, which is what keeps this equality true.
    """
    absent = Path("no-such-file")
    shipped = load_config(config_path=DEFAULT_CONFIG_PATH, dotenv_path=absent, env={})
    schema_defaults = load_config(config_path=absent, dotenv_path=absent, env={})
    assert shipped == schema_defaults
    assert set(yaml_keys(DEFAULT_CONFIG_PATH)) == set(Config.model_fields)


@pytest.mark.unit
def test_env_template_documents_every_setting() -> None:
    """``.env.example`` is the list a user reads; it must not fall behind.

    A setting the template omits is one nobody knows can be set from ``.env``.
    """
    template = ENV_TEMPLATE.read_text()
    missing = [
        env_var_for(field) for field in Config.model_fields if env_var_for(field) not in template
    ]
    assert not missing, f".env.example does not mention: {', '.join(missing)}"


@pytest.mark.unit
def test_names_are_derived_from_the_schema() -> None:
    """One field name yields its variable and its flag, so they cannot diverge."""
    assert env_var_for("screenshot_min_elements") == "WAA_SCREENSHOT_MIN_ELEMENTS"
    assert flag_for("screenshot_min_elements") == "--screenshot-min-elements"
