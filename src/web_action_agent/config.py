"""Run configuration: three layers, resolved once, then frozen.

Every tunable the agent has is resolved here, from three sources that override
one another in a fixed order — lowest precedence first::

    config.yaml  ->  .env / environment  ->  command-line flags

Each layer exists for a different reason:

* ``config.yaml`` is committed. It is the project's declared default for every
  tunable, and the file a reader opens to find out what can be tuned at all.
* ``.env`` is git-ignored and machine-local. It carries the secrets (API keys,
  read by the model client, not by this module) and any per-machine override —
  a slower box wanting longer timeouts, say.
* Command-line flags are per-run. They override everything, and they are the
  only layer that changes without editing a file.

Two properties of that arrangement are worth stating, because they are what the
design buys:

* **Secrets cannot leak through the committed file.** No field below holds a
  key, so an API key written into ``config.yaml`` is an *unknown key*, and
  unknown keys are an error (see :func:`load_config`). The failure is loud and
  immediate rather than a secret quietly riding into a public commit.
* **A typo is an error, not a silent default.** Configuration that ignores what
  it does not recognise is the worst kind: the run proceeds, at a value nobody
  chose. ``max_step: 40`` raises here instead.

The resolved :class:`Config` is frozen. Nothing may retune the agent halfway
through a run — a run's trace would no longer describe the run that produced
it, which the evaluation protocol depends on.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Annotated, Any

import yaml
from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, ValidationError

#: Prefix on every environment variable this project reads. A field named
#: ``max_steps`` is set by ``WAA_MAX_STEPS``; the mapping is mechanical, so
#: there is no table here to fall out of step with the schema.
ENV_PREFIX = "WAA_"

#: The committed defaults. The file sits inside the package rather than at the
#: repository root so that it ships in the wheel and is found the same way
#: whether the agent runs from a checkout or from ``pip install``.
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "config.yaml"

#: Where :func:`load_config` looks for the local secrets file when the caller
#: names none: the working directory, which for this project is the checkout.
DEFAULT_DOTENV_PATH = Path(".env")


class ConfigError(Exception):
    """A configuration layer was unreadable, or resolved to an invalid value.

    Every failure mode of this module raises this one type — a malformed YAML
    file, an unknown key, an out-of-range value, an unparsable ``.env`` line —
    so a caller can turn the whole of configuration loading into a single
    diagnostic without catching ``yaml`` and ``pydantic`` errors separately.
    """


def _split_domains(value: Any) -> Any:
    """Accept a comma-separated string wherever a list of domains is expected.

    YAML can express a list natively, but an environment variable cannot: the
    ``.env`` and flag layers can only ever hand over one string. This lets the
    same field be written ``["a.com", "b.com"]`` in ``config.yaml`` and
    ``WAA_ALLOWLIST_EXTRA=a.com,b.com`` in ``.env``.

    Args:
        value: The raw field value from whichever layer supplied it.

    Returns:
        Any: A tuple of trimmed, non-empty domains if ``value`` was a string;
        otherwise ``value`` untouched, leaving normal validation to report it.
    """
    if isinstance(value, str):
        return tuple(part.strip() for part in value.split(",") if part.strip())
    return value


#: A list of domain names that may also be written as one comma-separated string.
DomainList = Annotated[tuple[str, ...], BeforeValidator(_split_domains)]


class Config(BaseModel):
    """Every tunable of one run, resolved from all three layers.

    The field names are the schema. They are also, uppercased and prefixed, the
    environment variable names, and, hyphenated, the command-line flag names —
    so this class is the single place any of the three is declared.

    The bounds on each field are not decoration. They are the difference
    between a mistyped ``0`` failing at startup with a clear message and the
    agent taking zero steps and reporting an empty run as a legitimate result.
    """

    model_config = ConfigDict(
        # An unrecognised key is an error. This is the backstop; load_config
        # checks the file layer first so the message can name the file.
        extra="forbid",
        # Immutable once resolved: see the module docstring.
        frozen=True,
    )

    # --- Model endpoint -----------------------------------------------------
    # Which named endpoint the run's model calls go to. The endpoints themselves
    # are described in `profiles.yaml`, not here: this layer picks one and, at
    # most, adjusts two of its fields for a single run.
    profile: str = Field(
        default="openrouter",
        min_length=1,
        description="Name of the endpoint profile in profiles.yaml, e.g. openrouter.",
    )
    base_url: str | None = Field(
        default=None,
        description=(
            "Overrides the selected profile's base URL — for a self-hosted "
            "server on another host. None leaves the profile's own value."
        ),
    )
    model: str | None = Field(
        default=None,
        description=(
            "Overrides the selected profile's text model, for trying one model "
            "without editing profiles.yaml. None leaves the profile's own value."
        ),
    )
    # An API key is deliberately not a field here. Adding one would make it
    # writable in the committed config.yaml; keys are read from the environment
    # by the model client alone.

    # --- Agent loop ---------------------------------------------------------
    max_steps: int = Field(
        default=25,
        gt=0,
        description="Hard cap on steps per run; the loop's guarantee of termination.",
    )
    retry_budget: int = Field(
        default=2,
        ge=0,
        description="Validator rejections a single step may be retried through before it fails.",
    )
    screenshot_min_elements: int = Field(
        default=5,
        ge=0,
        description=(
            "Screenshots are cost-gated: below this many extracted interactive "
            "elements, the DOM signal counts as weak and vision is gated in."
        ),
    )
    screenshot_max_dom_failures: int = Field(
        default=2,
        ge=0,
        description="Consecutive DOM-only planner failures after which vision is gated in.",
    )
    text_token_budget: int = Field(
        default=1500,
        gt=0,
        description="Truncation budget for the distilled page-text extract.",
    )

    # --- Browser ------------------------------------------------------------
    viewport_width: int = Field(
        default=1280,
        gt=0,
        description="Viewport width; fixed, because element IDs are assigned viewport-only.",
    )
    viewport_height: int = Field(default=800, gt=0, description="Viewport height.")
    domain_delay_s: float = Field(
        default=1.0,
        ge=0.0,
        description="Politeness floor: minimum seconds between navigations to one domain.",
    )
    allowlist_extra: DomainList = Field(
        default=(),
        description=(
            "Domains the navigation allowlist accepts in addition to the start "
            "URL's own registrable domain, which is always allowed."
        ),
    )

    # --- Timeouts (seconds) -------------------------------------------------
    # Each expiry is a categorized failure the agent reports, never an
    # exception that ends the run without a verdict.
    action_timeout_s: float = Field(default=30.0, gt=0.0, description="One browser action.")
    llm_timeout_s: float = Field(default=90.0, gt=0.0, description="One model call.")
    run_timeout_s: float = Field(
        default=300.0, gt=0.0, description="Wall-clock ceiling for one whole run."
    )

    # --- Skill library ------------------------------------------------------
    skill_retirement_rate: float = Field(
        default=0.70,
        ge=0.0,
        le=1.0,
        description="Skills whose success rate falls below this are retired automatically.",
    )
    skill_retirement_min_sample: int = Field(
        default=5,
        gt=0,
        description=(
            "Recorded uses required before retirement may trigger, so one "
            "unlucky replay cannot retire a good skill."
        ),
    )

    # --- Evaluation harness -------------------------------------------------
    workers: int = Field(default=4, gt=0, description="Benchmark tasks run in parallel.")
    budget_usd: float = Field(
        default=15.0, gt=0.0, description="Spend ceiling for one evaluation sweep."
    )
    budget_hours: float = Field(
        default=6.0, gt=0.0, description="Wall-clock ceiling for one evaluation sweep."
    )


def env_var_for(field: str) -> str:
    """Return the environment variable that sets ``field``.

    Args:
        field: A :class:`Config` field name, e.g. ``"max_steps"``.

    Returns:
        str: The variable name, e.g. ``"WAA_MAX_STEPS"``.
    """
    return f"{ENV_PREFIX}{field.upper()}"


def flag_for(field: str) -> str:
    """Return the command-line flag that sets ``field``.

    The command layer is built elsewhere; it derives its flag names from here
    so that a field cannot acquire a flag spelled differently from its name.

    Args:
        field: A :class:`Config` field name, e.g. ``"max_steps"``.

    Returns:
        str: The flag, e.g. ``"--max-steps"``.
    """
    return "--" + field.replace("_", "-")


def read_dotenv(path: Path) -> dict[str, str]:
    """Parse a ``.env`` file into a mapping, without touching ``os.environ``.

    Deliberately a small parser rather than a dependency: the file format this
    project uses is ``NAME=value`` lines, blank lines, and ``#`` comments. An
    optional ``export`` prefix and surrounding quotes are tolerated because
    editors and shells add them; anything else is rejected rather than guessed
    at, since a silently skipped line here means a silently missing API key.

    Args:
        path: The file to read. A missing file is not an error — the file is
            machine-local and optional — and yields an empty mapping.

    Returns:
        dict[str, str]: Variable names mapped to their raw string values.

    Raises:
        ConfigError: If a non-blank, non-comment line is not an assignment.
    """
    if not path.exists():
        return {}

    values: dict[str, str] = {}
    for number, raw_line in enumerate(path.read_text().splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        line = line.removeprefix("export ").lstrip()
        name, separator, value = line.partition("=")
        if not separator or not name.strip():
            raise ConfigError(f"{path}:{number}: expected NAME=value, got {raw_line!r}")
        value = value.strip()
        # Strip one matched pair of quotes, so NAME="a b" and NAME='a b' both
        # yield `a b` rather than a value with quote characters in it.
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        values[name.strip()] = value
    return values


def _file_layer(path: Path) -> dict[str, Any]:
    """Read the committed defaults file — the lowest-precedence layer.

    Args:
        path: The YAML file to read. A missing file is not an error: the
            schema's own defaults then stand, which is what an installation
            with no config file should do.

    Returns:
        dict[str, Any]: The file's key/value pairs.

    Raises:
        ConfigError: If the file is not valid YAML, does not contain a mapping,
            or names a key that is not a :class:`Config` field. That last case
            is checked here rather than left to the schema so the message can
            name the file and the offending key.
    """
    if not path.exists():
        return {}

    try:
        parsed = yaml.safe_load(path.read_text())
    except yaml.YAMLError as error:
        raise ConfigError(f"{path}: not valid YAML: {error}") from error

    if parsed is None:  # An empty file is a legitimate "override nothing".
        return {}
    if not isinstance(parsed, dict):
        raise ConfigError(f"{path}: expected a mapping of settings, got {type(parsed).__name__}")

    unknown = sorted(set(parsed) - set(Config.model_fields))
    if unknown:
        raise ConfigError(
            f"{path}: unknown setting(s) {', '.join(unknown)}. "
            f"Known settings: {', '.join(sorted(Config.model_fields))}"
        )
    return parsed


def _env_layer(env: Mapping[str, str]) -> dict[str, str]:
    """Pick this project's settings out of an environment mapping.

    Only names matching a :class:`Config` field are read. That is what lets the
    same file hold ``WAA_OPENROUTER_API_KEY`` and other non-tunable variables
    without them being mistaken for settings — the unknown-key rule applies to
    the committed file, where a stray key is a typo, not to the environment,
    where other variables legitimately live.

    An empty value counts as unset, so that the committed ``.env.example``
    template can list every variable with a blank value and be copied verbatim
    without overriding anything.

    Args:
        env: The environment to read: ``.env`` merged under ``os.environ``.

    Returns:
        dict[str, str]: Field names mapped to their raw string values. Parsing
        each string into the field's type is the schema's job.
    """
    values: dict[str, str] = {}
    for field in Config.model_fields:
        raw = env.get(env_var_for(field))
        if raw is None or not raw.strip():
            continue
        values[field] = raw.strip()
    return values


def _cli_layer(overrides: Mapping[str, Any]) -> dict[str, Any]:
    """Take the per-run overrides — the highest-precedence layer.

    Args:
        overrides: Field names mapped to parsed flag values. ``None`` means the
            flag was not passed, which is how argument parsers report an absent
            option; those entries fall through to the layers below rather than
            overriding them with nothing.

    Returns:
        dict[str, Any]: The overrides that were actually supplied.

    Raises:
        ConfigError: If a name is not a :class:`Config` field. Unlike a typo in
            a file, this is a bug in the calling code, and it is caught here
            rather than shadowed by the schema's own error.
    """
    unknown = sorted(set(overrides) - set(Config.model_fields))
    if unknown:
        raise ConfigError(f"not a known setting: {', '.join(unknown)}")
    return {field: value for field, value in overrides.items() if value is not None}


def load_config(
    *,
    config_path: Path | None = None,
    dotenv_path: Path | None = None,
    env: Mapping[str, str] | None = None,
    cli: Mapping[str, Any] | None = None,
) -> Config:
    """Resolve the three layers into one frozen :class:`Config`.

    Later layers overwrite earlier ones key by key, so a ``.env`` that sets one
    value changes that value alone and leaves the rest of the committed
    defaults standing.

    Within the middle layer, an exported variable beats the same name in the
    ``.env`` file: both are machine-local, and the one typed on the command
    line for a single invocation is the more deliberate of the two.

    Args:
        config_path: The committed defaults file. Defaults to the packaged
            ``config.yaml``.
        dotenv_path: The local secrets/overrides file. Defaults to ``.env`` in
            the working directory. A missing file is fine.
        env: The process environment. Defaults to ``os.environ``; injectable so
            tests need not mutate the real one.
        cli: Per-run flag overrides, keyed by field name. ``None`` values are
            treated as "flag not passed".

    Returns:
        Config: The resolved, immutable configuration for this run.

    Raises:
        ConfigError: If any layer is unreadable, names an unknown setting, or
            the resolved values fail validation.
    """
    file_values = _file_layer(config_path if config_path is not None else DEFAULT_CONFIG_PATH)
    dotenv_values = read_dotenv(dotenv_path if dotenv_path is not None else DEFAULT_DOTENV_PATH)
    env_values = _env_layer({**dotenv_values, **(env if env is not None else os.environ)})
    cli_values = _cli_layer(cli or {})

    merged = {**file_values, **env_values, **cli_values}
    try:
        return Config(**merged)
    except ValidationError as error:
        # Re-raised as this module's own type: from the caller's side, "the
        # timeout you set is negative" is a configuration failure, and which
        # library noticed it is an implementation detail.
        raise ConfigError(f"invalid configuration: {error}") from error
