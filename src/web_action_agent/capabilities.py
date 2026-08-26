"""What this installation can actually do, given the keys it was given.

The project promises that the repository is runnable with local files alone —
no account required for anything but the model endpoint you choose, and none at
all for the offline test suite. A promise like that is only worth something if
it is checkable, so this module turns it into a report the program prints
before it does anything else::

    capabilities:
      [ok] agent    profile 'openrouter' -> nvidia/nemotron-3-nano-30b-a3b
      [ok] vision   screenshots go to nvidia/nemotron-3-nano-omni-30b-a3b
      [--] judge    no judge endpoint configured

Three things are worth noticing about that.

**It states what is missing, not just what is present.** The failure this
guards against is not a crash — it is a user who runs the benchmark, gets
numbers, and only later discovers the judge never ran. An unavailable
capability is named at startup, with the variable to set if it is wanted.

**Nothing here is fatal.** Running the agent needs one endpoint. Reproducing
the published evaluation additionally needs a judge, deliberately from a
different model family, because nothing grades its own homework. Neither is a
precondition of the other, so a missing judge is a line in the report, not an
error — the ``judge`` command can refuse for itself, when someone asks for it.

**No value is ever printed.** Every detail line names *variables*, and the
whole report goes through :func:`~web_action_agent.config.redact` before it is
returned, so a key cannot reach a terminal, a screen recording, or a log even
if some future detail line is built carelessly.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from web_action_agent.config import Config, redact, resolve_env
from web_action_agent.model_client import (
    MissingAPIKeyError,
    ProfileError,
    load_profile,
    resolve_api_key,
    resolve_profile,
)

#: The capability that runs a task: one endpoint, reachable, with a key.
AGENT = "agent"

#: The capability that lets an observation carry a screenshot. Derived from the
#: selected profile, which has a vision route or has not.
VISION = "vision"

#: The capability that grades a run. Evaluation only; see the module docstring.
JUDGE = "judge"

#: Variables describing the judge endpoint. They are read here directly rather
#: than through :class:`~web_action_agent.config.Config`, because the judge is
#: not yet a configured component — this module reports on it before anything
#: exists that could run it, which is the whole point of a startup check.
JUDGE_PROFILE_ENV = "WAA_JUDGE_PROFILE"
JUDGE_BASE_URL_ENV = "WAA_JUDGE_BASE_URL"
JUDGE_API_KEY_ENV = "WAA_JUDGE_API_KEY"
JUDGE_MODEL_ENV = "WAA_JUDGE_MODEL"

#: Markers on the rendered report. ASCII on purpose: this goes to a terminal
#: whose encoding the program does not control, and a mojibake tick mark in a
#: startup banner reads as a bug in the program itself.
AVAILABLE_MARK = "[ok]"
UNAVAILABLE_MARK = "[--]"


@dataclass(frozen=True)
class Capability:
    """One thing the installation can or cannot do, and why.

    Attributes:
        name: The capability's name, one of :data:`AGENT`, :data:`VISION`,
            :data:`JUDGE`.
        available: Whether it is usable as things stand.
        detail: One line of explanation. Where the capability is available this
            says what it resolved to; where it is not, it says what to set. It
            never contains a key value.
    """

    name: str
    available: bool
    detail: str

    def render(self) -> str:
        """Format this capability as one line of the report.

        Returns:
            str: The marker, the padded name, and the detail.
        """
        mark = AVAILABLE_MARK if self.available else UNAVAILABLE_MARK
        return f"  {mark} {self.name:<8} {self.detail}"


@dataclass(frozen=True)
class CapabilityReport:
    """What the installation can do, as printed at startup.

    Attributes:
        capabilities: The capabilities, in the order they are reported.
    """

    capabilities: tuple[Capability, ...]

    def get(self, name: str) -> Capability:
        """Look up one capability by name.

        Args:
            name: One of :data:`AGENT`, :data:`VISION`, :data:`JUDGE`.

        Returns:
            Capability: The named capability.

        Raises:
            KeyError: If no capability has that name.
        """
        for capability in self.capabilities:
            if capability.name == name:
                return capability
        raise KeyError(name)

    def can(self, name: str) -> bool:
        """Report whether one capability is available.

        Args:
            name: One of :data:`AGENT`, :data:`VISION`, :data:`JUDGE`.

        Returns:
            bool: True if the named capability is usable.

        Raises:
            KeyError: If no capability has that name.
        """
        return self.get(name).available

    def render(self) -> str:
        """Format the whole report, with every key value scrubbed out.

        The redaction pass is the backstop, not the mechanism: no line is built
        from a value in the first place. It is here because this string is the
        one thing in the program guaranteed to be printed on every run, which
        makes it the worst possible place for a key to appear.

        Returns:
            str: The report, one capability per line under a heading.
        """
        lines = ["capabilities:", *(capability.render() for capability in self.capabilities)]
        return redact("\n".join(lines))


def _agent_capability(
    config: Config, env: Mapping[str, str], profiles_path: Path | None
) -> tuple[Capability, Capability]:
    """Resolve the agent and vision capabilities, which stand or fall together.

    Vision is a property of the selected profile, so it cannot be established
    without first selecting one — and a profile that cannot be reached serves
    no screenshots either. Returning both from one place keeps those two facts
    from disagreeing.

    Args:
        config: The resolved run configuration, naming the profile.
        env: The environment to read keys from.
        profiles_path: Profiles file, or None for the packaged one.

    Returns:
        tuple[Capability, Capability]: The agent capability, then vision.
    """
    try:
        profile = resolve_profile(config, path=profiles_path)
    except ProfileError as error:
        return (
            Capability(AGENT, False, str(error)),
            Capability(VISION, False, "no usable profile, so no vision route"),
        )

    try:
        resolve_api_key(profile, env)
    except MissingAPIKeyError as error:
        return (
            Capability(AGENT, False, str(error)),
            Capability(VISION, False, f"needs the {profile.name!r} endpoint first"),
        )

    agent = Capability(AGENT, True, f"profile {profile.name!r} -> {profile.model}")
    if profile.vision is None:
        vision = Capability(
            VISION,
            False,
            f"profile {profile.name!r} serves no vision model; the agent runs on DOM text alone",
        )
    else:
        vision = Capability(VISION, True, f"screenshots go to {profile.vision.model}")
    return agent, vision


def _judge_capability(env: Mapping[str, str], profiles_path: Path | None) -> Capability:
    """Resolve whether a judge endpoint is configured and has a key.

    Two ways of naming one are accepted, matching the two ways the endpoint is
    described elsewhere: a profile by name, or an ad-hoc base URL and key for a
    judge that has no profile of its own.

    Args:
        env: The environment to read from.
        profiles_path: Profiles file, or None for the packaged one.

    Returns:
        Capability: The judge capability, available or not, with the variable
        to set named in its detail where it is not.
    """
    unset = Capability(
        JUDGE,
        False,
        f"no judge endpoint: set {JUDGE_PROFILE_ENV}, or {JUDGE_BASE_URL_ENV} "
        f"with {JUDGE_API_KEY_ENV}. Needed to reproduce the evaluation, not to run a task",
    )

    profile_name = (env.get(JUDGE_PROFILE_ENV) or "").strip()
    if profile_name:
        try:
            profile = load_profile(profile_name, path=profiles_path)
        except ProfileError as error:
            return Capability(JUDGE, False, str(error))
        try:
            resolve_api_key(profile, env)
        except MissingAPIKeyError as error:
            return Capability(JUDGE, False, str(error))
        return Capability(JUDGE, True, f"profile {profile.name!r} -> {profile.model}")

    base_url = (env.get(JUDGE_BASE_URL_ENV) or "").strip()
    api_key = (env.get(JUDGE_API_KEY_ENV) or "").strip()
    if not base_url:
        return unset
    if not api_key:
        return Capability(
            JUDGE,
            False,
            f"{JUDGE_BASE_URL_ENV} is set but {JUDGE_API_KEY_ENV} is not",
        )

    model = (env.get(JUDGE_MODEL_ENV) or "").strip()
    if not model:
        return Capability(
            JUDGE,
            False,
            f"{JUDGE_BASE_URL_ENV} is set but {JUDGE_MODEL_ENV} is not",
        )
    return Capability(JUDGE, True, f"{JUDGE_BASE_URL_ENV} -> {model}")


def report_capabilities(
    config: Config,
    *,
    env: Mapping[str, str] | None = None,
    profiles_path: Path | None = None,
) -> CapabilityReport:
    """Work out what this installation can do, without calling any endpoint.

    Everything here is decided from configuration and environment alone: no
    request is made, nothing is spent, and the check adds no latency to a run.
    That an endpoint answers is a different question, and the first real call
    asks it.

    Args:
        config: The resolved run configuration.
        env: The environment to read keys from. Defaults to ``.env`` merged
            under the process environment.
        profiles_path: Profiles file. Defaults to the packaged one.

    Returns:
        CapabilityReport: The three capabilities, agent first.
    """
    resolved = env if env is not None else resolve_env()
    agent, vision = _agent_capability(config, resolved, profiles_path)
    return CapabilityReport((agent, vision, _judge_capability(resolved, profiles_path)))
