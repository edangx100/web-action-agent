"""What this repository commits, and what it refuses to.

Three kinds of file must never land here: a secret, a private planning
document, and the bulk of the run artifacts — thousands of screenshots of live
third-party sites that the harness rewrites on every pass. A fourth kind must
land: the curated example run, the aggregated results tables, and the human
spot-check labels are deliverables, and they live inside the same ``runs/``
tree the rest of which is ignored.

Ignore rules that carve an exception out of a directory are easy to get subtly
wrong — Git will not look inside an excluded directory, so an exception under
one silently never matches — and the failure is invisible until a deliverable
turns out to be missing or a screenshot turns up in a public diff. So the rules
are asserted here against ``git check-ignore`` itself, rather than read.

``--no-index`` makes each check a question about the patterns alone. Without
it, ``git check-ignore`` answers "not ignored" for anything already tracked,
which would make these assertions pass for the wrong reason.
"""

import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
ENV_TEMPLATE = REPO_ROOT / ".env.example"


def is_ignored(path: str) -> bool:
    """Ask Git whether the ignore rules exclude ``path``.

    The path need not exist: ``git check-ignore`` matches patterns against the
    name, which is what lets this test describe run folders that only a real
    evaluation sweep would create.

    Args:
        path: A repository-relative path, in POSIX form.

    Returns:
        bool: True if the ignore rules exclude the path.

    Raises:
        AssertionError: If Git itself failed, rather than answering.
    """
    result = subprocess.run(
        ["git", "check-ignore", "--no-index", "--quiet", "--", path],
        cwd=REPO_ROOT,
        capture_output=True,
        check=False,
        text=True,
    )
    assert result.returncode in (0, 1), (
        f"git check-ignore failed on {path!r} ({result.returncode}):\n{result.stderr}"
    )
    return result.returncode == 0


IGNORED = [
    # A run folder is named for its timestamp and task; none of them are ours.
    "runs/2026-08-24T12-00-00Z-task-07/action_log.jsonl",
    "runs/2026-08-24T12-00-00Z-task-07/step-001.png",
    "runs/2026-08-24T12-00-00Z-task-07/screenshots/step-012.png",
    # Secrets, in every form the loader accepts.
    ".env",
    ".env.local",
    "credentials.key",
    # Private planning documents, if one is ever copied in by accident.
    "SPEC.md",
    "SPEC_QNS_SUGGESTED_ANS.md",
    "TASKS.md",
    "PROJ_IDEA.md",
    # Browser leavings: profiles, recordings, traces, captured traffic.
    "playwright-report/index.html",
    "test-results/run-1/video.webm",
    ".browser-profile/Default/Cookies",
    "trace.zip",
    "session.har",
    # The working skill library the inducer rewrites.
    "skills/amazon-search.json",
]

COMMITTED = [
    # The one curated example run, at every depth inside it.
    "runs/example/actions.jsonl",
    "runs/example/calls.jsonl",
    "runs/example/README.md",
    "runs/example/screenshots/step-00.png",
    # Aggregated results and human labels, wherever under runs/ they are written.
    "runs/results/success_rate.md",
    "runs/2026-08-24T12-00-00Z-task-07/results/verdicts.md",
    "runs/human_labels.jsonl",
    "runs/example/human_labels.jsonl",
    # The same two deliverables at the top level, where nothing excludes them.
    "results/cold_vs_warm.md",
    "human_labels.jsonl",
    # The secrets template, which is the one .env file that is committed.
    ".env.example",
]


@pytest.mark.unit
@pytest.mark.parametrize("path", IGNORED)
def test_local_only_files_are_ignored(path: str) -> None:
    """Scratch, secrets, and private documents stay out of the history."""
    assert is_ignored(path), f"{path} must be ignored but is not"


@pytest.mark.unit
@pytest.mark.parametrize("path", COMMITTED)
def test_deliverables_are_not_ignored(path: str) -> None:
    """The committed artifacts survive the rules that exclude their neighbours."""
    assert not is_ignored(path), f"{path} is a deliverable but the ignore rules exclude it"


def env_template_assignments() -> list[tuple[str, str]]:
    """Parse ``.env.example`` into (name, value) pairs, comments stripped.

    Returns:
        list[tuple[str, str]]: One pair per assignment, in file order.
    """
    pairs = []
    for line in ENV_TEMPLATE.read_text(encoding="utf-8").splitlines():
        statement = line.split("#", 1)[0].strip()
        if not statement:
            continue
        name, _, value = statement.partition("=")
        pairs.append((name.strip(), value.strip()))
    return pairs


@pytest.mark.unit
def test_env_template_declares_only_prefixed_variables() -> None:
    """Every variable the agent reads carries the project's prefix.

    The prefix is what keeps the agent's configuration from colliding with
    whatever else is exported in a developer's shell.
    """
    assignments = env_template_assignments()
    assert assignments, ".env.example declares no variables"

    unprefixed = [name for name, _ in assignments if not name.startswith("WAA_")]
    assert not unprefixed, f"variables missing the WAA_ prefix: {unprefixed}"


@pytest.mark.unit
def test_env_template_carries_no_values() -> None:
    """The template holds names, so filling in a key cannot commit it.

    A default belongs in ``config.yaml``; a value written here would either
    shadow that default invisibly or, for a key, be a secret in the history.
    """
    filled = [f"{name}={value}" for name, value in env_template_assignments() if value]
    assert not filled, f"values must be left empty in the template: {filled}"


@pytest.mark.unit
def test_env_template_declares_each_variable_once() -> None:
    """A repeated name means the later assignment silently wins."""
    names = [name for name, _ in env_template_assignments()]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    assert not duplicates, f"declared more than once: {duplicates}"
