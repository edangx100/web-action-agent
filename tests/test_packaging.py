"""The two install paths this project promises must both work from a clean tree.

Path one is the development path: a `uv`-managed `.venv` built from the
lockfile. Path two is the consumer path: a plain virtual environment and
`pip install -r requirements.txt`, with no `uv` on the machine at all. Both are
checked end to end here, because a lockfile and an exported manifest drift
apart silently and only bite someone else.
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
REQUIREMENTS = REPO_ROOT / "requirements.txt"


def run(command: list[str], *, env: dict[str, str] | None = None, timeout: int = 900):
    """Run a command, capturing output so a failure reports why."""
    return subprocess.run(
        command,
        cwd=REPO_ROOT,
        env={**os.environ, **(env or {})},
        capture_output=True,
        check=False,
        text=True,
        timeout=timeout,
    )


def assert_ok(result: subprocess.CompletedProcess, what: str) -> None:
    assert result.returncode == 0, f"{what} failed ({result.returncode}):\n{result.stderr}"


@pytest.mark.fixture
@pytest.mark.skipif(shutil.which("uv") is None, reason="uv is not installed")
def test_locked_sync_builds_a_working_environment(tmp_path):
    """`uv sync --frozen` resolves nothing new and yields a runnable `waa`."""
    env = {"UV_PROJECT_ENVIRONMENT": str(tmp_path / "env")}

    assert_ok(run(["uv", "sync", "--frozen"], env=env), "uv sync --frozen")
    assert_ok(run(["uv", "run", "--frozen", "waa", "--help"], env=env), "uv run waa --help")


@pytest.mark.fixture
def test_pip_install_from_requirements_provides_both_entry_points(tmp_path):
    """The exported manifest alone is enough — no `uv`, no `pyproject` extras."""
    assert REQUIREMENTS.is_file(), "requirements.txt must be exported and committed"

    venv = tmp_path / "consumer"
    assert_ok(run([sys.executable, "-m", "venv", venv.as_posix()]), "python -m venv")

    bin_dir = venv / ("Scripts" if os.name == "nt" else "bin")
    python = bin_dir / "python"
    install = run(
        [python.as_posix(), "-m", "pip", "install", "-r", REQUIREMENTS.as_posix()],
        env={"PIP_DISABLE_PIP_VERSION_CHECK": "1"},
    )
    assert_ok(install, "pip install -r requirements.txt")

    assert_ok(run([(bin_dir / "waa").as_posix(), "--help"]), "waa --help")
    assert_ok(
        run([python.as_posix(), "-m", "web_action_agent", "--help"]),
        "python -m web_action_agent --help",
    )
