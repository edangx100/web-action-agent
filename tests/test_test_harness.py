"""The suite's own safeguards: marker discipline, and what CI selects.

Two promises are checked here. The first is that every test declares what it
needs to run, so a selection by marker is a selection over the whole suite and
not just over the part someone remembered to label. The second is that the
workflow's selection cannot reach a test marked ``live`` -- one that would need
the real web or a real model endpoint, neither of which continuous integration
has.

Both are worth testing rather than trusting. A missing marker and a widened
marker expression are each a one-line change that no test would otherwise
notice, and the consequence -- tests that quietly stop running, or a CI job
that quietly reaches the network -- shows up much later and far from the cause.
"""

from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

import pytest
import yaml
from marker_policy import TEST_TYPE_MARKERS, applied_type_markers, policy_violations

from web_action_agent import config

REPO_ROOT = Path(__file__).resolve().parent.parent
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "ci.yml"


@dataclass
class FakeMark:
    """Stand-in for a pytest mark, carrying only the name the rule reads."""

    name: str


@dataclass
class FakeItem:
    """Stand-in for a collected test, for exercising the rule directly.

    Attributes:
        nodeid: Identifier the rule quotes when reporting a violation.
        marks: Marks applied to this pretend test.
    """

    nodeid: str = "tests/test_example.py::test_example"
    marks: list[FakeMark] = field(default_factory=list)

    def iter_markers(self) -> list[FakeMark]:
        """Yield the applied marks, as a real pytest item does."""
        return self.marks


def workflow() -> dict:
    """Parse the continuous integration workflow.

    Returns:
        The workflow as nested dictionaries and lists.
    """
    return yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))


def offline_marker_expression() -> str:
    """Read the marker expression the workflow runs the suite with.

    Returning it from the workflow rather than restating it here is the point:
    a test that hard-coded the expression would keep passing after someone
    widened the one CI actually uses.

    Returns:
        The marker expression, for example ``"unit or fixture or mock_llm"``.
    """
    expression = workflow()["env"]["OFFLINE_MARKERS"]
    assert isinstance(expression, str) and expression.strip(), (
        "the workflow must define OFFLINE_MARKERS as the suite it selects"
    )
    return expression


def collect(marker_expression: str) -> list[str]:
    """Collect, without running, the tests a marker expression selects.

    Args:
        marker_expression: A pytest ``-m`` expression.

    Returns:
        The node id of every selected test.

    Raises:
        AssertionError: If collection fails for any reason other than
            selecting nothing, which pytest reports with exit code 5.
    """
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q", "-m", marker_expression],
        cwd=REPO_ROOT,
        capture_output=True,
        check=False,
        text=True,
        timeout=300,
    )
    no_tests_collected = 5
    assert result.returncode in (0, no_tests_collected), (
        f"collection failed ({result.returncode}):\n{result.stdout}\n{result.stderr}"
    )
    return [line for line in result.stdout.splitlines() if "::" in line]


@pytest.mark.unit
def test_the_workflow_runs_on_pull_requests():
    """Assert the workflow is a merge gate, not a post-merge report.

    A suite that only runs after a merge reports damage instead of preventing
    it.
    """
    # PyYAML reads the unquoted key `on` as the boolean True, per YAML 1.1.
    triggers = workflow().get("on", workflow().get(True))

    assert "pull_request" in triggers


@pytest.mark.unit
def test_the_workflow_installs_a_browser():
    """Assert CI provisions Chromium, which the browser-backed tests need.

    Playwright installs as a Python package without any browser attached, so
    the download is a separate step that is easy to drop and whose absence
    surfaces only as a puzzling launch error inside an unrelated test.
    """
    steps = workflow()["jobs"]["test"]["steps"]
    commands = " ".join(step.get("run", "") for step in steps)

    assert "playwright install" in commands
    assert "chromium" in commands


@pytest.mark.unit
def test_the_workflow_selects_every_offline_marker():
    """Assert CI runs all three offline kinds of test, and does not name live.

    Dropping a marker from the expression would silently retire a whole class
    of test while CI stayed green.
    """
    expression = offline_marker_expression()

    for marker in ("unit", "fixture", "mock_llm"):
        assert marker in expression, f"CI does not run {marker!r} tests"
    assert "live" not in expression


@pytest.mark.unit
def test_no_live_test_is_reachable_from_the_ci_selection():
    """Assert nothing marked ``live`` survives the workflow's selection.

    This is the promise itself, checked against the real collection rather
    than by reading the expression: whatever the workflow selects, intersected
    with the live suite, must be empty.
    """
    reachable = collect(f"live and ({offline_marker_expression()})")

    assert reachable == [], f"CI would run live tests: {reachable}"


@pytest.mark.unit
def test_the_offline_selection_is_not_empty():
    """Assert the selection actually matches tests.

    A marker expression that selects nothing is green for the worst reason,
    and a typo in a marker name would produce exactly that.
    """
    assert collect(offline_marker_expression()) != []


@pytest.mark.unit
def test_the_whole_suite_declares_its_markers():
    """Assert collecting the suite passes the one-marker rule.

    The rule is enforced by a collection hook, so a suite that collects at all
    is a suite that complies. Collecting everything -- ``live`` included --
    is what makes this cover tests CI never runs.
    """
    assert collect("unit or fixture or mock_llm or live") != []


@pytest.mark.unit
@pytest.mark.parametrize("marker", TEST_TYPE_MARKERS)
def test_one_type_marker_satisfies_the_rule(marker):
    """Assert each of the four markers is accepted on its own.

    Args:
        marker: The single test-type marker under test.
    """
    item = FakeItem(marks=[FakeMark(marker), FakeMark("parametrize")])

    assert applied_type_markers(item) == [marker]
    assert policy_violations([item]) == []


@pytest.mark.unit
def test_an_unmarked_test_is_a_violation():
    """Assert a test with no type marker is reported, and reported by name."""
    item = FakeItem(nodeid="tests/test_x.py::test_forgot", marks=[FakeMark("skipif")])

    violations = policy_violations([item])

    assert len(violations) == 1
    assert "test_forgot" in violations[0]
    assert "none" in violations[0]


@pytest.mark.unit
def test_two_type_markers_are_a_violation():
    """Assert an ambiguous test is rejected rather than resolved by guesswork.

    A test marked both ``unit`` and ``live`` would run in CI under the first
    marker while claiming, under the second, that it must not.
    """
    item = FakeItem(marks=[FakeMark("unit"), FakeMark("live")])

    violations = policy_violations([item])

    assert len(violations) == 1
    assert "unit, live" in violations[0]


@pytest.mark.unit
def test_no_test_sees_this_machines_own_env_file_or_waa_variables() -> None:
    # A developer's .env holds their keys and choices; a test that read it
    # would behave differently on every machine. See tests/conftest.py.
    assert config.DEFAULT_DOTENV_PATH.name == "absent.env"
    assert not config.DEFAULT_DOTENV_PATH.exists()
    assert not [name for name in os.environ if name.startswith("WAA_")]
    assert not any(name.startswith("WAA_") for name in config.resolve_env())
