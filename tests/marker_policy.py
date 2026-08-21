"""The rule that every test declares what it needs to run.

A test in this suite is one of four kinds: pure Python, backed by the local
static site, backed by stubbed model responses, or backed by the real web and a
real model endpoint. Continuous integration runs the first three and must never
run the fourth, and it makes that choice purely from the marker each test
carries.

That only works if the markers are complete. An unmarked test is invisible to
every selection: it silently stops running in CI, or -- worse, if it turns out
to need the network -- silently starts running there and fails for reasons that
have nothing to do with the change under test. So the rule is enforced at
collection time rather than left to review, by
:func:`policy_violations` and the hook in ``tests/conftest.py`` that calls it.

This module holds only the rule, with no pytest imports, so the rule itself can
be tested directly.
"""

from __future__ import annotations

from typing import Protocol

TEST_TYPE_MARKERS = ("unit", "fixture", "mock_llm", "live")
"""tuple[str, ...]: The four markers, one of which every test must carry.

Kept in the order they appear in ``pyproject.toml``, cheapest first, which is
also the order in which a violation message lists the choices.
"""


class Markable(Protocol):
    """The slice of a pytest collection item this rule needs.

    Declared structurally so the rule can be exercised against plain stand-ins
    in a test, rather than against a real pytest session.
    """

    nodeid: str

    def iter_markers(self) -> object:
        """Yield the marks applied to this item, each with a ``name``."""
        ...


def applied_type_markers(item: Markable) -> list[str]:
    """List the test-type markers on one collected item.

    Args:
        item: A collected test, or any object exposing ``iter_markers()``.

    Returns:
        The names from :data:`TEST_TYPE_MARKERS` present on the item, in the
        order they are declared there. Other markers, such as
        ``pytest.mark.parametrize`` or ``skipif``, are ignored.
    """
    names = {mark.name for mark in item.iter_markers()}  # type: ignore[attr-defined]
    return [marker for marker in TEST_TYPE_MARKERS if marker in names]


def policy_violations(items: list[Markable]) -> list[str]:
    """Find every collected test that breaks the one-marker rule.

    Args:
        items: The tests collected for this session, before any marker-based
            deselection has been applied.

    Returns:
        A human-readable line per offending test, empty when the suite
        complies. Both failure modes are reported: no test-type marker at all,
        and more than one, which leaves it ambiguous what the test needs.
    """
    violations = []
    for item in items:
        applied = applied_type_markers(item)
        if len(applied) == 1:
            continue
        found = ", ".join(applied) if applied else "none"
        violations.append(f"{item.nodeid} (test-type markers found: {found})")
    return violations


def violation_report(violations: list[str]) -> str:
    """Render violations as the message a failing collection should print.

    Args:
        violations: Lines from :func:`policy_violations`.

    Returns:
        A message naming each offending test and the fix, ready to hand to
        ``pytest.UsageError``.
    """
    choices = ", ".join(f"@pytest.mark.{marker}" for marker in TEST_TYPE_MARKERS)
    offenders = "\n".join(f"  {line}" for line in violations)
    return (
        f"{len(violations)} test(s) do not declare exactly one test-type marker.\n"
        f"{offenders}\n"
        f"Apply exactly one of: {choices}\n"
        "The marker decides whether continuous integration runs the test, so an "
        "unmarked test is one nobody is running."
    )
