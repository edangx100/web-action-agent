"""The benchmark suite as committed, and the loader that reads it.

Two groups of tests. The first is about the real suite in ``benchmarks/``:
that it loads, that it is the right size, that it exercises all three kinds of
interaction, and that every task's start URL produces an allowlist covering
its own site. These are the checks that would otherwise fail on the first
morning of a sweep, after the browser had opened.

The second group is about the loader's refusals, each tried on a small suite
written to a temporary folder: a misspelled key, a start URL on the wrong site,
a task file nobody listed. Each is a mistake that is easy to make while editing
YAML by hand and that, unrefused, would quietly change what the benchmark
measures.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml
from pydantic import ValidationError

from web_action_agent import cli
from web_action_agent.allowlist import derive_domain, host_of
from web_action_agent.benchmark import (
    DIFFICULTIES,
    MAX_TASKS,
    MIN_TASKS,
    VERDICT_METHODS,
    BenchmarkError,
    Suite,
    load_suite,
    load_task,
)

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parent.parent
SUITE_PATH = REPO_ROOT / "benchmarks" / "suite.yaml"


@pytest.fixture(scope="module")
def suite() -> Suite:
    """Load the committed suite once for every test that reads it."""
    return load_suite(SUITE_PATH)


# --- The committed suite ------------------------------------------------------


def test_every_task_file_validates_against_the_schema(suite: Suite) -> None:
    # load_suite would have raised on the first bad file; this also checks
    # that the files it loaded are the files on disk, one for one.
    on_disk = sorted(p.stem for p in (SUITE_PATH.parent / "tasks").glob("*.yaml"))
    assert sorted(task.id for task in suite.tasks) == on_disk


def test_suite_size_is_within_bounds(suite: Suite) -> None:
    assert MIN_TASKS <= len(suite.tasks) <= MAX_TASKS


def test_site_count_is_four_or_five(suite: Suite) -> None:
    used = {task.site for task in suite.tasks}
    assert 4 <= len(used) <= 5
    # A site in the manifest that no task uses is a leftover, not a site.
    assert used == set(suite.sites)


def test_difficulty_labels_cover_all_three_patterns(suite: Suite) -> None:
    assert {task.difficulty for task in suite.tasks} == set(DIFFICULTIES)


def test_every_start_url_derives_an_allowlist_covering_its_site(suite: Suite) -> None:
    for task in suite.tasks:
        site = suite.site_of(task)
        allowlist = task.allowlist()
        assert site.domain in allowlist.entries, task.id
        assert allowlist.allows(task.start_url), task.id
        assert derive_domain(host_of(task.start_url)) == site.domain, task.id


def test_every_start_url_is_https(suite: Suite) -> None:
    # Not a schema rule (http is a website too), but every site chosen serves
    # https, and a task pinned to http would add a redirect to every run.
    for task in suite.tasks:
        assert task.start_url.startswith("https://"), task.id


def test_every_verdict_method_is_a_known_one(suite: Suite) -> None:
    for task in suite.tasks:
        assert set(task.methods) <= set(VERDICT_METHODS), task.id


def test_machine_checks_are_used_where_possible(suite: Suite) -> None:
    # The judge is the costly, arguable layer. Most tasks should be decidable
    # without it, so the suite's number is not mostly one model's opinion.
    judged_only = [task.id for task in suite.tasks if task.methods == ("judge",)]
    assert len(judged_only) <= len(suite.tasks) // 4, judged_only


def test_no_site_banner_costs_more_than_one_step(suite: Suite) -> None:
    # Sites were chosen for light or no consent interstitials. A site whose
    # banner costs more would spend part of every task's step budget before
    # the task begins, and would need saying so in the results.
    for site in suite.sites.values():
        assert site.consent_steps <= 1, site.id


def test_every_site_records_its_terms(suite: Suite) -> None:
    for site in suite.sites.values():
        assert site.terms.terms_url.startswith("https://"), site.id
        assert site.terms.robots_url.endswith("/robots.txt"), site.id
        assert site.terms.summary.strip(), site.id


def test_task_lookup_by_id(suite: Suite) -> None:
    first = suite.tasks[0]
    assert suite.task(first.id) is first
    with pytest.raises(KeyError):
        suite.task("no-such-task")


def test_cli_default_suite_is_the_committed_manifest() -> None:
    assert (REPO_ROOT / cli.DEFAULT_SUITE).resolve() == SUITE_PATH


# --- The loader's refusals ----------------------------------------------------


def a_site(**overrides: Any) -> dict[str, Any]:
    """Build the plain data of one valid manifest site entry.

    Args:
        **overrides: Fields to replace.

    Returns:
        dict[str, Any]: The site, ready to be written into a manifest.
    """
    site: dict[str, Any] = {
        "id": "example",
        "name": "Example",
        "domain": "example.org",
        "consent_steps": 0,
        "terms": {
            "terms_url": "https://example.org/terms",
            "robots_url": "https://example.org/robots.txt",
            "summary": "Allows automated read-only access.",
        },
    }
    site.update(overrides)
    return site


def a_task(task_id: str = "example-task", **overrides: Any) -> dict[str, Any]:
    """Build the plain data of one valid task.

    Args:
        task_id: The task's id, which is also its file name.
        **overrides: Fields to replace.

    Returns:
        dict[str, Any]: The task, ready to be written to a file.
    """
    task: dict[str, Any] = {
        "id": task_id,
        "site": "example",
        "difficulty": "single_page_search",
        "start_url": "https://www.example.org/",
        "instruction": "Report the page title.",
        "verdicts": [{"method": "answer_matches", "accepted": ["Example"]}],
    }
    task.update(overrides)
    return task


def write_suite(
    root: Path,
    tasks: list[dict[str, Any]],
    *,
    sites: list[dict[str, Any]] | None = None,
    listed: list[str] | None = None,
) -> Path:
    """Write a small suite to ``root`` and return the manifest's path.

    Args:
        root: An empty folder.
        tasks: The tasks, each written to ``tasks/<id>.yaml``.
        sites: The manifest's sites. One valid site if not given.
        listed: The task ids the manifest lists. Every task if not given.

    Returns:
        Path: The manifest file.
    """
    (root / "tasks").mkdir(parents=True)
    for task in tasks:
        (root / "tasks" / f"{task['id']}.yaml").write_text(yaml.safe_dump(task))
    manifest = {
        "name": "test-suite",
        "sites": sites if sites is not None else [a_site()],
        "tasks": listed if listed is not None else [task["id"] for task in tasks],
    }
    path = root / "suite.yaml"
    path.write_text(yaml.safe_dump(manifest))
    return path


def test_a_valid_small_suite_loads(tmp_path: Path) -> None:
    suite = load_suite(write_suite(tmp_path, [a_task()]))
    assert [task.id for task in suite.tasks] == ["example-task"]
    assert suite.site_of(suite.tasks[0]).domain == "example.org"


def test_unknown_key_is_refused(tmp_path: Path) -> None:
    # The classic hand-editing slip: a missing "s". Unrefused, the task would
    # run with no extra domains and nobody would know why it failed.
    task = a_task(extra_allowed_domain=["other.org"])
    with pytest.raises(BenchmarkError, match="extra_allowed_domain"):
        load_suite(write_suite(tmp_path, [task]))


@pytest.mark.parametrize("field", ["instruction", "start_url", "difficulty", "verdicts", "site"])
def test_each_required_field_is_required(tmp_path: Path, field: str) -> None:
    task = a_task()
    del task[field]
    with pytest.raises(BenchmarkError, match=field):
        load_suite(write_suite(tmp_path, [task]))


def test_unknown_difficulty_is_refused(tmp_path: Path) -> None:
    with pytest.raises(BenchmarkError, match="difficulty"):
        load_suite(write_suite(tmp_path, [a_task(difficulty="hard")]))


def test_empty_verdict_list_is_refused(tmp_path: Path) -> None:
    with pytest.raises(BenchmarkError, match="verdicts"):
        load_suite(write_suite(tmp_path, [a_task(verdicts=[])]))


def test_unknown_verdict_method_is_refused(tmp_path: Path) -> None:
    task = a_task(verdicts=[{"method": "vibes", "value": "good"}])
    with pytest.raises(BenchmarkError, match="vibes"):
        load_suite(write_suite(tmp_path, [task]))


def test_same_verdict_method_twice_is_refused(tmp_path: Path) -> None:
    task = a_task(
        verdicts=[
            {"method": "url_contains", "value": "a"},
            {"method": "url_contains", "value": "b"},
        ]
    )
    with pytest.raises(BenchmarkError, match="more than once"):
        load_suite(write_suite(tmp_path, [task]))


def test_verdict_with_missing_value_is_refused(tmp_path: Path) -> None:
    # A url_contains with nothing to look for would pass every run.
    task = a_task(verdicts=[{"method": "url_contains", "value": ""}])
    with pytest.raises(BenchmarkError):
        load_suite(write_suite(tmp_path, [task]))


@pytest.mark.parametrize(
    "url", ["javascript:alert(1)", "file:///etc/passwd", "www.example.org", "ftp://example.org/"]
)
def test_start_url_must_be_a_website(tmp_path: Path, url: str) -> None:
    with pytest.raises(BenchmarkError, match="start_url"):
        load_suite(write_suite(tmp_path, [a_task(start_url=url)]))


def test_start_url_on_a_different_site_is_refused(tmp_path: Path) -> None:
    task = a_task(start_url="https://www.example.com/")
    with pytest.raises(BenchmarkError, match="example.com"):
        load_suite(write_suite(tmp_path, [task]))


def test_lookalike_start_url_is_refused(tmp_path: Path) -> None:
    # Contains the site's domain, but is somebody else's.
    task = a_task(start_url="https://example.org.evil.net/")
    with pytest.raises(BenchmarkError, match="evil.net"):
        load_suite(write_suite(tmp_path, [task]))


def test_public_suffix_as_extra_domain_is_refused(tmp_path: Path) -> None:
    # "gov.sg" would admit every Singapore government site at once.
    task = a_task(extra_allowed_domains=["gov.sg"])
    with pytest.raises(BenchmarkError, match="public suffix"):
        load_suite(write_suite(tmp_path, [task]))


def test_extra_domain_is_added_to_the_allowlist(tmp_path: Path) -> None:
    suite = load_suite(write_suite(tmp_path, [a_task(extra_allowed_domains=["other.org"])]))
    assert suite.tasks[0].allowlist().allows("https://www.other.org/page")


def test_task_naming_an_undescribed_site_is_refused(tmp_path: Path) -> None:
    with pytest.raises(BenchmarkError, match="not described"):
        load_suite(write_suite(tmp_path, [a_task(site="elsewhere")]))


def test_id_must_match_file_name(tmp_path: Path) -> None:
    (tmp_path / "wrong-name.yaml").write_text(yaml.safe_dump(a_task("right-name")))
    with pytest.raises(BenchmarkError, match="file name"):
        load_task(tmp_path / "wrong-name.yaml")


def test_badly_shaped_id_is_refused(tmp_path: Path) -> None:
    (tmp_path / "Bad_Id.yaml").write_text(yaml.safe_dump(a_task("Bad_Id")))
    with pytest.raises(BenchmarkError, match="lowercase"):
        load_task(tmp_path / "Bad_Id.yaml")


def test_unlisted_task_file_is_refused(tmp_path: Path) -> None:
    tasks = [a_task("listed"), a_task("forgotten")]
    with pytest.raises(BenchmarkError, match="forgotten"):
        load_suite(write_suite(tmp_path, tasks, listed=["listed"]))


def test_listed_task_with_no_file_is_refused(tmp_path: Path) -> None:
    with pytest.raises(BenchmarkError, match="missing-task"):
        load_suite(write_suite(tmp_path, [a_task()], listed=["example-task", "missing-task"]))


def test_task_listed_twice_is_refused(tmp_path: Path) -> None:
    listed = ["example-task", "example-task"]
    with pytest.raises(BenchmarkError, match="listed twice"):
        load_suite(write_suite(tmp_path, [a_task()], listed=listed))


def test_site_listed_twice_is_refused(tmp_path: Path) -> None:
    with pytest.raises(BenchmarkError, match="listed twice"):
        load_suite(write_suite(tmp_path, [a_task()], sites=[a_site(), a_site()]))


def test_site_domain_must_be_registrable_domain(tmp_path: Path) -> None:
    # A host is not a site: the start-URL check compares registrable domains.
    with pytest.raises(BenchmarkError, match="example.org"):
        load_suite(write_suite(tmp_path, [a_task()], sites=[a_site(domain="www.example.org")]))


def test_costly_banner_needs_a_note(tmp_path: Path) -> None:
    with pytest.raises(BenchmarkError, match="consent_note"):
        load_suite(write_suite(tmp_path, [a_task()], sites=[a_site(consent_steps=2)]))
    noted = a_site(consent_steps=2, consent_note="Two-page consent wall.")
    assert load_suite(write_suite(tmp_path / "noted", [a_task()], sites=[noted]))


def test_unreviewed_sites_are_listed(tmp_path: Path) -> None:
    reviewed = a_site(id="done", terms={**a_site()["terms"], "reviewed_on": "2026-10-09"})
    pending = a_site(id="pending")
    suite = load_suite(write_suite(tmp_path, [a_task(site="done")], sites=[reviewed, pending]))
    assert [site.id for site in suite.unreviewed_sites()] == ["pending"]


def test_malformed_yaml_names_the_file(tmp_path: Path) -> None:
    path = write_suite(tmp_path, [a_task()])
    (tmp_path / "tasks" / "example-task.yaml").write_text("id: [unclosed\n")
    with pytest.raises(BenchmarkError, match="example-task.yaml"):
        load_suite(path)


def test_tasks_are_frozen(suite: Suite) -> None:
    # A task edited during a sweep would make its earlier trials incomparable
    # with its later ones.
    with pytest.raises(ValidationError):
        suite.tasks[0].instruction = "something else"  # type: ignore[misc]
