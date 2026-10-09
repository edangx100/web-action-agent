"""The benchmark: which tasks the agent is measured on, and the shape of each.

A benchmark result is only as good as the tasks behind it, so the tasks are
written down as files rather than as a list in code. Each task is one small
YAML file a person can read in a minute: what to ask the agent, which page to
start on, which other sites (if any) it may visit, which kind of interaction it
exercises, and how the result will be marked. A suite file -- the *manifest* --
names the tasks that make up the suite and records, for each site they use,
whether that site's terms permit an automated visitor at all.

This module is the shape of those files and the code that reads them. It does
not run tasks and it does not mark them; the evaluation harness and the
checkers do that, and both start from what :func:`load_suite` returns.

**Three kinds of task.** Every task is labelled with the interaction it
exercises, and the suite should contain all three:

* ``single_page_search`` -- the answer is on the page the task starts on, or
  one search away from it: type into a search box or read a table, then
  report;
* ``form_fill`` -- answer a form of several fields, possibly over several
  pages, and read the outcome;
* ``multi_page_navigation`` -- reach the answer by following links through
  more than one page.

They fail in different ways -- finding the right box, filling the right field,
choosing the right link -- which is why a suite made of only one kind would
say little about the agent.

**How a task is marked.** A task names one or more *verdict methods*. Three can
be decided by a machine with no AI involved: the final URL contains a value,
the agent's answer matches one of the accepted answers, or some text is visible
on the final page. The fourth, ``judge``, hands the result to a second model
with a sentence describing what success looks like, for tasks whose answer is
not a fixed string. Machine checks are preferred wherever a task allows one,
because they cannot disagree with themselves.

**What the loader refuses.** A task file that is malformed is a mistake to find
before a sweep, not halfway through one, so :func:`load_suite` checks the whole
suite up front: every file parses against the schema, every task names a site
the manifest describes, every start URL actually belongs to that site, and every
task file on disk is listed in the manifest (a forgotten file would otherwise be
silently left out of the results).
"""

from __future__ import annotations

import datetime as dt
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from .allowlist import ALLOWED_SCHEMES, Allowlist, AllowlistError, derive_domain, host_of

#: The three interaction patterns a task can exercise. A task's ``difficulty``
#: field is one of these.
DIFFICULTIES = ("single_page_search", "form_fill", "multi_page_navigation")

#: Fewest and most tasks a full suite should have. Fewer than ten and one lucky
#: or unlucky task moves the success rate by more than ten points; more than
#: fifteen and three trials per task across several models stops fitting the
#: time and money budget for a sweep.
MIN_TASKS = 10
MAX_TASKS = 15

#: What a task id looks like: lowercase words joined by hyphens, such as
#: ``wikipedia-architect``. The id is also the task file's name, and it ends up
#: in run folder names, so it is kept to characters that are safe everywhere.
TASK_ID = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")

#: The same rule for site ids in the manifest, such as ``python-docs``.
SITE_ID = TASK_ID

Difficulty = Literal["single_page_search", "form_fill", "multi_page_navigation"]


class BenchmarkError(Exception):
    """A suite or task file could not be loaded.

    The message always names the file at fault, since a suite is a dozen files
    and "invalid task" on its own would leave the reader opening each one.
    """


class _Strict(BaseModel):
    """Base for every model here: unknown keys are errors, and values are fixed.

    ``extra="forbid"`` is the important part. A misspelled key in a YAML file --
    ``extra_allowed_domain`` without the ``s`` -- would otherwise be dropped
    without a word, and the task would run with no extra domains at all.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)


# --- Verdict methods ---------------------------------------------------------
#
# Each verdict method is its own small model, told apart by the ``method`` key.
# That keeps each one's fields required where they should be: a ``url_contains``
# without a value is an error in the file, not a check that always passes.


class UrlContains(_Strict):
    """Pass if the page the run ended on has ``value`` in its URL.

    Attributes:
        method: Always ``"url_contains"``.
        value: Text the final URL must contain, such as ``Marina_Bay_Sands``.
    """

    method: Literal["url_contains"]
    value: str = Field(min_length=1)


class AnswerMatches(_Strict):
    """Pass if the agent's final answer matches one of the accepted answers.

    How strictly "matches" is meant -- case, spacing, punctuation -- is the
    checker's business, not the task's. The task only lists every answer that
    should count as right.

    Attributes:
        method: Always ``"answer_matches"``.
        accepted: Every answer that counts as correct, such as
            ``["£51.77", "51.77"]``. At least one.
    """

    method: Literal["answer_matches"]
    accepted: tuple[Annotated[str, Field(min_length=1)], ...] = Field(min_length=1)


class ElementVisible(_Strict):
    """Pass if ``text`` is visible on the page the run ended on.

    Attributes:
        method: Always ``"element_visible"``.
        text: Text that must be on screen at the end, such as the opening words
            of a search result the task should have produced.
    """

    method: Literal["element_visible"]
    text: str = Field(min_length=1)


class Judge(_Strict):
    """Hand the result to the judge model, with a description of success.

    Used where the right answer is not one fixed string: a description, or a
    fact that changes over time such as the date of the next public holiday.

    Attributes:
        method: Always ``"judge"``.
        expected_outcome: One or two sentences describing what a successful
            result looks like, written for the judge to compare against.
    """

    method: Literal["judge"]
    expected_outcome: str = Field(min_length=1)


#: Any one verdict method. Pydantic picks the right model by reading ``method``.
Verdict = Annotated[
    UrlContains | AnswerMatches | ElementVisible | Judge,
    Field(discriminator="method"),
]

#: Every verdict method name, in the order they are documented above.
VERDICT_METHODS = ("url_contains", "answer_matches", "element_visible", "judge")


# --- Tasks -------------------------------------------------------------------


class BenchmarkTask(_Strict):
    """One benchmark task, as read from one file in ``benchmarks/tasks/``.

    Attributes:
        id: Short name, the same as the file's name without ``.yaml``.
        instruction: What the agent is asked to do, exactly as it will be given
            to it. Says what form the answer should take, where that matters.
        site: The id of the site this task runs on, as listed in the manifest.
        start_url: The page the run opens on. Required: the agent never starts
            on a blank page and guesses an address. It also decides which site
            the run may visit -- see :meth:`allowlist`.
        extra_allowed_domains: Other sites this task may visit, for the rare
            task that genuinely needs two. Usually empty.
        difficulty: Which of the three interaction patterns the task exercises.
        verdicts: How the result is marked. At least one, and each method at
            most once.
        notes: Anything a person reading the task should know -- why the answer
            is what it is, or what might make the task drift. Not shown to the
            agent.
    """

    id: str
    instruction: str = Field(min_length=1)
    site: str
    start_url: str
    extra_allowed_domains: tuple[str, ...] = ()
    difficulty: Difficulty
    verdicts: tuple[Verdict, ...] = Field(min_length=1)
    notes: str = ""

    @field_validator("id")
    @classmethod
    def _id_shape(cls, value: str) -> str:
        """Refuse an id that would be awkward in a file or folder name."""
        if not TASK_ID.fullmatch(value):
            raise ValueError(
                f"task id {value!r} must be lowercase letters and digits joined by hyphens"
            )
        return value

    @field_validator("start_url")
    @classmethod
    def _start_url_is_a_website(cls, value: str) -> str:
        """Refuse a start URL the agent could never open.

        Only ``http`` and ``https`` are accepted, for the same reason the agent
        may only navigate to those: nothing else is a website.
        """
        scheme = value.split(":", 1)[0].lower() if ":" in value else ""
        if scheme not in ALLOWED_SCHEMES:
            raise ValueError(f"start_url {value!r} must be an http or https URL")
        return value

    @model_validator(mode="after")
    def _each_method_once(self) -> BenchmarkTask:
        """Refuse a task that names the same verdict method twice.

        Two ``url_contains`` entries would leave it unclear whether both must
        pass or either may. A task that needs two URL fragments is better
        written as one check on the more specific of them.
        """
        methods = [verdict.method for verdict in self.verdicts]
        repeated = sorted({method for method in methods if methods.count(method) > 1})
        if repeated:
            raise ValueError(f"verdict method(s) listed more than once: {', '.join(repeated)}")
        return self

    @property
    def methods(self) -> tuple[str, ...]:
        """The names of this task's verdict methods, in file order."""
        return tuple(verdict.method for verdict in self.verdicts)

    def allowlist(self, config_extra: tuple[str, ...] = ()) -> Allowlist:
        """Work out which sites a run of this task may visit.

        Uses the same rule as every other run: the site of the start URL, plus
        any domains from the configuration file, plus this task's own extras.

        Args:
            config_extra: Domains the configuration file adds for every run.

        Returns:
            Allowlist: The resolved list.

        Raises:
            AllowlistError: If the start URL or an extra domain is unusable.
        """
        return Allowlist.derive(
            self.start_url,
            config_extra=config_extra,
            task_extra=self.extra_allowed_domains,
        )


# --- The manifest --------------------------------------------------------------


class TermsReview(_Strict):
    """The record that a site may be visited by an automated, read-only agent.

    Drafted from the site's published terms and its ``robots.txt``, then
    confirmed by a person. ``reviewed_on`` stays empty until that person has
    read the terms themselves; a drafted summary alone is not a review.

    Attributes:
        terms_url: Where the site's terms of use are published.
        robots_url: The site's ``robots.txt``.
        summary: What the terms and ``robots.txt`` say about automated
            read-only access, and anything in them the tasks have to avoid.
        reviewed_on: The date a person confirmed the summary, or None if no one
            has yet.
    """

    terms_url: str
    robots_url: str
    summary: str = Field(min_length=1)
    reviewed_on: dt.date | None = None


class Site(_Strict):
    """One website the suite uses, as described in the manifest.

    Attributes:
        id: Short name tasks refer to it by, such as ``python-docs``.
        name: Its everyday name, for tables and reports.
        domain: Its registrable domain -- the part someone bought, such as
            ``wikipedia.org``. Every task on this site must start on a page
            under this domain.
        consent_steps: How many actions a cookie or consent banner costs before
            the page can be used: 0 if there is no banner or it does not get in
            the way. Sites were chosen to keep this at 0 or 1.
        consent_note: What the banner is and how it is dismissed. Required
            when ``consent_steps`` is more than 1, because such a site spends
            part of every task's step budget before the task begins, and a
            reader of the results has to be told.
        terms: The record of whether automated access is permitted.
    """

    id: str
    name: str = Field(min_length=1)
    domain: str
    consent_steps: int = Field(ge=0)
    consent_note: str = ""
    terms: TermsReview

    @field_validator("id")
    @classmethod
    def _id_shape(cls, value: str) -> str:
        """Refuse a site id that is not lowercase words joined by hyphens."""
        if not SITE_ID.fullmatch(value):
            raise ValueError(
                f"site id {value!r} must be lowercase letters and digits joined by hyphens"
            )
        return value

    @field_validator("domain")
    @classmethod
    def _domain_is_registrable(cls, value: str) -> str:
        """Refuse a domain that is not exactly a registrable domain.

        ``en.wikipedia.org`` is refused here, with a message suggesting
        ``wikipedia.org``: the start-URL check compares registrable domains,
        so the manifest must be written in the same terms. Which part counts
        as registrable comes from the Public Suffix List, not a guess --
        GOV.UK's is ``www.gov.uk``, because ``gov.uk`` itself is public.
        """
        try:
            derived = derive_domain(host_of(f"https://{value}/"))
        except AllowlistError as error:
            raise ValueError(str(error)) from error
        if derived != value:
            raise ValueError(f"site domain {value!r} should be written as {derived!r}")
        return value

    @model_validator(mode="after")
    def _costly_banner_is_explained(self) -> Site:
        """Require a note for any banner that costs more than one action."""
        if self.consent_steps > 1 and not self.consent_note.strip():
            raise ValueError(
                f"site {self.id!r}: a consent banner costing {self.consent_steps} steps "
                "needs a consent_note saying what it is"
            )
        return self


class Manifest(_Strict):
    """The suite file: its name, the sites it uses, and its tasks, in order.

    Attributes:
        name: The suite's name, recorded in every results table it produces.
        description: A sentence or two on what the suite is for.
        sites: Every site any task runs on.
        tasks: Task ids, in the order a sweep runs them. Each must have a file
            ``<id>.yaml`` in the tasks folder.
        tasks_dir: Where the task files are, relative to the manifest. Defaults
            to ``tasks``.
    """

    name: str = Field(min_length=1)
    description: str = ""
    sites: tuple[Site, ...] = Field(min_length=1)
    tasks: tuple[str, ...] = Field(min_length=1)
    tasks_dir: str = "tasks"


# --- The loaded suite ------------------------------------------------------------


@dataclass(frozen=True)
class Suite:
    """A whole benchmark suite, loaded and checked.

    Attributes:
        name: The suite's name, from the manifest.
        path: The manifest file it was loaded from.
        sites: Each site, keyed by id.
        tasks: Every task, in the manifest's order.
    """

    name: str
    path: Path
    sites: dict[str, Site]
    tasks: tuple[BenchmarkTask, ...]

    def site_of(self, task: BenchmarkTask) -> Site:
        """Return the site a task runs on.

        Args:
            task: A task from this suite.

        Returns:
            Site: Its entry in the manifest.
        """
        return self.sites[task.site]

    def task(self, task_id: str) -> BenchmarkTask:
        """Find one task by id.

        Args:
            task_id: The id to look for.

        Returns:
            BenchmarkTask: The task.

        Raises:
            KeyError: If the suite has no such task.
        """
        for candidate in self.tasks:
            if candidate.id == task_id:
                return candidate
        raise KeyError(task_id)

    def unreviewed_sites(self) -> tuple[Site, ...]:
        """List the sites whose terms no person has confirmed yet.

        Returns:
            tuple[Site, ...]: Sites whose ``terms.reviewed_on`` is still empty.
            Should be empty before any published sweep.
        """
        return tuple(site for site in self.sites.values() if site.terms.reviewed_on is None)


def _read_yaml(path: Path) -> Any:
    """Read one YAML file, turning every way that can fail into one error.

    Args:
        path: The file.

    Returns:
        Any: Whatever the file holds.

    Raises:
        BenchmarkError: If the file is missing or is not valid YAML.
    """
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise BenchmarkError(f"{path}: cannot be read: {error}") from error
    except yaml.YAMLError as error:
        raise BenchmarkError(f"{path}: not valid YAML: {error}") from error


def load_task(path: Path) -> BenchmarkTask:
    """Read and check one task file.

    Args:
        path: A file such as ``benchmarks/tasks/wikipedia-architect.yaml``.

    Returns:
        BenchmarkTask: The task.

    Raises:
        BenchmarkError: If the file cannot be read, does not match the task
            schema, or holds an id different from its own file name. That last
            rule means a task can always be found from its id by looking for
            the file of that name.
    """
    data = _read_yaml(path)
    try:
        task = BenchmarkTask.model_validate(data)
    except ValidationError as error:
        raise BenchmarkError(f"{path}: not a valid task:\n{error}") from error
    if task.id != path.stem:
        raise BenchmarkError(f"{path}: task id {task.id!r} does not match the file name")
    return task


def load_suite(manifest_path: Path) -> Suite:
    """Read a suite manifest and every task it lists, and check them together.

    Checks that each file is valid on its own, and also the things that only
    show when the files are read together: duplicate ids, a task naming a site
    the manifest does not describe, a start URL on a different site from the
    one the task claims, and a task file nobody listed.

    Args:
        manifest_path: The suite file, such as ``benchmarks/suite.yaml``.

    Returns:
        Suite: The checked suite.

    Raises:
        BenchmarkError: On the first problem found, naming the file at fault.
    """
    data = _read_yaml(manifest_path)
    try:
        manifest = Manifest.model_validate(data)
    except ValidationError as error:
        raise BenchmarkError(f"{manifest_path}: not a valid suite manifest:\n{error}") from error

    # Sites by id. Two sites with one id would make "site: wikipedia" ambiguous.
    sites: dict[str, Site] = {}
    for site in manifest.sites:
        if site.id in sites:
            raise BenchmarkError(f"{manifest_path}: site id {site.id!r} is listed twice")
        sites[site.id] = site

    duplicates = sorted({tid for tid in manifest.tasks if manifest.tasks.count(tid) > 1})
    if duplicates:
        raise BenchmarkError(f"{manifest_path}: task(s) listed twice: {', '.join(duplicates)}")

    tasks_dir = manifest_path.parent / manifest.tasks_dir
    tasks: list[BenchmarkTask] = []
    for task_id in manifest.tasks:
        task_path = tasks_dir / f"{task_id}.yaml"
        task = load_task(task_path)
        site = sites.get(task.site)
        if site is None:
            raise BenchmarkError(
                f"{task_path}: site {task.site!r} is not described in {manifest_path.name}"
            )
        _check_start_url(task, site, task_path)
        tasks.append(task)

    # A task file on disk that the manifest does not list would never run, and
    # nothing in the results would show it was missing. Refuse it instead.
    listed = set(manifest.tasks)
    unlisted = sorted(p.stem for p in tasks_dir.glob("*.yaml") if p.stem not in listed)
    if unlisted:
        raise BenchmarkError(
            f"{tasks_dir}: task file(s) not listed in {manifest_path.name}: {', '.join(unlisted)}"
        )

    return Suite(name=manifest.name, path=manifest_path, sites=sites, tasks=tuple(tasks))


def _check_start_url(task: BenchmarkTask, site: Site, task_path: Path) -> None:
    """Check a task's start URL gives an allowlist that covers its own site.

    This is the same allowlist a real run would build, so a task that passes
    here cannot fail at the start of a run for allowlist reasons.

    Args:
        task: The task.
        site: The manifest's entry for the site it names.
        task_path: The task's file, for the error message.

    Raises:
        BenchmarkError: If the allowlist cannot be built, or if the start URL is
            on a different site from the one the task names.
    """
    try:
        start_domain = derive_domain(host_of(task.start_url))
        task.allowlist()
    except AllowlistError as error:
        raise BenchmarkError(f"{task_path}: unusable start URL or domain: {error}") from error
    if start_domain != site.domain:
        raise BenchmarkError(
            f"{task_path}: start URL is on {start_domain!r}, but site {site.id!r} "
            f"is {site.domain!r}"
        )
