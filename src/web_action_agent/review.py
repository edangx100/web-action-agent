"""A person checking the machine's marks, one run at a time.

Every benchmark trial is marked by machine: by the checks of the answer and
the page, and by the AI judge. Those marks are the published results, so they
need checking themselves, and the only thing that can check them is a person.
This module is that person's tool. It shows one finished trial at a time --
the task, the agent's final answer, its final screenshot, and what the machine
decided -- asks for the person's own verdict, and writes it down.

**What gets written.** One line per label in ``results/human_labels.jsonl``,
a JSON object naming the trial folder, the person's verdict (pass or fail),
and for a fail, which of the four kinds of failure it was. The machine's mark
at the time is written next to it, so the file alone says where the two agreed
-- it is committed with the results, while the run folders it points to are
not.

**Resumable.** Nobody labels twenty runs in one sitting. Each label is saved
the moment it is given, so quitting -- or Ctrl-C, or closing the terminal --
loses nothing already answered. Starting again skips every trial that already
has a label and begins at the first one that does not. A trial is never
labelled twice.

**A fair sample, in a fixed order.** The trials are not shown sweep by sweep
and task by task: twenty labels taken from the front of that list would all
come from one sweep and three tasks. Instead they are shown in rounds. Round
one has one trial of every task in every sweep; round two has a second trial
of each; and so on. Within a round, the order comes from a hash of each
trial's name, which looks shuffled but is the same every time, so the order
does not change between sessions. So the first twenty labels cover twenty
different task-and-sweep pairs before any pair is seen twice.

**Only finished marks are shown.** A trial still waiting for its kind of
failure, or whose judging went wrong, has no machine verdict to check yet.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .benchmark import AnswerMatches, BenchmarkTask, ElementVisible, Suite, UrlContains
from .benchmark import Judge as JudgeMethod
from .checkers import final_state
from .eval_harness import SWEEP_FILE, TrialVerdict, iter_verdicts
from .metrics import find_sweeps, read_human_labels
from .reporting import describe_action
from .trace import TraceError
from .trace_reader import read_run

#: How many trials to label when no other number is asked for. About twenty
#: hand-labelled runs is enough to measure agreement with the judge without
#: turning the check into a second benchmark.
DEFAULT_TARGET = 20

#: What a person may type at the verdict question, and what each means. The
#: one-letter forms are what anyone actually types; the words are accepted too.
VERDICT_ANSWERS = {
    "p": "pass",
    "pass": "pass",
    "f": "fail",
    "fail": "fail",
    "s": "skip",
    "skip": "skip",
    "q": "quit",
    "quit": "quit",
}

#: What a person may type at the kind-of-failure question. ``q`` quits here
#: too, without labelling the trial on screen.
CATEGORY_ANSWERS = {
    "g": "grounding",
    "grounding": "grounding",
    "p": "planning",
    "planning": "planning",
    "n": "navigation",
    "navigation": "navigation",
    "s": "site_change",
    "site_change": "site_change",
    "site change": "site_change",
    "site-change": "site_change",
    "q": "quit",
    "quit": "quit",
}

#: Why a session ended, as :attr:`ReviewSummary.stopped` reports it.
QUIT = "quit"
TARGET_REACHED = "target"
END_OF_LIST = "end"


class ReviewError(Exception):
    """There is nothing to review where the person pointed."""


# --- What can be reviewed ------------------------------------------------------


@dataclass(frozen=True)
class ReviewItem:
    """One trial a person could label.

    Attributes:
        folder: The trial's run folder.
        sweep: The sweep folder's name, such as ``eval-openrouter-cold``.
        task_id: The benchmark task.
        trial: Which attempt, from 1.
        verdict: The machine's mark on it.
    """

    folder: Path
    sweep: str
    task_id: str
    trial: int
    verdict: TrialVerdict

    @property
    def name(self) -> str:
        """str: The trial's name, the same on every machine: sweep/task/trial-n."""
        return f"{self.sweep}/{self.task_id}/trial-{self.trial}"

    @property
    def position(self) -> str:
        """str: Where the trial comes within its round of the review.

        A hash of :attr:`name`. It looks random, so sweeps and tasks are mixed
        together, but the same name always gives the same position.
        """
        return hashlib.sha256(self.name.encode("utf-8")).hexdigest()


def find_reviewable(runs_dir: Path) -> list[ReviewItem]:
    """Find every trial with a finished machine mark, in review order.

    Args:
        runs_dir: One sweep folder, or a folder of them such as ``runs``.

    Returns:
        list[ReviewItem]: Trials whose mark is complete -- a pass, or a fail
        with its kind -- in rounds: one trial of every task in every sweep,
        then a second of each, and so on. Within a round they are in
        :attr:`ReviewItem.position` order.

    Raises:
        ReviewError: If the folder does not exist or holds no sweep.
    """
    if not runs_dir.is_dir():
        raise ReviewError(f"{runs_dir} is not a folder")
    # The folder may itself be a sweep, or hold several.
    sweeps = [runs_dir] if (runs_dir / SWEEP_FILE).exists() else find_sweeps(runs_dir)
    if not sweeps:
        raise ReviewError(f"{runs_dir} holds no benchmark sweeps (no {SWEEP_FILE} found)")

    items = [
        ReviewItem(
            folder=folder,
            sweep=sweep.name,
            task_id=folder.parent.name,
            trial=int(folder.name.removeprefix("trial-")),
            verdict=verdict,
        )
        for sweep in sweeps
        for folder, verdict in iter_verdicts(sweep)
        if verdict.complete
    ]
    items.sort(key=lambda item: item.position)
    # Number each trial within its own sweep and task -- 0 for the first in
    # hash order, 1 for the next -- and show all the 0s, then all the 1s.
    # sorted() keeps the hash order within each round.
    seen: dict[tuple[str, str], int] = {}
    rounds: list[int] = []
    for item in items:
        group = (item.sweep, item.task_id)
        rounds.append(seen.get(group, 0))
        seen[group] = rounds[-1] + 1
    return [item for _, item in sorted(zip(rounds, items, strict=True), key=lambda p: p[0])]


# --- One label -----------------------------------------------------------------


def _portable(folder: Path, base: Path) -> str:
    """Write a trial folder's path relative to the labels file's folder.

    Relative, so the labels file still points at the right runs when the
    project is on another machine. Relative to the labels file rather than to
    wherever the command happened to be run, so the path means the same thing
    whichever folder it is read from: ``results/human_labels.jsonl`` names a
    trial as ``../runs/<sweep>/<task>/trial-<n>``.

    Args:
        folder: A trial folder.
        base: The folder the labels file is in.

    Returns:
        str: The relative path, with forward slashes, or the absolute path if
        there is no relative one (another drive, on Windows).
    """
    try:
        return Path(os.path.relpath(folder.resolve(), base.resolve())).as_posix()
    except ValueError:
        return folder.resolve().as_posix()


@dataclass(frozen=True)
class HumanLabel:
    """A person's verdict on one trial, as one line of the labels file.

    Attributes:
        run: The trial folder, relative to the labels file's folder.
        sweep: The sweep's name.
        task_id: The benchmark task.
        trial: Which attempt.
        passed: The person's verdict.
        category: For a fail, the person's kind of failure. None for a pass.
        machine_status: The machine's mark when the person labelled it.
        machine_category: The machine's kind of failure, or None for a pass.
        labelled_at: When, RFC 3339 in UTC.
    """

    run: str
    sweep: str
    task_id: str
    trial: int
    passed: bool
    category: str | None
    machine_status: str
    machine_category: str | None
    labelled_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    @classmethod
    def for_item(
        cls, item: ReviewItem, passed: bool, category: str | None, base: Path
    ) -> HumanLabel:
        """Make the label for a trial from the person's answers.

        Args:
            item: The trial.
            passed: The person's verdict.
            category: The person's kind of failure, for a fail.
            base: The folder the labels file is in.

        Returns:
            HumanLabel: The label, with the machine's mark copied beside it.
        """
        return cls(
            run=_portable(item.folder, base),
            sweep=item.sweep,
            task_id=item.task_id,
            trial=item.trial,
            passed=passed,
            category=None if passed else category,
            machine_status=item.verdict.status,
            machine_category=item.verdict.category,
        )

    def to_dict(self) -> dict[str, Any]:
        """Turn the label into plain data, for one line of the file.

        Returns:
            dict[str, Any]: Every field.
        """
        return {
            "run": self.run,
            "sweep": self.sweep,
            "task_id": self.task_id,
            "trial": self.trial,
            "passed": self.passed,
            "category": self.category,
            "machine_status": self.machine_status,
            "machine_category": self.machine_category,
            "labelled_at": self.labelled_at,
        }


def append_label(path: Path, label: HumanLabel) -> None:
    """Add one label to the end of the labels file, safely.

    The line is pushed all the way to the disk before this returns, so a
    label the person has answered survives the program being killed straight
    afterwards.

    Args:
        path: The labels file. Created, with its folder, if missing.
        label: The label to add.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(label.to_dict()) + "\n"
    # A run killed halfway through writing a line leaves it without its
    # newline. Start on a fresh line, so the new label is not glued onto the
    # broken one (which the reader skips) and lost with it.
    if path.exists() and path.stat().st_size > 0:
        with path.open("rb") as existing:
            existing.seek(-1, os.SEEK_END)
            if existing.read(1) != b"\n":
                line = "\n" + line
    with path.open("a", encoding="utf-8") as labels:
        labels.write(line)
        labels.flush()
        os.fsync(labels.fileno())


# --- Showing a trial -----------------------------------------------------------


def _success_lines(task: BenchmarkTask) -> list[str]:
    """Say what counts as success for a task, from its task file.

    Args:
        task: The benchmark task.

    Returns:
        list[str]: One line per way the task is checked.
    """
    lines: list[str] = []
    for method in task.verdicts:
        if isinstance(method, AnswerMatches):
            accepted = ", ".join(repr(a) for a in method.accepted)
            lines.append(f"the answer matches one of: {accepted}")
        elif isinstance(method, UrlContains):
            lines.append(f"the final address contains {method.value!r}")
        elif isinstance(method, ElementVisible):
            lines.append(f"the final page shows {method.text!r}")
        elif isinstance(method, JudgeMethod):
            lines.append(f"(for the judge) {method.expected_outcome}")
    return lines


def _machine_lines(verdict: TrialVerdict) -> list[str]:
    """Say what the machine decided, and on what grounds.

    Args:
        verdict: The machine's mark.

    Returns:
        list[str]: The verdict and its reason, each check, what the judge
        said, and the kind of failure.
    """
    lines = [f"  verdict:  {verdict.status.upper()} -- {verdict.reason}"]
    for check in verdict.checks:
        result = "passed" if check.passed else "failed"
        lines.append(f"  check:    {check.method} {result} -- {check.detail}")
    if verdict.judge is not None:
        lines.append(f"  judge:    {verdict.judge.outcome} -- {verdict.judge.justification}")
    if verdict.categorized_by is not None:
        lines.append(f"  kind from judge: {verdict.categorized_by.justification}")
    lines.append(f"  kind of failure: {verdict.category or '(none: it passed)'}")
    return lines


def render_item(item: ReviewItem, suite: Suite | None = None) -> str:
    """Write out everything a person needs to label one trial.

    The evidence comes first and the machine's mark last, so the person reads
    what happened before reading what the machine made of it.

    Args:
        item: The trial.
        suite: The benchmark suite, to say what counts as success. None to
            leave that out.

    Returns:
        str: Several lines of text for the terminal.

    Raises:
        TraceError: If the run folder cannot be read back.
    """
    run = read_run(item.folder)
    final = final_state(run)

    lines = [
        f"Trial:   {item.name}",
        f"Folder:  {item.folder}",
        "",
        f"Task:    {run.task}",
    ]
    if suite is not None:
        try:
            task = suite.task(item.task_id)
        except KeyError:
            lines.append("What counts as success: (this task is not in the suite)")
        else:
            lines.append("What counts as success:")
            lines += [f"  - {line}" for line in _success_lines(task)]
    ended = "interrupted before it finished"
    if run.outcome is not None:
        ended = run.outcome.category + (f" ({run.outcome.reason})" if run.outcome.reason else "")

    # The last screenshot is the final page only when the agent looked at the
    # page after its last move; otherwise it shows the page before.
    screenshot = run.screenshots[-1] if run.screenshots else None
    if screenshot is None:
        shown = "(none recorded)"
    elif not screenshot.exists():
        shown = f"{screenshot} (missing from disk)"
    elif not final.page_is_current:
        shown = f"{screenshot} (the page before the last move; the run ended at {final.url})"
    else:
        shown = str(screenshot)

    lines += [
        "",
        f"Final answer:     {final.answer or '(no answer given)'}",
        f"Final page:       {final.url or '(none)'}",
        f"Final screenshot: {shown}",
        f"How it ended:     {ended}",
        "",
        "Steps:",
    ]
    lines += [
        f"  {step.number}. "
        f"{describe_action(step.action, step.observation) if step.action else '(no action)'}"
        for step in run.steps
    ] or ["  (none)"]
    lines += ["", "The machine's mark:", *_machine_lines(item.verdict)]
    return "\n".join(lines)


# --- A session -----------------------------------------------------------------


@dataclass(frozen=True)
class ReviewSummary:
    """How a review session went.

    Attributes:
        labelled_now: Labels given in this session.
        labelled_total: Trials with a label now, this session's included.
        reviewable: Trials that could be labelled at all.
        skipped: Trials passed over this session; still unlabelled.
        unreadable: Trials whose run folder could not be read, so not shown.
        stopped: Why the session ended: :data:`QUIT`,
            :data:`TARGET_REACHED`, or :data:`END_OF_LIST` (every trial was
            reached; any skipped ones are still unlabelled).
        target: How many labels the session was aiming for.
    """

    labelled_now: int
    labelled_total: int
    reviewable: int
    skipped: int
    unreadable: int
    stopped: str
    target: int

    def render(self) -> str:
        """Say how the session went, in a line or two.

        Returns:
            str: The counts, and what to do next.
        """
        line = (
            f"labelled {self.labelled_now} this session; "
            f"{self.labelled_total} of {self.reviewable} trials now have a label"
        )
        if self.skipped:
            line += f" ({self.skipped} skipped, shown first next time)"
        if self.unreadable:
            line += f"; {self.unreadable} could not be read and were not shown"
        if self.stopped == TARGET_REACHED:
            line += (
                f"\ntarget of {self.target} labels reached. "
                "Use --target with a higher number to label more."
            )
        elif self.stopped == QUIT:
            line += "\nstopped. Run the same command again to carry on where you left off."
        elif self.labelled_total == self.reviewable:
            line += "\nevery trial that can be labelled now has a label."
        return line


def _choose(
    ask: Callable[[str], str],
    say: Callable[[str], None],
    question: str,
    answers: dict[str, str],
) -> str:
    """Ask a question until the answer is one of the allowed ones.

    Args:
        ask: Shows the question and returns what the person typed.
        say: Prints a line.
        question: The question, listing the choices.
        answers: Allowed answers, lower case, and what each means.

    Returns:
        str: The meaning of the answer given.
    """
    while True:
        typed = ask(question).strip().lower()
        if typed in answers:
            return answers[typed]
        say(f"  please answer one of: {', '.join(k for k in answers if len(k) == 1)}")


def review(
    runs_dir: Path,
    labels_path: Path,
    *,
    ask: Callable[[str], str],
    say: Callable[[str], None],
    suite: Suite | None = None,
    target: int = DEFAULT_TARGET,
) -> ReviewSummary:
    """Show trials one at a time and record the person's verdicts.

    Trials that already have a label are passed over without being shown, so
    running this again carries on from the first trial with no label. The
    session ends when the person quits, when ``target`` trials have labels,
    or when every trial has one.

    Args:
        runs_dir: One sweep folder, or a folder of them.
        labels_path: The labels file, read for what is done and appended to.
        ask: Shows a question and returns the typed reply. Should return
            ``"q"`` when there is no more input, so the session ends cleanly.
        say: Prints one block of text.
        suite: The benchmark suite, to show what counts as success.
        target: Stop once this many trials have labels.

    Returns:
        ReviewSummary: How the session went.

    Raises:
        ReviewError: If there are no sweeps where ``runs_dir`` points.
    """
    items = find_reviewable(runs_dir)
    # Labels are keyed by the trial folder's full path, so a label written
    # from one directory is recognised from another.
    done = set(read_human_labels(labels_path))

    def is_labelled(item: ReviewItem) -> bool:
        return str(item.folder.resolve()) in done

    labelled_before = sum(is_labelled(i) for i in items)
    labelled_now = skipped = unreadable = 0
    stopped = END_OF_LIST

    for item in items:
        if is_labelled(item):
            continue
        if labelled_before + labelled_now >= target:
            stopped = TARGET_REACHED
            break
        try:
            shown = render_item(item, suite)
        except TraceError as error:
            # Nobody can judge a run they cannot see. Say so and move on; it
            # stays unlabelled.
            say(f"skipping {item.name}: its run folder cannot be read ({error})")
            unreadable += 1
            continue

        say(
            f"\n=== {labelled_before + labelled_now + 1} of {target} "
            f"({len(items)} trials can be labelled) ===\n{shown}\n"
        )
        verdict = _choose(ask, say, "Your verdict? [p]ass [f]ail [s]kip [q]uit: ", VERDICT_ANSWERS)
        if verdict == "quit":
            stopped = QUIT
            break
        if verdict == "skip":
            skipped += 1
            continue
        category = None
        if verdict == "fail":
            category = _choose(
                ask,
                say,
                "Kind of failure? [g]rounding [p]lanning [n]avigation [s]ite_change [q]uit: ",
                CATEGORY_ANSWERS,
            )
            if category == "quit":
                stopped = QUIT
                break
        # Saved at once, so a session cut short keeps every answer given.
        label = HumanLabel.for_item(item, verdict == "pass", category, labels_path.parent)
        append_label(labels_path, label)
        done.add(str(item.folder.resolve()))
        labelled_now += 1
    else:
        # Every trial was reached without a quit. Skips mean some are still
        # unlabelled, but there is nothing more to show this session.
        stopped = TARGET_REACHED if labelled_before + labelled_now >= target else END_OF_LIST

    return ReviewSummary(
        labelled_now=labelled_now,
        labelled_total=labelled_before + labelled_now,
        reviewable=len(items),
        skipped=skipped,
        unreadable=unreadable,
        stopped=stopped,
        target=target,
    )
