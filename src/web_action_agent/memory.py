"""What the planner is told about everything that happened before this step.

A language model has no recollection between calls. Each step's prompt is
assembled from scratch, so whatever the agent "remembers" is exactly what gets
written into that prompt -- and deciding what that is turns out to be a real
design problem rather than bookkeeping.

**Not the whole history.** The obvious answer is to include every step so far,
and it fails twice over. It gets *expensive*: a twenty-step run would send the
first step twenty times. And past a point it gets *worse*, not better -- the
current page, which is what the next decision is actually about, ends up
buried under a long transcript of pages the agent has already left.

**Two tiers instead.** The last few steps are kept word for word, in a rolling
window, because recent detail is what the next move most often depends on. Older
steps are *folded* into a running notepad -- the scratchpad -- which keeps the
parts of them still worth knowing: what was typed, where the agent went, and any
value it was told to hold on to. Nothing simply falls off the end.

**Folding is deterministic, not another model call.** The fashionable way to
fold history is to ask a model to summarise it. That costs a call on every step
past the window, and it can quietly get a value wrong -- "$12.99" becoming
"about $13" is exactly the loss the notepad exists to prevent. Here a folded
step is the same line it was in the window, and when even those lines would
grow too long they are *counted* rather than paraphrased. Nothing is written
into the notepad that did not happen.

**Bounded by construction.** Every part of what memory hands the planner has a
ceiling -- steps in the window, notes, folded lines, the length of any one line
-- so the prompt is the same size on step 20 as on step 200. A notepad that grew
by a line per step would only delay the problem it was built to solve.

**Values that must survive belong in a note.** Steps age out by step count;
notes do not. A note is keyed, so recording the same fact again replaces it
rather than repeating it, and it stays in the notepad for the whole run unless
the note ceiling is reached -- in which case the notepad says how many were
dropped rather than pretending there were never any.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol
from urllib.parse import urlsplit

from .actions import Action, Fill, Navigate
from .observation import Observation
from .planner import MAX_HISTORY_STEPS, render_history

if TYPE_CHECKING:
    from .config import Config

#: Steps kept word for word. Small on purpose: the planner's own evidence is the
#: page in front of it, and five steps is enough to see the last few moves and
#: whether they worked without crowding that page out.
DEFAULT_WINDOW = 5

#: Most notes kept at once. Notes are keyed, so this counts distinct facts, and
#: a task needing more than a dozen of those at once is a task whose notes
#: should be a table in the answer rather than a list in the prompt.
MAX_NOTES = 12

#: Most folded steps shown as their own line before the oldest are counted
#: instead. Enough to see the shape of the last stretch of a run; not so many
#: that the notepad becomes the transcript it replaced.
MAX_FOLDED_LINES = 6

#: Most pages, and most typed values, named in the one line that summarises the
#: steps too old to be shown individually.
MAX_TALLY_ITEMS = 5

#: Longest any single line memory writes, in characters. A note's value can come
#: from a page, and a page can hold a paragraph; one runaway note must not be
#: able to undo every other ceiling in this module.
MAX_LINE_CHARS = 200

#: Longest address shown in a step line. The planner sees the full current URL
#: in the observation; history only needs enough to tell pages apart.
MAX_URL_CHARS = 80


class StepResult(Protocol):
    """The slice of an execution result memory reads.

    Declared structurally so memory does not import the executor -- and with it
    the browser -- just to read three strings. The executor's own result has all
    three, and so can a stand-in in a test.
    """

    status: str
    detail: str
    url: str


@dataclass(frozen=True)
class StepRecord:
    """One step, as memory writes it down.

    Attributes:
        number: The step's position in the run, from 1. Kept on the record
            rather than inferred from where it sits, because the window starts
            at step 1 and then does not -- a history whose first line always
            said "1." would tell the planner the run had just begun.
        action: The action as :meth:`~web_action_agent.actions.BaseAction.describe`
            renders it, e.g. ``click 4``.
        target: What the element was, e.g. ``button "Search"``. Recorded
            because the number means nothing once the page is observed again:
            ``click 4`` on step 3 and ``click 4`` on step 9 are usually two
            different things, and ``button "Search"`` is not.
        outcome: What came of it, in a few words -- ``ok``, or the failure and
            its reason. Empty for a step nobody reported on.
        url: Where the page was afterwards, shortened, when the step moved it.
        entered: The text typed, for a fill. Kept separately so it survives even
            when this step's line has been counted away.
        kind: The action's name, for counting when lines are folded away.
    """

    number: int
    action: str
    target: str = ""
    outcome: str = ""
    url: str = ""
    entered: str = ""
    kind: str = ""

    def entry(self) -> dict[str, str]:
        """Render this step in the shape the planner's history renderer takes.

        Returns:
            dict[str, str]: An ``action`` and, when there is one, a ``result``.
            A step with no outcome is left without one, so that the renderer's
            own "no result recorded" says so rather than implying success.
        """
        action = f"{self.action} ({self.target})" if self.target else self.action
        result = self.outcome
        if self.url:
            result = f"{result}, now at {self.url}" if result else f"now at {self.url}"
        return {"action": action, "result": result} if result else {"action": action}

    def line(self) -> str:
        """Render this step as the one line both the window and the notepad show.

        Returns:
            str: e.g. ``7. click 4 (button "Search") -> ok, now at example.org/results``.
            The same line whether the step is recent or folded, so a step does
            not change its description by getting older.
        """
        (rendered,) = render_history([self.entry()], start=self.number)
        return _clip(rendered)

    def to_dict(self) -> dict[str, Any]:
        """Render this step as plain data, for the run's trace.

        Returns:
            dict[str, Any]: Every field, JSON-safe.
        """
        return {
            "number": self.number,
            "action": self.action,
            "target": self.target,
            "outcome": self.outcome,
            "url": self.url,
            "entered": self.entered,
            "kind": self.kind,
        }


@dataclass
class _Tally:
    """The steps too old to show individually, counted rather than paraphrased.

    Counting is the one form of compression that cannot misstate anything:
    "3 clicks" is true or it is not, where a summary sentence can be wrong in a
    way nobody will notice.

    Attributes:
        first: The earliest step counted.
        last: The latest step counted.
        kinds: How many of each action, in the order first seen.
        pages: Addresses reached, most recent last, each named once.
        entered: Text typed, most recent last, each named once.
        dropped_pages: Addresses no longer named because the list was full.
        dropped_entered: Typed values no longer named, likewise.
    """

    first: int = 0
    last: int = 0
    kinds: dict[str, int] = field(default_factory=dict)
    pages: list[str] = field(default_factory=list)
    entered: list[str] = field(default_factory=list)
    dropped_pages: int = 0
    dropped_entered: int = 0

    def add(self, step: StepRecord) -> None:
        """Count one more step into the tally.

        Args:
            step: The step that no longer gets a line of its own.
        """
        self.first = self.first or step.number
        self.last = step.number
        kind = step.kind or "step"
        self.kinds[kind] = self.kinds.get(kind, 0) + 1
        if step.url:
            self.dropped_pages += _remember(self.pages, step.url)
        if step.entered:
            self.dropped_entered += _remember(self.entered, step.entered)

    def line(self) -> str:
        """Render the tally as the notepad's first line about earlier steps.

        Values typed come before pages visited, deliberately. The line has a
        length ceiling, and a line that has to be cut should lose the addresses
        -- which the planner can rediscover by looking -- rather than the
        values, which it cannot.

        Returns:
            str: e.g. ``steps 1-9, counted: 5 click, 2 fill, 2 scroll; typed
            "rust async"; visited example.org/search, example.org/results``.
        """
        span = (
            f"step {self.first}" if self.first == self.last else f"steps {self.first}-{self.last}"
        )
        parts = [", ".join(f"{count} {kind}" for kind, count in self.kinds.items())]
        if self.entered:
            parts.append(
                "typed " + _listed([f'"{value}"' for value in self.entered], self.dropped_entered)
            )
        if self.pages:
            parts.append("visited " + _listed(self.pages, self.dropped_pages))
        return _clip(f"{span}, counted: " + "; ".join(parts))

    @property
    def empty(self) -> bool:
        """bool: Whether any step has been counted yet."""
        return not self.kinds


class Memory:
    """The agent's recollection of the run so far: a window and a notepad.

    One per run. Nothing here reads the browser or calls a model; it is handed
    each step after the step happens, and handed back the two strings the
    planner's prompt takes.

    Attributes:
        window: How many recent steps are kept word for word.
    """

    def __init__(
        self,
        window: int = DEFAULT_WINDOW,
        *,
        max_notes: int = MAX_NOTES,
        max_folded: int = MAX_FOLDED_LINES,
    ) -> None:
        """Start an empty memory.

        Args:
            window: Recent steps kept word for word. Capped at the planner's own
                ceiling, since a window the planner would cut short is a window
                that only looks bigger.
            max_notes: Most notes held at once. See :data:`MAX_NOTES`.
            max_folded: Most folded steps shown as their own line. See
                :data:`MAX_FOLDED_LINES`.

        Raises:
            ValueError: If any ceiling is below one. A window of zero is not
                "no memory", it is a planner that cannot see whether its last
                click did anything -- a setting that is easy to reach by
                accident and never what anyone meant.
        """
        for name, value in (
            ("window", window),
            ("max_notes", max_notes),
            ("max_folded", max_folded),
        ):
            if value < 1:
                raise ValueError(f"memory {name} must be at least 1, not {value}")
        self.window = min(window, MAX_HISTORY_STEPS)
        self._max_notes = max_notes
        self._max_folded = max_folded
        self._recent: list[StepRecord] = []
        self._folded: list[StepRecord] = []
        self._tally = _Tally()
        self._notes: dict[str, str] = {}
        self._dropped_notes = 0
        self._steps = 0

    @classmethod
    def from_config(cls, config: Config) -> Memory:
        """Start a run's memory from its settings.

        Args:
            config: The resolved configuration for this run.

        Returns:
            Memory: An empty memory with the configured window.
        """
        return cls(config.history_window)

    @property
    def steps_recorded(self) -> int:
        """int: How many steps this run has recorded, folded ones included."""
        return self._steps

    def record(
        self,
        action: Action,
        *,
        observation: Observation | None = None,
        result: StepResult | None = None,
    ) -> StepRecord:
        """Write down one step that has happened.

        Args:
            action: What was done.
            observation: The page the action was planned against. Used to name
                the element the action referred to -- its role and accessible
                name, which still mean something after the page has moved on,
                where its number does not.
            result: What executing it produced, when it was executed. None for
                a step with nothing to report, such as the run's final ``done``.

        Returns:
            StepRecord: The record, as it will appear in the history.
        """
        self._steps += 1
        step = StepRecord(
            number=self._steps,
            action=action.describe(),
            target=_target_of(action, observation),
            outcome=_outcome_of(result),
            url=_moved_to(action, result, observation),
            entered=_clip(action.text) if isinstance(action, Fill) else "",
            kind=action.action,
        )
        self._recent.append(step)
        # A step leaving the window is folded rather than dropped: it moves to
        # the notepad as the same line, and only when the notepad's own list is
        # full is the oldest of *those* counted into the tally.
        while len(self._recent) > self.window:
            self._folded.append(self._recent.pop(0))
        while len(self._folded) > self._max_folded:
            self._tally.add(self._folded.pop(0))
        return step

    def note(self, key: str, value: str) -> None:
        """Hold on to a fact for the rest of the run.

        This is how a value survives however long the run goes on: steps age
        out of the window by count, and notes do not. A second note under the
        same key replaces the first -- a page count that was 3 and is now 4 is
        one fact that changed, not two facts -- and moves it to the end, so
        that if the ceiling is ever reached it is the stalest note that goes.

        Args:
            key: What the fact is, in a word or two, e.g. ``price``.
            value: The fact itself, e.g. ``$12.99``. Kept exactly as given, up
                to :data:`MAX_LINE_CHARS`.

        Raises:
            ValueError: If the key is blank. A note nobody could name is a note
                nobody could ever update.
        """
        key = key.strip()
        if not key:
            raise ValueError("a note needs a key, so that it can be updated later")
        self._notes.pop(key, None)
        self._notes[key] = value.strip()
        while len(self._notes) > self._max_notes:
            oldest = next(iter(self._notes))
            del self._notes[oldest]
            self._dropped_notes += 1

    def history(self) -> tuple[str, ...]:
        """The recent steps, word for word, for the prompt.

        Returns:
            tuple[str, ...]: One line per step in the window, oldest first,
            numbered by their real position in the run.
        """
        return tuple(step.line() for step in self._recent)

    def scratchpad(self) -> str:
        """The running notepad, for the prompt.

        Returns:
            str: Notes first, because a value the agent was told to keep is the
            thing most likely to be needed; then what happened before the
            window. Empty when there is nothing yet, which the planner reads as
            "leave the section out" rather than as an empty heading.
        """
        sections = []
        if self._notes:
            lines = [_clip(f"- {key}: {value}") for key, value in self._notes.items()]
            if self._dropped_notes:
                lines.append(f"- ({self._dropped_notes} older notes dropped to stay within limits)")
            sections.append("Values noted:\n" + "\n".join(lines))

        earlier = []
        if not self._tally.empty:
            earlier.append(self._tally.line())
        earlier.extend(step.line() for step in self._folded)
        if earlier:
            sections.append("Earlier steps:\n" + "\n".join(earlier))

        return "\n\n".join(sections)

    def to_dict(self) -> dict[str, Any]:
        """Render memory's state as plain data, for the run's trace.

        Returns:
            dict[str, Any]: The window, the notes, and the notepad exactly as
            the planner was shown it -- so a trace can answer "what did the
            agent know at this step?" without reconstructing anything.
        """
        return {
            "steps_recorded": self._steps,
            "window": [step.to_dict() for step in self._recent],
            "notes": dict(self._notes),
            "dropped_notes": self._dropped_notes,
            "scratchpad": self.scratchpad(),
        }


def _target_of(action: Action, observation: Observation | None) -> str:
    """Name the element an action referred to, as it was when it was used.

    Args:
        action: The action.
        observation: The page it was planned against.

    Returns:
        str: The element's role and accessible name, e.g. ``button "Search"``.
        Empty for an action that names no element, or when the element cannot
        be found -- which the validator should already have prevented, and
        which is recorded as nothing rather than as a guess.
    """
    element_id = getattr(action, "element_id", None)
    if element_id is None or observation is None:
        return ""
    element = observation.element_by_id(element_id)
    if element is None:
        return ""
    return _clip(f'{element.role} "{element.name}"' if element.name else element.role)


def _outcome_of(result: StepResult | None) -> str:
    """Say in a few words what executing a step produced.

    Args:
        result: The execution result, or None.

    Returns:
        str: ``ok`` for success, the status and its sentence for a failure,
        and empty when nothing was reported.
    """
    if result is None:
        return ""
    if result.status == "ok":
        return "ok"
    return _clip(f"{result.status}: {result.detail}" if result.detail else result.status)


def _moved_to(
    action: Action,
    result: StepResult | None,
    observation: Observation | None,
) -> str:
    """Name where a step took the page, when that is news.

    Args:
        action: The action, for a navigation's own target.
        result: The execution result, carrying the address afterwards.
        observation: The page before, for comparison.

    Returns:
        str: The new address, shortened, or empty when saying it would add
        nothing: the page did not move -- most clicks on a form, every scroll --
        or a navigation arrived exactly where it was sent, which its own line
        already says. A navigation that was *redirected* still says where it
        ended up, because that is the case worth knowing.
    """
    if result is None or not result.url:
        return ""
    if observation is not None and result.url == observation.url:
        return ""
    if isinstance(action, Navigate) and result.url == action.url:
        return ""
    return _short_url(result.url)


def _short_url(url: str) -> str:
    """Shorten an address to the part that tells pages apart.

    Args:
        url: A full address.

    Returns:
        str: Host, path and query without the scheme or a trailing slash,
        cut to :data:`MAX_URL_CHARS`, e.g. ``example.org/search?q=rust``.
    """
    try:
        parts = urlsplit(url)
    except ValueError:
        return _clip(url, MAX_URL_CHARS)
    if not parts.netloc:
        return _clip(url, MAX_URL_CHARS)
    shown = parts.netloc + parts.path.rstrip("/")
    if parts.query:
        shown += f"?{parts.query}"
    return _clip(shown, MAX_URL_CHARS)


def _remember(items: list[str], value: str) -> int:
    """Add a value to a short ordered list, keeping each value once.

    Args:
        items: The list, most recent last. Changed in place.
        value: The value to add. Moved to the end if already present.

    Returns:
        int: How many values were pushed off the front to stay within
        :data:`MAX_TALLY_ITEMS` -- zero or one.
    """
    if value in items:
        items.remove(value)
    items.append(value)
    if len(items) > MAX_TALLY_ITEMS:
        items.pop(0)
        return 1
    return 0


def _listed(values: Iterable[str], dropped: int) -> str:
    """Join values for a tally line, saying how many were left out.

    Args:
        values: The values still named.
        dropped: How many earlier ones are not.

    Returns:
        str: e.g. ``a, b, c (and 2 earlier)``.
    """
    joined = ", ".join(values)
    return f"{joined} (and {dropped} earlier)" if dropped else joined


def _clip(text: str, limit: int = MAX_LINE_CHARS) -> str:
    """Cut text to a length, saying so with an ellipsis.

    Args:
        text: The text.
        limit: The longest it may be.

    Returns:
        str: The text unchanged when it fits, or its first ``limit - 1``
        characters and an ellipsis -- visibly cut, rather than silently.
    """
    return text if len(text) <= limit else text[: limit - 1] + "…"
