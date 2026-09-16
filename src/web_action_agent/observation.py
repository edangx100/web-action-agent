"""Everything the agent knows about the page at one moment, as one value.

An observation is taken once before the first step and once after every action
the agent executes. It is what the planner is shown, what the validator checks
a proposed action against, and what the run's trace records — so it is the one
place where "what the page was like" is defined, and every part of the agent
downstream of the browser reads it instead of asking the browser again.

**Three channels, two of them always there.**

1. The *interactive elements* — the numbered list of what can be clicked,
   filled or pressed in the current viewport. This is the primary channel, and
   the one the planner acts through: it names an element by its number.
2. The *page text* — the article, the search results, the prices. Elements
   alone say what the agent can do and never what the page says, so a task
   like "how much does it cost?" would be unanswerable without this. It is
   always present, and truncated to a budget, because page text is unbounded
   and a prompt is not.
3. The *screenshot* — optional. An image is worth roughly as many tokens as
   the two text channels together, but carrying one routes the step to a
   slower and pricier vision model, so it is sent only when the text channels
   look unreliable — and never at all when the configured model cannot see
   images. That last case is why the first two channels have to be enough on
   their own: nothing here may hard-depend on vision.

Alongside them, the URL and the page title — cheap, and the two things that
tell a reader of the trace *which* page all of the above describes.

**A snapshot, not a window.** Nothing in this record is live. It describes the
page as it was laid out at the instant it was taken, and the numbering inside
it is valid only within it: scroll, and the same button is a different number
in the next observation. That is why the validator checks a proposed action
against the observation the planner was actually shown rather than against the
browser — the two can already disagree, and the observation is the one the
planner had reason to believe.

**Why it serializes.** A finished run has to be auditable without being re-run,
and re-running is exactly what cannot be trusted to reproduce it: the site will
have moved on. So each observation is written to the run's trace as plain data
and can be read back into the same value. :meth:`Observation.to_dict` and
:meth:`Observation.from_dict` are that pair, and the round trip is what makes a
trace worth keeping.

**What is deliberately not here.** How an observation is phrased *for the
planner* is a prompt decision, and prompts live with the planner. This module
holds the facts; assembling them into the text a model is shown is somebody
else's job.

**Taking one, though, is here.** :func:`capture` is the one place the four
perception pieces are put together -- the element extractor, the text
distiller, the screenshot gate and the annotator -- because every caller that
needs an observation needs the same four in the same order, and a second
caller assembling them slightly differently is how two parts of a run end up
describing the same page two ways. It lives in this module rather than with the
extractor for a duller reason: the extractor is what this module imports, and
the reverse import would be a cycle.
"""

from __future__ import annotations

from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from playwright.sync_api import Error as BrowserFailure
from playwright.sync_api import Page

from .annotate import annotate
from .config import Config
from .distill import distill_from_config
from .observer import Element, extract_elements
from .screenshot_gate import GateDecision, ScreenshotGate

#: What :meth:`Observation.summary` prints where there is no screenshot. Spelt
#: out rather than left blank, because a blank means "nothing was recorded"
#: and this means "a decision was made not to take one".
NO_SCREENSHOT = "no screenshot"


@dataclass(frozen=True)
class Observation:
    """One complete look at the page: what it says, what it offers, where it is.

    Frozen on purpose. An observation is evidence — the state a particular
    decision was made against — and evidence that can be edited after the fact
    is worth much less. If the page changed, the answer is to observe it again,
    which produces a second record rather than quietly rewriting the first.

    Attributes:
        url: The page's address at the moment of the observation, after any
            redirect. Not necessarily the URL the agent asked to navigate to,
            which is the point of recording it.
        title: The page title, as the browser reports it. May be empty; some
            pages have none, and inventing one would misreport the page.
        elements: The interactive elements in the viewport, numbered from 1 in
            document order. Empty is a legitimate value, and a meaningful one —
            a page offering nothing to act on is one of the signals that a
            screenshot might be worth its cost.
        text: The distilled main-content text: the page's own words, with the
            navigation, footer and scripts stripped out. Also legitimately
            empty, on a page that is all controls and no prose.
        text_truncated: Whether ``text`` was cut short to fit its budget. Kept
            because the difference matters to whoever reads the trace: an agent
            that answered wrongly from a truncated page failed differently from
            one that answered wrongly from a complete one.
        screenshot: Where the annotated screenshot for this step is, when one
            forms part of this observation, and ``None`` when the cost gate
            kept it out or no vision-capable model is configured. ``None`` here
            does not mean no image exists — the run's trace keeps its own
            screenshot of every step for a human to look at afterwards. It
            means this observation does not carry one.
    """

    url: str
    title: str
    elements: tuple[Element, ...]
    text: str
    text_truncated: bool = False
    screenshot: Path | None = None

    @property
    def has_screenshot(self) -> bool:
        """bool: Whether this observation carries an image channel.

        Reads better at the call site than a ``is not None`` check, and gives
        the cost gate and the planner one shared way to ask.
        """
        return self.screenshot is not None

    def element_by_id(self, element_id: int) -> Element | None:
        """Look up the element the planner referred to by number.

        This is the lookup behind the validator's "does the element the planner
        named actually exist?" check, and it is written here so that every
        caller resolves a number the same way. A number the planner invented,
        or one carried over from a previous observation whose numbering no
        longer applies, simply is not found — which is a correctable mistake to
        hand back to the planner, not an error.

        Args:
            element_id: The number from this observation's element list.

        Returns:
            Element | None: The matching element, or ``None`` when this
            observation has no element with that number.
        """
        for element in self.elements:
            if element.id == element_id:
                return element
        return None

    def summary(self) -> str:
        """Render this observation as the single line the step log records.

        The action log keeps one record per step, and this is its "what the
        agent was looking at" field: enough to follow a run at a glance, and
        short enough that a hundred steps stay readable. The full observation
        is written alongside it, so nothing is lost by keeping this brief.

        Returns:
            str: The URL and title, then the size of each channel — for
            example ``http://localhost/search.html "Search" - 8 elements,
            412 chars of text, no screenshot``. Text is measured in characters
            rather than tokens because characters need no model to count and
            no dependency to explain.
        """
        text_size = f"{len(self.text)} chars of text"
        if self.text_truncated:
            text_size += " (truncated)"
        image = str(self.screenshot) if self.screenshot else NO_SCREENSHOT
        channels = f"{len(self.elements)} elements, {text_size}, {image}"
        return f'{self.url} "{self.title}" - {channels}'

    def to_dict(self) -> dict[str, Any]:
        """Render this observation as plain data, for the run's trace.

        Everything written here is JSON-safe: the element list becomes a list
        of plain records, and the screenshot becomes its path as a string
        rather than the image itself. That keeps traces small and readable as
        text, with the image sitting beside them in the run folder where a
        person can open it.

        Returns:
            dict[str, Any]: The whole record, under the keys
            :meth:`from_dict` reads back.
        """
        return {
            "url": self.url,
            "title": self.title,
            "elements": [element.to_dict() for element in self.elements],
            "text": self.text,
            "text_truncated": self.text_truncated,
            "screenshot": str(self.screenshot) if self.screenshot else None,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Observation:
        """Rebuild an observation from the plain data of a trace.

        The four channels that are always written are required, so a truncated
        or corrupt trace fails loudly here instead of yielding an observation
        that quietly claims the page was blank. The two that describe *how* the
        page was captured fall back to their defaults, which are the honest
        reading of an older trace that did not record them: nothing was cut,
        and no screenshot was carried.

        Args:
            data: One record, as written by :meth:`to_dict`.

        Returns:
            Observation: The same value the trace was written from.

        Raises:
            KeyError: If ``url``, ``title``, ``elements`` or ``text`` is
                missing.
        """
        # A missing key, an explicit null and an empty string all mean the same
        # thing here -- this observation carried no image -- so all three are
        # read the same way rather than one of them becoming Path("").
        screenshot = data.get("screenshot")
        return cls(
            url=data["url"],
            title=data["title"],
            elements=tuple(Element.from_dict(record) for record in data["elements"]),
            text=data["text"],
            text_truncated=data.get("text_truncated", False),
            screenshot=Path(screenshot) if screenshot else None,
        )


def capture(
    page: Page,
    *,
    config: Config,
    gate: ScreenshotGate | None = None,
    screenshot_dir: Path | None = None,
    label: str = "step",
) -> tuple[Observation, GateDecision | None]:
    """Look at a page and produce one observation of it.

    The order is not arbitrary. Elements come first because the screenshot gate
    decides from how many there are, and the annotated screenshot needs them to
    know what to number. The text extract is independent and always taken.

    An image is produced only when all three of these hold: a gate was given, it
    says yes for this page, and somewhere was given to put the file. Any one of
    them missing means a text-only observation -- which is a complete
    observation, not a degraded one. Nothing in this agent may depend on there
    being a picture.

    Args:
        page: The page to look at. The *active* page: after a click opened a
            tab, that is the tab, which is why callers ask the executor for it
            rather than remembering one.
        config: The run's settings, for the text budget and the viewport.
        gate: The run's screenshot gate, or None to take no screenshot at all.
        screenshot_dir: Where to write the annotated image, or None for the
            same. Created if it does not exist.
        label: Filename stem for this step's image, e.g. ``step-04``.

    Returns:
        tuple[Observation, GateDecision | None]: The observation, and the gate's
        decision with its grounds -- None when no gate was given. The decision
        is returned rather than kept because it belongs in the run's trace: "no
        screenshot" is a choice, and a reader should be able to see which rule
        made it.

    Raises:
        playwright.sync_api.Error: If the page cannot be read at all -- it has
            closed, or it navigated away mid-extraction. The controller treats
            that as a dead end rather than retrying it here, because a page
            that cannot be looked at cannot be acted on either.
    """
    elements = extract_elements(page)
    distilled = distill_from_config(page, config)
    decision = gate.decide(element_count=len(elements)) if gate else None

    screenshot: Path | None = None
    if decision is not None and decision.send_screenshot and screenshot_dir is not None:
        screenshot = _annotated_screenshot(page, elements, screenshot_dir, label)

    return (
        Observation(
            url=page.url,
            title=page.title(),
            elements=elements,
            text=distilled.text,
            text_truncated=distilled.truncated,
            screenshot=screenshot,
        ),
        decision,
    )


def _annotated_screenshot(
    page: Page,
    elements: tuple[Element, ...],
    directory: Path,
    label: str,
) -> Path | None:
    """Photograph the page and draw the element numbers onto the copy.

    Args:
        page: The page to photograph.
        elements: The elements to number, from this same look at the page.
        directory: Where the annotated image goes.
        label: Filename stem.

    Returns:
        Path | None: The annotated image, or None if the browser could not take
        the photograph. None rather than an exception on purpose: the image is
        the optional channel, and a run must not end because a screenshot
        failed when the two text channels are in hand.
    """
    directory.mkdir(parents=True, exist_ok=True)
    plain = directory / f"{label}-plain.png"
    marked = directory / f"{label}.png"
    try:
        page.screenshot(path=plain)
        annotate(plain, elements, marked)
    except (OSError, BrowserFailure):
        return None
    finally:
        # The unmarked photograph is scaffolding: the annotator needs a file to
        # read, and the numbered copy is the one anybody wants afterwards.
        with suppress(OSError):
            plain.unlink()
    return marked
