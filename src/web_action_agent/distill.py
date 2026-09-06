"""The page's own words, cut to a size a prompt can afford.

The element list tells the planner what it can *do*; it never tells it what the
page *says*. "How much does it cost?", "when does the archive open?", "what did
the confirmation number turn out to be?" — none of those are answerable from a
list of buttons. This module is the other half of reading a page, and it is the
channel that carries a run when no screenshot can.

Two jobs, split on purpose:

**Selecting.** :data:`SCRIPT_PATH` runs inside the browser and returns the
content text with the navigation, footer and scripts left out. It has to run
there because the questions involved -- is this hidden? what did the browser
actually lay out? -- are questions only a live page can answer.

**Fitting.** Everything after that is pure Python: estimate the size, cut it to
the budget if it is over, and say so. Keeping the cut here rather than in the
page means the rule that decides what a prompt can afford is testable without a
browser, and it is the rule most likely to be tuned.

**Why a budget at all.** Page text is unbounded and a prompt is not. A search
results page with two hundred rows would fill a context window on its own and
leave no room for the task, the history, or the element list -- and the agent
would fail on the page it could see *most* of, which is a strange way to lose.

**Why the cut is announced in the text as well as in the record.** The record's
flag is for whoever reads the trace afterwards. The marker left in the text is
for the model reading it now, and it prevents a specific bad answer: a model
shown the first 1,500 tokens of a long page, with no sign that anything was
removed, will say "the page does not mention it" with complete confidence. It
is much better placed to scroll, or to say it is not sure, if it knows the text
in front of it stops early.

**On counting tokens.** No tokenizer is used, and none is a dependency. The
exact count depends on the model, and the models here are configurable by
design, so an exact count for one of them would be a wrong count for the next.
:data:`CHARS_PER_TOKEN` is a deliberate approximation; see it for what that
costs and why it is the right trade here.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from playwright.sync_api import Page

from .config import Config

#: The script injected into the page, beside this module rather than inside it
#: as a string, so it stays readable JavaScript with its own comments.
SCRIPT_PATH = Path(__file__).resolve().parent / "distill_text.js"

#: Characters per token, for estimating size without a tokenizer.
#:
#: English prose runs about four characters to the token across the common
#: tokenizers, and the number is stable enough for a budget whose job is to
#: keep a prompt from overflowing rather than to fill it exactly. It errs in
#: the safe direction on the text that breaks it: code, URLs and long numbers
#: tokenize *worse* than four to one, so a page full of them is estimated low
#: and gets cut sooner than strictly necessary. Sending a slightly short
#: extract costs a little context; overflowing a context window costs the step.
CHARS_PER_TOKEN = 4

#: Appended where the text was cut. Written in words rather than as an ellipsis
#: because it is read by a model, and "..." is something prose contains anyway.
TRUNCATION_MARKER = "\n\n[page text truncated to fit the budget]"

#: How far back from the cut point to look for a word boundary, in characters.
#: Far enough to clear a long word, short enough that the search cannot eat a
#: meaningful amount of the extract on text with no spaces in it at all.
BOUNDARY_SEARCH_CHARS = 200


@dataclass(frozen=True)
class Distillation:
    """The page's content text, with the story of how it was obtained.

    Frozen for the same reason an observation is: this is evidence of what the
    agent was shown, and evidence that can be edited afterwards is worth less.

    Attributes:
        text: The content text, boilerplate removed and whitespace tidied,
            truncated to the budget if it was over. Legitimately empty on a
            page that is all controls and no prose -- a login form, say -- and
            an empty string says exactly that, where a missing value would not.
        truncated: Whether the text was cut to fit. The single most useful
            thing to know when an agent answers wrongly about a long page:
            answering from a complete page and answering from two-thirds of one
            are different failures with different fixes.
        root: Which strategy found the content -- ``main``, ``article``,
            ``largest_block`` or ``body``. The first two are the page telling
            us; the third is a guess; the fourth means no guess was confident
            enough. Recorded because a page that read oddly is much easier to
            diagnose when you know which of those happened.
        estimated_tokens: Size of :attr:`text` in tokens, by the approximation
            in :data:`CHARS_PER_TOKEN`. An estimate, and named as one.
    """

    text: str
    truncated: bool
    root: str
    estimated_tokens: int

    def summary(self) -> str:
        """Render this extract as the single line a step log records.

        Returns:
            str: Size, source strategy, and whether it was cut -- for example
            ``412 chars (~103 tokens) from main``. Characters come first
            because they are the number that is exactly true.
        """
        cut = ", truncated" if self.truncated else ""
        return f"{len(self.text)} chars (~{self.estimated_tokens} tokens) from {self.root}{cut}"

    def to_dict(self) -> dict[str, Any]:
        """Render this extract as plain data, for the run's trace.

        Returns:
            dict[str, Any]: Every field, all of them JSON-safe already.
        """
        return {
            "text": self.text,
            "truncated": self.truncated,
            "root": self.root,
            "estimated_tokens": self.estimated_tokens,
        }


@functools.cache
def _script() -> str:
    """Read the injected script, once per process.

    Returns:
        str: The contents of :data:`SCRIPT_PATH`.

    Raises:
        FileNotFoundError: If the script is missing, which means the package was
            built without its data files rather than that anything went wrong at
            run time.
    """
    return SCRIPT_PATH.read_text(encoding="utf-8")


def estimate_tokens(text: str) -> int:
    """Estimate how many tokens a string will cost.

    Args:
        text: The text to measure.

    Returns:
        int: The estimate, rounded up, so an extract is never reported as
        costing nothing at all. See :data:`CHARS_PER_TOKEN` for what the
        approximation is worth.
    """
    if not text:
        return 0
    return -(-len(text) // CHARS_PER_TOKEN)


def fit_to_budget(text: str, *, token_budget: int) -> tuple[str, bool]:
    """Cut text to a token budget, at a boundary a reader would recognise.

    The cut is placed at the last paragraph break inside the budget where there
    is one nearby, and at the last word break otherwise. Cutting mid-word costs
    nothing measurable in tokens and makes the extract look corrupted, which
    invites a reader -- human or model -- to distrust the whole of it rather
    than just its ending.

    Args:
        text: The full extract.
        token_budget: The ceiling, in estimated tokens. The marker announcing
            the cut is counted against it, so the returned text fits the budget
            including its own announcement.

    Returns:
        tuple[str, bool]: The text to use, and whether it had to be cut.
    """
    if estimate_tokens(text) <= token_budget:
        return text, False

    # Budget the marker first: a cut that overflowed the budget by announcing
    # itself would be its own small joke.
    allowance = token_budget * CHARS_PER_TOKEN - len(TRUNCATION_MARKER)
    if allowance <= 0:
        # A budget too small to hold even the announcement. Nothing useful can
        # be said, so say nothing rather than return a marker and no content.
        return "", True

    kept = text[:allowance]
    kept = _cut_at_boundary(kept)
    return kept + TRUNCATION_MARKER, True


def _cut_at_boundary(kept: str) -> str:
    """Retreat to the nearest sensible break at the end of a cut string.

    Retreating is a tidiness measure, so it is bounded twice over: the break has
    to be within :data:`BOUNDARY_SEARCH_CHARS` of the end, *and* it has to leave
    at least half the text standing. Without the second condition a short
    extract beginning with a heading would be cut back to the heading -- the
    paragraph break after it being, technically, the nearest one to the end.

    Args:
        kept: The text as cut by character count alone.

    Returns:
        str: The same text ending at a paragraph break, or failing that a word
        break, or unchanged where neither qualifies -- which is what happens to
        text with no spaces in it at all, where there is no better place to stop
        than where we stopped.
    """
    floor = max(len(kept) - BOUNDARY_SEARCH_CHARS, len(kept) // 2)
    paragraph = kept.rfind("\n\n")
    if paragraph >= floor:
        return kept[:paragraph].rstrip()
    word = max(kept.rfind(" "), kept.rfind("\n"))
    if word >= floor:
        return kept[:word].rstrip()
    return kept.rstrip()


def distill_text(page: Page, *, token_budget: int) -> Distillation:
    """Read one page's content text and fit it to the budget.

    Args:
        page: The Playwright page, already loaded. Read only -- the script runs
            against the live document and changes nothing in it.
        token_budget: Ceiling for the extract, in estimated tokens.

    Returns:
        Distillation: The text the planner will be shown, and how it was got.
    """
    result = page.evaluate(_script())
    text, truncated = fit_to_budget(result["text"], token_budget=token_budget)
    return Distillation(
        text=text,
        truncated=truncated,
        root=result["root"],
        estimated_tokens=estimate_tokens(text),
    )


def distill_from_config(page: Page, config: Config) -> Distillation:
    """Read one page's content text using the run's configured budget.

    The budget is configuration because the right value depends on the model's
    context window and on how much of it the history and element list are
    already taking, neither of which this module can see.

    Args:
        page: The Playwright page, already loaded.
        config: The resolved run configuration, supplying ``text_token_budget``.

    Returns:
        Distillation: The text the planner will be shown, and how it was got.
    """
    return distill_text(page, token_budget=config.text_token_budget)
