"""The verdicts a machine can settle on its own, with no AI involved.

A benchmark task says how it is to be marked (see
:mod:`~web_action_agent.benchmark`). Three of the four ways are simple enough
for plain code: did the run end on a URL containing some text, did the agent's
answer match one of the accepted answers, and was some text on the final page.
This module decides those three. The fourth, the judge, needs a model and lives
elsewhere.

These checks come first in the marking because they are cheap and because they
cannot disagree with themselves: the same run checked twice gets the same
answer, which is not true of any model.

**Checks read the run folder, not the browser.** Every check here is made
against :class:`FinalState`, which is built from a finished run's saved
records. Nothing reopens the website. That keeps a verdict reproducible -- it
can be recomputed next month from the folder alone -- and it means a verdict
always points at evidence somebody can inspect.

**How text is compared.** Agents and websites write the same thing in slightly
different ways: ``3,350`` and ``3350``, ``Raffles`` and ``RAFFLES``, a curly
apostrophe and a straight one, two spaces and one. A check that failed on those
would be marking typography rather than the agent. So text is *normalized*
before comparison, by :func:`normalize`, and the rules are written out there.
They are deliberately few. Anything that could change meaning -- a decimal
point, a percent sign, a minus sign, a currency symbol -- is left alone, and the
task lists every acceptable spelling in its ``accepted`` answers instead.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from urllib.parse import unquote_plus

from .benchmark import AnswerMatches, ElementVisible, UrlContains, Verdict
from .trace_reader import ReplayedRun

#: The verdict methods this module can decide. The fourth, ``judge``, needs a
#: model and is decided by the judge.
MACHINE_METHODS = ("url_contains", "answer_matches", "element_visible")

#: Characters that look alike to a reader but differ to a computer, mapped to
#: one plain form: curly quotes to straight, and the several dashes to a hyphen.
_LOOKALIKES = str.maketrans(
    {
        "‘": "'",  # left single quote
        "’": "'",  # right single quote, also the curly apostrophe
        "“": '"',  # left double quote
        "”": '"',  # right double quote
        "‐": "-",  # hyphen
        "‑": "-",  # non-breaking hyphen
        "‒": "-",  # figure dash
        "–": "-",  # en dash
        "—": "-",  # em dash
    }
)

#: A comma used as a thousands separator: a digit, a comma, then exactly three
#: digits and no fourth. ``3,350`` matches; ``1,2`` and ``3,3500`` do not, so a
#: comma that separates two numbers in a list is kept.
_THOUSANDS_COMMA = re.compile(r"(?<=\d),(?=\d{3}(?!\d))")

#: Punctuation that can be trimmed from the very start and end of a text
#: without changing what it says: a sentence's full stop, or quotes around it.
#: Not ``%``, ``$`` or ``-``, which carry meaning even at the edges.
_EDGE_PUNCTUATION = " .,;:!?'\"()[]"


def normalize(text: str) -> str:
    """Reduce a text to the one form comparisons are made in.

    The rules, in the order they are applied:

    1. Unicode compatibility form (NFKC): a full-width ``３`` becomes ``3``,
       and a non-breaking space becomes a normal space.
    2. Curly quotes become straight ones, and the various dashes become a
       plain hyphen.
    3. Letter case is ignored (``casefold``, which also handles letters such
       as the German ``ß`` that ``lower`` does not).
    4. Thousands separators are removed: ``3,350`` becomes ``3350``.
    5. Any run of whitespace becomes a single space.
    6. Punctuation at the very start and end is trimmed: ``"10."`` becomes
       ``10``.

    Nothing else changes. In particular ``3.9`` stays ``3.9``, ``15%`` stays
    ``15%``, and ``$3,350`` becomes ``$3350``, not ``3350``.

    Args:
        text: Any text: an answer, an accepted answer, a URL fragment.

    Returns:
        str: The normalized text.
    """
    text = unicodedata.normalize("NFKC", text)
    text = text.translate(_LOOKALIKES)
    text = text.casefold()
    text = _THOUSANDS_COMMA.sub("", text)
    text = " ".join(text.split())
    return text.strip(_EDGE_PUNCTUATION)


@dataclass(frozen=True)
class FinalState:
    """What a run ended with: the evidence every check is made against.

    Attributes:
        url: The address of the page the run ended on. Empty if the run never
            recorded a step.
        answer: The agent's final answer, or None if it gave none -- because
            it failed, was interrupted, or the task did not ask for one.
        page_texts: What the agent last saw of the page: the name and visible
            text of each on-screen element, then the page's main text.
        page_is_current: Whether ``page_texts`` show the page at ``url``.
            False when the run's last action moved to a new address after the
            agent last looked, so the final page's content was never recorded.
    """

    url: str
    answer: str | None
    page_texts: tuple[str, ...] = ()
    page_is_current: bool = True


def final_state(run: ReplayedRun) -> FinalState:
    """Work out what a run ended with, from its saved records.

    The agent looks at the page at the start of each step, then acts. So the
    last step's observation is the last page the agent saw, and the last
    step's result says where the browser was *after* that action. For a run
    that finished with ``done`` the two are the same page. For a run that
    stopped for any other reason, the last action may have moved the browser
    on, and that final page was never looked at.

    Args:
        run: A run read back from its folder.

    Returns:
        FinalState: The final URL, answer, and page contents.
    """
    answer = run.outcome.answer if run.outcome is not None else None
    if not run.steps:
        return FinalState(url="", answer=answer, page_is_current=False)

    last = run.steps[-1]
    seen = last.observation
    # The address after the last action, when the action recorded one; the
    # address the agent looked at otherwise (``done`` records no result).
    url_after = last.result.url if last.result is not None and last.result.url else seen.url

    texts: list[str] = []
    for element in seen.elements:
        texts.extend(part for part in (element.name, element.text) if part)
    if seen.text:
        texts.append(seen.text)

    return FinalState(
        url=url_after,
        answer=answer,
        page_texts=tuple(texts),
        page_is_current=url_after == seen.url,
    )


@dataclass(frozen=True)
class CheckResult:
    """The outcome of one check.

    Attributes:
        method: Which check it was, such as ``"url_contains"``.
        passed: Whether the run passed it.
        detail: One sentence saying what was compared with what, so a failed
            check can be understood from the results table alone.
    """

    method: str
    passed: bool
    detail: str


def check_url_contains(verdict: UrlContains, final: FinalState) -> CheckResult:
    """Pass if the final URL contains the task's value.

    Both sides are percent-decoded first, because a browser may show
    ``Integrated%20Programme``, ``Integrated+Programme`` or
    ``Integrated Programme`` for the same address, and the task author should
    not have to guess which. Then both are normalized as described in
    :func:`normalize`, so letter case does not matter.

    Args:
        verdict: The task's ``url_contains`` entry.
        final: What the run ended with.

    Returns:
        CheckResult: Passed if the value is found in the URL.
    """
    if not final.url:
        return CheckResult("url_contains", False, "the run recorded no final URL")
    wanted = normalize(unquote_plus(verdict.value))
    passed = wanted in normalize(unquote_plus(final.url))
    found = "contains" if passed else "does not contain"
    return CheckResult("url_contains", passed, f"final URL {final.url!r} {found} {verdict.value!r}")


def check_answer_matches(verdict: AnswerMatches, final: FinalState) -> CheckResult:
    """Pass if the agent's answer contains one of the accepted answers.

    "Contains" rather than "equals", because an agent asked for a number of
    days may answer "10" or "The entitlement is 10 days", and both are right.
    But it must contain the accepted answer *as a whole*: the characters on
    either side must not be letters or digits. So ``10`` is found in "10 days"
    and in "it is 10.", but not in "100", "2010", or "10.5".

    A known limit: an answer that lists several candidates ("10 or 11 days")
    passes if any of them is accepted. Task instructions ask for one answer,
    and the human spot-check exists to catch what this misses.

    Args:
        verdict: The task's ``answer_matches`` entry.
        final: What the run ended with.

    Returns:
        CheckResult: Passed if any accepted answer is found in the answer.
    """
    if final.answer is None or not final.answer.strip():
        return CheckResult("answer_matches", False, "the run gave no answer")
    answer = normalize(final.answer)
    for accepted in verdict.accepted:
        if _contains_whole(answer, normalize(accepted)):
            return CheckResult(
                "answer_matches", True, f"answer {final.answer!r} matches {accepted!r}"
            )
    return CheckResult(
        "answer_matches",
        False,
        f"answer {final.answer!r} matches none of {list(verdict.accepted)!r}",
    )


def check_element_visible(verdict: ElementVisible, final: FinalState) -> CheckResult:
    """Pass if the text appears on the last page the agent saw.

    Looks in the name and visible text of every on-screen element, and in the
    page's main text. A plain "contains" after :func:`normalize`, so a phrase
    is found inside a longer label.

    Fails, rather than guessing, when the final page was never looked at
    (see :attr:`FinalState.page_is_current`): the text might have been there,
    but nothing in the run folder can show it.

    Args:
        verdict: The task's ``element_visible`` entry.
        final: What the run ended with.

    Returns:
        CheckResult: Passed if the text is found.
    """
    if not final.page_is_current:
        return CheckResult(
            "element_visible",
            False,
            f"the run's final page ({final.url or 'none'}) was never observed, "
            "so what it showed is unknown",
        )
    wanted = normalize(verdict.text)
    passed = any(wanted in normalize(text) for text in final.page_texts)
    found = "is" if passed else "is not"
    return CheckResult("element_visible", passed, f"{verdict.text!r} {found} on the final page")


#: A verdict entry this module can decide.
MachineVerdict = UrlContains | AnswerMatches | ElementVisible


def check(verdict: MachineVerdict, final: FinalState) -> CheckResult:
    """Run whichever check a verdict entry asks for.

    Args:
        verdict: One machine-decidable verdict entry from a task.
        final: What the run ended with.

    Returns:
        CheckResult: The outcome.

    Raises:
        ValueError: For any other kind of verdict, such as ``judge``, which
            this module cannot decide.
    """
    if isinstance(verdict, UrlContains):
        return check_url_contains(verdict, final)
    if isinstance(verdict, AnswerMatches):
        return check_answer_matches(verdict, final)
    if isinstance(verdict, ElementVisible):
        return check_element_visible(verdict, final)
    raise ValueError(f"{getattr(verdict, 'method', verdict)!r} is not a machine-decidable check")


def run_checks(verdicts: tuple[Verdict, ...], final: FinalState) -> tuple[CheckResult, ...]:
    """Run every machine-decidable check in a task's verdict list.

    Entries this module cannot decide (the judge) are skipped, not failed:
    they are marked elsewhere, and the caller combines the two.

    Args:
        verdicts: A task's ``verdicts``, in file order.
        final: What the run ended with.

    Returns:
        tuple[CheckResult, ...]: One result per machine check, in file order.
    """
    return tuple(
        check(verdict, final)
        for verdict in verdicts
        if isinstance(verdict, UrlContains | AnswerMatches | ElementVisible)
    )


def _contains_whole(text: str, part: str) -> bool:
    """Report whether ``part`` appears in ``text`` with nothing glued to it.

    "Glued" means a letter or digit directly before or after, or a decimal
    point or comma leading into more digits -- so ``3`` is not found in
    ``3.5`` or ``1,3``. Spaces and other punctuation are fine boundaries.

    Args:
        text: A normalized text to search in.
        part: A normalized text to search for.

    Returns:
        bool: True if found with a boundary on both sides.
    """
    if not part:
        return False
    # \w covers letters, digits, and the underscore in every script. The
    # second look on each side stops a number matching part of a decimal.
    pattern = rf"(?<!\w)(?<!\d[.,]){re.escape(part)}(?!\w)(?![.,]\d)"
    return re.search(pattern, text) is not None
