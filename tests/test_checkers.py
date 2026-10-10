"""The three checks a machine can settle, and the text rules behind them.

Each check gets a case it must pass and a case it must fail. A check that only
ever passes looks exactly like a working one, so the failing cases matter as
much as the passing ones.

Most of the rest is about text comparison, because that is where a checker is
easiest to get wrong in both directions. Too strict and it marks punctuation
instead of the agent: ``3,350`` against ``3350``. Too loose and it marks a wrong
answer right: ``10`` found inside ``100``. Both kinds are pinned here.

A last group reads the committed example run folder, to check that the final
URL, answer, and page are taken from the right step.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from web_action_agent.benchmark import AnswerMatches, ElementVisible, Judge, UrlContains
from web_action_agent.checkers import (
    FinalState,
    check,
    check_answer_matches,
    check_element_visible,
    check_url_contains,
    final_state,
    normalize,
    run_checks,
)
from web_action_agent.trace_reader import read_run

pytestmark = pytest.mark.unit

TRACE = Path(__file__).resolve().parent / "fixtures" / "traces" / "search-and-open"


def ended(
    url: str = "https://www.moe.gov.sg/schoolfinder/primary%20school?area=Bishan",
    answer: str | None = "3",
    page_texts: tuple[str, ...] = ("Ai Tong School", "Kuo Chuan Presbyterian Primary School"),
    page_is_current: bool = True,
) -> FinalState:
    """Build the end of a run, with sensible defaults a test can override.

    Args:
        url: The final URL.
        answer: The agent's answer.
        page_texts: What was on the final page.
        page_is_current: Whether the page texts show the final URL's page.

    Returns:
        FinalState: The state to check against.
    """
    return FinalState(
        url=url, answer=answer, page_texts=page_texts, page_is_current=page_is_current
    )


def answer_ok(answer: str | None, *accepted: str) -> bool:
    """Report whether ``answer`` passes an ``answer_matches`` check.

    Args:
        answer: The agent's answer.
        *accepted: The task's accepted answers.

    Returns:
        bool: Whether the check passed.
    """
    verdict = AnswerMatches(method="answer_matches", accepted=accepted)
    return check_answer_matches(verdict, ended(answer=answer)).passed


# --- The normalization rules ---------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "normal"),
    [
        ("Raffles", "raffles"),  # case
        ("  ten   days ", "ten days"),  # spacing
        ("3,350", "3350"),  # thousands separator
        ("1,000,000", "1000000"),
        ("３５%", "35%"),  # full-width digits
        ("28 October", "28 october"),  # non-breaking space
        ("Catholic High School’s", "catholic high school's"),  # curly apostrophe
        ("“quoted”", "quoted"),  # curly quotes, trimmed at the edges
        ("10.", "10"),  # a sentence's full stop
        ("(24%)", "24%"),
    ],
)
def test_normalize_removes_what_does_not_matter(raw: str, normal: str) -> None:
    assert normalize(raw) == normal


@pytest.mark.parametrize(
    ("raw", "normal"),
    [
        ("3.9", "3.9"),  # a decimal point is meaning, not punctuation
        ("15%", "15%"),
        ("$3,350", "$3350"),  # the currency symbol stays
        ("-5", "-5"),  # so does a minus sign
        ("1,2", "1,2"),  # a comma between two numbers is not a separator
        ("3,3500", "3,3500"),
    ],
)
def test_normalize_keeps_what_does_matter(raw: str, normal: str) -> None:
    assert normalize(raw) == normal


# --- answer_matches ------------------------------------------------------------


def test_answer_matches_passes_the_exact_answer() -> None:
    assert answer_ok("10", "10")


def test_answer_matches_fails_a_wrong_answer() -> None:
    assert not answer_ok("11", "10")


@pytest.mark.parametrize(
    "answer",
    [
        "3,350",
        "3350",
        "$3,350",
        "The gross tax payable is $3,350.",
        "S$3,350",
    ],
)
def test_answer_matches_ignores_the_documented_differences(answer: str) -> None:
    assert answer_ok(answer, "3,350")


@pytest.mark.parametrize(
    ("answer", "accepted"),
    [
        ("3,305", "3,350"),  # digits swapped
        ("33500", "3,350"),  # a digit too many
        ("100", "10"),  # the right number inside a bigger one
        ("2010", "10"),
        ("10.5", "10"),  # the right number as part of a decimal
        ("1.3", "3"),  # ...at either end of it
        ("3.9", "39"),  # dropping the decimal point would match this
        ("24%", "15%"),
        ("Raffles Institution", "Catholic High School"),
    ],
)
def test_answer_matches_is_sensitive_to_a_genuine_mismatch(answer: str, accepted: str) -> None:
    assert not answer_ok(answer, accepted)


def test_answer_matches_accepts_the_answer_inside_a_sentence() -> None:
    assert answer_ok("In the 4th year an employee gets 10 days of leave.", "10")


def test_answer_matches_is_case_insensitive() -> None:
    assert answer_ok("ANGLO-CHINESE SCHOOL (INDEPENDENT)", "Anglo-Chinese School (Independent)")


def test_answer_matches_treats_dashes_alike() -> None:
    assert answer_ok("Anglo–Chinese School", "Anglo-Chinese School")


def test_answer_matches_any_of_several_accepted_answers() -> None:
    accepted = ("28 October 2027", "2027-10-28")
    assert answer_ok("2027-10-28", *accepted)
    assert answer_ok("Thursday, 28 October 2027", *accepted)
    assert not answer_ok("8 November 2026", *accepted)


@pytest.mark.parametrize("answer", [None, "", "   "])
def test_answer_matches_fails_when_there_is_no_answer(answer: str | None) -> None:
    assert not answer_ok(answer, "10")


def test_answer_matches_detail_names_both_sides() -> None:
    verdict = AnswerMatches(method="answer_matches", accepted=("10",))
    result = check_answer_matches(verdict, ended(answer="11"))
    assert "'11'" in result.detail and "'10'" in result.detail


# --- url_contains --------------------------------------------------------------


def test_url_contains_passes_when_present() -> None:
    verdict = UrlContains(method="url_contains", value="area=Bishan")
    assert check_url_contains(verdict, ended()).passed


def test_url_contains_fails_when_absent() -> None:
    verdict = UrlContains(method="url_contains", value="area=Tampines")
    assert not check_url_contains(verdict, ended()).passed


@pytest.mark.parametrize(
    "url",
    [
        "https://x.sg/find?special_programme=Integrated%20Programme",
        "https://x.sg/find?special_programme=Integrated+Programme",
        "https://x.sg/find?special_programme=Integrated Programme",
        "https://x.sg/find?special_programme=integrated%20programme",
    ],
)
def test_url_contains_ignores_how_spaces_and_case_are_written(url: str) -> None:
    verdict = UrlContains(method="url_contains", value="Integrated%20Programme")
    assert check_url_contains(verdict, ended(url=url)).passed


def test_url_contains_fails_with_no_url() -> None:
    verdict = UrlContains(method="url_contains", value="area=Bishan")
    result = check_url_contains(verdict, ended(url=""))
    assert not result.passed
    assert "no final URL" in result.detail


# --- element_visible -----------------------------------------------------------


def test_element_visible_passes_when_on_the_page() -> None:
    verdict = ElementVisible(method="element_visible", text="Kuo Chuan Presbyterian Primary School")
    assert check_element_visible(verdict, ended()).passed


def test_element_visible_fails_when_not_on_the_page() -> None:
    verdict = ElementVisible(method="element_visible", text="Raffles Institution")
    assert not check_element_visible(verdict, ended()).passed


def test_element_visible_finds_a_phrase_inside_longer_text() -> None:
    verdict = ElementVisible(method="element_visible", text="KUO CHUAN  presbyterian")
    assert check_element_visible(verdict, ended()).passed


def test_element_visible_fails_when_the_final_page_was_never_seen() -> None:
    # The text is in what the agent last saw, but the browser moved on after
    # that, so the final page's content is unknown. Guessing would be wrong.
    verdict = ElementVisible(method="element_visible", text="Ai Tong School")
    result = check_element_visible(verdict, ended(page_is_current=False))
    assert not result.passed
    assert "never observed" in result.detail


# --- Running a task's checks ---------------------------------------------------


def test_run_checks_skips_the_judge_and_keeps_file_order() -> None:
    verdicts = (
        AnswerMatches(method="answer_matches", accepted=("3",)),
        Judge(method="judge", expected_outcome="Three schools are listed."),
        UrlContains(method="url_contains", value="area=Bishan"),
    )
    results = run_checks(verdicts, ended())
    assert [r.method for r in results] == ["answer_matches", "url_contains"]
    assert all(r.passed for r in results)


def test_check_refuses_the_judge() -> None:
    with pytest.raises(ValueError, match="judge"):
        check(Judge(method="judge", expected_outcome="x"), ended())  # type: ignore[arg-type]


# --- Reading the end of a real run ---------------------------------------------


def test_final_state_of_a_finished_run() -> None:
    run = read_run(TRACE)
    final = final_state(run)
    # The run ended with "done" on the form page it had just opened.
    assert final.url == "http://northgate.example/form-step-1.html"
    assert final.answer == "Membership application"
    assert final.page_is_current
    assert any("Membership is free" in text for text in final.page_texts)


def test_final_state_uses_the_address_after_the_last_action() -> None:
    # Cut the run off after step 3, a click that opened a new page. The agent
    # never looked at that page, so the URL is known but the content is not.
    run = read_run(TRACE)
    cut = replace(run, steps=run.steps[:3], outcome=None)
    final = final_state(cut)
    assert final.url == "http://northgate.example/form-step-1.html"
    assert final.answer is None
    assert not final.page_is_current


def test_final_state_of_a_run_with_no_steps() -> None:
    run = replace(read_run(TRACE), steps=(), outcome=None)
    final = final_state(run)
    assert final.url == ""
    assert not final.page_is_current
