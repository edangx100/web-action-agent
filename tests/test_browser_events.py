"""What real websites do that nobody asked for, and the run surviving it.

Three things happen constantly on real sites and none of them is an action:
a link opens a second tab, the page opens one of the browser's own dialogs, a
click offers a file. Each one ends a run that is not ready for it, and each
ends it in a way that looks like something else:

* The unfollowed tab looks like an agent whose clicks stopped working. The page
  it is still staring at never changes again, so every later step reads as a
  wasted move, and the run ends at the step cap with no idea why.
* The unanswered dialog does not fail at all -- it *hangs*, because a page with
  a dialog open runs no code and answers no automation, until something times
  out and reports a reason that has nothing to do with what happened.
* The download is the quiet one: a click that produced no new page, no error,
  and no explanation.

So the tests here are less about the handling than about the *record*. Every
one of them asserts twice: that the run is still alive and doing the right
thing, and that what happened is written down where the trace will find it. A
silently-handled surprise is only half handled, because the next person to read
the run cannot tell it from a page that did nothing.
"""

import pytest
from browser_steps import act, element_id_named, observe

from web_action_agent.browser import (
    BROWSER_EVENTS,
    DIALOG_DISMISSED,
    DOWNLOAD_IGNORED,
    PAGE_CLOSED,
    POPUP_ADOPTED,
    BrowserEvent,
    TaskSession,
)
from web_action_agent.executor import BROWSER_ERROR
from web_action_agent.observation import Observation


def kinds(result):
    """List the kinds of event a step reported.

    Args:
        result: One execution result.

    Returns:
        list[str]: One entry per event, in order.
    """
    return [event.kind for event in result.events]


def one_event(result, kind):
    """Find the single event of a given kind on a step.

    Args:
        result: One execution result.
        kind: The kind to look for.

    Returns:
        BrowserEvent: The matching event.

    Raises:
        AssertionError: If there is not exactly one, which would make any
            assertion about "the" event meaningless.
    """
    matches = [event for event in result.events if event.kind == kind]
    assert len(matches) == 1, f"expected one {kind}, got {kinds(result)}"
    return matches[0]


def click_named(target, page, name, **kwargs):
    """Click the element with a given accessible name.

    Args:
        target: The task session to act through.
        page: The page to look at, which may not be the session's active one.
        name: The element's accessible name.
        **kwargs: Passed through to the executor.

    Returns:
        ExecutionResult: What the browser did.
    """
    observation = observe(page)
    element_id = element_id_named(observation, name)
    return act(
        target,
        {"action": "click", "element_id": element_id},
        observation=observation,
        **kwargs,
    )


# --- A link that opens a second tab -----------------------------------------


@pytest.mark.fixture
def test_a_link_that_opens_a_tab_moves_the_agent_to_it(task_session, page, fixture_site):
    """The site moved the conversation to a new tab, so the agent goes there."""
    page.goto(fixture_site.url_for("popup.html"))

    result = click_named(task_session, page, "Reading room refurbishment")

    assert result.ok
    assert task_session.page is not page, "the new tab should be the active page"
    assert task_session.page.url.endswith("article.html")


@pytest.mark.fixture
def test_a_tab_opened_by_a_script_is_adopted_too(task_session, page, fixture_site):
    """Markup and script open a tab by different routes; both have to be followed."""
    page.goto(fixture_site.url_for("popup.html"))

    result = click_named(task_session, page, "Open the archive ledger")

    assert result.ok
    assert task_session.page.url.endswith("ledger.html")


@pytest.mark.fixture
def test_the_adoption_is_written_into_the_step(task_session, page, fixture_site):
    """Without the record, the next observation describes a page from nowhere."""
    page.goto(fixture_site.url_for("popup.html"))

    result = click_named(task_session, page, "Reading room refurbishment")

    event = one_event(result, POPUP_ADOPTED)
    assert "new tab" in event.detail
    assert POPUP_ADOPTED in result.summary()


@pytest.mark.fixture
def test_the_next_action_lands_on_the_adopted_tab(task_session, page, fixture_site):
    """Adoption is worth nothing if the step after it goes to the old page."""
    page.goto(fixture_site.url_for("popup.html"))
    click_named(task_session, page, "Open the archive ledger")

    assert act(task_session, {"action": "scroll", "direction": "down"}).ok

    assert task_session.page.evaluate("window.scrollY") > 0, "the new tab should have scrolled"
    assert page.evaluate("window.scrollY") == 0, "the page left behind should not have"


@pytest.mark.fixture
def test_an_ordinary_link_adopts_nothing(task_session, page, fixture_site):
    """Most links open in the tab you are already on, and that is not an event."""
    page.goto(fixture_site.url_for("popup.html"))

    result = click_named(task_session, page, "Archive ledger")

    assert result.ok
    assert result.events == ()
    assert task_session.page is page
    assert page.url.endswith("ledger.html")


@pytest.mark.fixture
def test_closing_the_adopted_tab_falls_back_to_the_one_underneath(task_session, page, fixture_site):
    """A site that closes its own tab must not leave the run with no page."""
    page.goto(fixture_site.url_for("popup.html"))
    click_named(task_session, page, "Reading room refurbishment")
    adopted = task_session.page

    adopted.close()

    assert task_session.page is page
    result = act(task_session, {"action": "navigate", "url": fixture_site.url_for("search.html")})
    assert result.ok, "the run should carry on from the page underneath"
    assert one_event(result, PAGE_CLOSED).detail


@pytest.mark.fixture
def test_a_tab_the_agent_did_not_start_with_is_watched_too(task_session, page, fixture_site):
    """The classic version of this bug: handlers on the first page only.

    Everything works, right up to the first dialog raised in a tab that opened
    during the run -- which is exactly where a site is most likely to raise
    one, and exactly the run that hangs.
    """
    page.goto(fixture_site.url_for("popup.html"))
    click_named(task_session, page, "Cancel a reservation")
    adopted = task_session.page
    assert adopted.url.endswith("confirm.html")

    result = click_named(task_session, adopted, "Cancel this reservation")

    assert result.ok
    assert "Cancel reservation 4471?" in one_event(result, DIALOG_DISMISSED).detail


@pytest.mark.fixture
def test_losing_the_last_tab_is_reported_rather_than_raised(task_session, page, fixture_site):
    """Nothing left to act on is still a step with a result, not a crash."""
    page.goto(fixture_site.url_for("popup.html"))

    page.close()

    # No observation: there is no page to take one of, which is the situation
    # the executor has to survive.
    result = act(
        task_session,
        {"action": "navigate", "url": fixture_site.url_for("index.html")},
        observation=Observation(url="", title="", elements=(), text=""),
    )
    assert result.status == BROWSER_ERROR, result.summary()
    assert "no page left" in one_event(result, PAGE_CLOSED).detail


# --- The browser's own dialogs ----------------------------------------------


@pytest.mark.fixture
def test_a_confirm_box_is_answered_and_the_step_completes(task_session, page, fixture_site):
    """A dialog left open does not fail the step -- it hangs it. So it is answered."""
    page.goto(fixture_site.url_for("confirm.html"))

    result = click_named(task_session, page, "Cancel this reservation")

    assert result.ok
    assert page.evaluate("document.body.dataset.dialog") == "kept"


@pytest.mark.fixture
def test_a_confirm_box_is_declined_rather_than_agreed_to(task_session, page, fixture_site):
    """Nobody read the question, so the safe answer is no.

    The page's own status line is the evidence: it says the reservation is
    still held, which is what Cancel does. Accepting would have cancelled a
    reservation on the strength of a click the agent made for other reasons.
    """
    page.goto(fixture_site.url_for("confirm.html"))

    click_named(task_session, page, "Cancel this reservation")

    assert "still held" in page.text_content("#dialog-status")


@pytest.mark.fixture
def test_an_alert_is_dismissed_too(task_session, page, fixture_site):
    """A one-button notice blocks the page just as thoroughly as a question."""
    page.goto(fixture_site.url_for("confirm.html"))

    result = click_named(task_session, page, "Opening hours")

    assert result.ok
    assert page.evaluate("document.body.dataset.dialog") == "notified"
    assert "alert" in one_event(result, DIALOG_DISMISSED).detail


@pytest.mark.fixture
def test_the_dialog_record_keeps_what_the_page_asked(task_session, page, fixture_site):
    """A run blocked by a dialog it needed to accept should say so, in words."""
    page.goto(fixture_site.url_for("confirm.html"))

    result = click_named(task_session, page, "Cancel this reservation")

    event = one_event(result, DIALOG_DISMISSED)
    assert "confirm" in event.detail
    assert "Cancel reservation 4471?" in event.detail


# --- A click that offers a file ---------------------------------------------


@pytest.mark.fixture
def test_a_download_is_declined_and_written_down(task_session, page, fixture_site):
    """Files are out of scope, but a click that produced no page needs explaining."""
    page.goto(fixture_site.url_for("download.html"))

    result = click_named(task_session, page, "Download ledger-extract.csv")

    assert result.ok
    event = one_event(result, DOWNLOAD_IGNORED)
    assert "ledger-extract.csv" in event.detail
    assert page.url.endswith("download.html"), "a download does not move the page"


@pytest.mark.fixture
def test_the_page_stays_usable_after_offering_a_file(task_session, page, fixture_site):
    """The click reached the page; only the file was refused."""
    page.goto(fixture_site.url_for("download.html"))

    click_named(task_session, page, "Download ledger-extract.csv")

    assert page.text_content("#download-status") == "Export offered."
    assert act(task_session, {"action": "scroll", "direction": "down"}).ok


# --- How events reach the trace ---------------------------------------------


@pytest.mark.fixture
def test_a_quiet_step_reports_nothing(task_session, page, fixture_site):
    """Most steps have no events, and must not be given any."""
    page.goto(fixture_site.url_for("search.html"))
    observation = observe(page)

    result = act(
        task_session,
        {
            "action": "fill",
            "element_id": element_id_named(observation, "Search terms"),
            "text": "x",
        },
        observation=observation,
    )

    assert result.events == ()
    assert "|" not in result.summary()


@pytest.mark.fixture
def test_an_event_is_reported_once_and_not_again(task_session, page, fixture_site):
    """Each step carries its own events; none inherits the last step's."""
    page.goto(fixture_site.url_for("popup.html"))
    first = click_named(task_session, page, "Reading room refurbishment")

    second = act(task_session, {"action": "scroll", "direction": "down"})

    assert kinds(first) == [POPUP_ADOPTED]
    assert second.events == ()


@pytest.mark.fixture
def test_something_that_happens_between_steps_arrives_with_the_next_one(
    task_session, page, fixture_site
):
    """A tab can open while the agent is looking at the page, not acting on it.

    The click here is made directly, behind the executor's back, to stand in
    for a page that opens a tab on a timer. The event still has to reach the
    trace, and the next step is where a reader would look for it.
    """
    page.goto(fixture_site.url_for("popup.html"))
    page.click("#popup-link")
    # Standing in for the time an observation takes. The executor waits for a
    # tab it knows a click will open; this one is opened behind its back, so
    # the tab is reported whenever the browser gets round to saying so.
    page.wait_for_timeout(250)

    result = act(task_session, {"action": "scroll", "direction": "down"})

    assert kinds(result) == [POPUP_ADOPTED]


@pytest.mark.fixture
def test_the_events_are_written_into_the_trace(task_session, page, fixture_site):
    """A trace is read after the run, so an event has to survive being written down."""
    page.goto(fixture_site.url_for("popup.html"))

    result = click_named(task_session, page, "Reading room refurbishment")

    record = result.to_dict()["events"]
    assert [event["kind"] for event in record] == [POPUP_ADOPTED]
    assert record[0]["url"].endswith("article.html")


# --- The event vocabulary ---------------------------------------------------


@pytest.mark.unit
def test_there_are_four_event_kinds_and_they_are_distinct():
    """Events are counted in run summaries, so the set may not drift quietly."""
    assert len(BROWSER_EVENTS) == len(set(BROWSER_EVENTS)) == 4
    assert set(BROWSER_EVENTS) == {POPUP_ADOPTED, PAGE_CLOSED, DIALOG_DISMISSED, DOWNLOAD_IGNORED}


@pytest.mark.unit
def test_an_event_reads_as_one_fragment_of_a_log_line():
    """A step log stays one line per step, events included."""
    event = BrowserEvent(kind=POPUP_ADOPTED, detail="a new tab opened", url="http://x/a.html")

    assert event.summary() == "popup_adopted: a new tab opened"
    assert event.to_dict() == {
        "kind": POPUP_ADOPTED,
        "detail": "a new tab opened",
        "url": "http://x/a.html",
    }


@pytest.mark.unit
def test_a_session_with_nothing_to_watch_still_answers_for_its_page():
    """A script or a test may hand over a bare page; it just gets no watching."""
    stand_in = object()

    session = TaskSession(stand_in)

    assert session.page is stand_in
    assert session.drain_events() == ()
