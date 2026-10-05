"""The three ceilings, and the difference between failing and hanging.

A benchmark that hangs is worse than one that fails. A failure is a row in a
results table with a reason in it; a hang is a process nobody notices for an
hour, holding a browser open, with no verdict and nothing to publish. So the
property under test throughout this module is not "the timeout is thirty
seconds" -- it is **every expiry ends in a stated outcome**, and stated in the
right words.

The words matter because the three ceilings mean three different things. "The
site was too slow", "the model endpoint was too slow" and "the whole task took
too long" send whoever reads the results in three different directions, and one
category called *timeout* would send them nowhere.

Two testing decisions matter more than the rest here:

* **The run's clock is injected.** Otherwise a test of a five-minute ceiling
  takes five minutes, so it gets tested at one second and hoped about at five.
* **The hanging page hangs honestly.** Its button is animated rather than
  broken: automation refuses to click a moving target, because the click would
  land somewhere unintended. That is a carousel, a ticker, a live counter --
  the ordinary reason a real click waits forever.
"""

import time

import pytest
from browser_steps import act, element_id_named, observe
from openai import APITimeoutError

from web_action_agent.actions import parse_action
from web_action_agent.config import Config
from web_action_agent.executor import OK, TIMEOUT, Executor
from web_action_agent.model_client import (
    ModelClient,
    ModelClientError,
    ModelTimeoutError,
    load_profile,
)
from web_action_agent.timeouts import (
    ACTION_TIMEOUT,
    LLM_TIMEOUT,
    RUN_TIMEOUT,
    TIMEOUT_CATEGORIES,
    RunBudget,
    timed_out,
)

RUN_SECONDS = 300.0
"""float: The configured wall-clock ceiling for one run, in seconds."""


class FakeClock:
    """A clock a test moves by hand.

    Reading the time never advances it: a budget asked twice in the same test
    step must give the same answer, or every assertion below would be racing
    the machine it runs on.

    Attributes:
        now: The current reading, in seconds.
    """

    def __init__(self, now=0.0):
        """Args: now: The starting reading."""
        self.now = now

    def __call__(self):
        """Returns: float: The current reading, unchanged by being read."""
        return self.now

    def advance(self, seconds):
        """Move the clock forward.

        Args:
            seconds: How far to move it.
        """
        self.now += seconds


def a_budget(seconds=RUN_SECONDS):
    """Build a run budget on a clock the test controls.

    Args:
        seconds: The ceiling.

    Returns:
        tuple[RunBudget, FakeClock]: The budget and the clock to move.
    """
    clock = FakeClock()
    return RunBudget(seconds, clock=clock), clock


# --- What an expiry produces ------------------------------------------------


@pytest.mark.unit
def test_there_are_three_ceilings_and_they_are_distinct():
    """Three scopes, three names. Results are counted by these."""
    assert len(TIMEOUT_CATEGORIES) == len(set(TIMEOUT_CATEGORIES)) == 3
    assert set(TIMEOUT_CATEGORIES) == {ACTION_TIMEOUT, LLM_TIMEOUT, RUN_TIMEOUT}


@pytest.mark.unit
@pytest.mark.parametrize("category", TIMEOUT_CATEGORIES)
def test_every_ceiling_ends_the_run_with_a_verdict(category):
    """Not an exception, not silence: the same terminal shape a success has."""
    ended = timed_out(category, "something took too long")

    assert ended.action == "done"
    assert ended.status == "failed"
    assert ended.reason.startswith(f"{category}: ")


@pytest.mark.unit
def test_the_reason_leads_with_the_category():
    """So that counting failures by cause is a prefix match, not a text search."""
    ended = timed_out(ACTION_TIMEOUT, "click 4 took longer than 30.0s")

    assert ended.reason == "action_timeout: click 4 took longer than 30.0s"
    assert "action_timeout" in ended.describe()


@pytest.mark.unit
def test_a_misspelt_category_is_refused():
    """A category nothing counts would be a failure that vanishes from the table."""
    with pytest.raises(ValueError, match="not a timeout category"):
        timed_out("timeout", "something")


# --- The run's wall clock ---------------------------------------------------


@pytest.mark.unit
def test_a_fresh_run_has_its_whole_budget():
    """Nothing has happened yet, so nothing has been spent."""
    budget, _ = a_budget()

    assert budget.remaining_s == RUN_SECONDS
    assert not budget.expired
    assert budget.check() is None


@pytest.mark.unit
def test_a_run_that_uses_its_time_is_told_to_stop():
    """The check a loop makes at the top of every iteration."""
    budget, clock = a_budget()
    clock.advance(RUN_SECONDS + 1)

    ended = budget.check()

    assert budget.expired
    assert ended is not None
    assert ended.reason.startswith(f"{RUN_TIMEOUT}: ")


@pytest.mark.unit
def test_a_run_still_inside_its_budget_carries_on():
    """The ceiling has to be reachable without being touched by ordinary runs."""
    budget, clock = a_budget()
    clock.advance(RUN_SECONDS - 0.5)

    assert budget.check() is None


@pytest.mark.unit
def test_an_overrun_reads_as_nothing_left_rather_than_negative_time():
    """Nobody should have to remember that "remaining" can be below zero."""
    budget, clock = a_budget()
    clock.advance(RUN_SECONDS * 3)

    assert budget.remaining_s == 0.0
    assert budget.elapsed_s == pytest.approx(RUN_SECONDS * 3)


@pytest.mark.unit
def test_a_long_call_is_cut_down_to_what_the_run_has_left():
    """A 90-second call started with 10 seconds left would overshoot by a minute."""
    budget, clock = a_budget()
    clock.advance(RUN_SECONDS - 10)

    assert budget.clamp(90.0) == pytest.approx(10.0)


@pytest.mark.unit
def test_a_short_call_is_left_alone():
    """Clamping is a ceiling on the ceiling, not a replacement for it."""
    budget, _ = a_budget()

    assert budget.clamp(90.0) == 90.0


@pytest.mark.unit
def test_a_spent_run_offers_no_time_at_all():
    """Zero is the signal not to start the next thing."""
    budget, clock = a_budget()
    clock.advance(RUN_SECONDS)

    assert budget.clamp(90.0) == 0.0


@pytest.mark.unit
def test_a_run_with_no_time_is_a_setup_mistake():
    """The other reading -- "no limit" -- would silently remove the ceiling."""
    with pytest.raises(ValueError, match="must be positive"):
        RunBudget(0)


@pytest.mark.unit
def test_the_budget_reads_as_one_line_of_progress():
    """A run's log says where it is against its ceiling, in whole seconds."""
    budget, clock = a_budget()
    clock.advance(42.4)

    assert budget.summary() == "42s of 300s used"


@pytest.mark.unit
def test_a_run_takes_its_ceiling_from_its_settings():
    """One place reads the setting, so there is one place to change it."""
    budget = RunBudget.from_config(Config(run_timeout_s=120.0))

    assert budget.seconds == 120.0
    assert budget.check() is None


@pytest.mark.unit
def test_the_budget_uses_a_clock_that_cannot_go_backwards():
    """A machine adjusting its clock mid-run must not extend or end a run."""
    budget = RunBudget(RUN_SECONDS)

    assert budget.elapsed_s >= 0.0
    assert budget.remaining_s <= RUN_SECONDS


# --- The model endpoint's ceiling -------------------------------------------


class SlowEndpoint:
    """A transport that accepts a request and never answers.

    Shaped like the one attribute of the OpenAI client the model client
    actually uses, and raising what that client raises when a request runs out
    of time.

    Attributes:
        requests: Every request body it was handed, so a test can read the
            ceiling that was sent with it.
    """

    def __init__(self, error=None):
        """Args: error: What to raise. Defaults to a timeout."""
        self.requests = []
        # The real exception type, which is the part that matters: the client
        # tells a timeout from every other failure by type, not by message. The
        # request it would normally carry is only there for debugging, so None
        # is enough here.
        self._error = error or APITimeoutError(request=None)
        self.chat = self

    @property
    def completions(self):
        """SlowEndpoint: Itself, so ``chat.completions.create`` resolves."""
        return self

    def create(self, **request):
        """Record the request, then fail the way the endpoint would.

        Args:
            **request: The call body.

        Raises:
            Exception: Whatever this endpoint was built to raise.
        """
        self.requests.append(request)
        raise self._error


#: A shipped profile, used rather than an invented one so these tests exercise
#: the same shape a run does. Nothing is sent anywhere: the transport is a
#: stand-in and no key is read.
PROFILE = load_profile("local")


def a_client(transport, *, timeout_s=90.0):
    """Build a model client over a stand-in transport.

    Args:
        transport: The stand-in.
        timeout_s: The client's own ceiling.

    Returns:
        ModelClient: The client.
    """
    return ModelClient(PROFILE, transport=transport, env={}, timeout_s=timeout_s)


@pytest.mark.mock_llm
def test_a_stalled_endpoint_is_reported_as_a_timeout():
    """Not "the call failed": a slow endpoint and a broken one need different fixes."""
    client = a_client(SlowEndpoint())

    with pytest.raises(ModelTimeoutError) as raised:
        client.complete([{"role": "user", "content": "hello"}])

    assert "no answer" in str(raised.value)
    assert "90s" in str(raised.value), "the ceiling it waited for belongs in the message"


@pytest.mark.mock_llm
def test_a_timeout_is_still_an_ordinary_call_failure():
    """A caller that does not care about the distinction must not have to know."""
    client = a_client(SlowEndpoint())

    with pytest.raises(ModelClientError):
        client.complete([{"role": "user", "content": "hello"}])


@pytest.mark.mock_llm
def test_a_broken_endpoint_is_not_reported_as_a_slow_one():
    """The distinction only helps if it is actually a distinction."""
    client = a_client(SlowEndpoint(error=ConnectionRefusedError("no route to host")))

    with pytest.raises(ModelClientError) as raised:
        client.complete([{"role": "user", "content": "hello"}])

    assert not isinstance(raised.value, ModelTimeoutError)


@pytest.mark.mock_llm
def test_one_call_can_be_given_less_time_than_the_client_allows():
    """What lets a run refuse to start a 90-second call with 10 seconds left."""
    endpoint = SlowEndpoint()
    client = a_client(endpoint)

    with pytest.raises(ModelTimeoutError) as raised:
        client.complete([{"role": "user", "content": "hello"}], timeout_s=10.0)

    assert endpoint.requests[0]["timeout"] == 10.0
    assert "10s" in str(raised.value)


@pytest.mark.mock_llm
def test_a_run_short_on_time_asks_for_less_than_the_full_call():
    """The two ceilings meeting: the run's clock shortens the endpoint's."""
    budget, clock = a_budget()
    clock.advance(RUN_SECONDS - 12)
    endpoint = SlowEndpoint()
    client = a_client(endpoint)

    with pytest.raises(ModelTimeoutError):
        client.complete([{"role": "user", "content": "hi"}], timeout_s=budget.clamp(90.0))

    assert endpoint.requests[0]["timeout"] == pytest.approx(12.0)


@pytest.mark.mock_llm
def test_a_stalled_call_becomes_a_stated_run_outcome():
    """The bridge every ceiling crosses: an exception in, a verdict out."""
    client = a_client(SlowEndpoint())

    try:
        client.complete([{"role": "user", "content": "hello"}])
    except ModelTimeoutError as error:
        ended = timed_out(LLM_TIMEOUT, str(error))

    assert ended.status == "failed"
    assert ended.reason.startswith(f"{LLM_TIMEOUT}: ")
    assert PROFILE.model in ended.reason, "the trace should name the endpoint that stalled"


# --- The browser's ceiling, against a page that never settles ---------------


@pytest.mark.fixture
def test_a_page_that_never_settles_trips_the_action_ceiling(page, fixture_site):
    """The control is visible, enabled, and moving, so the click can never land."""
    page.goto(fixture_site.url_for("hang.html"))
    observation = observe(page)

    result = act(
        page,
        {"action": "click", "element_id": element_id_named(observation, "Reserve from trolley")},
        observation=observation,
        action_timeout_s=1.0,
    )

    assert result.status == TIMEOUT
    assert not result.ok


@pytest.mark.fixture
def test_the_ceiling_that_is_used_is_the_one_configured(page, fixture_site):
    """A ceiling nobody can change is a constant, not a setting."""
    page.goto(fixture_site.url_for("hang.html"))
    observation = observe(page)

    started = time.monotonic()
    act(
        page,
        {"action": "click", "element_id": element_id_named(observation, "Reserve from trolley")},
        observation=observation,
        action_timeout_s=1.0,
    )
    elapsed = time.monotonic() - started

    assert 0.9 <= elapsed < 3.0, "a one-second ceiling should end the step in about a second"


@pytest.mark.fixture
def test_the_rest_of_the_page_still_works(page, fixture_site):
    """The page is healthy; one control on it is not. Those are different things."""
    page.goto(fixture_site.url_for("hang.html"))
    observation = observe(page)

    result = act(
        page,
        {"action": "click", "element_id": element_id_named(observation, "Back to the home page")},
        observation=observation,
        action_timeout_s=5.0,
    )

    assert result.ok
    assert page.url.endswith("index.html")


@pytest.mark.fixture
def test_a_timed_out_step_becomes_a_stated_run_outcome(page, fixture_site):
    """What the controller will do with it: a category, and words underneath."""
    page.goto(fixture_site.url_for("hang.html"))
    observation = observe(page)

    result = act(
        page,
        {"action": "click", "element_id": element_id_named(observation, "Reserve from trolley")},
        observation=observation,
        action_timeout_s=1.0,
    )
    ended = timed_out(ACTION_TIMEOUT, result.summary())

    assert ended.status == "failed"
    assert ended.reason.startswith(f"{ACTION_TIMEOUT}: click ")


# --- One ceiling for the whole step, not for one call inside it -------------


class Waiter:
    """A locator whose click takes a given number of seconds and succeeds."""

    def __init__(self, calls, seconds):
        """Args: calls: Shared call list. seconds: How long the click takes."""
        self._calls = calls
        self._seconds = seconds
        self.first = self

    def count(self):
        """Returns: int: One, so the element is always found."""
        return 1

    def click(self, **kwargs):
        """Spend the step's budget, then succeed.

        Args:
            **kwargs: Ignored; the timeout is not enforced by this stand-in.
        """
        self._calls.append(("click", kwargs.get("timeout")))
        time.sleep(self._seconds)

    def get_attribute(self, name):
        """Returns: None: This element declares nothing about opening tabs.

        Args:
            name: The attribute asked for.
        """


class SlowClickPage:
    """A page whose click takes time, recording the waits that follow it.

    Attributes:
        calls: Every call made on it, in order, as ``(name, argument)``.
        url: The address reported back in the result.
    """

    def __init__(self, click_seconds):
        """Args: click_seconds: How long a click on this page takes."""
        self.calls = []
        self.url = "http://localhost/slow.html"
        self._click_seconds = click_seconds

    def locator(self, selector):
        """Returns: Waiter: An element that is always present, and slow."""
        return Waiter(self.calls, self._click_seconds)

    def wait_for_load_state(self, state, **kwargs):
        """Record the settle wait and the time it was given.

        Args:
            state: The load state waited for.
            **kwargs: Carries the timeout, which is what the test reads.
        """
        self.calls.append(("wait_for_load_state", kwargs.get("timeout")))

    def wait_for_timeout(self, milliseconds):
        """Pause, as the browser would, while the step listens for a new tab.

        Args:
            milliseconds: How long to pause.
        """
        time.sleep(milliseconds / 1000)


@pytest.mark.unit
def test_a_step_that_spends_its_ceiling_does_not_then_wait_to_settle():
    """Thirty seconds per action has to mean the step, not one call inside it.

    Without this, a click that used its whole budget would go on to wait five
    seconds more for the page to settle, and a ceiling of thirty would quietly
    be a ceiling of thirty-five.
    """
    page = SlowClickPage(click_seconds=0.3)

    result = Executor(page, action_timeout_s=0.1).execute(
        parse_action({"action": "click", "element_id": 1})
    )

    assert result.status == OK, "the click itself succeeded; only the extra waiting is skipped"
    assert [name for name, _ in page.calls] == ["click"], "no settle should have been attempted"


@pytest.mark.unit
def test_a_quick_step_still_waits_for_the_page_to_settle():
    """The clamp must not turn into "never settle", which would be worse."""
    page = SlowClickPage(click_seconds=0.0)

    Executor(page, action_timeout_s=30.0, settle_timeout_s=5.0).execute(
        parse_action({"action": "click", "element_id": 1})
    )

    settles = [given for name, given in page.calls if name == "wait_for_load_state"]
    assert settles, "a step with time to spare should wait for the page"
    assert 0 < settles[0] <= 5000, "and wait no longer than the settle ceiling"
