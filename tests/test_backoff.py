"""Waiting out a busy service, and knowing when to stop waiting.

An AI endpoint answering "too many requests" is not a broken endpoint. It is a
busy one, and the designed response is to wait and ask again -- which is a
different thing from every other failure in this project, and the reason it
gets a file of its own.

**Two retry mechanisms, deliberately not one.** The validator already retries:
when the model proposes something that makes no sense, it is handed a complaint
and asked again. That counts *bad answers*. This one counts *refusals*, and the
two must not be confused, because each would then hide the other -- a run out
of validation retries would look rate-limited, and a rate-limited run would
look like a model that cannot follow a schema. Nothing in this module touches
the validator, and there is a test that says so in the only way a test can: by
checking the two budgets are read from different settings and spent on
different things.

**Waiting is a cost, so it is recorded.** A task that took four minutes because
the endpoint made it queue looks, in a trace with no record of the queueing,
exactly like a task that took four minutes because the agent was floundering.
Every pause is written onto the call it belongs to.

**No test here actually waits.** The client is handed its sleep and its
randomness, so a test of the backoff does not take the backoff -- which is what
keeps the real defaults under test instead of a hurried one-tenth-of-a-second
version of them.
"""

from types import SimpleNamespace

import pytest
from openai import APITimeoutError, RateLimitError

from web_action_agent.config import Config
from web_action_agent.model_client import (
    ENDPOINT_UNAVAILABLE,
    RATE_LIMITED,
    Backoff,
    BackoffEvent,
    CallRecord,
    ModelBusyError,
    ModelClient,
    ModelClientError,
    ModelTimeoutError,
    load_profile,
    run_failure,
)
from web_action_agent.trace import new_id

PROFILE = load_profile("local")
"""Profile: A shipped profile. Nothing is sent anywhere; the transport is a stub."""

REPLY = '{"action": "scroll", "direction": "down"}'
"""str: What a successful call returns, shaped like a planner's answer."""


def refusal(status, *, retry_after=None):
    """Build the exception an endpoint raises when it turns a call away.

    The real class, not a stand-in: the client decides what to retry by reading
    the status off the exception, so a test that invented its own type could
    pass while the real thing was mishandled.

    Args:
        status: The HTTP status to refuse with.
        retry_after: Seconds to name in a ``Retry-After`` header, if any.

    Returns:
        RateLimitError: The refusal, carrying that status and header.
    """
    headers = {"retry-after": str(retry_after)} if retry_after is not None else {}
    response = SimpleNamespace(status_code=status, headers=headers, request=None)
    return RateLimitError("too many requests", response=response, body=None)


class Endpoint:
    """A transport that refuses a given number of times, then answers.

    Attributes:
        attempts: How many times it was called.
        requests: Every request body it was handed.
    """

    def __init__(self, *, refusals=(), reply=REPLY):
        """Set up the endpoint.

        Args:
            refusals: The exceptions to raise, in order, before answering. A
                single exception repeated forever is spelled by passing enough
                of them; a permanently busy endpoint is easier said with
                :class:`AlwaysBusy`.
            reply: The content to return once it does answer.
        """
        self.attempts = 0
        self.requests = []
        self._refusals = list(refusals)
        self._reply = reply
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **request):
        """Refuse, or answer, depending on how many times it has been asked.

        Args:
            **request: The call body.

        Returns:
            SimpleNamespace: A completion shaped like the SDK's.

        Raises:
            Exception: The next refusal, while there are any left.
        """
        self.attempts += 1
        self.requests.append(request)
        if self._refusals:
            raise self._refusals.pop(0)
        message = SimpleNamespace(content=self._reply)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=message, finish_reason="stop")],
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=4),
        )


class AlwaysBusy(Endpoint):
    """A transport that never lets anyone in."""

    def __init__(self, status=429, *, retry_after=None):
        """Args: status: What to refuse with. retry_after: Header to send."""
        super().__init__()
        self._status = status
        self._retry_after = retry_after

    def _create(self, **request):
        """Refuse, always.

        Args:
            **request: The call body.

        Raises:
            RateLimitError: Every time.
        """
        self.attempts += 1
        self.requests.append(request)
        raise refusal(self._status, retry_after=self._retry_after)


class Recorder:
    """Stands in for sleeping, and remembers what it was asked to wait.

    Attributes:
        waits: Every pause, in order. The whole point of injecting this: the
            real ceilings stay under test, and the test still runs instantly.
    """

    def __init__(self):
        """Start with nothing waited."""
        self.waits = []

    def __call__(self, seconds):
        """Record a pause instead of taking it.

        Args:
            seconds: How long the client wanted to wait.
        """
        self.waits.append(seconds)


def a_client(transport, *, backoff=None, sleep=None, spread=0.0):
    """Build a client over a stand-in endpoint.

    Args:
        transport: The stand-in.
        backoff: The policy. Defaults to the shipped one.
        sleep: Where pauses go. A fresh recorder by default.
        spread: The jitter draw, fixed so a test can assert an exact pause.

    Returns:
        tuple[ModelClient, Recorder]: The client and its sleep recorder.
    """
    recorder = sleep or Recorder()
    client = ModelClient(
        PROFILE,
        transport=transport,
        env={},
        backoff=backoff or Backoff(),
        sleep=recorder,
        spread=lambda: spread,
    )
    return client, recorder


def call(client):
    """Make one ordinary call.

    Args:
        client: The client to call.

    Returns:
        ModelResponse: What came back.
    """
    return client.complete([{"role": "user", "content": "what now?"}])


# --- Waiting, and then getting in -------------------------------------------


@pytest.mark.mock_llm
def test_a_busy_endpoint_is_waited_out_and_the_call_goes_through():
    """Two refusals, then an answer. The step completes; nothing is lost."""
    endpoint = Endpoint(refusals=[refusal(429), refusal(429)])
    client, _ = a_client(endpoint)

    response = call(client)

    assert response.text == REPLY
    assert endpoint.attempts == 3


@pytest.mark.mock_llm
def test_the_waiting_is_recorded_on_the_call_it_paid_for():
    """A run held up by a queue must not read as a run that was thinking slowly."""
    client, _ = a_client(Endpoint(refusals=[refusal(429), refusal(429)]))

    record = call(client).record

    assert [event.attempt for event in record.backoffs] == [1, 2]
    assert {event.status for event in record.backoffs} == {429}
    assert record.waited_s == pytest.approx(3.0), "one second, then two"


@pytest.mark.mock_llm
def test_each_wait_is_longer_than_the_last():
    """A fixed pause is the wrong shape: too short retries into the same wall."""
    client, sleeps = a_client(Endpoint(refusals=[refusal(429), refusal(429), refusal(503)]))

    call(client)

    assert sleeps.waits == [1.0, 2.0, 4.0]


@pytest.mark.mock_llm
def test_a_call_that_is_let_straight_in_records_no_waiting():
    """The common case has to stay free of ceremony."""
    client, sleeps = a_client(Endpoint())

    record = call(client).record

    assert record.backoffs == ()
    assert record.waited_s == 0.0
    assert sleeps.waits == []


@pytest.mark.mock_llm
def test_the_same_waiting_applies_to_a_schema_constrained_call():
    """Planner, judge and inducer differ in what they ask for, not in how they ask.

    All three call the same method, so a retry that lived in one of them would
    be missing from the other two. Asking with a schema exercises the other
    branch of the request builder and must be retried just the same.
    """
    endpoint = Endpoint(refusals=[refusal(429)])
    client, _ = a_client(endpoint)

    response = client.complete(
        [{"role": "user", "content": "what now?"}],
        schema={"type": "object"},
    )

    assert response.data == {"action": "scroll", "direction": "down"}
    assert endpoint.attempts == 2


# --- Giving up, with a reason -----------------------------------------------


@pytest.mark.mock_llm
def test_an_endpoint_that_never_relents_ends_the_call():
    """Bounded, because the point of retrying is still to finish."""
    endpoint = AlwaysBusy()
    client, sleeps = a_client(endpoint, backoff=Backoff(attempts=3))

    with pytest.raises(ModelBusyError) as raised:
        call(client)

    assert endpoint.attempts == 3, "three attempts means two waits"
    assert len(sleeps.waits) == 2
    assert raised.value.status == 429


@pytest.mark.mock_llm
def test_giving_up_says_it_was_the_rate_limit():
    """ "Too many requests" is a finding about the run; it should survive to the table."""
    client, _ = a_client(AlwaysBusy(), backoff=Backoff(attempts=2))

    with pytest.raises(ModelBusyError) as raised:
        call(client)

    assert raised.value.category == RATE_LIMITED
    ended = run_failure(raised.value)
    assert ended.status == "failed"
    assert ended.reason.startswith(f"{RATE_LIMITED}: ")


@pytest.mark.mock_llm
def test_a_broken_provider_is_waited_out_too_but_named_differently():
    """Same behaviour, different finding: one blames the run, the other the provider."""
    client, _ = a_client(AlwaysBusy(status=503), backoff=Backoff(attempts=2))

    with pytest.raises(ModelBusyError) as raised:
        call(client)

    assert raised.value.category == ENDPOINT_UNAVAILABLE
    assert run_failure(raised.value).reason.startswith(f"{ENDPOINT_UNAVAILABLE}: ")


@pytest.mark.mock_llm
def test_the_reason_says_how_long_was_spent_waiting():
    """Whoever reads the trace should not have to add up the pauses."""
    client, _ = a_client(AlwaysBusy(), backoff=Backoff(attempts=3))

    with pytest.raises(ModelBusyError) as raised:
        call(client)

    assert "3 attempts" in str(raised.value)
    assert "3.0s of waiting" in str(raised.value), "one second, then two"


@pytest.mark.mock_llm
def test_the_waits_spent_on_a_lost_cause_are_still_reported():
    """The time was spent whether or not the call worked."""
    client, _ = a_client(AlwaysBusy(), backoff=Backoff(attempts=3))

    with pytest.raises(ModelBusyError) as raised:
        call(client)

    assert [event.attempt for event in raised.value.backoffs] == [1, 2]


# --- What is not worth waiting for ------------------------------------------


@pytest.mark.mock_llm
@pytest.mark.parametrize("status", [400, 401, 403, 404, 422])
def test_a_refusal_that_waiting_cannot_fix_is_not_retried(status):
    """A bad key fails identically on the second attempt, and on the twentieth."""
    endpoint = AlwaysBusy(status=status)
    client, sleeps = a_client(endpoint)

    with pytest.raises(ModelClientError) as raised:
        call(client)

    assert not isinstance(raised.value, ModelBusyError)
    assert endpoint.attempts == 1, "one attempt, and no waiting"
    assert sleeps.waits == []


@pytest.mark.mock_llm
def test_a_failure_with_no_http_answer_behind_it_is_not_retried():
    """A refused connection is not a busy service, and never becomes one."""
    endpoint = Endpoint(refusals=[ConnectionRefusedError("no route to host")])
    client, _ = a_client(endpoint)

    with pytest.raises(ModelClientError):
        call(client)

    assert endpoint.attempts == 1


@pytest.mark.mock_llm
def test_a_timeout_is_not_retried():
    """The ceiling has just been spent waiting; waiting again is the wrong answer."""
    endpoint = Endpoint(refusals=[APITimeoutError(request=None)])
    client, sleeps = a_client(endpoint)

    with pytest.raises(ModelTimeoutError):
        call(client)

    assert endpoint.attempts == 1
    assert sleeps.waits == []


# --- Listening to what the endpoint asks for --------------------------------


@pytest.mark.mock_llm
def test_an_endpoint_asking_for_longer_is_obeyed():
    """It knows when it will be ready; the doubling is only a guess."""
    client, sleeps = a_client(Endpoint(refusals=[refusal(429, retry_after=5)]))

    call(client)

    assert sleeps.waits == [5.0]


@pytest.mark.mock_llm
def test_an_endpoint_asking_for_less_is_not_obeyed():
    """The pause is this project's floor as well as the endpoint's request."""
    client, sleeps = a_client(Endpoint(refusals=[refusal(429, retry_after=0.1)]))

    call(client)

    assert sleeps.waits == [1.0]


@pytest.mark.mock_llm
def test_what_the_endpoint_asked_for_is_recorded_beside_what_was_waited():
    """So a trace shows whether a long pause was ours or theirs."""
    client, _ = a_client(Endpoint(refusals=[refusal(429, retry_after=5)]))

    event = call(client).record.backoffs[0]

    assert event.asked_s == 5.0
    assert event.waited_s == 5.0


@pytest.mark.mock_llm
def test_an_unreadable_retry_after_falls_back_to_the_doubling():
    """Some endpoints send a date here; unknown must not mean "do not wait"."""
    client, sleeps = a_client(Endpoint(refusals=[refusal(429, retry_after="Wed, 21 Oct 2026")]))

    call(client)

    assert sleeps.waits == [1.0]


# --- The policy on its own --------------------------------------------------


@pytest.mark.unit
def test_the_pause_doubles_each_time():
    """Small first, so a one-second hiccup does not cost a minute."""
    backoff = Backoff(base_s=1.0, max_s=60.0)

    assert [backoff.wait_after(n) for n in (1, 2, 3, 4)] == [1.0, 2.0, 4.0, 8.0]


@pytest.mark.unit
def test_no_single_pause_exceeds_the_ceiling():
    """Doubling forever would eventually stall a run inside one wait."""
    backoff = Backoff(base_s=1.0, max_s=5.0)

    assert backoff.wait_after(10) == 5.0


@pytest.mark.unit
def test_a_ceiling_also_bounds_what_the_endpoint_asks_for():
    """An endpoint asking for an hour does not get an hour."""
    backoff = Backoff(base_s=1.0, max_s=5.0)

    assert backoff.wait_after(1, asked_s=3600) == 5.0


@pytest.mark.unit
def test_jitter_only_ever_adds():
    """Four workers refused together must not retry in lockstep.

    Adding rather than spreading either way matters: a pause the endpoint asked
    for must never come back shorter than it asked.
    """
    backoff = Backoff(base_s=4.0, max_s=60.0, jitter=0.25)

    assert backoff.wait_after(1, spread=0.0) == 4.0
    assert backoff.wait_after(1, spread=1.0) == 5.0
    assert backoff.wait_after(1, spread=0.5) == 4.5


@pytest.mark.unit
def test_jitter_cannot_push_a_pause_past_the_ceiling():
    """The ceiling is the ceiling, whatever the randomness draws."""
    backoff = Backoff(base_s=4.0, max_s=4.0, jitter=0.5)

    assert backoff.wait_after(1, spread=1.0) == 4.0


@pytest.mark.unit
def test_the_policy_comes_from_the_run_configuration():
    """One place reads the settings, so there is one place to change them."""
    backoff = Backoff.from_config(Config(backoff_attempts=6, backoff_base_s=2.0, backoff_max_s=9.0))

    assert (backoff.attempts, backoff.base_s, backoff.max_s) == (6, 2.0, 9.0)


@pytest.mark.unit
def test_the_two_retry_budgets_are_different_settings():
    """The validator counts bad answers; this counts refusals. Never one number.

    Merging them would make each hide the other: a run out of validation
    retries would look rate-limited, and a rate-limited run would look like a
    model that cannot follow a schema.
    """
    config = Config(retry_budget=2, backoff_attempts=6)

    assert Backoff.from_config(config).attempts == 6
    assert config.retry_budget == 2


@pytest.mark.unit
def test_a_pause_reads_as_one_fragment_of_a_call_log():
    """A hundred calls have to stay readable."""
    plain = BackoffEvent(attempt=1, status=429, waited_s=1.0)
    asked = BackoffEvent(attempt=2, status=429, waited_s=5.0, asked_s=5.0)

    assert plain.summary() == "attempt 1 refused with 429, waited 1.0s"
    assert asked.summary() == "attempt 2 refused with 429, waited 5.0s (asked for 5.0s)"
    assert plain.to_dict() == {"attempt": 1, "status": 429, "waited_s": 1.0, "asked_s": None}


@pytest.mark.unit
def test_a_record_with_no_waiting_says_zero_rather_than_nothing():
    """Every call answers the question, so nothing downstream needs a special case."""
    record = CallRecord(
        call_id=new_id("call"),
        started_at="2026-09-17T08:00:00.000Z",
        ended_at="2026-09-17T08:00:00.100Z",
        profile="local",
        model="test",
        vision=False,
        prompt_tokens=1,
        completion_tokens=1,
        latency_s=0.1,
        finish_reason="stop",
    )

    assert record.waited_s == 0.0
    assert record.backoffs == ()
