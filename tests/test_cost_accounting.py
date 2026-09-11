"""What a run cost, and the one number that must never be invented.

Two of the three cost numbers are facts: the endpoint reports how many tokens
it read and wrote, and a clock says how long it took. The third -- dollars --
is a *lookup*, and that makes it the one that can be quietly wrong.

Almost every test here is about the same failure: a cost that is unknown being
written down as zero. It is tempting, because zero adds up and ``None`` does
not, and it is the kind of wrong that never announces itself. One unpriced
model in a benchmark makes the whole cost-per-task column wrong, in the
flattering direction, with nothing on the page to say so. So the tests assert
the awkward behaviour on purpose: unknown stays unknown, a free route is not a
price of zero, and a total containing an unpriced call is itself unknown -- with
the count of unpriced calls beside it, so the gap is visible.

The other half is the lookup being *specific enough*. The same open-weight
model is served by an aggregator, by its vendor, and by a server on the
developer's own machine at three different prices, the last of which is not a
per-token price at all. A table keyed on the model name alone would bill a
self-hosted run for tokens that cost nothing, so the tests check that the same
model on two profiles prices differently.
"""

from types import SimpleNamespace

import pytest
import yaml

from web_action_agent.cost import (
    DEFAULT_PRICING_PATH,
    Price,
    PriceList,
    PricingError,
    RunCost,
    shipped_prices,
    total_cost,
)
from web_action_agent.model_client import (
    BackoffEvent,
    CallRecord,
    ModelClient,
    load_profile,
    load_profiles,
)

PROFILE = load_profile("openrouter")
"""Profile: A shipped profile. Nothing is sent anywhere; the transport is a stub."""

REPLY = '{"action": "scroll", "direction": "down"}'
"""str: What a call returns, shaped like a planner's answer."""

PRICED = {
    "openrouter": {
        "nvidia/nemotron-3-nano-30b-a3b": {
            "input_usd_per_mtok": 2.0,
            "output_usd_per_mtok": 8.0,
            "checked": "2026-09-11",
        }
    }
}
"""dict: A price table with one real price in it.

Invented for the tests on purpose. The shipped table's prices are whatever the
providers charge on the day somebody fills them in, and a test that asserted
those would be a test that fails when a provider changes its mind.
"""


class Endpoint:
    """A transport that answers with fixed token counts.

    Attributes:
        calls: How many times it was asked.
    """

    def __init__(self, *, prompt_tokens=1000, completion_tokens=100):
        """Args: prompt_tokens, completion_tokens: What usage to report, or None for silence."""
        self.calls = 0
        self._usage = (
            SimpleNamespace(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens)
            if prompt_tokens is not None or completion_tokens is not None
            else None
        )
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **request):
        """Answer one call.

        Args:
            **request: The call body.

        Returns:
            SimpleNamespace: A completion shaped like the SDK's.
        """
        self.calls += 1
        message = SimpleNamespace(content=REPLY)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=message, finish_reason="stop")],
            usage=self._usage,
        )


def a_price_list(table=None, tmp_path=None):
    """Build a price list from a table written to a temporary file.

    Written to disk rather than constructed in memory so the loader is
    exercised too: the file is the thing that ships, and a loader nobody runs
    in a test is a loader that breaks the first time somebody edits the table.

    Args:
        table: The table to write. The one-price table by default.
        tmp_path: Where to write it. Required.

    Returns:
        PriceList: The prices it declares.
    """
    path = tmp_path / "pricing.yaml"
    path.write_text(yaml.safe_dump(table if table is not None else PRICED), encoding="utf-8")
    return PriceList.load(path)


def a_client(prices, *, transport=None, profile=PROFILE):
    """Build a model client on a stub transport and a given price list.

    Args:
        prices: The prices to apply.
        transport: The stub. A fresh one by default.
        profile: The profile to run under.

    Returns:
        tuple[ModelClient, Endpoint]: The client and its endpoint.
    """
    endpoint = transport or Endpoint()
    return ModelClient(profile, transport=endpoint, env={}, prices=prices), endpoint


def call(client):
    """Make one ordinary call.

    Args:
        client: The client to call.

    Returns:
        ModelResponse: What came back.
    """
    return client.complete([{"role": "user", "content": "what now?"}])


def a_record(**overrides):
    """Build a call record with plausible values.

    Args:
        **overrides: Fields to replace.

    Returns:
        CallRecord: The record.
    """
    fields = {
        "profile": "openrouter",
        "model": "nvidia/nemotron-3-nano-30b-a3b",
        "vision": False,
        "prompt_tokens": 1000,
        "completion_tokens": 100,
        "latency_s": 1.5,
        "finish_reason": "stop",
        "cost_usd": 0.0028,
    }
    return CallRecord(**{**fields, **overrides})


# --- What one call costs ----------------------------------------------------


@pytest.mark.unit
def test_a_call_is_priced_per_million_tokens():
    """The unit providers quote in, so a reader can check it against their page."""
    price = Price(input_usd_per_mtok=2.0, output_usd_per_mtok=8.0)

    # A thousand input tokens at $2 per million is $0.002; a hundred output
    # tokens at $8 per million is $0.0008.
    assert price.cost_of(1000, 100) == pytest.approx(0.0028)


@pytest.mark.unit
def test_input_and_output_are_priced_separately():
    """They differ by a factor of three or four, so one rate would be wrong twice."""
    price = Price(input_usd_per_mtok=1.0, output_usd_per_mtok=10.0)

    assert price.cost_of(1_000_000, 0) == pytest.approx(1.0)
    assert price.cost_of(0, 1_000_000) == pytest.approx(10.0)


@pytest.mark.unit
def test_an_unknown_price_is_unknown_rather_than_free():
    """The failure this whole module exists to prevent."""
    assert Price().cost_of(1000, 100) is None


@pytest.mark.unit
def test_half_a_price_prices_nothing():
    """A table with an input rate and no output rate knows half a price.

    Half a price can only produce a number that is wrong and looks right.
    """
    assert Price(input_usd_per_mtok=2.0).cost_of(1000, 100) is None
    assert Price(output_usd_per_mtok=8.0).cost_of(1000, 100) is None


@pytest.mark.unit
def test_tokens_nobody_counted_cannot_be_priced():
    """Some endpoints report no usage at all; a price cannot rescue that."""
    price = Price(input_usd_per_mtok=2.0, output_usd_per_mtok=8.0)

    assert price.cost_of(None, 100) is None
    assert price.cost_of(1000, None) is None


# --- Finding the right price ------------------------------------------------


@pytest.mark.unit
def test_a_price_is_found_by_profile_and_model(tmp_path):
    """Both halves of the key matter; neither alone identifies a price."""
    prices = a_price_list(tmp_path=tmp_path)

    assert prices.cost_of(
        "openrouter", "nvidia/nemotron-3-nano-30b-a3b", 1000, 100
    ) == pytest.approx(0.0028)


@pytest.mark.unit
def test_the_same_model_costs_differently_on_a_different_profile(tmp_path):
    """A self-hosted run must not be billed at an aggregator's rate.

    The same open-weight model runs on somebody's own hardware, where tokens
    cost electricity rather than dollars. Keyed on the model name alone, this
    run would be charged for every token it ever generated.
    """
    prices = a_price_list(tmp_path=tmp_path)

    assert prices.cost_of("local", "nvidia/nemotron-3-nano-30b-a3b", 1000, 100) is None


@pytest.mark.unit
def test_a_model_the_table_does_not_name_is_unpriced(tmp_path):
    """A judge nobody has priced yet is an ordinary situation, not an error."""
    prices = a_price_list(tmp_path=tmp_path)

    assert prices.price_for("openrouter", "some/other-model") is None
    assert prices.cost_of("openrouter", "some/other-model", 1000, 100) is None


@pytest.mark.unit
def test_a_nearly_matching_name_is_not_priced_by_resemblance(tmp_path):
    """`...-reasoning` and `...-reasoning:free` differ by a suffix and by everything."""
    prices = a_price_list(tmp_path=tmp_path)

    assert prices.cost_of("openrouter", "nvidia/nemotron-3-nano-30b-a3b:free", 1000, 100) is None


# --- The table that ships ---------------------------------------------------


@pytest.mark.unit
def test_the_shipped_table_loads():
    """It travels inside the package, so an installed copy prices like a checkout."""
    prices = shipped_prices()

    assert len(prices) > 0
    assert prices.source.endswith("pricing.yaml")
    assert DEFAULT_PRICING_PATH.exists()


@pytest.mark.unit
def test_every_shipped_route_has_an_entry_in_the_table():
    """Adding a profile without a price should fail here, not in a cost column.

    The entry may well say "unknown" -- most of them do until somebody fills
    them in. What must not happen is a route the table has never heard of,
    because that reads as "unpriced" in exactly the same way as a genuine gap.
    """
    prices = shipped_prices()
    missing = []
    for name, profile in load_profiles().items():
        for route in (profile.text, profile.vision):
            if route is not None and prices.price_for(name, route.model) is None:
                missing.append(f"{name}/{route.model}")

    assert missing == [], f"routes with no entry in the price table: {missing}"


@pytest.mark.unit
def test_the_shipped_table_records_unknown_prices_as_unknown():
    """Until somebody fills them in, dollars are unknown -- and say so."""
    prices = shipped_prices()
    local = prices.price_for("local", "nvidia/nemotron-3-nano-30b-a3b")

    assert local is not None
    assert not local.known, "a self-hosted route has no per-token price to record"
    assert local.note, "and the table should say why, for whoever reads it"


# --- A table that is wrong should fail loudly -------------------------------


@pytest.mark.unit
def test_a_zero_price_is_refused(tmp_path):
    """Zero is a claim that the calls were free, and it is almost always a typo."""
    table = {"openrouter": {"a/model": {"input_usd_per_mtok": 0, "output_usd_per_mtok": 8.0}}}

    with pytest.raises(PricingError, match="not a price"):
        a_price_list(table, tmp_path=tmp_path)


@pytest.mark.unit
def test_a_negative_price_is_refused(tmp_path):
    """Nobody pays you to run a model."""
    table = {"openrouter": {"a/model": {"input_usd_per_mtok": -2.0, "output_usd_per_mtok": 8.0}}}

    with pytest.raises(PricingError, match="not a price"):
        a_price_list(table, tmp_path=tmp_path)


@pytest.mark.unit
def test_a_price_that_is_not_a_number_is_refused(tmp_path):
    """A price copied as "2.00 USD" would otherwise silently price nothing."""
    table = {"openrouter": {"a/model": {"input_usd_per_mtok": "2.00 USD"}}}

    with pytest.raises(PricingError, match="should be a number"):
        a_price_list(table, tmp_path=tmp_path)


@pytest.mark.unit
def test_a_missing_table_is_refused():
    """A broken build, not a run that quietly prices nothing."""
    with pytest.raises(PricingError, match="cannot read"):
        PriceList.load(DEFAULT_PRICING_PATH.parent / "no-such-pricing.yaml")


@pytest.mark.unit
def test_an_empty_table_prices_nothing_without_complaining(tmp_path):
    """Legitimate: it means nobody has filled anything in, which is a state."""
    path = tmp_path / "pricing.yaml"
    path.write_text("", encoding="utf-8")

    prices = PriceList.load(path)

    assert len(prices) == 0
    assert prices.cost_of("openrouter", "a/model", 10, 10) is None


# --- Rolling a run up -------------------------------------------------------


@pytest.mark.unit
def test_a_run_totals_its_calls():
    """The sum of the parts, which is the only claim a total should make."""
    total = total_cost([a_record(), a_record(), a_record()])

    assert total.calls == 3
    assert total.prompt_tokens == 3000
    assert total.completion_tokens == 300
    assert total.total_tokens == 3300
    assert total.cost_usd == pytest.approx(0.0084)


@pytest.mark.unit
def test_a_total_containing_an_unpriced_call_is_itself_unknown():
    """The heart of it: a total that omits some of its parts is not a total."""
    total = total_cost([a_record(), a_record(cost_usd=None)])

    assert total.cost_usd is None
    assert total.unpriced_calls == 1


@pytest.mark.unit
def test_what_could_be_priced_is_still_reported():
    """Unknown must not mean the known half is thrown away.

    Read together the two say "at least this much, and one call we cannot
    price", which is the honest sentence and the useful one.
    """
    total = total_cost([a_record(), a_record(cost_usd=None)])

    assert total.known_cost_usd == pytest.approx(0.0028)
    assert "unknown" in total.summary()
    assert "1 unpriced" in total.summary()


@pytest.mark.unit
def test_tokens_nobody_reported_make_the_token_total_unknown_too():
    """Same rule, applied to the numbers the endpoint was supposed to give."""
    total = total_cost([a_record(), a_record(prompt_tokens=None)])

    assert total.prompt_tokens is None
    assert total.total_tokens is None
    assert total.completion_tokens == 200, "the half that was reported is still summed"


@pytest.mark.unit
def test_thinking_time_and_queueing_time_are_totalled_apart():
    """A run held up by a rate limit needs a different fix from a slow model."""
    waited = a_record(backoffs=(BackoffEvent(attempt=1, status=429, waited_s=2.0),))

    total = total_cost([a_record(), waited])

    assert total.latency_s == pytest.approx(3.0)
    assert total.waited_s == pytest.approx(2.0)


@pytest.mark.unit
def test_a_run_that_made_no_calls_costs_nothing():
    """Zero calls really did cost zero -- the one place zero is the true answer."""
    total = total_cost([])

    assert total == RunCost()
    assert total.cost_usd == 0.0
    assert total.total_tokens == 0


@pytest.mark.unit
def test_the_roll_up_writes_itself_as_plain_data():
    """A trace is read after the run, so the totals have to survive being written."""
    record = total_cost([a_record(), a_record(cost_usd=None)]).to_dict()

    assert record["calls"] == 2
    assert record["cost_usd"] is None
    assert record["known_cost_usd"] == pytest.approx(0.0028)
    assert record["unpriced_calls"] == 1
    assert record["total_tokens"] == 2200


# --- End to end, through the client -----------------------------------------


@pytest.mark.mock_llm
def test_three_calls_produce_three_records_that_add_up(tmp_path):
    """The task's own test: the parts equal the whole."""
    client, endpoint = a_client(a_price_list(tmp_path=tmp_path))

    records = [call(client).record for _ in range(3)]
    total = total_cost(records)

    assert endpoint.calls == 3
    assert total.calls == 3
    assert total.prompt_tokens == sum(record.prompt_tokens for record in records)
    assert total.completion_tokens == sum(record.completion_tokens for record in records)
    assert total.cost_usd == pytest.approx(sum(record.cost_usd for record in records))


@pytest.mark.mock_llm
def test_a_model_with_no_price_still_records_tokens_and_wall_clock(tmp_path):
    """Dollars unknown; everything that is a fact is recorded anyway.

    Tokens and wall-clock are the cost measure a self-hosted run has, and they
    are what makes two runs comparable when neither has a price.
    """
    empty = tmp_path / "empty.yaml"
    empty.write_text("", encoding="utf-8")
    client, _ = a_client(PriceList.load(empty))

    record = call(client).record

    assert record.cost_usd is None
    assert record.prompt_tokens == 1000
    assert record.completion_tokens == 100
    assert record.latency_s >= 0.0


@pytest.mark.mock_llm
def test_an_endpoint_that_reports_no_usage_prices_nothing(tmp_path):
    """A priced model cannot be priced without the counts to price."""
    client, _ = a_client(
        a_price_list(tmp_path=tmp_path),
        transport=Endpoint(prompt_tokens=None, completion_tokens=None),
    )

    record = call(client).record

    assert record.prompt_tokens is None
    assert record.cost_usd is None
    assert record.latency_s >= 0.0, "the clock still ran"


@pytest.mark.mock_llm
def test_the_call_is_priced_under_the_profile_that_served_it(tmp_path):
    """Profile and model together, resolved at the call rather than guessed later."""
    prices = a_price_list(tmp_path=tmp_path)
    priced, _ = a_client(prices)
    unpriced, _ = a_client(prices, profile=load_profile("local"))

    assert call(priced).record.cost_usd == pytest.approx(0.0028)
    assert call(unpriced).record.cost_usd is None


@pytest.mark.mock_llm
def test_a_client_with_no_price_list_uses_the_shipped_one(tmp_path):
    """The default has to be the checked-in table, not "no prices at all"."""
    client = ModelClient(PROFILE, transport=Endpoint(), env={})

    assert client.prices.source.endswith("pricing.yaml")
