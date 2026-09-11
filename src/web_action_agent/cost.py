"""What a run cost, and the difference between free and unknown.

Three numbers describe one model call: how many tokens went in and out, how
long it took, and how many dollars it was. The first two are facts the endpoint
reports. The third is a *lookup* -- the endpoint does not say -- and that makes
it the one that can be quietly wrong.

**Unknown is recorded as unknown.** The tempting shortcut is to write ``0.0``
for a model whose price nobody has filled in, because zero is a number and
``None`` is awkward to add up. It is also a lie, and a peculiarly damaging one:
a single unpriced model in a benchmark makes the whole cost-per-task column
wrong, in the flattering direction, with nothing on the page to say so. So an
unpriced call records ``None``, and a total containing one is itself ``None``,
with the count of unpriced calls beside it so the gap is visible rather than
absorbed.

**A free route is not a price of zero either.** A free tier is a promotional
arrangement with a rate limit attached; it can end, and it applies to a route
rather than to the work. Recording it as ``0.0`` would let the project's
headline comparison -- the same benchmark with and without a skill library --
report a cost saving that was really somebody's promotion.

**Prices are keyed by profile *and* model.** The same open-weight model is
served by an aggregator, by its vendor, and by a server on the developer's own
machine, at three different prices; the last has no per-token price at all. A
table keyed on the model name alone would bill a self-hosted run for tokens
that cost nothing, which is the same class of error as the false zero and
harder to notice.

**Tokens and wall-clock are always recorded, whatever the dollars do.** They
are the cost measure a self-hosted run has, and they are what makes two runs
comparable even when neither has a price.
"""

from __future__ import annotations

import functools
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

if TYPE_CHECKING:
    # For the annotation only. The record is defined by the model client, which
    # reads this module for prices -- importing it back at runtime would be a
    # cycle, and this module needs the shape, not the class.
    from .model_client import CallRecord

#: The checked-in price table, shipped inside the package so that an installed
#: copy prices the same way a checkout does.
DEFAULT_PRICING_PATH = Path(__file__).resolve().parent / "pricing.yaml"

#: Providers quote prices per million tokens, so the table does too and the
#: division happens here rather than in everybody's head.
TOKENS_PER_PRICE_UNIT = 1_000_000


class PricingError(Exception):
    """The price table is missing, malformed, or says something impossible.

    Raised at load time rather than at the first call that needs a price: a
    broken table is a setup mistake, and finding it out halfway through a
    benchmark means the first half is priced and the second is not.
    """


@dataclass(frozen=True)
class Price:
    """What one profile charges for one model, per million tokens.

    Attributes:
        input_usd_per_mtok: Dollars per million input tokens, or None when the
            price is unknown or the route is free. Both are recorded the same
            way on purpose -- see the module docstring -- and ``note`` is where
            the difference is written down for a reader.
        output_usd_per_mtok: Dollars per million output tokens, or None.
        checked: The date the price was last verified against the provider's
            own page, as written in the table. Empty when nobody has. A price
            with no date is a price nobody should publish a number from.
        note: Why this entry says what it says, in a few words.
    """

    input_usd_per_mtok: float | None = None
    output_usd_per_mtok: float | None = None
    checked: str = ""
    note: str = ""

    @property
    def known(self) -> bool:
        """bool: Whether this entry can price a call at all.

        Both halves are required. A table giving an input price and leaving the
        output blank knows half a price, and half a price cannot produce a
        number that is anything but misleading.
        """
        return self.input_usd_per_mtok is not None and self.output_usd_per_mtok is not None

    def cost_of(self, prompt_tokens: int | None, completion_tokens: int | None) -> float | None:
        """Price one call.

        Args:
            prompt_tokens: Input tokens the endpoint reported, or None if it
                reported none.
            completion_tokens: Output tokens reported, or None.

        Returns:
            float | None: Dollars, or None when either the price or the token
            counts are unknown. Never ``0.0`` as a stand-in for either.
        """
        if not self.known or prompt_tokens is None or completion_tokens is None:
            return None
        # Written as one expression per side so a reader can check it against a
        # provider's pricing page without rearranging anything.
        input_cost = prompt_tokens * self.input_usd_per_mtok / TOKENS_PER_PRICE_UNIT
        output_cost = completion_tokens * self.output_usd_per_mtok / TOKENS_PER_PRICE_UNIT
        return input_cost + output_cost


class PriceList:
    """Every price the project knows, from the checked-in table.

    Lookup is by profile and model together, and it is exact. A model whose
    name nearly matches an entry is not priced by it: names like
    ``...-30b-a3b`` and ``...-30b-a3b:free`` differ by one suffix and by their
    entire price, so guessing from a resemblance would produce a confident
    wrong number -- which is worse than the honest ``None`` that a miss gives.

    Attributes:
        source: Where the table was read from, for the run's trace. A published
            cost figure should be able to name the file it was priced under.
    """

    def __init__(self, prices: Mapping[tuple[str, str], Price], *, source: str = "") -> None:
        """Build a price list.

        Args:
            prices: Prices by ``(profile, model)``.
            source: Where they came from.
        """
        self._prices = dict(prices)
        self.source = source

    @classmethod
    def load(cls, path: Path | None = None) -> PriceList:
        """Read a price table from disk.

        Args:
            path: The table to read. Defaults to the checked-in one.

        Returns:
            PriceList: The prices it declares.

        Raises:
            PricingError: If the file is missing, is not a mapping of profiles
                to models, or gives a price that is not a positive number.
        """
        path = path or DEFAULT_PRICING_PATH
        try:
            raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        except OSError as error:
            raise PricingError(f"cannot read the price table at {path}: {error}") from error
        except yaml.YAMLError as error:
            raise PricingError(f"{path} is not valid YAML: {error}") from error

        if raw is None:
            # An empty table is legitimate -- it prices nothing, and everything
            # is recorded as unknown, which is what an empty table means.
            return cls({}, source=str(path))
        if not isinstance(raw, Mapping):
            raise PricingError(
                f"{path} should map profile names to models, not {type(raw).__name__}"
            )

        prices: dict[tuple[str, str], Price] = {}
        for profile, models in raw.items():
            if not isinstance(models, Mapping):
                raise PricingError(f"{path}: profile {profile!r} should map model names to prices")
            for model, entry in models.items():
                prices[(str(profile), str(model))] = _price_from(entry, path, profile, model)
        return cls(prices, source=str(path))

    def price_for(self, profile: str, model: str) -> Price | None:
        """Find what one profile charges for one model.

        Args:
            profile: The profile name, as it appears in ``profiles.yaml``.
            model: The exact model identifier the call used -- the route's
                resolved model, not the one configured, since those differ on
                a vision call.

        Returns:
            Price | None: The entry, or None when the table has none. A miss is
            an ordinary answer, not an error: a run may legitimately use a
            judge model nobody has priced yet.
        """
        return self._prices.get((profile, model))

    def cost_of(
        self,
        profile: str,
        model: str,
        prompt_tokens: int | None,
        completion_tokens: int | None,
    ) -> float | None:
        """Price one call, by profile and model.

        Args:
            profile: The profile that served the call.
            model: The exact model identifier used.
            prompt_tokens: Input tokens reported, or None.
            completion_tokens: Output tokens reported, or None.

        Returns:
            float | None: Dollars, or None when the price or the tokens are
            unknown.
        """
        price = self.price_for(profile, model)
        return price.cost_of(prompt_tokens, completion_tokens) if price else None

    def __len__(self) -> int:
        """Returns: int: How many priced or unpriced entries the table holds."""
        return len(self._prices)

    def entries(self) -> tuple[tuple[str, str], ...]:
        """List what the table has an entry for.

        Returns:
            tuple[tuple[str, str], ...]: Every ``(profile, model)`` pair, in the
            order the file declares them. Used by the test that every shipped
            profile is priced, so that adding a profile without adding a price
            is caught here rather than in a benchmark's cost column.
        """
        return tuple(self._prices)


@functools.cache
def shipped_prices() -> PriceList:
    """The checked-in table, read once per process.

    Returns:
        PriceList: The prices this version of the project ships.

    Raises:
        PricingError: If the shipped table cannot be read. That is a broken
            build rather than a user mistake, and it should stop everything.
    """
    return PriceList.load()


@dataclass(frozen=True)
class RunCost:
    """What a whole run's model calls cost, rolled up.

    The awkward part of this record is deliberate. ``cost_usd`` is None when
    *any* call could not be priced, because a total that silently omits some of
    its parts is not a total -- while ``known_cost_usd`` still reports what the
    priced calls came to, so the figure is not simply lost. Read together they
    say "at least this much, and this many calls we cannot price", which is the
    honest sentence.

    Attributes:
        calls: How many model calls the run made, including the ones that
            failed after being billed.
        prompt_tokens: Input tokens across the run, or None if any call's
            endpoint reported none.
        completion_tokens: Output tokens, or None on the same condition.
        cost_usd: Dollars for the whole run, or None if any call was unpriced.
        known_cost_usd: Dollars for the calls that could be priced. Always a
            number, and never to be published as the run's cost when
            ``unpriced_calls`` is above zero.
        unpriced_calls: How many calls had no price to apply.
        latency_s: Seconds the endpoints spent thinking.
        waited_s: Seconds spent waiting to be let in after a refusal. Kept
            apart from latency because a run held up by a rate limit and a run
            held up by a slow model need different fixes.
    """

    calls: int = 0
    prompt_tokens: int | None = 0
    completion_tokens: int | None = 0
    cost_usd: float | None = 0.0
    known_cost_usd: float = 0.0
    unpriced_calls: int = 0
    latency_s: float = 0.0
    waited_s: float = 0.0

    @property
    def total_tokens(self) -> int | None:
        """int | None: Input plus output, or None if either is unknown."""
        if self.prompt_tokens is None or self.completion_tokens is None:
            return None
        return self.prompt_tokens + self.completion_tokens

    def summary(self) -> str:
        """Render the roll-up as the line a run's report prints.

        Returns:
            str: e.g. ``12 calls, 8,431 tokens, $0.0042, 9.1s thinking`` --
            with dollars shown as ``unknown (2 calls unpriced)`` where they
            are, because a missing number should read as missing rather than
            as nothing.
        """
        tokens = (
            f"{self.total_tokens:,} tokens" if self.total_tokens is not None else "tokens unknown"
        )
        if self.cost_usd is not None:
            money = f"${self.cost_usd:.4f}"
        else:
            money = f"unknown (${self.known_cost_usd:.4f} known, {self.unpriced_calls} unpriced)"
        return f"{self.calls} calls, {tokens}, {money}, {self.latency_s:.1f}s thinking"

    def to_dict(self) -> dict[str, Any]:
        """Render the roll-up as plain data, for the run's trace.

        Returns:
            dict[str, Any]: JSON-safe values. Dollars are rounded to six
            decimal places, which is a hundredth of a cent -- fine enough for a
            single call and coarse enough not to publish float noise.
        """
        return {
            "calls": self.calls,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "cost_usd": round(self.cost_usd, 6) if self.cost_usd is not None else None,
            "known_cost_usd": round(self.known_cost_usd, 6),
            "unpriced_calls": self.unpriced_calls,
            "latency_s": round(self.latency_s, 3),
            "waited_s": round(self.waited_s, 3),
        }


def total_cost(records: Iterable[CallRecord]) -> RunCost:
    """Roll up every call a run made.

    Args:
        records: The run's call records, in any order.

    Returns:
        RunCost: The totals, with unknowns propagated rather than absorbed.
        A run that made no calls comes back as zeros, which is the true answer
        for a run that spent nothing rather than a run nobody measured.
    """
    calls = 0
    prompt_tokens: int | None = 0
    completion_tokens: int | None = 0
    known_cost = 0.0
    unpriced = 0
    latency = 0.0
    waited = 0.0

    for record in records:
        calls += 1
        # Once one call's tokens are unknown, the run's total is unknown too --
        # and stays that way however many later calls report theirs.
        if record.prompt_tokens is None or prompt_tokens is None:
            prompt_tokens = None
        else:
            prompt_tokens += record.prompt_tokens
        if record.completion_tokens is None or completion_tokens is None:
            completion_tokens = None
        else:
            completion_tokens += record.completion_tokens

        if record.cost_usd is None:
            unpriced += 1
        else:
            known_cost += record.cost_usd

        latency += record.latency_s
        waited += record.waited_s

    return RunCost(
        calls=calls,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        cost_usd=known_cost if unpriced == 0 else None,
        known_cost_usd=known_cost,
        unpriced_calls=unpriced,
        latency_s=latency,
        waited_s=waited,
    )


def _price_from(entry: Any, path: Path, profile: object, model: object) -> Price:
    """Turn one entry of the price table into a :class:`Price`.

    Args:
        entry: The mapping under one model name, or None for an entry left
            entirely blank.
        path: The file being read, for error messages.
        profile: The profile the entry sits under.
        model: The model the entry names.

    Returns:
        Price: The entry, with blanks read as "unknown".

    Raises:
        PricingError: If the entry is not a mapping, or gives a price that is
            not a positive number. A negative or zero price is a typo rather
            than a bargain, and letting one through would put a wrong number
            into a published table.
    """
    where = f"{path}: {profile}/{model}"
    if entry is None:
        return Price()
    if not isinstance(entry, Mapping):
        raise PricingError(f"{where} should be a mapping of prices, not {type(entry).__name__}")

    return Price(
        input_usd_per_mtok=_rate_from(entry.get("input_usd_per_mtok"), where, "input"),
        output_usd_per_mtok=_rate_from(entry.get("output_usd_per_mtok"), where, "output"),
        checked=str(entry.get("checked") or ""),
        note=str(entry.get("note") or ""),
    )


def _rate_from(value: Any, where: str, side: str) -> float | None:
    """Read one half of a price.

    Args:
        value: What the table says, which may be blank.
        where: File and entry, for the error message.
        side: ``input`` or ``output``, for the error message.

    Returns:
        float | None: The rate, or None for a blank -- which means unknown,
        and is the table's way of saying "nobody has filled this in".

    Raises:
        PricingError: If the value is not a positive number.
    """
    if value is None or value == "":
        return None
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise PricingError(f"{where}: {side} price should be a number, not {value!r}")
    if value <= 0:
        raise PricingError(
            f"{where}: {side} price is {value}, which is not a price. Leave it blank to record "
            "the cost as unknown; zero would claim the calls were free"
        )
    return float(value)
