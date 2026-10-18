"""The two limits on a benchmark sweep: money and time.

A sweep makes hundreds of paid AI calls and can run for hours, unattended.
Without a limit, the first sign that something went wrong -- a model that
loops, a price that was higher than expected, a site that makes every run time
out -- is the bill, or the morning after. So a sweep has two ceilings, set in
``config.yaml`` and changeable per command:

* **money** -- ``budget_usd``, $15 by default: what the sweep's AI calls may
  cost in total;
* **time** -- ``budget_hours``, 6 by default: how long the sweep may run.

**Checked before each run.** Before a worker starts a trial, it asks
:meth:`Budget.check`. Once either ceiling has been reached, no new trial
starts. Trials already running are allowed to finish -- stopping one halfway
would waste what it had already spent -- so a sweep can end slightly over the
ceiling, by at most what the trials in progress cost. That is the "stops after
the run that crosses it" behaviour, and it is said plainly rather than hidden.

**Stopping is not losing.** Everything finished stays finished, and the sweep
is resumable as always: raise the ceiling and run the same command again, and
it carries on with the trials that are left.

**What counts as spent.** Every AI call the sweep made, read from the sweep
folder itself rather than kept in memory, so a restart counts what earlier
attempts spent too:

* the agent's calls, from each trial's ``calls.jsonl`` -- including trials
  killed halfway, which cost money even though they produced no result;
* the judge's calls, from the sweep's ``judge_calls.jsonl``, which the harness
  appends to every time it asks the judge.

**Calls with no price.** A call to a model with no entry in ``pricing.yaml``
has an unknown cost. It is not counted as zero -- that would be a guess -- and
it cannot be counted as anything else either, so the money ceiling cannot see
it. Such calls are counted and reported, and :func:`unpriced_routes` lets the
command warn before the sweep starts that the ceiling will be partly blind.
The time ceiling has no such gap: time is always known.

**What counts as elapsed.** The time the sweep has actually been *running*,
added up over every time it was started. Not the time since it was first
started: a sweep stopped on Monday and resumed on Wednesday has not run for two
days. The running total is saved in ``budget.json`` in the sweep folder after
each trial, so a sweep killed without warning loses at most the time since its
last finished trial from the count.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass
from pathlib import Path

from .config import Config
from .cost import PriceList, shipped_prices
from .model_client import CallRecord, Profile
from .trace import COST_LOG

#: The sweep's running total of time spent, saved between invocations.
LEDGER_FILE = "budget.json"

#: Every judge call the sweep made, one JSON record per line, append-only.
JUDGE_LOG = "judge_calls.jsonl"

#: The connection-test calls made before the sweep's trials start, one per
#: route each time the sweep starts. A fraction of a cent, but paid for, so
#: counted.
PREFLIGHT_LOG = "preflight_calls.jsonl"

#: Seconds in an hour, named so the conversions below read as what they are.
SECONDS_PER_HOUR = 3600.0


@dataclass(frozen=True)
class Spend:
    """What a sweep's AI calls have cost so far.

    Attributes:
        known_usd: Dollars for every call that had a price.
        calls: How many calls in total.
        unpriced_calls: How many had no price, and so are not in
            ``known_usd``. The real total is at least ``known_usd``.
    """

    known_usd: float = 0.0
    calls: int = 0
    unpriced_calls: int = 0

    def summary(self) -> str:
        """Render the spend for a message.

        Returns:
            str: e.g. ``$0.0123 over 61 calls`` -- with the unpriced calls
            named where there are any, so a total that is a lower bound never
            reads like an exact one.
        """
        text = f"${self.known_usd:.4f} over {self.calls} calls"
        if self.unpriced_calls:
            text += f" ({self.unpriced_calls} with no price, not counted)"
        return text


def read_calls(path: Path) -> list[CallRecord]:
    """Read every call record in one ``.jsonl`` file.

    Tolerant of a half-written last line: the file may be being written by a
    trial that is still running, and a partial line is a call that has not
    finished being recorded yet, not an error.

    Args:
        path: The file.

    Returns:
        list[CallRecord]: The complete records.
    """
    records: list[CallRecord] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return records
    for line in lines:
        try:
            records.append(CallRecord.from_dict(json.loads(line)))
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            continue
    return records


def sweep_spend(sweep_dir: Path) -> Spend:
    """Add up what a sweep has spent, from the files in its folder.

    Args:
        sweep_dir: The sweep's folder.

    Returns:
        Spend: Every agent call in every trial folder -- finished, running,
        or killed halfway -- plus every judge call in the judge log, and every
        connection-test call made before the trials started.
    """
    files = [*sweep_dir.rglob(COST_LOG), sweep_dir / JUDGE_LOG, sweep_dir / PREFLIGHT_LOG]
    known, calls, unpriced = 0.0, 0, 0
    for path in files:
        for record in read_calls(path):
            calls += 1
            if record.cost_usd is None:
                unpriced += 1
            else:
                known += record.cost_usd
    return Spend(known_usd=known, calls=calls, unpriced_calls=unpriced)


def record_judge_call(sweep_dir: Path, record: CallRecord, *, log: str = JUDGE_LOG) -> None:
    """Add one call to one of the sweep's call logs: the judge's by default.

    Append-only, so a trial that is marked again later -- after a judging
    error, say -- adds its new call rather than replacing the old one. Both
    calls were paid for.

    Args:
        sweep_dir: The sweep's folder.
        record: The call's cost record.
        log: Which log: :data:`JUDGE_LOG`, or :data:`PREFLIGHT_LOG` for a
            connection test.
    """
    with (sweep_dir / log).open("a", encoding="utf-8") as out:
        out.write(json.dumps(asdict(record)) + "\n")


def unpriced_routes(profiles: Iterable[Profile], prices: PriceList | None = None) -> list[str]:
    """List the models a sweep may call that have no price.

    Args:
        profiles: The planner's profile, and the judge's if there is one.
        prices: The price table. Defaults to the shipped one.

    Returns:
        list[str]: ``profile -> model`` for each route without a price. Each
        is a place the money ceiling cannot see.
    """
    table = prices if prices is not None else shipped_prices()
    missing: list[str] = []
    for profile in profiles:
        routes = [profile.text] + ([profile.vision] if profile.vision is not None else [])
        for route in routes:
            entry = table.price_for(profile.name, route.model)
            if entry is None or not entry.known:
                label = f"{profile.name} -> {route.model}"
                if label not in missing:
                    missing.append(label)
    return missing


class Budget:
    """The money and time ceilings for one sweep, checked before each trial.

    Attributes:
        sweep_dir: The sweep's folder, where spending is read from and the
            time total is saved.
        max_usd: The money ceiling, in dollars.
        max_hours: The time ceiling, in hours.
    """

    def __init__(
        self,
        sweep_dir: Path,
        *,
        max_usd: float,
        max_hours: float,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        """Set up the ceilings. Nothing is read until :meth:`start`.

        Args:
            sweep_dir: The sweep's folder.
            max_usd: The money ceiling, in dollars.
            max_hours: The time ceiling, in hours.
            clock: Where the current time comes from, in seconds. A test
                passes one it can move forward by hand.
        """
        self.sweep_dir = sweep_dir
        self.max_usd = max_usd
        self.max_hours = max_hours
        self._clock = clock
        self._earlier_s = 0.0
        self._started: float | None = None

    @classmethod
    def from_config(cls, sweep_dir: Path, config: Config) -> Budget:
        """Build the ceilings a configuration asks for.

        Args:
            sweep_dir: The sweep's folder.
            config: The settings, whose ``budget_usd`` and ``budget_hours``
                are the ceilings.

        Returns:
            Budget: The ceilings.
        """
        return cls(sweep_dir, max_usd=config.budget_usd, max_hours=config.budget_hours)

    def start(self) -> None:
        """Begin timing this invocation, carrying on from earlier ones.

        Reads how long earlier invocations of the same sweep ran, from the
        ledger, so the time ceiling covers the whole sweep and not just the
        latest start.
        """
        try:
            ledger = json.loads((self.sweep_dir / LEDGER_FILE).read_text(encoding="utf-8"))
            self._earlier_s = float(ledger.get("elapsed_s", 0.0))
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            self._earlier_s = 0.0
        self._started = self._clock()

    def elapsed_s(self) -> float:
        """Return how long the sweep has run, over every invocation.

        Returns:
            float: Seconds.
        """
        now = self._clock() - self._started if self._started is not None else 0.0
        return self._earlier_s + now

    def save(self) -> None:
        """Save the running time total, so a restart carries on from it.

        Written to a temporary name and renamed, so a kill mid-write cannot
        leave a broken ledger.
        """
        partial = self.sweep_dir / f"{LEDGER_FILE}.partial"
        partial.write_text(json.dumps({"elapsed_s": self.elapsed_s()}) + "\n", encoding="utf-8")
        partial.replace(self.sweep_dir / LEDGER_FILE)

    def spend(self) -> Spend:
        """Return what the sweep has spent so far.

        Returns:
            Spend: Read fresh from the folder each time, so it includes the
            steps of trials still running.
        """
        return sweep_spend(self.sweep_dir)

    def check(self) -> str | None:
        """Decide whether another trial may start.

        Returns:
            str | None: None if it may. Otherwise one sentence saying which
            ceiling was reached, by how much, and how to carry on.
        """
        spend = self.spend()
        if spend.known_usd >= self.max_usd:
            return (
                f"the money ceiling of ${self.max_usd:.2f} has been reached "
                f"({spend.summary()}). Finished trials are kept: raise the ceiling "
                "with --budget-usd and run the same command to carry on"
            )
        hours = self.elapsed_s() / SECONDS_PER_HOUR
        if hours >= self.max_hours:
            return (
                f"the time ceiling of {self.max_hours:g} hours has been reached "
                f"({hours:.2f} hours run). Finished trials are kept: raise the ceiling "
                "with --budget-hours and run the same command to carry on"
            )
        return None
