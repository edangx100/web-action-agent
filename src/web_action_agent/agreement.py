"""How often two markers agree: a plain percentage, and Cohen's kappa.

The benchmark's marks come from a machine -- the checks and the AI judge --
and a person labels a sample of the same trials by hand. This module measures
how well the two agree. It knows nothing about files or trials: it takes
pairs of labels, one from each marker, and returns the numbers.

**The plain percentage** is the headline: of the trials both marked, how many
got the same label. It is easy to read and is what other published agent
benchmarks report, so ours can be compared with theirs.

**Why it is not enough on its own.** Suppose the agent passes 90% of its
trials. A judge that says "pass" to everything, without looking, would agree
with a careful human about 90% of the time -- and would have measured
nothing. A percentage gives no credit for that being luck.

**Cohen's kappa** (written κ) corrects for it. It first works out how often
the two would agree *by chance alone*, from how often each one uses each
label, and then asks how much of the remaining room the real agreement
fills::

    observed  = share of trials where the two agree
    by chance = for each label: (share the machine gave it) x (share the human gave it),
                added up over the labels
    kappa     = (observed - by chance) / (1 - by chance)

So κ = 1 is perfect agreement, κ = 0 is no better than chance, and below 0
is worse than chance. With 90% "pass" from both sides, chance alone gives
about 82% agreement, so a 90% raw figure is only κ ≈ 0.44.

**When kappa cannot be worked out.** If both sides gave every trial the same
single label -- say, all "pass" -- chance alone would agree every time, and
the formula becomes 0 divided by 0. That is reported as undefined, with the
reason, never as a number: a 100% that measured nothing should not be
dressed up as a perfect score.

**Small samples are shown, not hidden.** Fewer labelled trials than the
configured minimum (20 by default) still get their numbers, alongside how
many trials they rest on and a warning. Hiding a small sample would hide the
fact that the check was barely done.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from fractions import Fraction

#: The fewest labelled trials an agreement figure should rest on before it is
#: relied on. Below this it is still shown, with a warning.
DEFAULT_MIN_LABELS = 20

#: Plain-word bands for kappa, from Landis and Koch (1977), the scale most
#: papers quote. Each pair is (lowest kappa in the band, its name).
KAPPA_BANDS = (
    (0.81, "almost perfect"),
    (0.61, "substantial"),
    (0.41, "moderate"),
    (0.21, "fair"),
    (0.0, "slight"),
)


def kappa_band(kappa: float) -> str:
    """Name the band a kappa falls in, in plain words.

    Args:
        kappa: A value of Cohen's kappa.

    Returns:
        str: ``almost perfect``, ``substantial``, ``moderate``, ``fair``,
        ``slight``, or ``worse than chance`` for a negative value.
    """
    # Rounded first, so 0.605 shows as "0.61" and is named "substantial" too.
    shown = round(kappa, 2)
    for lowest, name in KAPPA_BANDS:
        if shown >= lowest:
            return name
    return "worse than chance"


@dataclass(frozen=True)
class Agreement:
    """How well two markers agreed over some trials.

    Attributes:
        n: Trials both marked.
        agreed: Trials where they gave the same label.
        chance: How often they would agree by chance alone, from 0 to 1, or
            None when there are no trials.
        kappa: Cohen's kappa, or None when it cannot be worked out.
        kappa_note: Why kappa is None, in a sentence. Empty otherwise.
        minimum: The fewest trials the figure should rest on.
    """

    n: int
    agreed: int
    chance: float | None
    kappa: float | None
    kappa_note: str
    minimum: int = DEFAULT_MIN_LABELS

    @property
    def raw(self) -> float | None:
        """float | None: The share of trials agreed on, or None with no trials."""
        return self.agreed / self.n if self.n else None

    @property
    def enough(self) -> bool:
        """bool: Whether there are at least :attr:`minimum` trials."""
        return self.n >= self.minimum

    def render(self, empty: str = "not yet measured (no human labels)") -> str:
        """Write the agreement as one line for a results table.

        Args:
            empty: What to say when there are no trials to measure.

        Returns:
            str: For example ``85.0% (17 of 20 trials) · Cohen's κ 0.66
            (substantial)``. With no trials, says it is not yet measured.
            Below the minimum, the numbers are still given, followed by how
            few trials they rest on.
        """
        if self.n == 0:
            return empty
        text = f"{self.raw:.1%} ({self.agreed} of {self.n} trials) · "
        if self.kappa is None:
            text += f"Cohen's κ undefined: {self.kappa_note}"
        else:
            text += f"Cohen's κ {self.kappa:.2f} ({kappa_band(self.kappa)})"
        if not self.enough:
            text += (
                f" · only {self.n} labelled, fewer than the {self.minimum} needed "
                "to rely on this: treat as a rough guide"
            )
        return text


def measure(pairs: Iterable[tuple[str, str]], minimum: int = DEFAULT_MIN_LABELS) -> Agreement:
    """Measure how well two markers agree.

    Args:
        pairs: One pair per trial: the first marker's label, then the
            second's. Labels are any strings, such as ``pass`` and ``fail``,
            or the kinds of failure.
        minimum: The fewest trials the figure should rest on.

    Returns:
        Agreement: The plain agreement, the agreement expected by chance,
        and Cohen's kappa -- or why kappa cannot be worked out.
    """
    pairs = list(pairs)
    n = len(pairs)
    if n == 0:
        return Agreement(0, 0, None, None, "no trials were labelled", minimum)

    agreed = sum(first == second for first, second in pairs)
    # How many times each marker used each label.
    first_counts = Counter(first for first, _ in pairs)
    second_counts = Counter(second for _, second in pairs)

    # Worked in exact fractions, not floating point, so that "chance is
    # exactly 1" is a reliable test rather than a rounding accident.
    observed = Fraction(agreed, n)
    chance = sum(
        (
            Fraction(first_counts[label], n) * Fraction(second_counts[label], n)
            for label in first_counts.keys() | second_counts.keys()
        ),
        Fraction(0),
    )

    if chance == 1:
        # Both markers used one and the same label for everything: kappa
        # would be 0 / 0. (Chance can only be 1 if both used a single label.)
        [only] = first_counts
        note = (
            f"every trial was labelled {only!r} by both sides, so chance alone "
            "would agree every time"
        )
        return Agreement(n, agreed, 1.0, None, note, minimum)

    kappa = (observed - chance) / (1 - chance)
    return Agreement(n, agreed, float(chance), float(kappa), "", minimum)
