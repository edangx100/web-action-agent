"""When a screenshot is worth what it costs, and why that went the way it did.

A screenshot is the costliest channel an observation has, though not for the
reason people assume. At this project's viewport an image is worth roughly
1,100-1,400 tokens, which is comparable to the text channels rather than many
times them. What makes it expensive is everything around the tokens: it routes
the step to a different and pricier vision model, it is slower, and one of the
shipped vision routes cannot constrain decoding to a schema at all, so those
steps also buy extra validator retries. Paying all of that on every step would
buy very little, because most steps are ordinary — a search box and a submit
button are perfectly legible as text, and nothing is clearer for having been
drawn.

So the screenshot is the *secondary* channel, sent only when the text channels
look unreliable. This module is that judgement, and nothing else: it takes the
facts of the step and returns a decision. It does not capture the screenshot, it
does not annotate it, and it does not send it. Keeping the rule separate from the
machinery is what lets the rule be tested with no browser and no model.

**Two signals of a weak DOM.**

1. *Too few interactive elements.* An extraction that finds two elements on a
   page a person would call busy is usually not a sparse page — it is a page
   built in a way the extractor reads badly: a canvas, a custom widget with no
   accessible role, content the viewport filter cut away. The element count is
   therefore a cheap proxy for "the text channel may be missing things", and it
   is available before anything is sent to the model, which is what makes it
   worth acting on.

2. *Repeated planner failures on DOM-only input.* The first signal is a guess
   made in advance; this one is evidence after the fact. If the planner has
   failed twice running while looking at text alone, whatever it needs to see
   is plausibly not in the text, and an image is worth trying. Failures on a
   step that already carried a screenshot do not count toward this — the run
   had already escalated, so those say nothing about whether text was enough.

**One signal that overrules both.** If the configured profile serves no vision
model, no screenshot is ever sent, whatever the two rules above would say. That
is not an optimisation but the shape of a guarantee: the agent has to run end to
end on text alone, so the possibility of a screenshot may never become a
requirement for one. Deciding it here, in the same place as the cost rules,
means there is exactly one function in the agent that can answer "is an image
going in this prompt", and it says no before it considers anything else.

**Every decision is recorded, including the boring ones.** :class:`GateDecision`
is returned for every step, gated in or out, and carries the trigger that
decided it and the counts it was decided from. A run whose cost came out high
should be answerable from its own trace — which step sent an image, and on what
grounds — without re-running anything or reasoning backwards from a total.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .config import Config

#: Gated **out**: the profile serves no vision model, so there is no image
#: channel to gate in the first place. Checked before the cost rules and
#: overrules them both.
TRIGGER_NO_VISION_ROUTE = "no_vision_route"

#: Gated **in**: fewer interactive elements were extracted than the configured
#: floor, so the element list is suspected of under-reporting the page.
TRIGGER_FEW_ELEMENTS = "few_elements"

#: Gated **in**: the planner has failed this many steps in a row on DOM-only
#: input, which is evidence the text channels are missing what it needs.
TRIGGER_PLANNER_FAILURES = "planner_failures"

#: Gated **out**: neither rule fired. The ordinary outcome, and the reason the
#: cost figure stays where it does.
TRIGGER_DOM_SUFFICIENT = "dom_sufficient"


@dataclass(frozen=True)
class GateDecision:
    """One step's screenshot decision, with the grounds for it.

    Frozen, like the observation it accompanies: it is the record of a choice
    already made, and the run's trace is the only thing that reads it back.

    Attributes:
        send_screenshot: Whether this step's observation carries an image.
        trigger: Which rule decided it — one of :data:`TRIGGER_NO_VISION_ROUTE`,
            :data:`TRIGGER_FEW_ELEMENTS`, :data:`TRIGGER_PLANNER_FAILURES` or
            :data:`TRIGGER_DOM_SUFFICIENT`. A short stable token rather than
            prose, so a hundred runs can be counted by trigger.
        detail: The same decision as one readable sentence, naming the numbers
            it turned on. This is what a person reads in a trace; ``trigger``
            is what a script groups by.
        element_count: Interactive elements extracted for this step.
        dom_failures: Consecutive DOM-only planner failures at this step, before
            this step's own outcome is known.
    """

    send_screenshot: bool
    trigger: str
    detail: str
    element_count: int
    dom_failures: int

    def summary(self) -> str:
        """Render this decision as the single line the step log records.

        Returns:
            str: For example ``screenshot: yes (few_elements) - 2 elements
            extracted, below the floor of 5``.
        """
        answer = "yes" if self.send_screenshot else "no"
        return f"screenshot: {answer} ({self.trigger}) - {self.detail}"

    def to_dict(self) -> dict[str, Any]:
        """Render this decision as plain data, for the run's trace.

        Returns:
            dict[str, Any]: Every field, all of them JSON-safe already.
        """
        return {
            "send_screenshot": self.send_screenshot,
            "trigger": self.trigger,
            "detail": self.detail,
            "element_count": self.element_count,
            "dom_failures": self.dom_failures,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> GateDecision:
        """Rebuild one decision from the plain data of a trace.

        The inverse of :meth:`to_dict`. Every field is required: the whole
        point of recording a decision is that a reader can see the grounds, and
        a decision rebuilt with its counts guessed at would be evidence of
        nothing.

        Args:
            data: One decision, as written by :meth:`to_dict`.

        Returns:
            GateDecision: The same value the trace was written from.

        Raises:
            KeyError: If any field is missing.
        """
        return cls(
            send_screenshot=data["send_screenshot"],
            trigger=data["trigger"],
            detail=data["detail"],
            element_count=data["element_count"],
            dom_failures=data["dom_failures"],
        )


class ScreenshotGate:
    """The cost rule for the screenshot channel, across one run.

    One gate is built per run and asked once per step. It is stateful for a
    single reason: "failures in a row" is a fact about the run so far, not about
    the current step, and it has to live somewhere. Keeping it here rather than
    in the controller means the whole rule — both signals, the streak, and the
    vision override — is in one readable place, and can be exercised without a
    loop around it.

    Typical use, once per step::

        decision = gate.decide(element_count=len(elements))
        if decision.send_screenshot:
            ...  # attach the annotated screenshot to the observation
        trace.record(decision)
        ...  # ask the planner
        gate.record_planner_outcome(succeeded=ok, had_screenshot=decision.send_screenshot)
    """

    def __init__(
        self,
        *,
        min_elements: int,
        max_dom_failures: int,
        supports_vision: bool,
    ) -> None:
        """Build a gate for one run.

        Args:
            min_elements: Extract fewer interactive elements than this and the
                DOM signal counts as weak. ``0`` disables this rule, since no
                count can fall below zero.
            max_dom_failures: Consecutive DOM-only planner failures after which
                vision is gated in. ``0`` gates vision in from the first step,
                which is the honest reading of "tolerate no failures" and the
                way to force the image channel on for a comparison run.
            supports_vision: Whether the run's profile has a vision route. False
                pins every decision to "no screenshot", regardless of the other
                two arguments. Required rather than defaulted: a gate that
                assumed vision it did not have would send an image to a model
                that cannot read one.
        """
        self._min_elements = min_elements
        self._max_dom_failures = max_dom_failures
        self._supports_vision = supports_vision
        # Steps the planner has failed in a row while looking at text alone.
        # Reset by any success; see record_planner_outcome.
        self._dom_failures = 0

    @classmethod
    def from_config(cls, config: Config, *, supports_vision: bool) -> ScreenshotGate:
        """Build a gate from the run's resolved configuration.

        The thresholds are configuration because the right values depend on the
        sites being visited, and the exact numbers are the sort of thing an
        experiment should be able to move without a code change.

        Args:
            config: The resolved run configuration, supplying
                ``screenshot_min_elements`` and ``screenshot_max_dom_failures``.
            supports_vision: Whether the run's profile has a vision route. Comes
                from the profile rather than the config, because it is a fact
                about the endpoint and not a setting anyone chooses.

        Returns:
            ScreenshotGate: A gate for one run, with no failures recorded yet.
        """
        return cls(
            min_elements=config.screenshot_min_elements,
            max_dom_failures=config.screenshot_max_dom_failures,
            supports_vision=supports_vision,
        )

    @property
    def dom_failures(self) -> int:
        """int: Consecutive DOM-only planner failures recorded so far."""
        return self._dom_failures

    @property
    def text_only(self) -> bool:
        """bool: Whether this run can never carry a screenshot.

        True when the profile serves no vision model, in which case
        :meth:`decide` answers "no" for every step whatever the thresholds say.
        Exposed so a caller can skip work it already knows is wasted -- reading
        a PNG off disk and base64-encoding it for a prompt it will not go in --
        and so a trace can state the mode the run was in once, instead of
        leaving a reader to infer it from every step's trigger.

        This is a fact about the endpoint, not a setting. Nothing in the agent
        may turn it off, which is the point: the two text channels have to be
        enough on their own, so the possibility of an image never becomes a
        requirement for one.
        """
        return not self._supports_vision

    def decide(self, *, element_count: int) -> GateDecision:
        """Decide whether this step's observation carries a screenshot.

        The checks run in order of authority, not of cost: the vision override
        first, because it is a guarantee and the other two are heuristics; then
        the failure streak, because evidence from this run beats a guess made
        about the page; then the element count.

        Args:
            element_count: Interactive elements extracted for this step.

        Returns:
            GateDecision: The decision and the grounds for it, ready to be
            written to the trace whichever way it went.
        """
        # The guarantee, first and unconditionally. Nothing below this line can
        # turn the image channel on for a profile that has no vision route.
        if not self._supports_vision:
            return self._decision(
                send=False,
                trigger=TRIGGER_NO_VISION_ROUTE,
                detail="the profile serves no vision model; running on DOM text alone",
                element_count=element_count,
            )

        if self._dom_failures >= self._max_dom_failures:
            return self._decision(
                send=True,
                trigger=TRIGGER_PLANNER_FAILURES,
                detail=(
                    f"{self._dom_failures} DOM-only planner "
                    f"{_failures(self._dom_failures)} in a row, at a limit of "
                    f"{self._max_dom_failures}"
                ),
                element_count=element_count,
            )

        if element_count < self._min_elements:
            return self._decision(
                send=True,
                trigger=TRIGGER_FEW_ELEMENTS,
                detail=(
                    f"{element_count} elements extracted, below the floor of {self._min_elements}"
                ),
                element_count=element_count,
            )

        return self._decision(
            send=False,
            trigger=TRIGGER_DOM_SUFFICIENT,
            detail=(
                f"{element_count} elements extracted, at or above the floor of "
                f"{self._min_elements}, and no DOM-only failure streak"
            ),
            element_count=element_count,
        )

    def record_planner_outcome(self, *, succeeded: bool, had_screenshot: bool) -> None:
        """Tell the gate how the planner did on the step it just decided.

        Success and failure are reported through one method on purpose. Two
        methods would be two things a caller could forget to wire up, and
        forgetting the success half is the worse bug of the pair: the streak
        would only ever grow, and every run would end up paying for images.

        Args:
            succeeded: Whether the planner produced a usable action for the
                step. Any success clears the streak — that is what "consecutive"
                means, and it lets a run that escalated to images drop back to
                text once it is unstuck.
            had_screenshot: Whether that step's observation carried an image. A
                failure with one is left out of the count: the run had already
                escalated, so the failure is not evidence that text was the
                thing missing. The streak is neither raised nor cleared, so the
                next text-only step resumes from where it left off.
        """
        if succeeded:
            self._dom_failures = 0
        elif not had_screenshot:
            self._dom_failures += 1

    def _decision(
        self,
        *,
        send: bool,
        trigger: str,
        detail: str,
        element_count: int,
    ) -> GateDecision:
        """Stamp a decision with the run state it was made against.

        Args:
            send: Whether a screenshot goes to the planner.
            trigger: The rule that decided it.
            detail: That decision as a readable sentence.
            element_count: Elements extracted for this step.

        Returns:
            GateDecision: The finished record.
        """
        return GateDecision(
            send_screenshot=send,
            trigger=trigger,
            detail=detail,
            element_count=element_count,
            dom_failures=self._dom_failures,
        )

    def __repr__(self) -> str:
        """Describe the gate's thresholds and current streak.

        Returns:
            str: The two thresholds, whether vision is available, and how many
            DOM-only failures are outstanding.
        """
        return (
            f"ScreenshotGate(min_elements={self._min_elements}, "
            f"max_dom_failures={self._max_dom_failures}, "
            f"supports_vision={self._supports_vision}, "
            f"dom_failures={self._dom_failures})"
        )


def _failures(count: int) -> str:
    """Pluralise "failure" for a detail line.

    A trace is read by people, and "1 DOM-only planner failures" is the kind of
    small wrongness that makes a reader trust the rest of the line less.

    Args:
        count: How many failures the line is about.

    Returns:
        str: ``"failure"`` for exactly one, ``"failures"`` otherwise.
    """
    return "failure" if count == 1 else "failures"
