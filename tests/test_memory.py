"""What the agent remembers, and why it is not everything.

A language model forgets between calls. Whatever the agent "knows" about the
run so far is exactly what memory writes into the next prompt, so every test
here is really asking one question: *is the prompt the right thing to be
reading?*

Two failures pull in opposite directions, and memory has to avoid both:

* **Forgetting.** Keep only the last few steps and a value the agent found on
  step 1 -- a search term, a price -- is gone by step 8. The agent then does
  the work again, or answers without it.
* **Hoarding.** Keep everything and the prompt grows every step. That costs
  money on every call, and past a point it makes decisions *worse*: the page in
  front of the agent is buried under a transcript of pages it already left.

So the tests come in pairs. Something recorded early must still be there late;
and the prompt on step 200 must be no bigger than on step 30. The task's own
tests are the first two in each half -- N+3 steps showing exactly N verbatim
with a value from step 1 surviving, and a twenty-step run through the planner
whose prompt stops growing.

Nothing here calls a model except the one test that measures prompts through
the real planner, and that one talks to a stub.
"""

from types import SimpleNamespace

import pytest

from web_action_agent.actions import parse_action
from web_action_agent.config import Config
from web_action_agent.memory import (
    DEFAULT_WINDOW,
    MAX_FOLDED_LINES,
    MAX_LINE_CHARS,
    MAX_NOTES,
    MAX_TALLY_ITEMS,
    Memory,
)
from web_action_agent.model_client import ModelClient, load_profile
from web_action_agent.observation import Observation
from web_action_agent.observer import Box, Element
from web_action_agent.planner import MAX_HISTORY_STEPS, Planner, render_history

N = DEFAULT_WINDOW
"""int: The window size the tests reason about."""

TASK = "Find the opening hours of the Hollow Lane branch."
"""str: A task for the prompt tests. Its content never matters."""

PAGE = "http://localhost:8000/search.html"
"""str: Where the stand-in page is."""


def a_page(url=PAGE, *, names=("Home", "Search terms", "Search", "Results")):
    """Build an observation with a few named elements.

    Args:
        url: The page's address.
        names: Accessible names for elements 1, 2, 3, ...

    Returns:
        Observation: The page.
    """
    roles = ("link", "searchbox", "button", "link")
    return Observation(
        url=url,
        title="Search",
        elements=tuple(
            Element(
                id=number,
                tag="a",
                role=roles[(number - 1) % len(roles)],
                name=name,
                text=name,
                box=Box(x=0, y=20 * number, width=80, height=18),
            )
            for number, name in enumerate(names, start=1)
        ),
        text="Search the catalogue.",
    )


def ok(url=PAGE):
    """A step that worked, leaving the page at ``url``.

    Args:
        url: Where the page ended up.

    Returns:
        SimpleNamespace: Shaped like an execution result.
    """
    return SimpleNamespace(status="ok", detail="", url=url)


def failed(status, detail, url=PAGE):
    """A step that did not work.

    Args:
        status: The failure's name.
        detail: Its sentence.
        url: Where the page was afterwards.

    Returns:
        SimpleNamespace: Shaped like an execution result.
    """
    return SimpleNamespace(status=status, detail=detail, url=url)


def act(proposal):
    """Parse an action from the plain data a planner emits.

    Args:
        proposal: The action as a dict.

    Returns:
        Action: The typed action.
    """
    return parse_action(proposal)


def scroll_steps(memory, count):
    """Record a run of uneventful steps.

    Args:
        memory: The memory to record into.
        count: How many.
    """
    for _ in range(count):
        memory.record(
            act({"action": "scroll", "direction": "down"}), observation=a_page(), result=ok()
        )


def a_planner():
    """Build a planner over a stub transport, for rendering prompts.

    Returns:
        Planner: A planner whose model is never actually called by the prompt
        tests; rendering a prompt makes no call.
    """
    client = ModelClient(load_profile("openrouter"), transport=Transport(), env={})
    return Planner(client)


def recent_steps_in(prompt):
    """Pull the verbatim step lines out of a rendered prompt.

    Args:
        prompt: The planner's user message.

    Returns:
        list[str]: The lines under "# Recent steps", or none.
    """
    if "# Recent steps" not in prompt:
        return []
    section = prompt.split("# Recent steps\n", 1)[1].split("\n\n", 1)[0]
    return section.splitlines()


class Transport:
    """A model endpoint that always answers with a scroll, and keeps the prompts.

    Attributes:
        prompts: The user message of every call, in order.
    """

    def __init__(self):
        """Start with nothing asked."""
        self.prompts = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **request):
        """Record the prompt and answer.

        Args:
            **request: The call body.

        Returns:
            SimpleNamespace: A completion shaped like the SDK's.
        """
        self.prompts.append(request["messages"][-1]["content"])
        message = SimpleNamespace(content='{"action": "scroll", "direction": "down"}')
        return SimpleNamespace(
            choices=[SimpleNamespace(message=message, finish_reason="stop")],
            usage=SimpleNamespace(prompt_tokens=900, completion_tokens=20),
        )


# --- The task's own tests ---------------------------------------------------


@pytest.mark.unit
def test_after_n_plus_three_steps_the_prompt_shows_n_steps_and_a_notepad():
    """Exactly the window, word for word, and the rest folded rather than lost."""
    memory = Memory(N)
    scroll_steps(memory, N + 3)

    prompt = a_planner().render_prompt(
        task=TASK,
        observation=a_page(),
        history=memory.history(),
        scratchpad=memory.scratchpad(),
    )

    assert len(recent_steps_in(prompt)) == N
    assert "# Notes so far" in prompt
    assert memory.scratchpad().count("scroll down") == 3, (
        "the three older steps are folded, not dropped"
    )


@pytest.mark.unit
def test_a_value_noted_at_step_one_survives_to_step_n_plus_three():
    """The fact the agent found first is still there after its step has aged out."""
    memory = Memory(N)
    memory.record(act({"action": "scroll", "direction": "down"}), observation=a_page(), result=ok())
    memory.note("opening hours", "09:00 to 17:30")
    scroll_steps(memory, N + 2)

    assert memory.steps_recorded == N + 3
    assert "opening hours: 09:00 to 17:30" in memory.scratchpad()
    assert all("09:00" not in line for line in memory.history()), "it lives in the notepad now"


@pytest.mark.unit
def test_what_was_typed_at_step_one_survives_after_its_step_leaves_the_window():
    """A search term is exactly the kind of value that must not simply disappear."""
    memory = Memory(N)
    memory.record(
        act({"action": "fill", "element_id": 2, "text": "Hollow Lane"}),
        observation=a_page(),
        result=ok(),
    )
    scroll_steps(memory, N + 2)

    assert '"Hollow Lane"' in memory.scratchpad()


# --- What a step says about itself ------------------------------------------


@pytest.mark.unit
def test_the_window_is_numbered_by_real_step_not_by_position():
    """A window starting at "1." on step 14 would tell the model it had just begun."""
    memory = Memory(N)
    scroll_steps(memory, N + 3)

    numbers = [line.split(".", 1)[0] for line in memory.history()]

    assert numbers == [str(n) for n in range(4, N + 4)]


@pytest.mark.unit
def test_a_step_reads_the_same_before_and_after_it_leaves_the_window():
    """Getting older must not change how a step is described."""
    memory = Memory(2)
    memory.record(
        act({"action": "click", "element_id": 3}), observation=a_page(), result=ok(f"{PAGE}?q=x")
    )
    in_window = memory.history()[0]
    scroll_steps(memory, 2)

    assert in_window in memory.scratchpad()
    assert in_window not in memory.history()


@pytest.mark.unit
def test_a_step_records_what_the_element_was_not_just_its_number():
    """`click 4` means nothing once the page is observed again; `button "Search"` does."""
    memory = Memory()

    step = memory.record(
        act({"action": "click", "element_id": 3}), observation=a_page(), result=ok()
    )

    assert step.target == 'button "Search"'
    assert 'click 3 (button "Search")' in memory.history()[0]


@pytest.mark.unit
def test_a_step_that_moved_the_page_says_where_it_went():
    """The most common effect of a step, and the one a reader most needs."""
    memory = Memory()

    memory.record(
        act({"action": "click", "element_id": 4}),
        observation=a_page(),
        result=ok("http://localhost:8000/results.html?q=hollow"),
    )

    assert memory.history()[0].endswith("-> ok, now at localhost:8000/results.html?q=hollow")


@pytest.mark.unit
def test_a_step_that_left_the_page_where_it_was_does_not_repeat_the_address():
    """Twenty steps on one page must not name the page twenty times."""
    memory = Memory()

    memory.record(
        act({"action": "fill", "element_id": 2, "text": "x"}), observation=a_page(), result=ok()
    )

    assert "now at" not in memory.history()[0]


@pytest.mark.unit
def test_a_navigation_that_arrived_where_it_was_sent_does_not_repeat_the_address():
    """Its own line already says where it went; saying it twice costs a URL a step."""
    memory = Memory()
    target = "http://localhost:8000/article.html"

    memory.record(
        act({"action": "navigate", "url": target}), observation=a_page(), result=ok(target)
    )

    assert memory.history()[0] == f"1. navigate {target} -> ok"


@pytest.mark.unit
def test_a_redirected_navigation_says_where_it_actually_ended_up():
    """The one navigation case where the address afterwards is news."""
    memory = Memory()

    memory.record(
        act({"action": "navigate", "url": "http://localhost:8000/old.html"}),
        observation=a_page(),
        result=ok("http://localhost:8000/new.html"),
    )

    assert memory.history()[0].endswith("-> ok, now at localhost:8000/new.html")


@pytest.mark.unit
def test_a_failed_step_records_why():
    """The planner deciding its next move needs to know the last one did not work."""
    memory = Memory()

    memory.record(
        act({"action": "click", "element_id": 3}),
        observation=a_page(),
        result=failed("element_detached", "element 3 is no longer on the page"),
    )

    assert "element_detached: element 3 is no longer on the page" in memory.history()[0]


@pytest.mark.unit
def test_a_step_nobody_reported_on_says_so_rather_than_implying_success():
    """Silence is not "ok"."""
    memory = Memory()

    memory.record(act({"action": "done", "status": "success", "answer": "09:00"}))

    assert memory.history()[0].endswith("-> no result recorded")


@pytest.mark.unit
def test_a_fresh_memory_gives_the_planner_no_empty_headings():
    """A heading with nothing under it invites the model to wonder what it is missing."""
    memory = Memory()

    prompt = a_planner().render_prompt(
        task=TASK, observation=a_page(), history=memory.history(), scratchpad=memory.scratchpad()
    )

    assert memory.history() == ()
    assert memory.scratchpad() == ""
    assert "# Recent steps" not in prompt
    assert "# Notes so far" not in prompt


# --- Notes: the values that must last ---------------------------------------


@pytest.mark.unit
def test_noting_the_same_fact_again_replaces_it():
    """A page count that was 3 and is now 4 is one fact that changed, not two facts."""
    memory = Memory()

    memory.note("pages", "3")
    memory.note("pages", "4")

    assert "pages: 4" in memory.scratchpad()
    assert "pages: 3" not in memory.scratchpad()


@pytest.mark.unit
def test_notes_outlive_any_number_of_steps():
    """Steps age out by count; notes do not."""
    memory = Memory(N)
    memory.note("branch", "Hollow Lane")
    scroll_steps(memory, 200)

    assert "branch: Hollow Lane" in memory.scratchpad()


@pytest.mark.unit
def test_past_the_note_ceiling_the_stalest_note_goes_and_the_notepad_says_so():
    """A bounded notepad must admit what it has let go of."""
    memory = Memory(max_notes=3)
    for key in ("a", "b", "c", "d"):
        memory.note(key, key.upper())

    pad = memory.scratchpad()

    assert "a: A" not in pad
    assert "d: D" in pad
    assert "1 older notes dropped" in pad


@pytest.mark.unit
def test_updating_a_note_keeps_it_from_being_the_first_to_go():
    """Fresh information is the last thing that should be evicted."""
    memory = Memory(max_notes=2)
    memory.note("a", "1")
    memory.note("b", "2")
    memory.note("a", "3")
    memory.note("c", "4")

    pad = memory.scratchpad()

    assert "a: 3" in pad
    assert "b: 2" not in pad


@pytest.mark.unit
def test_a_note_needs_a_key():
    """A note nobody could name is a note nobody could ever update."""
    with pytest.raises(ValueError, match="needs a key"):
        Memory().note("  ", "value")


# --- Folding: the steps that got old ----------------------------------------


@pytest.mark.unit
def test_steps_too_old_to_show_are_counted_rather_than_paraphrased():
    """Counting cannot misstate anything; a summary sentence can."""
    memory = Memory(N)
    scroll_steps(memory, N + MAX_FOLDED_LINES + 4)

    pad = memory.scratchpad()

    assert "steps 1-4, counted: 4 scroll" in pad
    assert pad.count("scroll down") == MAX_FOLDED_LINES, "the rest keep their own lines"


@pytest.mark.unit
def test_the_count_keeps_where_the_agent_went_and_what_it_typed():
    """The two things about an old step most likely to matter later."""
    memory = Memory(1, max_folded=1)
    memory.record(
        act({"action": "fill", "element_id": 2, "text": "Hollow Lane"}),
        observation=a_page(),
        result=ok(),
    )
    memory.record(
        act({"action": "click", "element_id": 3}),
        observation=a_page(),
        result=ok("http://localhost:8000/results.html"),
    )
    scroll_steps(memory, 2)

    tally = memory.scratchpad().split("Earlier steps:\n", 1)[1].splitlines()[0]

    assert tally.startswith("steps 1-2, counted: 1 fill, 1 click")
    assert "visited localhost:8000/results.html" in tally
    assert 'typed "Hollow Lane"' in tally


@pytest.mark.unit
def test_when_the_count_must_be_cut_it_is_the_addresses_that_go():
    """The planner can rediscover a page by looking; it cannot rediscover a value.

    Five long addresses are enough to push the count past its length ceiling.
    What was typed has to survive the cut, so it comes first.
    """
    memory = Memory(1, max_folded=1)
    memory.record(
        act({"action": "fill", "element_id": 2, "text": "Hollow Lane"}),
        observation=a_page(),
        result=ok(),
    )
    for number in range(MAX_TALLY_ITEMS):
        memory.record(
            act({"action": "click", "element_id": 4}),
            observation=a_page(),
            result=ok(
                f"http://localhost:8000/a-rather-long-results-path/page-{number}?query=branch"
            ),
        )
    scroll_steps(memory, 2)

    tally = memory.scratchpad().split("Earlier steps:\n", 1)[1].splitlines()[0]

    assert len(tally) <= MAX_LINE_CHARS
    assert tally.endswith("…"), "this line really was cut"
    assert 'typed "Hollow Lane"' in tally


@pytest.mark.unit
def test_the_count_names_only_recent_values_and_says_how_many_it_left_out():
    """Bounded, and honest about the bound."""
    memory = Memory(1, max_folded=1)
    for number in range(MAX_TALLY_ITEMS + 3):
        memory.record(
            act({"action": "fill", "element_id": 2, "text": f"term {number}"}),
            observation=a_page(),
            result=ok(),
        )
    scroll_steps(memory, 2)

    pad = memory.scratchpad()

    assert '"term 0"' not in pad
    assert f'"term {MAX_TALLY_ITEMS + 2}"' in pad
    assert "(and 3 earlier)" in pad


@pytest.mark.unit
def test_one_enormous_note_cannot_undo_every_other_ceiling():
    """A note's value can come from a page, and a page can hold a paragraph."""
    memory = Memory()

    memory.note("paragraph", "word " * 1000)

    longest = max(len(line) for line in memory.scratchpad().splitlines())
    assert longest <= MAX_LINE_CHARS
    assert memory.scratchpad().rstrip().endswith("…"), "cut visibly, not silently"


# --- Bounded, whatever the run does -----------------------------------------


def a_busy_run_step(memory, number):
    """Record one step of a run that does a bit of everything.

    Every step types something new, reaches a new page, or notes a new fact, so
    nothing stays small by being repeated: if memory grows, this shows it.

    The step number is written into each value at a *fixed width*. Otherwise the
    test's own data would get longer as the numbers gain digits, and a test
    meant to measure memory's growth would be measuring its own.

    Args:
        memory: The memory to record into.
        number: The step number, which decides what the step does.
    """
    tag = f"{number:05d}"
    url = f"http://localhost:8000/results.html?q=branch-{tag}"
    if number % 3 == 1:
        action = act({"action": "fill", "element_id": 2, "text": f"branch number {tag}"})
    elif number % 3 == 2:
        action = act({"action": "click", "element_id": 4})
    else:
        action = act({"action": "navigate", "url": url})
    memory.record(action, observation=a_page(), result=ok(url))
    memory.note(f"fact {tag}", f"value seen on step {tag}")


def a_busy_run(memory, steps):
    """Record a whole busy run.

    Args:
        memory: The memory to record into.
        steps: How many steps.
    """
    for number in range(1, steps + 1):
        a_busy_run_step(memory, number)


@pytest.mark.unit
def test_memory_is_no_bigger_on_step_300_than_on_step_30():
    """A notepad that grew by a line per step would only delay the problem.

    Both runs stop at the same point in the busy run's three-step cycle. That
    matters: a fill line is nearly twice as long as a navigate line, so two runs
    that stopped at different points would differ by *which* steps sit in the
    window -- a difference in content, not in growth -- and the test would be
    measuring the wrong thing.
    """
    thirty, three_hundred = Memory(N), Memory(N)
    a_busy_run(thirty, 30)
    a_busy_run(three_hundred, 300)

    def size(memory):
        return len("\n".join(memory.history())) + len(memory.scratchpad())

    # Allowed to differ by a few characters, and only because memory's own step
    # numbers and counts gain a digit between 30 and 300 -- one character on
    # each of a dozen lines. That growth is logarithmic in the length of the
    # run, not linear: ten times the steps, the same notepad.
    assert size(three_hundred) <= size(thirty) + 30


@pytest.mark.mock_llm
def test_prompt_size_stays_bounded_across_a_twenty_step_run():
    """The task's own test, through the real planner: the prompt stops growing.

    Early steps are allowed to grow the prompt, because memory is filling up:
    the window is full after 5 steps, the folded lines after 11, the notes after
    12, and the counted line reaches its length ceiling a few steps after that.
    From about step 16 every ceiling has been reached -- and from then on the
    prompt stays flat, even though every step still adds a new page, a new value
    or a new note.
    """
    transport = Transport()
    client = ModelClient(load_profile("openrouter"), transport=transport, env={})
    planner = Planner(client)
    memory = Memory(N)

    for number in range(1, 21):
        planner.decide(
            task=TASK,
            observation=a_page(),
            history=memory.history(),
            scratchpad=memory.scratchpad(),
        )
        a_busy_run_step(memory, number)

    sizes = [len(prompt) for prompt in transport.prompts]
    ceiling = (
        len(transport.prompts[0])
        + (N + MAX_NOTES + MAX_FOLDED_LINES + 3) * (MAX_LINE_CHARS + 1)
        + 200  # headings and blank lines between sections
    )

    assert max(sizes) <= ceiling, f"prompt reached {max(sizes)} characters"
    # The prompt for step 20 against the prompt for step 17. Each carries memory
    # of steps ending at the same point in the busy run's three-step cycle, so
    # the window holds the same *kind* of steps and only growth could make one
    # longer than the other. A few characters are digits; a line would not be.
    after_16, after_19 = sizes[16], sizes[19]
    assert after_19 <= after_16 + 10, f"still growing once every ceiling is reached: {sizes}"
    assert sizes[-1] > sizes[0], "and the memory did reach the prompt at all"


# --- Setup ------------------------------------------------------------------


@pytest.mark.unit
def test_a_window_of_zero_is_refused():
    """Not "no memory": a planner that cannot see whether its last click did anything."""
    with pytest.raises(ValueError, match="at least 1"):
        Memory(0)


@pytest.mark.unit
def test_a_window_larger_than_the_planner_shows_is_cut_to_what_it_shows():
    """A window the planner would trim anyway would only look larger."""
    assert Memory(MAX_HISTORY_STEPS + 5).window == MAX_HISTORY_STEPS


@pytest.mark.unit
def test_the_configured_window_is_bounded_by_the_planners_own_limit():
    """Two numbers that must agree are held level by a test, not by a comment."""
    limits = Config.model_fields["history_window"].metadata
    upper = next(limit.le for limit in limits if hasattr(limit, "le"))

    assert upper == MAX_HISTORY_STEPS


@pytest.mark.unit
def test_memory_takes_its_window_from_the_settings():
    """One place reads the setting, so there is one place to change it."""
    assert Memory.from_config(Config(history_window=3)).window == 3


@pytest.mark.unit
def test_what_the_agent_knew_at_a_step_can_be_written_into_the_trace():
    """ "What did it know at step 12?" should be answerable from the folder alone."""
    memory = Memory(2)
    scroll_steps(memory, 3)
    memory.note("branch", "Hollow Lane")

    record = memory.to_dict()

    assert record["steps_recorded"] == 3
    assert [step["number"] for step in record["window"]] == [2, 3]
    assert record["notes"] == {"branch": "Hollow Lane"}
    assert record["scratchpad"] == memory.scratchpad()


@pytest.mark.unit
def test_history_can_be_numbered_from_a_later_step():
    """The planner's renderer, used for a window that does not start at the start."""
    lines = render_history([{"action": "scroll down", "result": "ok"}], start=14)

    assert lines == ("14. scroll down -> ok",)
