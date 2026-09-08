"""Driving one agent step in a test, the way a run drives one.

Two test modules put actions through the executor: one about what each action
does to a page, one about the surprises a page springs while it happens. Both
need the same three moves -- look at the page, find the number a thing was
given, execute one action against it -- and both would otherwise carry their
own copy, which is how two files end up disagreeing about what a step is.

The order matters and is the reason these live together: looking at the page is
what writes the marks the executor acts through, so a test that skips it leaves
a click with nothing to find. Doing it here means no test has to remember.
"""

from web_action_agent.actions import parse_action
from web_action_agent.executor import Executor
from web_action_agent.observation import Observation
from web_action_agent.observer import extract_elements


def observe(page):
    """Look at the page the way the agent does, marks and all.

    Args:
        page: The live page.

    Returns:
        Observation: The elements and the address. The text channel is left
        empty; nothing that uses this reads it.
    """
    return Observation(
        url=page.url,
        title=page.title(),
        elements=extract_elements(page),
        text="",
    )


def element_id_named(observation, name):
    """Find the number the observation gave to the element with a given name.

    Args:
        observation: The observation to look in.
        name: The element's accessible name.

    Returns:
        int: Its number.

    Raises:
        AssertionError: If the page offers no such element, which would make
            the rest of the test assert nothing.
    """
    for element in observation.elements:
        if element.name == name:
            return element.id
    raise AssertionError(f"no element named {name!r} in {[e.name for e in observation.elements]}")


def act(target, proposal, *, observation=None, **kwargs):
    """Execute one proposed action against a page or a task session.

    Args:
        target: The task's session, or a bare page.
        proposal: The action as plain data, parsed here so that every test
            names its action in the same shape the planner emits.
        observation: The observation to plan against. Taken fresh when not
            given; passed explicitly by tests that change the page in between.
        **kwargs: Passed to :class:`~web_action_agent.executor.Executor`, for
            the tests that need a short timeout.

    Returns:
        ExecutionResult: What the browser did.
    """
    page = getattr(target, "page", target)
    if observation is None:
        observation = observe(page)
    return Executor(target, **kwargs).execute(parse_action(proposal), observation=observation)
