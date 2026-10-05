"""Being a guest on somebody else's server.

Three habits are asserted here, and each of them costs the agent something,
which is the point: politeness that costs nothing is politeness nobody had to
choose.

**The delay has to hold across workers.** The easy version keeps the time of
the last visit in memory, and it is wrong in exactly the case that matters:
four workers each waiting a second between their own visits still put four
requests a second on one site. So the tests below run separate throttle
objects, separate threads, and separate *processes*, and assert they queue
behind one another -- because if they do not, nothing else here matters.

**robots.txt has to be read before the fetch, not after.** A page fetched and
then discarded is precisely what that file exists to prevent, so the refusal
belongs in the validator, one stage before the browser. There is a test that
the file is not even requested for an address the run was never going to
visit.

**The user agent has to be what the site sees.** What a browser claims about
itself in JavaScript and what arrives in an HTTP header are different
questions, so the fixture server records every request it serves and the test
reads it from there.

Almost all of this runs with no network at all: the robots policy takes its
fetcher as an argument, so the rules can be tested without a server. The
handful of tests that do use the fixture site are the ones where the point is
what really goes over the wire.
"""

import multiprocessing
import threading
import time
from pathlib import Path

import pytest

from web_action_agent.actions import parse_action
from web_action_agent.allowlist import Allowlist
from web_action_agent.browser import BrowserProcess, BrowserSettings
from web_action_agent.config import Config
from web_action_agent.observation import Observation
from web_action_agent.politeness import (
    DEFAULT_USER_AGENT,
    ROBOTS_DISALLOWED,
    RobotsPolicy,
    Throttle,
    politeness_for_run,
    site_of,
)
from web_action_agent.validator import BLOCKED_NAVIGATION, Validator

SITE = "https://example.org"
"""str: A site to be polite to. No test here contacts it."""

DELAY_S = 0.3
"""float: The gap the throttle tests use.

Short enough that a dozen tests stay fast, long enough to be measurable
against the scheduling noise of a loaded machine.
"""

ROBOTS_FOR_EVERYONE = "User-agent: *\nDisallow: /private/\nCrawl-delay: 4\n"
"""str: Rules addressed to every automated visitor."""

EMPTY_OBSERVATION = Observation(url=SITE, title="", elements=(), text="")
"""Observation: A page with nothing on it.

Every validator test here is about a URL, and a ``navigate`` names no element,
so the page itself never matters.
"""


class Fetcher:
    """A stand-in for reading robots.txt, that counts what it was asked for.

    Attributes:
        asked: Every URL requested, in order. The count is what two tests
            check: rules must be fetched once per site, and not at all for a
            site the run has already refused on other grounds.
    """

    def __init__(self, text=ROBOTS_FOR_EVERYONE, *, by_url=None):
        """Build the stand-in.

        Args:
            text: What to return for any site not named in ``by_url``. None
                stands for "this site has no robots.txt".
            by_url: Per-URL answers, for tests that need two sites to differ.
        """
        self.asked = []
        self._text = text
        self._by_url = by_url or {}

    def __call__(self, url):
        """Answer one request for a robots.txt.

        Args:
            url: The file's address.

        Returns:
            str | None: The rules, or None when this site has none.
        """
        self.asked.append(url)
        return self._by_url.get(url, self._text)


def a_throttle(directory, *, delay_s=DELAY_S, robots=None):
    """Build a throttle with its own coordination directory.

    Args:
        directory: Where the per-site files live. Two throttles sharing one
            directory are two workers sharing one machine.
        delay_s: Seconds between visits.
        robots: A robots policy, for the site-declared delay.

    Returns:
        Throttle: The throttle.
    """
    return Throttle(delay_s, directory=directory, robots=robots)


def gap_between(throttle, first_url, second_url, *, second=None):
    """Measure how far apart two visits to a site are actually allowed to be.

    The throttle's promise is about the *gap between visits*, not about how
    long any one caller sits waiting: a worker that arrives a moment after
    another correctly waits a moment less. Asserting on the gap therefore
    tests the guarantee rather than an accident of timing -- and it cannot be
    broken by a machine stalling mid-test, since a stall can only make a gap
    wider.

    Args:
        throttle: The throttle taking the first visit.
        first_url: The first address.
        second_url: The second address.
        second: The throttle taking the second visit, for the tests about two
            workers. Defaults to the same one.

    Returns:
        float: Seconds from the first visit being allowed to the second being
        allowed.
    """
    started = time.monotonic()
    throttle.wait(first_url)
    (second or throttle).wait(second_url)
    return time.monotonic() - started


def visit(directory, url, delay_s=DELAY_S):
    """Take one turn at a site, through a throttle of this worker's own.

    Written at module level so that it can be run in a separate *process*,
    which is the case the whole design exists for: a worker built by
    ``multiprocessing`` cannot be handed an object, only a function and some
    strings.

    Args:
        directory: The shared coordination directory, as a string.
        url: The address being visited.
        delay_s: Seconds between visits.

    Returns:
        tuple[float, float]: When the visit was allowed to happen, and how long
        it had to wait first.
    """
    waited = Throttle(delay_s, directory=Path(directory)).wait(url)
    return time.time(), waited


# --- The delay, and who honours it ------------------------------------------


@pytest.mark.unit
def test_the_first_visit_to_a_site_does_not_wait(tmp_path):
    """Politeness is about the gap between visits, not a toll on the first."""
    assert a_throttle(tmp_path).wait(f"{SITE}/a") == 0.0


@pytest.mark.unit
def test_a_second_visit_to_the_same_site_waits(tmp_path):
    """The whole feature, in its simplest form."""
    throttle = a_throttle(tmp_path)

    gap = gap_between(throttle, f"{SITE}/a", f"{SITE}/b")

    assert gap >= DELAY_S * 0.9, "the second visit should have been held back"


@pytest.mark.unit
def test_visits_to_different_sites_do_not_wait_for_each_other(tmp_path):
    """A delay per site, not a delay per agent: the point is not to hammer one host."""
    throttle = a_throttle(tmp_path)
    throttle.wait("https://example.org/a")

    assert throttle.wait("https://example.net/a") == 0.0


@pytest.mark.unit
def test_subdomains_of_one_site_share_its_queue(tmp_path):
    """Usually one machine, and always one organisation's bandwidth."""
    throttle = a_throttle(tmp_path)

    gap = gap_between(throttle, "https://news.example.org/a", "https://www.example.org/b")

    assert gap >= DELAY_S * 0.9


@pytest.mark.unit
def test_a_second_worker_queues_behind_the_first(tmp_path):
    """Two throttle objects, no shared memory: the file is the only link.

    This is the failure the file exists to prevent. Two workers each holding
    their own idea of "when I last visited" would both go at once, and the site
    would see two requests where the run promised one.
    """
    first = a_throttle(tmp_path)
    second = a_throttle(tmp_path)

    gap = gap_between(first, f"{SITE}/a", f"{SITE}/b", second=second)

    assert gap >= DELAY_S * 0.9


@pytest.mark.unit
def test_two_workers_arriving_together_are_served_one_after_the_other(tmp_path):
    """Simultaneous arrivals get different slots, not the same one.

    The interesting half of the design: both workers reserve while holding the
    lock and sleep after releasing it, so they queue on the *site* rather than
    on the lock -- and neither is left waiting for a slot the other already
    took.
    """
    arrivals = []

    def worker():
        """Take one turn, recording when it was allowed to happen."""
        a_throttle(tmp_path).wait(f"{SITE}/a")
        arrivals.append(time.time())

    threads = [threading.Thread(target=worker) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)

    assert len(arrivals) == 2
    assert abs(arrivals[1] - arrivals[0]) >= DELAY_S * 0.9


@pytest.mark.unit
def test_two_worker_processes_queue_behind_each_other(tmp_path):
    """The case the requirement is actually about: separate processes.

    An evaluation runs its workers as processes, which share no memory at all.
    A delay kept anywhere but on disk would be four independent delays wearing
    one name.
    """
    # Spawn rather than fork: the test process has browser machinery and
    # threads in it, and a forked child inherits both.
    context = multiprocessing.get_context("spawn")
    with context.Pool(2) as pool:
        results = pool.starmap(visit, [(str(tmp_path), f"{SITE}/a", DELAY_S)] * 2)

    times = sorted(moment for moment, _ in results)
    assert times[1] - times[0] >= DELAY_S * 0.9, "the site saw two visits a delay apart"
    # One of them had to wait; which one, and for exactly how long, depends on
    # how far apart the two processes started. Asserting a *full* delay here
    # would be asserting that they started at the same instant, which is not
    # something a machine promises -- and it is not what the throttle promises
    # either. The gap above is the guarantee; this is only that coordination
    # happened at all rather than both going straight through.
    assert max(waited for _, waited in results) > 0


@pytest.mark.unit
def test_a_reservation_outlives_the_object_that_made_it(tmp_path):
    """A worker that finishes and exits still counts as having visited."""
    gap = gap_between(a_throttle(tmp_path), f"{SITE}/a", f"{SITE}/b", second=a_throttle(tmp_path))

    assert gap >= DELAY_S * 0.9


@pytest.mark.unit
def test_a_zero_delay_turns_the_throttle_off(tmp_path):
    """A test against a local fixture is not being impolite to anyone."""
    throttle = a_throttle(tmp_path, delay_s=0.0)
    throttle.wait(f"{SITE}/a")

    assert throttle.wait(f"{SITE}/b") == 0.0


@pytest.mark.unit
def test_a_site_asking_for_a_longer_gap_gets_it(tmp_path):
    """robots.txt can name a crawl-delay, and a site asking for more is given more.

    Whole seconds, because that is all the standard library's parser keeps --
    which is why this test waits a real second rather than a fraction of one.
    """
    robots = RobotsPolicy(fetch=Fetcher("User-agent: *\nCrawl-delay: 1\n"))
    throttle = a_throttle(tmp_path, delay_s=0.05, robots=robots)

    gap = gap_between(throttle, f"{SITE}/a", f"{SITE}/b")

    assert gap >= 0.9, "the site's own delay should have replaced the shorter default"


@pytest.mark.unit
def test_a_site_asking_for_a_shorter_gap_is_not_obeyed(tmp_path):
    """The configured delay is a floor, not a suggestion the site can lower."""
    robots = RobotsPolicy(fetch=Fetcher("User-agent: *\nCrawl-delay: 0\n"))
    throttle = a_throttle(tmp_path, robots=robots)

    gap = gap_between(throttle, f"{SITE}/a", f"{SITE}/b")

    assert gap >= DELAY_S * 0.9


@pytest.mark.unit
def test_a_url_nobody_can_parse_still_gets_a_queue(tmp_path):
    """Every path through the throttle has to end in a visit or a wait, not a crash."""
    throttle = a_throttle(tmp_path)

    assert throttle.wait("not a url at all") == 0.0
    assert gap_between(throttle, "not a url at all", "not a url at all") >= DELAY_S * 0.9


@pytest.mark.unit
def test_a_site_name_cannot_escape_the_politeness_directory(tmp_path):
    """The name comes from a URL, so it is not something this code chose."""
    throttle = a_throttle(tmp_path)

    # The first visit to a site never waits, so this records the visit and
    # returns immediately -- the file it writes is what the test is about.
    throttle.wait("https://exam..ple/../../etc/passwd")

    written = [path for path in tmp_path.rglob("*") if path.is_file()]
    assert written, "the visit should have been recorded somewhere"
    for path in written:
        assert path.parent == tmp_path, f"{path} escaped the politeness directory"


# --- What the site says it wants --------------------------------------------


@pytest.mark.unit
def test_an_excluded_path_is_refused():
    """The rule that makes robots.txt worth reading."""
    policy = RobotsPolicy(fetch=Fetcher())

    assert not policy.allows(f"{SITE}/private/report.html")


@pytest.mark.unit
def test_a_path_the_site_permits_is_allowed():
    """Excluding one folder does not exclude a site."""
    policy = RobotsPolicy(fetch=Fetcher())

    assert policy.allows(f"{SITE}/search.html")
    assert policy.refusal(f"{SITE}/search.html") is None


@pytest.mark.unit
def test_the_refusal_says_what_was_refused_and_by_whom():
    """Like every refusal in this agent, it is written to be acted on."""
    policy = RobotsPolicy(fetch=Fetcher())

    complaint = policy.refusal(f"{SITE}/private/report.html?page=2")

    assert "robots.txt" in complaint
    assert "/private/report.html" in complaint
    assert len(complaint) > 40, "a code with no sentence gives a model nothing to change"


@pytest.mark.unit
def test_a_site_with_no_robots_file_allows_everything():
    """The overwhelmingly common case, and unambiguous: no rules means no rules."""
    policy = RobotsPolicy(fetch=Fetcher(None))

    assert policy.allows(f"{SITE}/anything/at/all")


@pytest.mark.unit
def test_rules_are_read_once_per_site():
    """A run visiting forty pages on one site asks for its rules once."""
    fetcher = Fetcher()
    policy = RobotsPolicy(fetch=fetcher)

    for page in range(5):
        policy.allows(f"{SITE}/page-{page}.html")

    assert fetcher.asked == [f"{SITE}/robots.txt"]


@pytest.mark.unit
def test_a_site_with_no_rules_is_not_asked_again():
    """Remembering the absence matters as much as remembering the rules."""
    fetcher = Fetcher(None)
    policy = RobotsPolicy(fetch=fetcher)

    policy.allows(f"{SITE}/a")
    policy.allows(f"{SITE}/b")

    assert len(fetcher.asked) == 1


@pytest.mark.unit
def test_every_site_is_asked_for_its_own_rules():
    """robots.txt belongs to one scheme, host and port -- never to a neighbour."""
    fetcher = Fetcher(
        None,
        by_url={"https://strict.example/robots.txt": "User-agent: *\nDisallow: /\n"},
    )
    policy = RobotsPolicy(fetch=fetcher)

    assert not policy.allows("https://strict.example/a")
    assert policy.allows("https://open.example/a")
    assert len(fetcher.asked) == 2


@pytest.mark.unit
def test_rules_addressed_to_this_agent_by_name_apply():
    """A site can single this agent out, and being named is the point of a truthful name."""
    policy = RobotsPolicy(
        user_agent="WebActionAgent/0.0.1 (+link)",
        fetch=Fetcher("User-agent: WebActionAgent\nDisallow: /reports/\n"),
    )

    assert not policy.allows(f"{SITE}/reports/2026.html")
    assert policy.allows(f"{SITE}/index.html")


@pytest.mark.unit
def test_rules_addressed_to_a_different_agent_do_not_apply():
    """Obeying somebody else's rules would be as wrong as ignoring our own."""
    policy = RobotsPolicy(
        user_agent="WebActionAgent/0.0.1 (+link)",
        fetch=Fetcher("User-agent: SomeOtherBot\nDisallow: /\n"),
    )

    assert policy.allows(f"{SITE}/index.html")


@pytest.mark.unit
def test_an_address_with_no_site_in_it_is_allowed():
    """Nothing to ask, nobody to ask: the allowlist has already refused this shape."""
    fetcher = Fetcher()
    policy = RobotsPolicy(fetch=fetcher)

    assert policy.allows("not a url at all")
    assert fetcher.asked == []


# --- Refused before the fetch, not after ------------------------------------


def a_validator(**kwargs):
    """Build a validator for a run that starts on the test site.

    Args:
        **kwargs: Passed to :class:`~web_action_agent.validator.Validator`.

    Returns:
        Validator: The gate.
    """
    return Validator(Allowlist.derive(SITE), **kwargs)


@pytest.mark.unit
def test_an_excluded_address_is_a_validation_refusal():
    """The browser is never asked, so the page is never fetched."""
    validator = a_validator(robots=RobotsPolicy(fetch=Fetcher()))

    verdict = validator.check(
        {"action": "navigate", "url": f"{SITE}/private/report.html"},
        EMPTY_OBSERVATION,
    )

    assert not verdict.ok
    assert verdict.rejection.code == BLOCKED_NAVIGATION
    assert verdict.rejection.cause == ROBOTS_DISALLOWED
    assert verdict.action is None, "a refused navigation must not hand back an action"


@pytest.mark.unit
def test_a_permitted_address_passes_the_gate():
    """Reading robots.txt must not turn into refusing everything."""
    validator = a_validator(robots=RobotsPolicy(fetch=Fetcher()))

    verdict = validator.check({"action": "navigate", "url": f"{SITE}/search"}, EMPTY_OBSERVATION)

    assert verdict.ok


@pytest.mark.unit
def test_a_site_the_run_may_not_visit_is_never_asked_for_its_rules():
    """Order matters: the allowlist first, because robots.txt is itself a request.

    Fetching a file from a site to find out whether we are allowed to fetch
    from it -- having already decided not to go there -- would be a request
    made for nothing, at the site's expense.
    """
    fetcher = Fetcher()
    validator = a_validator(robots=RobotsPolicy(fetch=fetcher))

    verdict = validator.check(
        {"action": "navigate", "url": "https://elsewhere.example/page"},
        EMPTY_OBSERVATION,
    )

    assert not verdict.ok
    assert verdict.rejection.cause != ROBOTS_DISALLOWED
    assert fetcher.asked == [], "the off-allowlist address should have settled it"


@pytest.mark.unit
def test_actions_that_are_not_navigations_ask_nothing():
    """robots.txt governs what is fetched, and a scroll fetches nothing."""
    fetcher = Fetcher()
    validator = a_validator(robots=RobotsPolicy(fetch=fetcher))

    assert validator.check({"action": "scroll", "direction": "down"}, EMPTY_OBSERVATION).ok
    assert fetcher.asked == []


@pytest.mark.unit
def test_a_run_without_a_robots_policy_still_validates():
    """Reading robots.txt is a setting, and off is a state the gate must survive."""
    verdict = a_validator().check(
        parse_action({"action": "navigate", "url": f"{SITE}/private/report.html"}),
        EMPTY_OBSERVATION,
    )

    assert verdict.ok


# --- Saying who we are ------------------------------------------------------


@pytest.mark.unit
def test_the_default_user_agent_does_not_impersonate_a_person():
    """The one property of this string that is not cosmetic."""
    assert "WebActionAgent" in DEFAULT_USER_AGENT
    assert "Mozilla" not in DEFAULT_USER_AGENT
    assert "Chrome" not in DEFAULT_USER_AGENT
    assert "github.com" in DEFAULT_USER_AGENT, "a site owner needs somewhere to complain"


@pytest.mark.unit
def test_a_run_builds_its_politeness_from_its_settings():
    """One call, so the two pieces are always wired to each other the same way."""
    throttle, robots = politeness_for_run(Config(domain_delay_s=2.5, user_agent="Test/1 (+x)"))

    assert throttle.delay_s == 2.5
    assert robots is not None
    assert robots.user_agent == "Test/1 (+x)"


@pytest.mark.unit
def test_switching_robots_off_is_visible_in_what_is_built():
    """Not reading robots.txt is a decision, and it should look like one."""
    _, robots = politeness_for_run(Config(respect_robots=False))

    assert robots is None


@pytest.mark.unit
def test_the_browser_settings_carry_the_configured_user_agent():
    """The chain from a setting to the browser has one link that is easy to forget."""
    settings = BrowserSettings.from_config(Config(user_agent="Test/1 (+x)"))

    assert settings.user_agent == "Test/1 (+x)"


@pytest.mark.fixture
def test_the_user_agent_reaching_the_server_is_the_configured_one(playwright, fixture_site):
    """What a browser claims in JavaScript and what a server receives can differ.

    So this reads the server's own record of the request rather than asking the
    page about itself.
    """
    agent = "WebActionAgent/test (+politeness check)"
    settings = BrowserSettings.from_config(Config(user_agent=agent))

    with (
        BrowserProcess(settings, playwright=playwright) as browser,
        browser.task_session() as session,
    ):
        fixture_site.requests.clear()
        session.page.goto(fixture_site.url_for("search.html"))

    served = dict(fixture_site.requests)
    assert served["/search.html"] == agent


@pytest.mark.fixture
def test_robots_is_fetched_under_the_same_name(fixture_site):
    """A site that excludes this agent by name must be able to see the name coming."""
    agent = "WebActionAgent/test (+politeness check)"
    fixture_site.requests.clear()

    RobotsPolicy(user_agent=agent).allows(fixture_site.url_for("search.html"))

    assert ("/robots.txt", agent) in fixture_site.requests


@pytest.mark.fixture
def test_the_fixture_sites_own_rules_are_obeyed(fixture_site):
    """The rules are read over HTTP from a real file, not from a stub."""
    policy = RobotsPolicy()

    assert not policy.allows(fixture_site.url_for("ledger.html"))
    assert policy.allows(fixture_site.url_for("search.html"))


@pytest.mark.fixture
def test_a_crawl_delay_in_a_real_file_is_read(fixture_site):
    """The fixture site asks for two seconds; the throttle would honour it."""
    policy = RobotsPolicy()

    assert policy.crawl_delay(fixture_site.url_for("index.html")) == 2.0


@pytest.mark.fixture
def test_site_of_names_a_local_server_by_its_address(fixture_site):
    """A development host is its own site: there is no registrable domain to find."""
    assert site_of(fixture_site.url_for("index.html")) == "127.0.0.1"
