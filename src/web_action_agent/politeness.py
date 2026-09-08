"""How this agent behaves towards somebody else's server.

Everything in this module is about being a guest. The agent visits public
pages on machines nobody is paying to serve it, and three habits separate a
tool people tolerate from one they block:

**Say who you are.** The browser announces itself on every request with a
*user agent* string. The tempting move is to copy a real Chrome one, so sites
treat the agent as a person. This one does not: it says
``WebActionAgent/0.0.1`` and gives a link to the project. That costs something
real -- a site that blocks automated visitors will block this -- and it is
still the right trade. A tool that lies about being a person cannot then
complain about being treated as one, and an evaluation whose numbers depend on
impersonation is not measuring the agent.

**Do not hammer.** :class:`Throttle` keeps a minimum gap between visits to the
same site, and it keeps it *across workers*: an evaluation runs several tasks
at once, and four workers each politely waiting a second between their own
visits still add up to four requests a second at the site. The gap is enforced
through a small file, so it holds between processes that know nothing about
each other.

**Read the sign on the door.** :class:`RobotsPolicy` reads ``robots.txt`` --
the file a site uses to tell automated visitors which paths to leave alone --
and a disallowed address is refused by the validator, before the browser is
asked. Refused, not fetched: asking and then discarding the answer is exactly
the thing the file exists to prevent.

**Where each piece is enforced.** The two are deliberately in different places,
because they answer different questions. Whether the agent *may* visit an
address is a policy question, so it belongs with the validator's other four
checks. How *fast* it may visit is a timing question with no opinion about the
address, so it belongs where the navigating happens.

**What this is not.** None of this makes the agent invisible or hard to block,
and none of it is meant to. A site that does not want this traffic can say so
in ``robots.txt``, or by user agent, and be obeyed in both cases.

**Where it stops.** ``robots.txt`` is checked for an address the agent decides
to *navigate* to. Following a link by clicking it is not checked, because the
address is only known once the click has happened -- by which point the fetch
is made and refusing it would be theatre. That is a real gap rather than a
hidden one: an agent told to click through a site can reach a path the site
excludes. Closing it properly means reading each link's address at observation
time, which is a change to what an observation is, not to this module.
"""

from __future__ import annotations

import os
import re
import tempfile
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Iterator
from contextlib import contextmanager, suppress
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import urlsplit
from urllib.robotparser import RobotFileParser

from . import __version__
from .allowlist import AllowlistError, derive_domain, host_of

if TYPE_CHECKING:
    # Imported for the annotation only. Settings read this module for the
    # default user agent, so importing settings back at runtime would be a
    # cycle -- and this module needs the name, not the class.
    from .config import Config

#: What the browser says it is, on every request to every site. Truthful on
#: purpose: the project's name, its version, and where to find it -- which is
#: also where a site owner would go to complain. Configurable, because a fork
#: or a deployment should say what *it* is rather than inherit this.
DEFAULT_USER_AGENT = (
    f"WebActionAgent/{__version__} (+https://github.com/edangx100/web-action-agent)"
)

#: Why a navigation was refused, when ``robots.txt`` is the reason. Sits under
#: the validator's ``blocked_navigation`` code as its finer cause, beside the
#: allowlist's own three: a wrong domain and a disallowed path are both refused
#: navigations, and they are not the same event.
ROBOTS_DISALLOWED = "robots_disallowed"

#: Seconds between visits to one site, when nothing says otherwise.
DEFAULT_DELAY_S = 1.0

#: How long to wait for a ``robots.txt``, in seconds. Short: the file is small,
#: it is fetched once per site per run, and a site too slow to serve it is not
#: a site worth blocking a run over.
ROBOTS_TIMEOUT_S = 5.0

#: Most of a ``robots.txt`` this reads, in bytes. A cap rather than a limit
#: anyone should reach -- these files are a few kilobytes -- so that a
#: misconfigured server streaming megabytes cannot stall a run.
ROBOTS_MAX_BYTES = 256 * 1024

#: How often to re-check a politeness lock held by another worker, in seconds.
LOCK_POLL_S = 0.01

#: When a lock left behind by a worker that died is broken open, in seconds.
#: Long enough that no healthy worker holds one this long -- the work under it
#: is reading and writing one short file -- and short enough that a crash
#: cannot wedge an evaluation.
LOCK_STALE_S = 30.0

#: Characters kept when a site name becomes a filename. Everything else is
#: replaced, so that a host with a colon in it (an IPv6 address, a port) cannot
#: produce a path with surprises in it.
_UNSAFE_IN_FILENAME = re.compile(r"[^a-z0-9._-]")

#: Reads a ``robots.txt`` and returns its text, or None if there is not one to
#: be had. Injectable so the rules can be tested without a server, and so an
#: offline test never depends on a fetch.
Fetch = Callable[[str], str | None]


def site_of(url: str) -> str:
    """Name the site a URL belongs to, for politeness purposes.

    The registrable domain rather than the host, so that a run does not treat
    ``news.example.com`` and ``www.example.com`` as two servers to be hammered
    independently. They are usually one machine, and even where they are not,
    they are one organisation's bandwidth.

    Args:
        url: The address about to be visited.

    Returns:
        str: The site's name -- its registrable domain, or the host itself for
        an address that has none, such as an IP literal or ``localhost``.
        Falls back to the whole URL for something unparseable, which keeps
        every caller of this function total: a URL nobody can name still gets
        a queue of its own rather than an exception.
    """
    with suppress(AllowlistError, ValueError):
        return derive_domain(host_of(url))
    return url


def _origin_of(url: str) -> str | None:
    """Name the server a URL belongs to, for ``robots.txt`` purposes.

    Scheme, host and port together, because ``robots.txt`` belongs to exactly
    that: the rules for ``example.com`` say nothing about ``example.com:8443``
    and nothing about the same host over plain HTTP.

    Args:
        url: The address about to be visited.

    Returns:
        str | None: The origin, e.g. ``https://example.com``, or None for a URL
        with no host to speak of.
    """
    with suppress(ValueError):
        parts = urlsplit(url)
        if parts.scheme and parts.netloc:
            return f"{parts.scheme}://{parts.netloc}"
    return None


class RobotsPolicy:
    """What a site's ``robots.txt`` says this agent may fetch.

    One of these serves a whole run. Each site's file is fetched the first time
    the run considers an address there and remembered afterwards, so a run that
    visits forty pages on one site asks for its rules once.

    **A file that cannot be read allows everything.** A 404 means the site has
    no rules, which is the overwhelmingly common case and unambiguous. A
    timeout or a server error is less clear, and the strict reading -- used by
    crawlers, and written into RFC 9309 -- is to treat it as "stay away
    entirely". This agent does not: it is a tool a person points at a handful
    of pages, not a crawler working through a site unattended, and refusing to
    open a page because an unrelated status endpoint hiccupped would fail a run
    for a reason nobody could act on. The choice is written down here because
    it is a choice, not an oversight.

    Attributes:
        user_agent: The name the rules are matched against. ``robots.txt``
            addresses agents by name, so this must be the same name the browser
            sends -- a policy read as somebody else is not this agent's policy.
    """

    def __init__(
        self,
        *,
        user_agent: str = DEFAULT_USER_AGENT,
        fetch: Fetch | None = None,
        timeout_s: float = ROBOTS_TIMEOUT_S,
    ) -> None:
        """Build the policy for one run.

        Args:
            user_agent: The agent name to match rules against, and the name
                sent when fetching the file itself.
            fetch: How to read a ``robots.txt``. Defaults to an ordinary HTTP
                GET; a test passes its own and touches no network.
            timeout_s: How long the default fetcher waits.
        """
        self.user_agent = user_agent
        self._fetch = fetch or self._http_fetch
        self._timeout_s = timeout_s
        # Origin -> its parsed rules, or None for a site whose file could not
        # be read. Both are answers, and both are worth remembering: without
        # the None, every step of a run would re-request a file that is not
        # there.
        self._rules: dict[str, RobotFileParser | None] = {}

    def allows(self, url: str) -> bool:
        """Say whether the site permits this agent to fetch an address.

        Args:
            url: The address about to be visited.

        Returns:
            bool: True if the site's rules permit it, or if there are no rules.
        """
        return self.refusal(url) is None

    def refusal(self, url: str) -> str | None:
        """Explain why an address may not be fetched, if it may not.

        The sentence is written to be handed to the planner, like every other
        refusal in this agent: it says what was refused, who refused it, and
        what to do instead. "Blocked" on its own tells a model nothing it can
        act on.

        Args:
            url: The address about to be visited.

        Returns:
            str | None: The complaint, or None when the address is allowed.
        """
        rules = self._rules_for(url)
        if rules is None or rules.can_fetch(self.user_agent, url):
            return None
        return (
            f"the site's robots.txt asks automated visitors not to fetch {_path_of(url)}; "
            "this run respects that. Choose a page the site does not exclude"
        )

    def crawl_delay(self, url: str) -> float | None:
        """Read the gap a site asks visitors to leave between requests.

        Args:
            url: An address on the site.

        Only whole seconds are understood, because that is what the standard
        library's parser keeps: a site asking for ``Crawl-delay: 0.5`` is read
        as asking for nothing, and gets this run's own delay instead. That
        errs towards waiting longer, which is the safe direction.

        Returns:
            float | None: The site's requested seconds between requests, or
            None where it asks for nothing this can read.
        """
        rules = self._rules_for(url)
        if rules is None:
            return None
        declared = rules.crawl_delay(self.user_agent)
        with suppress(TypeError, ValueError):
            return float(declared) if declared is not None else None
        return None

    def _rules_for(self, url: str) -> RobotFileParser | None:
        """Get one site's rules, fetching them the first time they are needed.

        Args:
            url: An address on the site.

        Returns:
            RobotFileParser | None: The parsed rules, or None when the site has
            none that could be read.
        """
        origin = _origin_of(url)
        if origin is None:
            return None
        if origin not in self._rules:
            self._rules[origin] = self._read(f"{origin}/robots.txt")
        return self._rules[origin]

    def _read(self, robots_url: str) -> RobotFileParser | None:
        """Fetch and parse one ``robots.txt``.

        Args:
            robots_url: The file's address.

        Returns:
            RobotFileParser | None: The parsed rules, or None if the file could
            not be read or was not a usable ``robots.txt``.
        """
        text = self._fetch(robots_url)
        if text is None:
            return None
        rules = RobotFileParser()
        rules.set_url(robots_url)
        # `parse` reads lines rather than fetching, which keeps the fetching in
        # one replaceable place -- and keeps the parser from opening a URL of
        # its own with a user agent we did not choose.
        rules.parse(text.splitlines())
        return rules

    def _http_fetch(self, robots_url: str) -> str | None:
        """Read a ``robots.txt`` over HTTP, under this run's own name.

        Args:
            robots_url: The file's address.

        Returns:
            str | None: The file's text, or None for any reason it could not be
            had -- missing, refused, unreachable, too slow. See the class
            docstring for why every one of those means "no rules".
        """
        request = urllib.request.Request(robots_url, headers={"User-Agent": self.user_agent})
        try:
            with urllib.request.urlopen(request, timeout=self._timeout_s) as response:
                raw = response.read(ROBOTS_MAX_BYTES)
        except (urllib.error.URLError, OSError, ValueError):
            return None
        return raw.decode("utf-8", errors="replace")


class Throttle:
    """A minimum gap between visits to one site, shared by every worker.

    The naive version of this is a timestamp in memory, and it is wrong in the
    one case that matters. An evaluation runs several tasks at once; four
    workers each leaving a second between their own visits still put four
    requests a second on one site, which is the behaviour the delay exists to
    prevent. So the gap is kept in a file per site, and any worker on the
    machine honours it.

    **Reserve, then sleep.** A worker takes the lock, reads when the next visit
    may happen, writes back the one *after* that, and releases the lock before
    waiting. Two workers arriving together therefore leave with different
    slots -- now, and now plus the delay -- rather than both waiting for the
    same moment and then both going at once. Holding the lock while sleeping
    would work too, and would serialize the workers on the lock instead of on
    the site.

    Attributes:
        delay_s: Seconds between visits to one site.
        directory: Where the per-site files live.
    """

    def __init__(
        self,
        delay_s: float = DEFAULT_DELAY_S,
        *,
        directory: Path | None = None,
        robots: RobotsPolicy | None = None,
    ) -> None:
        """Build the throttle.

        Args:
            delay_s: Seconds to leave between visits to one site. Zero turns
                the throttle off, which is what a test that is not testing
                politeness wants.
            directory: Where to keep the per-site files. Defaults to a fixed
                folder in the system temporary directory, which is what makes
                separate worker processes -- and separate runs on one machine
                -- honour each other's visits.
            robots: The run's robots policy, consulted for a site's own
                requested delay. A site asking for longer gets longer; a site
                asking for less is ignored, because this delay is also the
                project's own floor.
        """
        self.delay_s = delay_s
        self.directory = directory or Path(tempfile.gettempdir()) / "web-action-agent-politeness"
        self._robots = robots
        self.directory.mkdir(parents=True, exist_ok=True)

    def wait(self, url: str) -> float:
        """Wait, if need be, until this run may visit the URL's site.

        Args:
            url: The address about to be visited.

        Returns:
            float: Seconds actually waited. Zero when the site has not been
            visited recently, which is the common case -- the agent spends
            seconds per step thinking, and a site is rarely visited twice in
            quick succession by one worker.
        """
        delay = self._delay_for(url)
        if delay <= 0:
            return 0.0

        slot = self._reserve(site_of(url), delay)
        pause = slot - time.time()
        if pause <= 0:
            return 0.0
        time.sleep(pause)
        return pause

    def _delay_for(self, url: str) -> float:
        """Decide how long a gap this site should get.

        Args:
            url: The address about to be visited.

        Returns:
            float: The configured delay, or the site's own requested delay
            where that is longer.
        """
        if self._robots is None:
            return self.delay_s
        declared = self._robots.crawl_delay(url)
        return max(self.delay_s, declared) if declared is not None else self.delay_s

    def _reserve(self, site: str, delay: float) -> float:
        """Claim the next moment this run may visit a site.

        Args:
            site: The site's name, from :func:`site_of`.
            delay: Seconds that must separate visits.

        Returns:
            float: The wall-clock time at which the caller may go, as a Unix
            timestamp. Wall clock rather than a process clock on purpose: the
            whole point is that separate processes compare the same numbers.
        """
        path = self.directory / f"{_filename_for(site)}.slot"
        with self._lock(path):
            earliest = _read_timestamp(path)
            now = time.time()
            # The caller goes at the later of "now" and "when the site is next
            # free", and the slot after that is written back before the lock is
            # released -- so the next worker in queues behind this visit rather
            # than beside it.
            slot = max(now, earliest)
            path.write_text(f"{slot + delay}\n", encoding="utf-8")
            return slot

    @contextmanager
    def _lock(self, path: Path) -> Iterator[None]:
        """Hold a lock on one site's file, across processes.

        Created with ``O_EXCL``, which either creates the file or fails --
        atomically, and on every platform Python runs on. The alternatives
        (``fcntl``, a third-party lock library) are either POSIX-only or a
        dependency, and the work being protected here is reading and writing
        one short line.

        Args:
            path: The site's slot file. The lock sits beside it.

        Yields:
            None: With the lock held.
        """
        lock_path = path.with_suffix(".lock")
        while True:
            try:
                descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                break
            except FileExistsError:
                # Either another worker is mid-reservation, or one died holding
                # the lock. Waiting forever on the second case would wedge an
                # evaluation, so a lock nobody could still be holding is broken.
                if _is_stale(lock_path):
                    with suppress(OSError):
                        lock_path.unlink()
                time.sleep(LOCK_POLL_S)
        try:
            yield
        finally:
            os.close(descriptor)
            with suppress(OSError):
                lock_path.unlink()


def politeness_for_run(config: Config) -> tuple[Throttle, RobotsPolicy | None]:
    """Build a run's politeness from its settings.

    One function so that the two pieces are wired together the same way
    everywhere: the throttle is given the robots policy, so a site's own
    requested delay is honoured without every caller remembering to ask.

    Args:
        config: The resolved configuration for this run.

    Returns:
        tuple[Throttle, RobotsPolicy | None]: The throttle to wait on before
        navigating, and the policy for the validator to check addresses
        against -- None when the run is configured not to read ``robots.txt``,
        which is a decision the person running it has to make explicitly.
    """
    robots = RobotsPolicy(user_agent=config.user_agent) if config.respect_robots else None
    return Throttle(config.domain_delay_s, robots=robots), robots


def _path_of(url: str) -> str:
    """Render the part of a URL that ``robots.txt`` rules talk about.

    Args:
        url: The address.

    Returns:
        str: Its path, with any query attached, e.g. ``/search?q=rust``. Just
        ``/`` for an address with no path, which is what a site's rules call
        its front page.
    """
    with suppress(ValueError):
        parts = urlsplit(url)
        path = parts.path or "/"
        return f"{path}?{parts.query}" if parts.query else path
    return url


def _filename_for(site: str) -> str:
    """Turn a site's name into a filename that cannot surprise anyone.

    Args:
        site: The site's name, which comes from a URL and is therefore not
            something this code chose.

    Returns:
        str: The name with anything but lowercase letters, digits, dots,
        dashes and underscores replaced, so it cannot escape the politeness
        directory or collide with the lock files beside it.
    """
    return _UNSAFE_IN_FILENAME.sub("_", site.lower())[:120] or "site"


def _read_timestamp(path: Path) -> float:
    """Read a slot file, treating anything unreadable as "no wait needed".

    Args:
        path: The slot file.

    Returns:
        float: The timestamp it holds, or 0.0 when the file is missing, empty,
        or corrupt. A politeness file nobody can read should slow a run down,
        not stop it.
    """
    with suppress(OSError, ValueError):
        return float(path.read_text(encoding="utf-8").strip())
    return 0.0


def _is_stale(lock_path: Path) -> bool:
    """Say whether a lock is old enough to have been abandoned.

    Args:
        lock_path: The lock file.

    Returns:
        bool: True if it was created longer ago than :data:`LOCK_STALE_S`.
        False if it has since vanished, which means the worker holding it
        finished and there is nothing to break.
    """
    with suppress(OSError):
        return time.time() - lock_path.stat().st_mtime > LOCK_STALE_S
    return False
