"""Which sites one run may visit, and the boundary that decides it.

The agent is given a task and a start URL, and from then on it chooses its own
navigation. That is the point of it, and it is also the risk: a planner that
has misread the page can propose any URL at all, and a page under someone
else's control can suggest one. So navigation is gated. Every ``navigate``
passes through here before the browser is asked, and a URL outside the run's
allowlist is refused and handed back to the planner as a correctable error —
never a crash, and never a request that leaves the machine.

**Where the list comes from.** Three sources, resolved once at the start of a
run and then fixed:

* the *registrable domain* of the start URL, always;
* domains listed in the configuration file;
* extra domains declared by the task itself, for the rare task that genuinely
  spans two sites.

Nothing widens the list mid-run. A run's reachable surface is decided before
its first step, which is what makes it a property of the run rather than of
what the run happened to encounter.

**Why the registrable domain, and not the host.** The host alone is too
narrow — a run that starts on ``www.straitstimes.com`` and is refused
``search.straitstimes.com`` is useless. The domain suffix alone is far too
wide. The registrable domain is the boundary that matches what a person means
by "this website": the part someone actually bought.

Getting it right is not string manipulation, because how much of the ending is
*public* varies and follows no pattern. ``catalogue.nlb.gov.sg`` belongs to
``nlb.gov.sg``, since ``gov.sg`` is a public suffix beneath which every
Singapore government agency registers separately; a naive "last two labels"
rule yields ``gov.sg`` and quietly admits all of them. The same rule applied to
``www.straitstimes.com`` yields ``straitstimes.com`` and happens to be right.
There is no rule, only a list — the Public Suffix List, which this module
consults from a packaged copy rather than guessing at.

The same list draws the boundary tighter where a site hosts strangers:
``sgfoodie.blogspot.com`` and ``sgtravel.blogspot.com`` are separate
registrable domains, so a run that starts on one may not wander into the
other.

**Matching.** An allowlist entry admits itself and its subdomains, and nothing
else. That is deliberately not a substring test: ``straitstimes.com.evil.net``
and ``notstraitstimes.com`` both contain the entry and both are refused, which
is the lookalike case this gate exists for. Comparison happens on normalized hosts —
lowercased, trailing dot removed, internationalized names converted to their
ASCII form — so that a Unicode homograph cannot equal an ASCII entry by
accident of encoding.

**What is refused before the list is even consulted.** A URL with no host, and
any scheme other than ``http`` or ``https``. The closed action space (no script
execution) is a safety property of the whole design, and ``javascript:`` or
``file:`` reaching :meth:`Allowlist.check` is exactly how that property would
be lost — so it is refused here as well, rather than trusted to be impossible.
"""

from __future__ import annotations

import ipaddress
from collections.abc import Iterable
from dataclasses import dataclass
from urllib.parse import urlsplit

from publicsuffixlist import PublicSuffixList

#: The only schemes the agent may navigate to. Anything else — ``javascript:``,
#: ``data:``, ``file:``, ``chrome:`` — is refused unconditionally, whatever the
#: allowlist says, because none of them is a website.
ALLOWED_SCHEMES = frozenset({"http", "https"})

#: The Public Suffix List, from the packaged snapshot: no network call at
#: import, and the same answers on a laptop and in continuous integration.
#:
#: ``accept_unknown=False`` matters. Left at its default, the library treats
#: any unrecognised final label as a public suffix, which would make
#: ``localhost`` one and put a development start URL on the same footing as
#: ``.com``. With it off, an unknown ending simply has no registrable domain,
#: and :func:`derive_domain` falls back to the literal host — the narrow,
#: obviously-safe answer.
_PSL = PublicSuffixList(accept_unknown=False)

#: Refusal codes. The planner is retried with the message; the caller branches
#: on the code, so a wording change never breaks a caller.
UNSUPPORTED_SCHEME = "unsupported_scheme"
MALFORMED_URL = "malformed_url"
OFF_ALLOWLIST = "off_allowlist"


class AllowlistError(Exception):
    """The allowlist could not be built from the values it was given.

    Distinct from a refusal. A refusal is a normal event during a run — the
    planner proposed somewhere it may not go, and is told so. This is a
    configuration failure at startup: an unusable start URL, or an entry that
    would not restrict anything. It stops the run before its first step, when
    the operator can still fix it.
    """


@dataclass(frozen=True)
class Refusal:
    """Why one URL may not be navigated to.

    Attributes:
        code: One of :data:`UNSUPPORTED_SCHEME`, :data:`MALFORMED_URL`, or
            :data:`OFF_ALLOWLIST`. Stable, and the thing to branch on.
        message: A sentence naming the URL and the reason, written to be shown
            to the planner as the feedback for its bounded retry.
    """

    code: str
    message: str


def normalize_host(host: str) -> str:
    """Reduce a host to the one spelling comparisons are made in.

    Hosts have several spellings that name the same place: case is irrelevant,
    a trailing dot is the explicit root, and an internationalized name has both
    a Unicode and an ASCII (punycode) form. Comparing unnormalized hosts means
    ``StraitsTimes.com`` failing to match ``straitstimes.com``, and — worse — a
    homograph being compared as though it were something else entirely.

    Args:
        host: A host as written, in any of those spellings.

    Returns:
        str: The lowercased, dot-stripped, ASCII form.

    Raises:
        AllowlistError: If ``host`` is empty, has an empty label, or cannot be
            converted to ASCII. A host with no single agreed spelling is
            refused rather than compared in whichever form it arrived in.
    """
    cleaned = host.strip().rstrip(".").lower()
    if not cleaned:
        raise AllowlistError("empty host")
    if any(not label for label in cleaned.split(".")):
        raise AllowlistError(f"host {host!r} has an empty label")
    if cleaned.isascii():
        return cleaned
    try:
        return cleaned.encode("idna").decode("ascii")
    except UnicodeError as error:
        raise AllowlistError(
            f"host {host!r} is not a valid internationalized name: {error}"
        ) from error


def is_ip_literal(host: str) -> bool:
    """Report whether a host is a literal IP address rather than a name.

    Worth its own check because the Public Suffix List answers nonsense for
    one: asked for the registrable domain of ``127.0.0.1`` it returns ``0.1``,
    having read the address as a dotted name. Addresses are therefore taken out
    of its path entirely.

    Args:
        host: A normalized host. IPv6 addresses are expected without the
            brackets a URL wraps them in, which is how :func:`host_of` yields
            them.

    Returns:
        bool: True for an IPv4 or IPv6 literal.
    """
    try:
        ipaddress.ip_address(host)
    except ValueError:
        return False
    return True


def registrable_domain(host: str) -> str | None:
    """Return the part of ``host`` that someone registered.

    Args:
        host: A normalized host.

    Returns:
        str | None: The registrable domain — ``nlb.gov.sg`` for
        ``catalogue.nlb.gov.sg`` — or None where there is none to compute: an
        IP literal, a single-label host such as ``localhost``, a bare public
        suffix such as ``gov.sg``, or a name under an ending the list does not
        know. A None result is not an error; it means the caller must fall back
        to something narrower than a domain, which :func:`derive_domain` does.
    """
    if is_ip_literal(host):
        return None
    return _PSL.privatesuffix(host)


def is_public_suffix(host: str) -> bool:
    """Report whether ``host`` is an ending under which anyone may register.

    Args:
        host: A normalized host.

    Returns:
        bool: True for ``gov.sg``, ``com``, or ``blogspot.com``; False for
        ``nlb.gov.sg`` and for ``localhost``.
    """
    return not is_ip_literal(host) and _PSL.is_public(host)


def host_of(url: str) -> str:
    """Extract the normalized host from a URL, refusing what has none.

    Args:
        url: An absolute URL.

    Returns:
        str: The normalized host.

    Raises:
        AllowlistError: If the URL is unparsable, carries no host, or is
            relative. Note that the host is taken from the parsed authority, so
            a URL like ``https://straitstimes.com@evil.example/`` yields
            ``evil.example`` — the host a browser would contact, not the one it
            is dressed as.
    """
    try:
        parts = urlsplit(url)
        host = parts.hostname
    except ValueError as error:
        raise AllowlistError(f"{url!r} is not a parsable URL: {error}") from error
    if not host:
        raise AllowlistError(f"{url!r} names no host")
    return normalize_host(host)


def derive_domain(host: str) -> str:
    """Turn one host into the allowlist entry that admits its site.

    Args:
        host: A normalized host.

    Returns:
        str: Its registrable domain where it has one, and the host itself
        where it does not — an IP literal or a development host such as
        ``localhost``, both of which name a single machine and are their own
        narrowest sensible boundary.

    Raises:
        AllowlistError: If ``host`` is itself a public suffix. ``gov.sg`` has
            no registrable domain, and admitting it literally would admit every
            site registered beneath it. There is no narrow reading of such an
            entry, so it is rejected rather than interpreted.
    """
    domain = registrable_domain(host)
    if domain is not None:
        return domain
    if is_public_suffix(host):
        raise AllowlistError(
            f"{host!r} is a public suffix, not a site: anyone may register beneath it, "
            "so allowing it would allow every site that has"
        )
    return host


def _entry_from(value: str) -> str:
    """Read one configured allowlist entry.

    Entries are written as bare domains, but a pasted URL is a natural mistake
    and an unambiguous one, so both are accepted.

    Args:
        value: A domain such as ``channelnewsasia.com``, or a full URL.

    Returns:
        str: The normalized entry.

    Raises:
        AllowlistError: If the value has no usable host, or is a public suffix.
    """
    text = value.strip()
    if not text:
        raise AllowlistError("empty allowlist entry")
    host = host_of(text) if "//" in text else normalize_host(text.split("/", 1)[0])
    if is_public_suffix(host):
        raise AllowlistError(
            f"allowlist entry {value!r} is a public suffix: it would admit every site "
            "registered beneath it"
        )
    return host


@dataclass(frozen=True)
class Allowlist:
    """The set of sites one run may navigate to. Built once, then fixed.

    Frozen for the same reason the configuration is: a run whose reachable
    surface changed halfway through is a run whose trace no longer describes
    what it was allowed to do.

    Attributes:
        entries: Normalized hosts, deduplicated and sorted. Each admits itself
            and its subdomains. Sorted rather than kept in source order so that
            two runs configured the same way produce the same list — this
            string ends up in the run's trace and in the message the planner
            sees on refusal.
    """

    entries: tuple[str, ...]

    @classmethod
    def derive(
        cls,
        start_url: str,
        *,
        config_extra: Iterable[str] = (),
        task_extra: Iterable[str] = (),
    ) -> Allowlist:
        """Build the allowlist for a run from its three sources.

        Args:
            start_url: The URL the run opens on. Its registrable domain is
                always allowed — a run cannot be forbidden the page it starts
                on — and no other source is required for the list to be usable.
            config_extra: Domains from the configuration file, applying to
                every run on this machine.
            task_extra: Domains declared by this task, for the rare task that
                legitimately spans two sites.

        Returns:
            Allowlist: The resolved list.

        Raises:
            AllowlistError: If the start URL has no usable host or names a
                public suffix, or if any extra entry is unusable. Every such
                failure is a mistake in the run's own setup, and is raised
                before the browser opens.
        """
        entries = {derive_domain(host_of(start_url))}
        for value in (*config_extra, *task_extra):
            entries.add(_entry_from(value))
        return cls(entries=tuple(sorted(entries)))

    def admits_host(self, host: str) -> bool:
        """Report whether a normalized host is admitted by some entry.

        Args:
            host: A normalized host.

        Returns:
            bool: True if the host is an entry or a subdomain of one. The
            subdomain test is on a label boundary, so ``notstraitstimes.com``
            and ``straitstimes.com.evil.net`` are both outside a
            ``straitstimes.com`` entry.
            An entry that is an IP literal admits only itself: an address has
            no subdomains, and a *name* ending in one is not the machine.
        """
        for entry in self.entries:
            if host == entry:
                return True
            if not is_ip_literal(entry) and host.endswith(f".{entry}"):
                return True
        return False

    def check(self, url: str) -> Refusal | None:
        """Decide whether the run may navigate to ``url``.

        Args:
            url: The URL the planner proposed.

        Returns:
            Refusal | None: None if the navigation is permitted; otherwise the
            reason, worded for the planner to act on. Nothing raises here: a
            proposed URL being unusable is an ordinary event in a run, and the
            answer to it is feedback for the retry, not an end to the run.
        """
        try:
            parts = urlsplit(url)
            scheme = parts.scheme.lower()
        except ValueError as error:
            return Refusal(MALFORMED_URL, f"{url!r} is not a parsable URL: {error}")

        if scheme not in ALLOWED_SCHEMES:
            named = f"scheme {scheme!r}" if scheme else "no scheme"
            return Refusal(
                UNSUPPORTED_SCHEME,
                f"cannot navigate to {url!r}: {named}; only "
                f"{' and '.join(sorted(ALLOWED_SCHEMES))} URLs may be navigated to",
            )

        try:
            host = host_of(url)
        except AllowlistError as error:
            return Refusal(MALFORMED_URL, f"cannot navigate to {url!r}: {error}")

        if not self.admits_host(host):
            return Refusal(
                OFF_ALLOWLIST,
                f"cannot navigate to {url!r}: host {host!r} is outside this run's "
                f"allowlist ({self.describe()}). Stay within the allowed sites, or "
                "finish with what the current page provides",
            )
        return None

    def allows(self, url: str) -> bool:
        """Report whether ``url`` may be navigated to.

        Args:
            url: The URL the planner proposed.

        Returns:
            bool: True if :meth:`check` returns no refusal.
        """
        return self.check(url) is None

    def describe(self) -> str:
        """Render the list for a trace, a log line, or a planner prompt.

        Returns:
            str: The entries, comma-separated, each shown as admitting its own
            subdomains — which is the part a reader otherwise has to be told.
        """
        return ", ".join(f"{entry} (and subdomains)" for entry in self.entries)
