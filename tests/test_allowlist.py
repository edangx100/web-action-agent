"""What the navigation gate admits, and — mostly — what it refuses.

The allowlist is a safety boundary, so these tests are weighted towards the
refusals. A gate that is too tight fails loudly during a run and gets fixed; a
gate that is too loose looks exactly like a working one until the day it
matters. Every way the boundary could be drawn one label too far out is pinned
here: the sibling registration under a multi-label suffix, the lookalike that
merely contains the allowed name, the homograph that differs only in encoding,
and the URL dressed up to read as an allowed host.

The registrable-domain cases are table-driven because the interesting content
is the table itself. ``nlb.gov.sg`` and ``straitstimes.com`` are one rule to
the code and two entirely different rules to a reader, and the table is where
that distinction is visible.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from web_action_agent.allowlist import (
    MALFORMED_URL,
    OFF_ALLOWLIST,
    UNSUPPORTED_SCHEME,
    Allowlist,
    AllowlistError,
    derive_domain,
    host_of,
    is_ip_literal,
    is_public_suffix,
    normalize_host,
    registrable_domain,
)

pytestmark = pytest.mark.unit


# --- The domain boundary itself ---------------------------------------------


@pytest.mark.parametrize(
    ("host", "expected"),
    [
        # A single-label suffix: the familiar case.
        ("straitstimes.com", "straitstimes.com"),
        ("www.straitstimes.com", "straitstimes.com"),
        ("a.b.c.straitstimes.com", "straitstimes.com"),
        ("channelnewsasia.com", "channelnewsasia.com"),
        # A multi-label suffix. The whole reason this is not string slicing:
        # the last two labels of `catalogue.nlb.gov.sg` are `gov.sg`, which is
        # not a site but an ending every Singapore agency registers under.
        ("nlb.gov.sg", "nlb.gov.sg"),
        ("catalogue.nlb.gov.sg", "nlb.gov.sg"),
        ("a.b.nlb.gov.sg", "nlb.gov.sg"),
        ("library.nus.edu.sg", "nus.edu.sg"),
        ("sph.com.sg", "sph.com.sg"),
        ("site.co.jp", "site.co.jp"),
        # The list's private section draws the boundary between tenants of one
        # host, so two blogs on one platform are two registrable domains.
        ("sgfoodie.blogspot.com", "sgfoodie.blogspot.com"),
        ("post.sgfoodie.blogspot.com", "sgfoodie.blogspot.com"),
        # Nothing to compute: a bare suffix, a lone label, an address.
        ("gov.sg", None),
        ("edu.sg", None),
        ("sg", None),
        ("com", None),
        ("blogspot.com", None),
        ("localhost", None),
        ("127.0.0.1", None),
        ("::1", None),
    ],
)
def test_registrable_domain(host: str, expected: str | None) -> None:
    assert registrable_domain(host) == expected


def test_ip_literal_is_not_read_as_a_dotted_name() -> None:
    """An address must leave the suffix list's path before it reaches it.

    Handed ``127.0.0.1`` directly, the list reads the dots as labels and
    answers ``0.1`` — a plausible-looking string that is not a domain at all.
    """
    assert is_ip_literal("127.0.0.1")
    assert is_ip_literal("::1")
    assert not is_ip_literal("straitstimes.com")
    assert registrable_domain("127.0.0.1") is None


@pytest.mark.parametrize(
    ("host", "expected"),
    [
        ("com", True),
        ("sg", True),
        ("gov.sg", True),
        ("edu.sg", True),
        ("blogspot.com", True),
        ("nlb.gov.sg", False),
        ("straitstimes.com", False),
        ("localhost", False),
    ],
)
def test_is_public_suffix(host: str, expected: bool) -> None:
    assert is_public_suffix(host) is expected


@pytest.mark.parametrize(
    ("host", "expected"),
    [
        ("catalogue.nlb.gov.sg", "nlb.gov.sg"),
        ("www.straitstimes.com", "straitstimes.com"),
        # No registrable domain to be had, so the entry narrows to the host
        # itself rather than widening to its ending.
        ("localhost", "localhost"),
        ("127.0.0.1", "127.0.0.1"),
    ],
)
def test_derive_domain(host: str, expected: str) -> None:
    assert derive_domain(host) == expected


def test_derive_domain_refuses_a_public_suffix() -> None:
    """There is no narrow reading of ``gov.sg``, so it is not given one."""
    with pytest.raises(AllowlistError, match="public suffix"):
        derive_domain("gov.sg")


# --- Normalization: one spelling to compare in ------------------------------


@pytest.mark.parametrize(
    ("written", "expected"),
    [
        ("STRAITSTIMES.com", "straitstimes.com"),
        ("straitstimes.com.", "straitstimes.com"),
        ("  StraitsTimes.COM.  ", "straitstimes.com"),
        ("Catalogue.NLB.GOV.SG", "catalogue.nlb.gov.sg"),
        # Unicode and punycode are the same name and must compare equal.
        ("пример.com", "xn--e1afmkfd.com"),
        ("xn--e1afmkfd.com", "xn--e1afmkfd.com"),
    ],
)
def test_normalize_host(written: str, expected: str) -> None:
    assert normalize_host(written) == expected


@pytest.mark.parametrize("written", ["", "   ", ".", "a..b.com"])
def test_normalize_host_refuses_what_it_cannot_represent(written: str) -> None:
    with pytest.raises(AllowlistError):
        normalize_host(written)


def test_host_comes_from_the_authority_not_the_userinfo() -> None:
    """``https://straitstimes.com@evil.example/`` contacts the second host.

    The classic disguise: a URL that reads as an allowed site to a human
    skimming it, and resolves to somewhere else in the browser. The gate must
    agree with the browser, not with the reader.
    """
    disguised = "https://www.straitstimes.com@evil.example/story"
    assert host_of(disguised) == "evil.example"
    assert not Allowlist.derive("https://www.straitstimes.com/").allows(disguised)


@pytest.mark.parametrize("url", ["/relative/path", "https://", "not a url", "http://[oops"])
def test_host_of_refuses_urls_with_no_host(url: str) -> None:
    with pytest.raises(AllowlistError):
        host_of(url)


# --- Derivation from the three sources --------------------------------------


def test_start_url_contributes_its_registrable_domain() -> None:
    assert Allowlist.derive("https://catalogue.nlb.gov.sg/search").entries == ("nlb.gov.sg",)


def test_config_and_task_domains_are_added() -> None:
    allowlist = Allowlist.derive(
        "https://www.straitstimes.com/singapore",
        config_extra=["channelnewsasia.com"],
        task_extra=["mothership.sg"],
    )
    assert allowlist.entries == ("channelnewsasia.com", "mothership.sg", "straitstimes.com")


def test_entries_are_normalized_deduplicated_and_ordered() -> None:
    """Two runs configured the same way produce the same list, spelled once.

    The rendered list reaches the trace and the planner's refusal message, so
    an incidental ordering would show up as a difference between two runs that
    are not different.
    """
    allowlist = Allowlist.derive(
        "https://StraitsTimes.COM/",
        config_extra=["STRAITSTIMES.com.", "mothership.sg", "https://channelnewsasia.com/asia"],
        task_extra=["mothership.sg"],
    )
    assert allowlist.entries == ("channelnewsasia.com", "mothership.sg", "straitstimes.com")


def test_a_start_url_on_a_public_suffix_is_refused_up_front() -> None:
    """Better a run that will not start than one allowed every agency at once.

    ``gov.sg`` is not a website. Every Singapore government body registers its
    own name beneath it, so an allowlist entry of ``gov.sg`` would admit all of
    them — which is the opposite of what someone typing it would mean.
    """
    with pytest.raises(AllowlistError, match="public suffix"):
        Allowlist.derive("https://gov.sg/")


@pytest.mark.parametrize("entry", ["gov.sg", "com.sg", "edu.sg", "com", "", "   ", "https://"])
def test_an_unusable_extra_entry_is_refused_up_front(entry: str) -> None:
    with pytest.raises(AllowlistError):
        Allowlist.derive("https://straitstimes.com/", config_extra=[entry])


def test_the_resolved_allowlist_is_frozen() -> None:
    allowlist = Allowlist.derive("https://straitstimes.com/")
    with pytest.raises(FrozenInstanceError):
        allowlist.entries = ("evil.example",)  # type: ignore[misc]


# --- What the gate admits and refuses ---------------------------------------


#: A run of the library-catalogue task, with two news sites added — one by
#: configuration, one by the task. The start URL sits under a two-label public
#: suffix and the added ones under a one-label suffix, so both shapes of the
#: boundary are exercised by the same list.
ALLOWLIST = Allowlist.derive(
    "https://catalogue.nlb.gov.sg/search",
    config_extra=["straitstimes.com"],
    task_extra=["channelnewsasia.com"],
)


@pytest.mark.parametrize(
    ("url", "why"),
    [
        ("https://catalogue.nlb.gov.sg/search/results", "the start URL's own host"),
        ("https://nlb.gov.sg/", "the registrable domain itself"),
        ("https://www.nlb.gov.sg/about", "a sibling subdomain"),
        ("https://a.b.nlb.gov.sg/deep", "a deeper subdomain"),
        ("http://nlb.gov.sg/", "plain http as well as https"),
        ("https://NLB.GOV.SG./", "case and a trailing dot are the same host"),
        ("https://nlb.gov.sg:8443/", "a port does not change the host"),
        (
            "https://nlb.gov.sg/x?q=https://evil.example#f",
            "another URL in the query is not the host",
        ),
        ("https://straitstimes.com/singapore", "added by configuration"),
        ("https://www.straitstimes.com/singapore", "and its subdomains"),
        ("https://channelnewsasia.com/singapore", "added by the task"),
        ("https://cnalifestyle.channelnewsasia.com/", "and its subdomains too"),
    ],
)
def test_admitted(url: str, why: str) -> None:
    assert ALLOWLIST.allows(url), why


@pytest.mark.parametrize(
    ("url", "why"),
    [
        # The multi-label suffix case, and the reason this project needs a real
        # list: every Singapore agency registers directly under `gov.sg`, so a
        # neighbour differs from the allowed site by exactly one label and is a
        # completely different organisation.
        ("https://mom.gov.sg/", "another agency under the same multi-label suffix"),
        ("https://iras.gov.sg/", "and another"),
        ("https://nus.edu.sg/", "a neighbour under a sibling multi-label suffix"),
        ("https://gov.sg/", "the suffix itself, which is nobody's website"),
        # Lookalikes that a substring test would wave through.
        ("https://notnlb.gov.sg/", "a longer name ending in the allowed one"),
        ("https://nlb.gov.sg.evil.example/", "the allowed name as a subdomain of a hostile one"),
        ("https://nlb-gov-sg.example/", "the allowed name with the dots changed"),
        ("https://nlb.com.sg/", "the same name under a different Singapore suffix"),
        ("https://straitstimes.sg/", "an allowed brand under a suffix nobody listed"),
        # A homograph is a different registration and must not normalize into
        # the allowed one. The leading character here is Cyrillic.
        ("https://сhannelnewsasia.com/", "Cyrillic homograph of an allowed domain"),
        # Nothing was said about these at all.
        ("https://mothership.sg/", "never listed"),
        ("http://127.0.0.1:8000/", "never listed"),
    ],
)
def test_refused(url: str, why: str) -> None:
    refusal = ALLOWLIST.check(url)
    assert refusal is not None, why
    assert refusal.code == OFF_ALLOWLIST


def test_blog_platform_tenants_are_separate_sites() -> None:
    """Two strangers' blogs on one host are two sites, and the list knows it.

    ``blogspot.com`` is in the suffix list's private section — the part naming
    companies that hand out subdomains to the public. Honouring it draws the
    boundary tighter for free: a run that starts on one person's blog cannot
    wander into another's.
    """
    allowlist = Allowlist.derive("https://sgfoodie.blogspot.com/")
    assert allowlist.entries == ("sgfoodie.blogspot.com",)
    assert allowlist.allows("https://sgfoodie.blogspot.com/2026/08/hawker-centres.html")
    assert not allowlist.allows("https://sgtravel.blogspot.com/")
    assert not allowlist.allows("https://blogspot.com/")


@pytest.mark.parametrize(
    "url",
    [
        "javascript:alert(document.cookie)",
        "data:text/html,<script>x()</script>",
        "file:///etc/passwd",
        "chrome://settings",
        "//nlb.gov.sg/protocol-relative",
    ],
)
def test_only_http_urls_may_be_navigated_to(url: str) -> None:
    """The closed action space is not left resting on this gate being unreachable.

    A ``javascript:`` URL is script execution wearing a navigation's clothes.
    It is refused whatever the allowlist says, including for a host that is on
    it, so that no future caller can reach the browser with one.
    """
    refusal = ALLOWLIST.check(url)
    assert refusal is not None
    assert refusal.code == UNSUPPORTED_SCHEME


@pytest.mark.parametrize("url", ["https://", "http://[oops", "https:///path"])
def test_a_malformed_url_is_a_refusal_not_an_exception(url: str) -> None:
    """During a run, a bad URL is the planner's mistake to correct, not a crash."""
    refusal = ALLOWLIST.check(url)
    assert refusal is not None
    assert refusal.code == MALFORMED_URL


def test_a_refusal_tells_the_planner_what_it_needs_to_retry() -> None:
    """The message is the planner's only feedback, so it names both sides."""
    refusal = ALLOWLIST.check("https://evil.example/x")
    assert refusal is not None
    assert "evil.example" in refusal.message
    assert "nlb.gov.sg" in refusal.message


# --- Local and development hosts --------------------------------------------


def test_a_local_start_url_allows_that_host_and_no_other() -> None:
    """The fixture site is addressed by IP, which has no registrable domain.

    The entry narrows to the address itself. An address has no subdomains, so
    a *name* ending in one — which is a perfectly legal DNS name — is not the
    machine and is refused.
    """
    allowlist = Allowlist.derive("http://127.0.0.1:8123/search.html")
    assert allowlist.entries == ("127.0.0.1",)
    assert allowlist.allows("http://127.0.0.1:8123/results.html")
    assert allowlist.allows("http://127.0.0.1/other")
    assert not allowlist.allows("http://127.0.0.2/")
    assert not allowlist.allows("http://evil.127.0.0.1/")


def test_localhost_is_a_host_not_a_suffix() -> None:
    allowlist = Allowlist.derive("http://localhost:8123/")
    assert allowlist.entries == ("localhost",)
    assert allowlist.allows("http://localhost/page")
    assert not allowlist.allows("https://straitstimes.com/")
