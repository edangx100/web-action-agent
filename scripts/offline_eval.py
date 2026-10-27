"""Run the whole evaluation on this machine alone: no account, no key, no internet.

The real benchmark needs two things a newcomer may not have: an AI key, and
the live Singapore websites. This script swaps both for stand-ins that run on
this computer, and changes nothing else:

* the **websites** become the project's practice site (``tests/fixtures/site``),
  served on ``127.0.0.1``, with its five-task benchmark suite;
* the **AI model** becomes a stand-in that answers in the same format a real
  model service does, also on ``127.0.0.1`` (``tests/fixtures/model_server.py``).

Then it runs the same ``waa eval`` command the real benchmark uses: real
browsers, the real agent, the real checks, the real results tables. If this
passes on a fresh machine, every part of the evaluation except the model and
the websites works there.

With ``--watch-network``, every network call the evaluation makes (the agent
and its browsers alike) is recorded by ``strace``, a standard Linux tool that
watches a program from the outside. The run fails if any call opened a
connection to another machine, sent another machine data, or looked up a
name. A browser's route check, which sends nothing, is counted and allowed.

Usage, from the project folder (a ``pip install -r requirements.txt`` and
``playwright install chromium`` are enough, or the same through ``uv run``)::

    python scripts/offline_eval.py [--trials N] [--workdir DIR] [--watch-network]

Exit status: 0 when every trial passed and every check here held, 1 when
something did not, 2 when the run could not be set up.
"""

from __future__ import annotations

import argparse
import ipaddress
import os
import re
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
"""Path: The project folder."""

# The two stand-in servers live with the tests, which import them the same way.
sys.path.insert(0, str(REPO / "tests"))

from fixtures.model_server import StandInModel
from fixtures.site_server import StaticSite

from web_action_agent.eval_harness import iter_verdicts
from web_action_agent.judge import PASS

SUITE_DIR = REPO / "tests" / "fixtures" / "benchmark"
"""Path: The practice site's benchmark suite: a manifest and its task files."""

#: The made-up address the practice suite's files use for the site. It is
#: swapped for the real local address once the site is running.
PLACEHOLDER_URL = "https://fixture.test/"
PLACEHOLDER_DOMAIN = "domain: fixture.test"

#: Settings removed from the environment the evaluation runs in. Every setting
#: of this project starts with WAA_; the others are names AI services commonly
#: read their keys from. So no key from this machine can reach the run.
DROPPED_PREFIX = "WAA_"
DROPPED_MARKERS = ("API_KEY", "API_TOKEN")

#: The folder, inside the working folder, that the sweep is written to.
SWEEP_NAME = "offline-local-cold"

#: The system calls through which a program reaches another machine: opening
#: a connection, and sending data (with an address, or on a socket already
#: connected to one).
WATCHED_CALLS = "connect,sendto,sendmsg,sendmmsg,write,writev"

#: The calls among those that send data, as opposed to opening a connection.
SENDING_CALLS = ("sendto", "sendmsg", "sendmmsg", "write", "writev")

#: The port for name lookups (DNS). Everything in this run is addressed by
#: number, 127.0.0.1, so a lookup means something tried to find an outside
#: host, even when the lookup itself goes to a helper on this machine.
DNS_PORT = 53

#: The start of a watched call on an internet socket, as ``strace -yy`` writes
#: it: the process number, the call, then the socket and its kind, such as
#: ``1234 connect(18<UDPv6:[285619]>`` or
#: ``1234 sendto(3<TCP:[127.0.0.1:52344->127.0.0.1:44535]>``.
#: Files, pipes and local (UNIX) sockets have other kinds and are skipped:
#: they cannot leave the machine.
INTERNET_CALL = re.compile(r"^\d+ (\w+)\(\d+<(TCP|UDP)(?:v6)?:\[(.*?)\]>")

#: The far end of a connected socket, inside its description:
#: ``127.0.0.1:52344->127.0.0.1:44535`` or ``[::1]:5000->[::1]:6000``.
REMOTE_END = re.compile(r"->\[?([0-9A-Fa-f:.]+?)\]?:(\d+)$")

#: An IPv4 address given to the call: ``sin_port=htons(80), sin_addr=inet_addr("1.2.3.4")``.
IPV4 = re.compile(r'sin_port=htons\((\d+)\), sin_addr=inet_addr\("([^"]+)"\)')

#: An IPv6 address given to the call: ``sin6_port=htons(443), ... inet_pton(AF_INET6, "::1", ...``.
IPV6 = re.compile(r'sin6_port=htons\((\d+)\).*?inet_pton\(AF_INET6, "([^"]+)"')


@dataclass(frozen=True)
class NetworkCall:
    """One call on an internet socket, from the strace log.

    Attributes:
        call: The system call, such as ``connect`` or ``sendto``.
        protocol: ``TCP`` or ``UDP``.
        address: Where it went, such as ``127.0.0.1``.
        port: The port number.
        line: The log line it came from, to show if it is a problem.
    """

    call: str
    protocol: str
    address: str
    port: int
    line: str

    @property
    def on_this_machine(self) -> bool:
        """bool: True when the address is this computer's own (loopback)."""
        address = ipaddress.ip_address(self.address)
        # "::ffff:127.0.0.1" is 127.0.0.1 written the IPv6 way.
        if isinstance(address, ipaddress.IPv6Address) and address.ipv4_mapped:
            address = address.ipv4_mapped
        return address.is_loopback

    @property
    def is_route_check(self) -> bool:
        """bool: True for a UDP ``connect``, which sends nothing at all.

        Connecting a UDP socket only asks this computer's own network code
        which way an address would be reached. No message leaves. Browsers
        use it to see whether IPv6 works: Chromium checks a route to one of
        Google's public DNS addresses each time it starts. Any data later
        sent on such a socket is a separate call, and is checked as one.
        """
        return self.protocol == "UDP" and self.call == "connect"

    @property
    def problem(self) -> str | None:
        """str | None: What is wrong with this call, or None if nothing is."""
        if self.port == DNS_PORT:
            return f"a name lookup ({self.call} to {self.address} port {self.port})"
        if self.on_this_machine or self.is_route_check:
            return None
        doing = "data sent" if self.call in SENDING_CALLS else "a connection opened"
        return f"{doing} to {self.address} port {self.port} ({self.protocol})"


def network_calls_in(log_lines: Iterable[str]) -> list[NetworkCall]:
    """Find every call on an internet socket in an strace log, with where it went.

    Where it went is the address handed to the call, when there is one, or
    else the far end of the socket it was made on. A call with neither -- data
    on a socket not connected to anything -- went nowhere and is skipped.

    Args:
        log_lines: The lines ``strace -f -yy`` wrote.

    Returns:
        list[NetworkCall]: The calls, in log order.
    """
    found = []
    for line in log_lines:
        start = INTERNET_CALL.match(line)
        if start is None:
            continue
        call, protocol, socket = start.groups()
        given = IPV4.findall(line) + IPV6.findall(line)
        remote = REMOTE_END.search(socket)
        ends = given or ([(remote.group(2), remote.group(1))] if remote else [])
        for port, address in ends:
            found.append(NetworkCall(call, protocol, address, int(port), line.strip()))
    return found


def network_problems(calls: list[NetworkCall]) -> list[str]:
    """Say what is wrong with the network calls a run made, if anything.

    Args:
        calls: Every call, from :func:`network_calls_in`.

    Returns:
        list[str]: One line per call that opened a connection to another
        machine, sent it data, or looked up a name; or one line saying no
        connection was seen at all. Empty when everything stayed here.
    """
    # The run must at least reach the local site and the stand-in model.
    # Seeing no connection means the watching failed, not that the run was clean.
    if not any(c.protocol == "TCP" and c.call == "connect" for c in calls):
        return ["no connections were seen at all, so the watching did not work"]
    return [f"{c.problem}: {c.line}" for c in calls if c.problem]


def local_suite(site_url: str, folder: Path) -> Path:
    """Copy the practice suite, pointed at the practice site's real address.

    Args:
        site_url: Where the site is being served, such as
            ``http://127.0.0.1:41234/``.
        folder: Where to write the copy.

    Returns:
        Path: The copied manifest.
    """
    (folder / "tasks").mkdir(parents=True, exist_ok=True)
    for source in [SUITE_DIR / "suite.yaml", *sorted((SUITE_DIR / "tasks").glob("*.yaml"))]:
        text = source.read_text(encoding="utf-8")
        text = text.replace(PLACEHOLDER_URL, site_url)
        text = text.replace(PLACEHOLDER_DOMAIN, "domain: 127.0.0.1")
        (folder / source.relative_to(SUITE_DIR)).write_text(text, encoding="utf-8")
    return folder / "suite.yaml"


def clean_environment(model_url: str) -> tuple[dict[str, str], list[str]]:
    """Build the environment the evaluation runs in: no keys, the stand-in model.

    Args:
        model_url: The stand-in model's address.

    Returns:
        tuple[dict[str, str], list[str]]: The environment, and the names of
        the settings removed from this one (never their values).
    """
    environment, dropped = {}, []
    for name, value in os.environ.items():
        if name.startswith(DROPPED_PREFIX) or any(m in name.upper() for m in DROPPED_MARKERS):
            dropped.append(name)
        else:
            environment[name] = value
    # The local profile, pointed at the stand-in instead of a real server.
    environment["WAA_BASE_URL"] = model_url
    return environment, sorted(dropped)


def main(argv: list[str] | None = None) -> int:
    """Run the offline evaluation and check it.

    Args:
        argv: Command-line arguments, or None for the real ones.

    Returns:
        int: 0 when everything held, 1 when something did not, 2 when the run
        could not be set up.
    """
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--trials", type=int, default=1, help="times to run each task (default 1)")
    parser.add_argument("--workdir", type=Path, help="folder for the results (default: a new one)")
    parser.add_argument(
        "--watch-network",
        action="store_true",
        help="record every network call with strace, and fail if any leaves this machine",
    )
    args = parser.parse_args(argv)

    strace = shutil.which("strace")
    if args.watch_network and strace is None:
        print(
            "offline_eval: --watch-network needs strace (apt-get install strace)", file=sys.stderr
        )
        return 2
    workdir = (args.workdir or Path(tempfile.mkdtemp(prefix="waa-offline-"))).resolve()
    workdir.mkdir(parents=True, exist_ok=True)
    network_log = workdir / "network.log"

    with StaticSite() as site, StandInModel() as model:
        suite = local_suite(site.base_url, workdir / "benchmark")
        environment, dropped = clean_environment(model.base_url)
        print(f"practice site:     {site.base_url}")
        print(f"stand-in model:    {model.base_url}")
        print(f"settings removed:  {', '.join(dropped) or 'none were set'}")
        print(f"working folder:    {workdir}")

        command = [
            sys.executable, "-m", "web_action_agent", "eval",
            "--suite", str(suite),
            "--trials", str(args.trials),
            "--cold",
            "--profile", "local",
            "--out", str(Path("runs") / SWEEP_NAME),
        ]  # fmt: skip
        if args.watch_network:
            # -f follows every process the evaluation starts (the browsers
            # too); --seccomp-bpf has the kernel stop only at the watched
            # calls, so the run is not slowed by the watching; -yy writes
            # each socket's kind and, once connected, both of its ends.
            command = [
                strace, "-f", "--seccomp-bpf", "-qq", "-yy",
                "-e", f"trace={WATCHED_CALLS}", "-e", "signal=none",
                "-o", str(network_log), *command,
            ]  # fmt: skip
        # Run from the working folder: the sweep and its tables are written
        # there, and there is no .env file there for a key to come from.
        finished = subprocess.run(command, cwd=workdir, env=environment, check=False)

    problems = []
    if finished.returncode != 0:
        problems.append(f"waa eval ended with exit status {finished.returncode}")
    verdicts = [verdict for _, verdict in iter_verdicts(workdir / "runs" / SWEEP_NAME)]
    expected = len(list((SUITE_DIR / "tasks").glob("*.yaml"))) * args.trials
    passed = sum(verdict.status == PASS for verdict in verdicts)
    if passed != expected:
        problems.append(f"{passed} of {expected} trials passed")
    if not (workdir / "results" / "README.md").is_file():
        problems.append("the results tables were not written")
    if args.watch_network:
        calls = network_calls_in(network_log.read_text(encoding="utf-8").splitlines())
        problems += network_problems(calls)

    print()
    print(f"trials passed:     {passed} of {expected}")
    print(f"model calls:       {len(model.calls)} (all to the stand-in on this machine)")
    if args.watch_network:
        local = sum(c.on_this_machine for c in calls)
        checks = sum(c.is_route_check and not c.on_this_machine for c in calls)
        print(f"network calls:     {len(calls)} seen, {local} within this machine")
        if checks:
            print(
                f"                   {checks} route checks (nothing sent), as browsers make on start"
            )
    print(f"results tables:    {workdir / 'results'}")
    for problem in problems:
        print(f"PROBLEM: {problem}", file=sys.stderr)
    if not problems:
        print("offline evaluation: everything held")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
