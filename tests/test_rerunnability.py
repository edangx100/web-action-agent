"""Anyone can re-run the evaluation: no account, no key, nothing outside this machine.

``scripts/offline_eval.py`` runs ``waa eval`` end to end with the two outside
things swapped for stand-ins on this computer: the practice site instead of
the live websites, and a stand-in model instead of an AI service. Continuous
integration runs it on a fresh machine, installed from ``requirements.txt``
alone, with every network call recorded. These tests hold the parts:

- **The stand-in model** reads the page title it is sent and answers from it,
  so a pass shows the page really reached it.
- **The network check** reads a log from ``strace`` and tells calls that stay
  on this machine from ones that leave it. Lines here are written the way
  real logs write them, and one test runs the real tool, so a change in its
  format is noticed.
- **The whole run**, in a real browser: it passes with no key set, and it
  fails when the answers are wrong.
- **The CI job** that does all this on a fresh machine: what it installs,
  what it runs, and that no secret can reach it.
"""

from __future__ import annotations

import importlib.util
import json
import shutil
import socket
import subprocess
import sys
import threading
import urllib.error
import urllib.request
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
from fixtures.model_server import StandInModel, prompt_text, reply_to

from web_action_agent.benchmark import load_suite

REPO = Path(__file__).resolve().parent.parent
SCRIPT = REPO / "scripts" / "offline_eval.py"


def load_script() -> ModuleType:
    """Import the script as a module, so its parts can be tested one by one.

    Returns:
        ModuleType: ``scripts/offline_eval.py``.
    """
    spec = importlib.util.spec_from_file_location("offline_eval", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # Registered first: the script's dataclasses look themselves up by name.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


offline_eval = load_script()


def chat(*texts: str) -> dict[str, Any]:
    """A chat request whose user messages are ``texts``."""
    return {"model": "m", "messages": [{"role": "user", "content": t} for t in texts]}


def page(title: str) -> str:
    """The page part of a planner prompt, as the planner writes it."""
    return f"# Current page\nURL: http://127.0.0.1:1/x.html\nTitle: {title}\n\n## Page text\n..."


# --- The stand-in model ----------------------------------------------------------------


@pytest.mark.unit
def test_the_connection_test_is_answered_with_ok() -> None:
    request = chat('This is a connection test. Reply with the JSON object {"ok": true}.')

    assert json.loads(reply_to(request)) == {"ok": True}


@pytest.mark.unit
@pytest.mark.parametrize(
    ("title", "name"),
    [
        ("Search the catalogue — Northgate Public Library", "Northgate Public Library"),
        ("Northgate Public Library", "Northgate Public Library"),
        ("A - B - Some Library", "Some Library"),
    ],
)
def test_the_answer_is_read_from_the_page_title(title: str, name: str) -> None:
    action = json.loads(reply_to(chat(page(title))))

    assert action == {"action": "done", "status": "success", "answer": name, "reason": None}


@pytest.mark.unit
def test_with_no_title_to_read_the_stand_in_reports_failure() -> None:
    # The page never loaded: nothing to read, so no answer to give.
    action = json.loads(reply_to(chat(page(""))))

    assert action["status"] == "failed"
    assert action["answer"] is None


@pytest.mark.unit
def test_the_latest_page_is_the_one_read() -> None:
    request = chat(page("Old page — Wrong Name"), page("New page — Right Name"))

    assert json.loads(reply_to(request))["answer"] == "Right Name"


@pytest.mark.unit
def test_text_is_read_from_a_message_with_a_picture_too() -> None:
    request = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "first"},
                    {"type": "image_url", "image_url": {"url": "data:..."}},
                    {"type": "text", "text": "second"},
                ],
            }
        ]
    }

    assert prompt_text(request) == "first\nsecond"


@pytest.mark.unit
def test_the_server_answers_in_the_openai_format() -> None:
    def post(url: str, body: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        """POST JSON and return the status and the JSON reply."""
        request = urllib.request.Request(url, data=json.dumps(body).encode(), method="POST")
        try:
            with urllib.request.urlopen(request) as response:
                return response.status, json.load(response)
        except urllib.error.HTTPError as error:
            return error.code, json.load(error)

    with StandInModel() as model:
        status, body = post(f"{model.base_url}/chat/completions", chat(page("A — B")))
        missing, _ = post(f"{model.base_url}/elsewhere", {})
        address = model.base_url

    assert status == 200
    assert json.loads(body["choices"][0]["message"]["content"])["answer"] == "B"
    assert body["choices"][0]["finish_reason"] == "stop"
    assert body["usage"]["prompt_tokens"] > 0
    assert missing == 404
    assert len(model.calls) == 1
    assert address.startswith("http://127.0.0.1:")


# --- The network check -----------------------------------------------------------------

#: Lines as ``strace -f -yy`` writes them; the route check is from a real log of this run.
TCP_TO_SITE = (
    "101 connect(5<TCP:[1001]>, {sa_family=AF_INET, sin_port=htons(44535), "
    'sin_addr=inet_addr("127.0.0.1")}, 16) = -1 EINPROGRESS (Operation now in progress)'
)
SEND_ON_LOCAL_TCP = (
    '101 sendto(5<TCP:[127.0.0.1:52344->127.0.0.1:44535]>, "GET / HTTP/1.1\\r\\n"..., 120, '
    "MSG_NOSIGNAL, NULL, 0) = 120"
)
ROUTE_CHECK = (
    "102 connect(18<UDPv6:[285619]>, {sa_family=AF_INET6, sin6_port=htons(443), "
    'sin6_flowinfo=htonl(0), inet_pton(AF_INET6, "2001:4860:4860::8888", &sin6_addr), '
    "sin6_scope_id=0}, 28) = -1 ENETUNREACH (Network is unreachable)"
)
UNIX_SOCKET = '103 connect(7<UNIX:[2002]>, {sa_family=AF_UNIX, sun_path="/run/dbus"}, 110) = 0'
PIPE_WRITE = '103 write(9<pipe:[3003]>, "x", 1) = 1'


def tcp_connect(address: str, port: int) -> str:
    """A TCP connect line, to the given IPv4 address."""
    return (
        f"104 connect(6<TCP:[4004]>, {{sa_family=AF_INET, sin_port=htons({port}), "
        f'sin_addr=inet_addr("{address}")}}, 16) = -1 EINPROGRESS'
    )


def udp_send(address: str, port: int) -> str:
    """A UDP sendto line with an address, as a name lookup is sent."""
    return (
        f'105 sendto(8<UDP:[5005]>, "\\x12\\x34"..., 40, MSG_NOSIGNAL, {{sa_family=AF_INET, '
        f'sin_port=htons({port}), sin_addr=inet_addr("{address}")}}, 16) = 40'
    )


@pytest.mark.unit
def test_calls_on_this_machine_and_route_checks_are_fine() -> None:
    calls = offline_eval.network_calls_in(
        [TCP_TO_SITE, SEND_ON_LOCAL_TCP, ROUTE_CHECK, UNIX_SOCKET, PIPE_WRITE]
    )

    # Files, pipes and local sockets are not internet calls and are skipped.
    assert [(c.call, c.protocol, c.address, c.port) for c in calls] == [
        ("connect", "TCP", "127.0.0.1", 44535),
        ("sendto", "TCP", "127.0.0.1", 44535),
        ("connect", "UDP", "2001:4860:4860::8888", 443),
    ]
    assert offline_eval.network_problems(calls) == []


@pytest.mark.unit
@pytest.mark.parametrize(
    ("line", "problem"),
    [
        (tcp_connect("93.184.215.14", 443), "a connection opened to 93.184.215.14 port 443"),
        (udp_send("8.8.8.8", 443), "data sent to 8.8.8.8 port 443"),
        # A name lookup, even to a helper on this machine, is a sign something
        # tried to find an outside host.
        (udp_send("127.0.0.53", 53), "a name lookup"),
        (
            # Data on a UDP socket connected to an outside address: the
            # address is only in the socket's description.
            '106 write(18<UDPv6:[[::]:5->[2001:4860:4860::8888]:443]>, "x", 1) = 1',
            "data sent to 2001:4860:4860::8888 port 443",
        ),
        (
            '107 sendto(5<TCP:[10.0.0.2:5->140.82.112.3:443]>, "x", 1, 0, NULL, 0) = 1',
            "data sent to 140.82.112.3 port 443",
        ),
    ],
)
def test_a_call_leaving_this_machine_is_a_problem(line: str, problem: str) -> None:
    problems = offline_eval.network_problems(offline_eval.network_calls_in([TCP_TO_SITE, line]))

    assert len(problems) == 1
    assert problems[0].startswith(problem)
    # The log line is shown, so the problem can be found in the log.
    assert line.strip() in problems[0]


@pytest.mark.unit
def test_ipv6_spellings_of_this_machine_count_as_this_machine() -> None:
    lines = [
        TCP_TO_SITE,
        (
            "108 connect(5<TCPv6:[6006]>, {sa_family=AF_INET6, sin6_port=htons(80), "
            'sin6_flowinfo=htonl(0), inet_pton(AF_INET6, "::ffff:127.0.0.1", &sin6_addr), '
            "sin6_scope_id=0}, 28) = 0"
        ),
        '108 sendto(5<TCPv6:[[::1]:5000->[::1]:6000]>, "x", 1, 0, NULL, 0) = 1',
    ]

    assert offline_eval.network_problems(offline_eval.network_calls_in(lines)) == []


@pytest.mark.unit
def test_lines_with_a_padded_process_number_are_read_too() -> None:
    # strace 6.1, in the CI container, writes two spaces after the process
    # number where 5.16 writes one. Copied from that log.
    lines = [
        (
            "1791  connect(3<TCP:[16707]>, {sa_family=AF_INET, sin_port=htons(39753), "
            'sin_addr=inet_addr("127.0.0.1")}, 16) = -1 EINPROGRESS (Operation now in progress)'
        ),
        (
            '1791  sendto(3<TCP:[127.0.0.1:55022->127.0.0.1:39753]>, "POST /v1/chat/completions '
            'HTTP/1"..., 525, 0, NULL, 0) = 525'
        ),
        (
            "2017  connect(18<UDPv6:[18403]>, {sa_family=AF_INET6, sin6_port=htons(443), "
            'sin6_flowinfo=htonl(0), inet_pton(AF_INET6, "2001:4860:4860::8888", &sin6_addr), '
            "sin6_scope_id=0}, 28) = -1 ENETUNREACH (Network is unreachable)"
        ),
    ]

    calls = offline_eval.network_calls_in(lines)

    assert [(c.call, c.protocol, c.port) for c in calls] == [
        ("connect", "TCP", 39753),
        ("sendto", "TCP", 39753),
        ("connect", "UDP", 443),
    ]
    assert offline_eval.network_problems(calls) == []


@pytest.mark.unit
def test_seeing_no_connection_at_all_is_a_problem() -> None:
    # The run always reaches the local site, so an empty log means the
    # watching failed -- not that the run was clean.
    assert offline_eval.network_problems(offline_eval.network_calls_in([UNIX_SOCKET])) == [
        "no connections were seen at all, so the watching did not work"
    ]


@pytest.mark.unit
@pytest.mark.skipif(shutil.which("strace") is None, reason="strace is not installed")
def test_the_real_strace_writes_what_the_check_reads(tmp_path: Path) -> None:
    # A real program, watched by the real tool: one TCP connection to a
    # server on this machine, and one UDP route check, which sends nothing.
    listener = socket.create_server(("127.0.0.1", 0))
    port = listener.getsockname()[1]
    threading.Thread(target=lambda: listener.accept()[0].close(), daemon=True).start()
    program = (
        "import socket\n"
        f"socket.create_connection(('127.0.0.1', {port})).sendall(b'hello')\n"
        "probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)\n"
        "try:\n"
        "    probe.connect(('192.0.2.1', 443))\n"  # a documentation address
        "except OSError:\n"
        "    pass\n"
    )
    log = tmp_path / "network.log"
    command = ["strace", "-f", "-qq", "-yy", "-e", f"trace={offline_eval.WATCHED_CALLS}"]
    subprocess.run(
        [*command, "-e", "signal=none", "-o", str(log), sys.executable, "-c", program],
        check=True,
    )
    listener.close()

    calls = offline_eval.network_calls_in(log.read_text().splitlines())

    seen = {(c.call, c.protocol, c.address, c.port) for c in calls}
    assert ("connect", "TCP", "127.0.0.1", port) in seen
    assert ("sendto", "TCP", "127.0.0.1", port) in seen
    assert ("connect", "UDP", "192.0.2.1", 443) in seen
    assert offline_eval.network_problems(calls) == []


# --- Setting the run up ----------------------------------------------------------------


@pytest.mark.unit
def test_no_key_or_project_setting_reaches_the_run(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WAA_OPENROUTER_API_KEY", "secret-1")
    monkeypatch.setenv("WAA_JUDGE_PROFILE", "openrouter-judge")
    monkeypatch.setenv("OPENAI_API_KEY", "secret-2")
    monkeypatch.setenv("HOME_TOOL_api_token", "secret-3")
    monkeypatch.setenv("PATH_TO_SOMETHING", "kept")

    environment, dropped = offline_eval.clean_environment("http://127.0.0.1:9/v1")

    assert dropped == sorted(
        ["WAA_OPENROUTER_API_KEY", "WAA_JUDGE_PROFILE", "OPENAI_API_KEY", "HOME_TOOL_api_token"]
    )
    assert not any("secret" in value for value in environment.values())
    assert environment["PATH_TO_SOMETHING"] == "kept"
    # The only project setting is the one pointing the local profile at the
    # stand-in model.
    assert {k: v for k, v in environment.items() if k.startswith("WAA_")} == {
        "WAA_BASE_URL": "http://127.0.0.1:9/v1"
    }


@pytest.mark.unit
def test_the_suite_copy_points_every_task_at_the_local_site(tmp_path: Path) -> None:
    manifest = offline_eval.local_suite("http://127.0.0.1:41234/", tmp_path)

    suite = load_suite(manifest)

    assert len(suite.tasks) == 5
    assert all(t.start_url.startswith("http://127.0.0.1:41234/") for t in suite.tasks)
    assert [site.domain for site in suite.sites.values()] == ["127.0.0.1"]
    # Nothing in the copy still names the made-up address, outside comments.
    settings = [
        line
        for path in tmp_path.rglob("*.yaml")
        for line in path.read_text().splitlines()
        if not line.lstrip().startswith("#")
    ]
    assert not any("fixture.test" in line for line in settings)


# --- The whole run ---------------------------------------------------------------------


@pytest.mark.fixture
def test_the_whole_evaluation_runs_and_passes_with_no_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # A key in the environment must not matter: it is removed for the run.
    monkeypatch.setenv("WAA_OPENROUTER_API_KEY", "not-a-real-key")

    status = offline_eval.main(["--workdir", str(tmp_path)])

    printed = capsys.readouterr().out
    assert status == 0
    assert "trials passed:     5 of 5" in printed
    assert "settings removed:  WAA_OPENROUTER_API_KEY" in printed
    assert (tmp_path / "results" / "README.md").is_file()
    assert (tmp_path / "results" / "offline-local-cold.md").is_file()
    # Each trial left a full run folder, with its screenshots.
    trials = sorted((tmp_path / "runs" / "offline-local-cold").glob("*/trial-1"))
    assert len(trials) == 5
    assert all(any((t / "screenshots").glob("*.png")) for t in trials)


@pytest.mark.fixture
def test_wrong_answers_make_the_run_fail(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # The stand-in runs inside this process, so its rule can be swapped for
    # one that always answers wrongly.
    def wrong(request: dict[str, Any]) -> str:
        if "connection test" in prompt_text(request):
            return json.dumps({"ok": True})
        return json.dumps(
            {"action": "done", "status": "success", "answer": "Southgate", "reason": None}
        )

    monkeypatch.setattr("fixtures.model_server.reply_to", wrong)

    status = offline_eval.main(["--workdir", str(tmp_path)])

    assert status == 1
    assert "0 of 5 trials passed" in capsys.readouterr().err


@pytest.mark.fixture
@pytest.mark.skipif(shutil.which("strace") is None, reason="strace is not installed")
def test_watching_the_network_finds_nothing_leaving_this_machine(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    status = offline_eval.main(["--workdir", str(tmp_path), "--watch-network"])

    printed = capsys.readouterr().out
    assert status == 0
    assert (tmp_path / "network.log").is_file()
    assert "within this machine" in printed


# --- The continuous-integration job ----------------------------------------------------

WORKFLOW = REPO / ".github" / "workflows" / "ci.yml"


def rerun_job() -> dict[str, Any]:
    """The workflow's job that re-runs the evaluation from scratch."""
    import yaml

    return yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))["jobs"]["rerun-from-scratch"]


def job_commands(job: dict[str, Any]) -> str:
    """Every shell command the job runs, joined."""
    return "\n".join(step.get("run", "") for step in job["steps"])


@pytest.mark.unit
def test_the_job_starts_from_a_fresh_container_and_installs_from_requirements_alone() -> None:
    job = rerun_job()
    commands = job_commands(job)

    assert job["container"]["image"].startswith("python:")
    assert "pip install --disable-pip-version-check -q -r requirements.txt" in commands
    # Not the development tools: someone re-running this need not have them.
    assert "uv " not in commands
    assert "pytest" not in commands


@pytest.mark.unit
def test_the_job_runs_the_evaluation_watching_the_network() -> None:
    commands = job_commands(rerun_job())

    assert "scripts/offline_eval.py" in commands
    assert "--watch-network" in commands
    assert "apt-get install" in commands and "strace" in commands


@pytest.mark.unit
def test_the_job_is_given_no_secret_and_checks_it_has_no_key() -> None:
    job = rerun_job()

    # A secret can only reach a job by being named in it.
    assert "secrets." not in WORKFLOW.read_text(encoding="utf-8")
    assert "env" not in job
    assert "API_KEY" in job_commands(job)
