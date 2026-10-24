"""The demo recordings, and the script that makes them.

``scripts/demo.sh`` runs one task live, with the browser showing, and turns it
into the two recordings the README shows: ``docs/demo.cast`` (the terminal,
recorded with asciinema) and ``docs/demo.gif`` (the browser, from the video
``waa run --record-video`` saves). These tests hold two things:

- **The script.** Run with stand-ins for ``waa``, asciinema and ffmpeg, so
  nothing is spent: it runs the task exactly as its file words it, replaces the
  recordings only after a run that finished, and refuses a GIF that is too big.
- **The committed recordings.** The GIF is small enough for the README, and
  the cast shows the same task as the committed example run.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

REPO = Path(__file__).resolve().parent.parent
"""Path: The repository's top folder."""

SCRIPT = REPO / "scripts" / "demo.sh"
"""Path: The script under test."""

TASK_FILE = REPO / "benchmarks" / "tasks" / "iras-tax-on-first-80k.yaml"
"""Path: The task the demo runs by default."""

EXAMPLE_RUN = REPO / "runs" / "example" / "run.json"
"""Path: The committed example run's description: its task and start page."""

MAX_GIF_BYTES = 5_000_000
"""int: The largest GIF the README should carry; the script holds the same limit."""


# --- Stand-ins -----------------------------------------------------------------------------

#: A stand-in for ``waa``: logs its arguments, one per line. On success it
#: leaves a run folder where a real run would -- a copy of the committed
#: example run, a real pass of the default task, with its answer changed to
#: FAKE_ANSWER if that is set -- and a video in it. Then it exits with FAKE_EXIT.
FAKE_WAA = """#!/usr/bin/env bash
printf '%s\\n' "$@" > "$FAKE_LOG.waa"
if [[ "${FAKE_EXIT:-0}" == "0" ]]; then
  run=runs/2026-10-24T01-02-03Z
  mkdir -p runs
  cp -r "$FAKE_EXAMPLE" "$run"
  if [[ -n "${FAKE_ANSWER:-}" ]]; then
    sed -i "s/\\"answer\\": \\"[^\\"]*\\"/\\"answer\\": \\"$FAKE_ANSWER\\"/g" "$run/outcome.json"
  fi
  mkdir -p "$run/video"
  printf 'video' > "$run/video/page.webm"
fi
exit "${FAKE_EXIT:-0}"
"""

#: A stand-in for asciinema: runs the command it is given, and saves what the
#: command printed as the "recording".
FAKE_ASCIINEMA = """#!/usr/bin/env bash
echo "$*" > "$FAKE_LOG.asciinema"
cmd=""
out=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    -c) cmd="$2"; shift 2 ;;
    --cols|--rows|--idle-time-limit|--title) shift 2 ;;
    --*) shift ;;
    *) out="$1"; shift ;;
  esac
done
bash -c "$cmd" > "$out"
"""

#: A stand-in for ffmpeg: logs its arguments, and writes FAKE_GIF_BYTES bytes
#: to the file named last.
FAKE_FFMPEG = """#!/usr/bin/env bash
echo "$*" > "$FAKE_LOG.ffmpeg"
out="${@: -1}"
head -c "${FAKE_GIF_BYTES:-1000}" /dev/zero > "$out"
"""


def make(
    tmp_path: Path,
    *args: str,
    waa_exit: int = 0,
    answer: str = "",
    gif_bytes: int = 1000,
    with_asciinema: bool = True,
) -> tuple[int, str]:
    """Run the script in a scratch folder, with stand-ins for every tool.

    Args:
        tmp_path: The folder to run in; ``runs/`` and ``docs/`` end up here.
        *args: Arguments to the script.
        waa_exit: The exit code of the stand-in ``waa run``.
        answer: An answer to give the run instead of the example's right one.
        gif_bytes: How big the stand-in ffmpeg makes the GIF.
        with_asciinema: False to leave asciinema out, as if not installed.

    Returns:
        tuple[int, str]: The script's exit code, and what it printed to stderr.
    """
    tools = {"fake-waa": FAKE_WAA, "fake-ffmpeg": FAKE_FFMPEG}
    if with_asciinema:
        tools["fake-asciinema"] = FAKE_ASCIINEMA
    for name, text in tools.items():
        (tmp_path / name).write_text(text)
        (tmp_path / name).chmod(0o755)
    env = {
        **os.environ,
        "WAA": str(tmp_path / "fake-waa"),
        "ASCIINEMA": str(tmp_path / "fake-asciinema"),
        "FFMPEG": str(tmp_path / "fake-ffmpeg"),
        # The test's own Python, which has PyYAML, reads the task file.
        "PYTHON": sys.executable,
        "FAKE_LOG": str(tmp_path / "calls"),
        "FAKE_EXIT": str(waa_exit),
        "FAKE_EXAMPLE": str(EXAMPLE_RUN.parent),
        "FAKE_ANSWER": answer,
        "FAKE_GIF_BYTES": str(gif_bytes),
    }
    done = subprocess.run(
        ["bash", str(SCRIPT), *args],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    return done.returncode, done.stderr


def waa_args(tmp_path: Path) -> list[str]:
    """The arguments the stand-in ``waa`` was called with, or [] if never."""
    log = tmp_path / "calls.waa"
    return log.read_text().splitlines() if log.exists() else []


def old_recordings(tmp_path: Path) -> None:
    """Put last time's recordings in place, to see whether they survive."""
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "demo.gif").write_text("old gif")
    (tmp_path / "docs" / "demo.cast").write_text("old cast")


def example_task() -> dict[str, Any]:
    """The committed example run's description."""
    return json.loads(EXAMPLE_RUN.read_text(encoding="utf-8"))


# --- The script ---------------------------------------------------------------------------


@pytest.mark.unit
def test_the_script_runs_the_task_cold_headed_with_video_and_saves_both(tmp_path: Path) -> None:
    code, said = make(tmp_path, "--task-file", str(TASK_FILE))

    assert code == 0, said
    task = example_task()
    assert waa_args(tmp_path) == [
        "run",
        "--cold",
        "--headed",
        "--record-video",
        "--start-url",
        task["start_url"],
        task["task"],
    ], "the task's words reach waa exactly as written, dollar signs and all"
    assert (tmp_path / "docs" / "demo.gif").stat().st_size == 1000
    assert (tmp_path / "docs" / "demo.cast").exists()


@pytest.mark.unit
def test_the_cast_shows_the_command_that_was_run(tmp_path: Path) -> None:
    make(tmp_path, "--task-file", str(TASK_FILE))

    cast = (tmp_path / "docs" / "demo.cast").read_text()
    task = example_task()
    assert cast.splitlines()[0] == (
        f'$ waa run --cold --headed --record-video --start-url {task["start_url"]} "{task["task"]}"'
    )


@pytest.mark.unit
def test_a_profile_is_passed_to_waa_and_shown_in_the_cast(tmp_path: Path) -> None:
    # For when the usual service is down: the run goes elsewhere, and the
    # recording says so.
    code, said = make(tmp_path, "--task-file", str(TASK_FILE), "--profile", "nvidia")

    assert code == 0, said
    args = waa_args(tmp_path)
    assert args[args.index("--profile") + 1] == "nvidia"
    cast = (tmp_path / "docs" / "demo.cast").read_text()
    assert cast.startswith("$ waa run --cold --headed --record-video --profile nvidia --start-url")


@pytest.mark.unit
def test_the_gif_is_made_from_the_runs_video(tmp_path: Path) -> None:
    make(tmp_path, "--task-file", str(TASK_FILE))

    ffmpeg = (tmp_path / "calls.ffmpeg").read_text()
    assert "-i runs/2026-10-24T01-02-03Z/video/page.webm" in ffmpeg
    # Real time, five frames a second, still frames thinned out, 800 wide.
    assert "setpts=PTS/1,fps=5,mpdecimate=max=30,setpts=N/5/TB,scale=800:-1" in ffmpeg


@pytest.mark.unit
def test_the_default_task_is_the_example_runs_task(tmp_path: Path) -> None:
    # The script's default task file is a path inside the project, so give the
    # scratch folder the same file at the same place.
    tasks = tmp_path / "benchmarks" / "tasks"
    tasks.mkdir(parents=True)
    shutil.copy(TASK_FILE, tasks)

    code, said = make(tmp_path)

    assert code == 0, said
    task = example_task()
    assert waa_args(tmp_path)[-2:] == [task["start_url"], task["task"]]


@pytest.mark.unit
def test_a_run_that_did_not_finish_keeps_the_old_recordings(tmp_path: Path) -> None:
    old_recordings(tmp_path)

    code, said = make(tmp_path, "--task-file", str(TASK_FILE), waa_exit=1)

    assert code == 1
    assert "did not finish the task" in said
    assert (tmp_path / "docs" / "demo.gif").read_text() == "old gif"
    assert (tmp_path / "docs" / "demo.cast").read_text() == "old cast"


@pytest.mark.unit
def test_the_run_is_marked_by_the_tasks_own_checks(tmp_path: Path) -> None:
    code, said = make(tmp_path, "--task-file", str(TASK_FILE))

    assert code == 0, said
    verdict = json.loads((tmp_path / "runs" / "2026-10-24T01-02-03Z" / "verdict.json").read_text())
    assert verdict["status"] == "pass"
    assert "the task's checks passed the run" in said


@pytest.mark.unit
def test_a_wrong_answer_the_agent_called_success_is_not_recorded(tmp_path: Path) -> None:
    # The agent can be sure of a wrong answer: here it reports success with
    # the tax on the next $40,000 instead of the first $80,000.
    old_recordings(tmp_path)

    code, said = make(tmp_path, "--task-file", str(TASK_FILE), answer="4600")

    assert code == 1
    assert "marked the run 'fail'" in said
    assert "4600" in said, "the reason names the wrong answer"
    assert (tmp_path / "docs" / "demo.gif").read_text() == "old gif"
    assert (tmp_path / "docs" / "demo.cast").read_text() == "old cast"


@pytest.mark.unit
def test_a_run_that_could_not_start_stops_with_two(tmp_path: Path) -> None:
    old_recordings(tmp_path)

    code, said = make(tmp_path, "--task-file", str(TASK_FILE), waa_exit=2)

    assert code == 2
    assert "could not start" in said
    assert (tmp_path / "docs" / "demo.gif").read_text() == "old gif"


@pytest.mark.unit
def test_a_gif_over_the_limit_is_refused_and_the_old_one_kept(tmp_path: Path) -> None:
    old_recordings(tmp_path)

    code, said = make(tmp_path, "--task-file", str(TASK_FILE), gif_bytes=MAX_GIF_BYTES + 1)

    assert code == 2
    assert "over the 5000000 limit" in said
    assert "--video runs/2026-10-24T01-02-03Z/video/page.webm" in said, "how to retry for free"
    assert (tmp_path / "docs" / "demo.gif").read_text() == "old gif"
    # The run passed, so its terminal recording is kept with it for the retry.
    assert (tmp_path / "runs" / "2026-10-24T01-02-03Z" / "demo.cast").exists()


@pytest.mark.unit
def test_video_makes_only_the_gif_without_running_anything(tmp_path: Path) -> None:
    old_recordings(tmp_path)
    video = tmp_path / "earlier.webm"
    video.write_text("video")

    code, said = make(tmp_path, "--video", str(video), "--width", "640", "--fps", "4")

    assert code == 0, said
    assert waa_args(tmp_path) == [], "nothing run, nothing spent"
    assert (
        "fps=4,mpdecimate=max=30,setpts=N/4/TB,scale=640:-1"
        in (tmp_path / "calls.ffmpeg").read_text()
    )
    assert (tmp_path / "docs" / "demo.gif").stat().st_size == 1000
    assert (tmp_path / "docs" / "demo.cast").read_text() == "old cast", "the cast is left alone"


@pytest.mark.unit
def test_video_brings_the_runs_own_terminal_recording_with_it(tmp_path: Path) -> None:
    # A GIF that came out too big is made again from the same run, and the
    # terminal recording that run left in its folder goes into docs/ with it.
    old_recordings(tmp_path)
    run = tmp_path / "runs" / "2026-10-24T01-02-03Z"
    (run / "video").mkdir(parents=True)
    (run / "video" / "page.webm").write_text("video")
    (run / "demo.cast").write_text("this run's cast")

    code, said = make(tmp_path, "--video", "runs/2026-10-24T01-02-03Z/video/page.webm")

    assert code == 0, said
    assert waa_args(tmp_path) == []
    assert (tmp_path / "docs" / "demo.cast").read_text() == "this run's cast"


@pytest.mark.unit
def test_without_asciinema_nothing_is_run(tmp_path: Path) -> None:
    code, said = make(tmp_path, "--task-file", str(TASK_FILE), with_asciinema=False)

    assert code == 2
    assert "uv run --extra demo" in said
    assert waa_args(tmp_path) == []


@pytest.mark.unit
def test_the_script_refuses_an_unknown_argument(tmp_path: Path) -> None:
    code, _ = make(tmp_path, "--slow")

    assert code == 2
    assert waa_args(tmp_path) == []


@pytest.mark.unit
def test_the_script_explains_itself() -> None:
    done = subprocess.run(
        ["bash", str(SCRIPT), "--help"], capture_output=True, text=True, check=False
    )

    assert done.returncode == 0
    assert "uv run --extra demo scripts/demo.sh" in done.stdout
    assert "Exit codes" in done.stdout, "the whole header is shown"
    assert "set -euo pipefail" not in done.stdout, "and nothing after it"


@pytest.mark.unit
def test_the_script_is_executable() -> None:
    assert os.access(SCRIPT, os.X_OK)


# --- The committed recordings -----------------------------------------------------------


@pytest.mark.unit
def test_the_committed_gif_is_a_gif_under_the_limit() -> None:
    gif = REPO / "docs" / "demo.gif"

    assert gif.read_bytes()[:6] in (b"GIF87a", b"GIF89a")
    assert gif.stat().st_size <= MAX_GIF_BYTES


@pytest.mark.unit
def test_the_committed_cast_shows_the_example_runs_task() -> None:
    # An asciicast (version 2) is a header line of JSON, then one JSON event
    # per line: [seconds, "o", text written to the terminal].
    lines = (REPO / "docs" / "demo.cast").read_text(encoding="utf-8").splitlines()
    header = json.loads(lines[0])
    shown = "".join(event[2] for event in map(json.loads, lines[1:]) if event[1] == "o")

    assert header["version"] == 2
    assert example_task()["task"] in shown
    assert "success in" in shown, "the recorded run finished its task"
