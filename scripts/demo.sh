#!/usr/bin/env bash
# The demo recordings, from one live run.
#
# Runs one benchmark task with the browser window showing and no learned skills
# (`waa run --cold`, so it looks the same on a fresh clone as here), and makes
# the two recordings the README shows:
#
#   docs/demo.cast   the terminal, recorded with asciinema. Play it with
#                    `uv run --extra demo asciinema play docs/demo.cast`.
#   docs/demo.gif    the browser, from the video Playwright records of the
#                    page (`waa run --record-video`), sped up into a GIF.
#
# Usage (from the project folder; the `demo` extra installs both recorders):
#   uv run --extra demo scripts/demo.sh [--task-file FILE] [--profile NAME]
#                                       [--speed N] [--width PX] [--fps N]
#                                       [--video FILE]
#
#   --task-file  the benchmark task to run (default:
#                benchmarks/tasks/iras-tax-on-first-80k.yaml, the task of the
#                committed example run in runs/example/)
#   --profile    the AI service to run on, such as nvidia when OpenRouter is
#                down (default: the configured one, openrouter)
#   --speed      how many times faster the GIF plays than the run (default: 1;
#                long waits for the AI are shortened either way)
#   --width      the GIF's width in pixels (default: 800)
#   --fps        frames per second in the GIF (default: 5)
#   --video      make the GIF again from a video a run already recorded, such
#                as runs/<run>/video/<name>.webm; nothing is run and nothing
#                is spent, so the GIF settings can be tried until it looks right.
#                If this script recorded that run, its terminal recording
#                (runs/<run>/demo.cast) is used too.
#
# The run is live: it needs the profile's key in .env (WAA_OPENROUTER_API_KEY
# for openrouter, about $0.002 a run; WAA_NVIDIA_API_KEY for nvidia, on free
# credits). It also needs a screen for the browser window; on Windows with WSL,
# WSLg provides one.
#
# The recordings are only replaced by a run that passes the task's own
# benchmark checks (the right answer, on the right page), not merely one where
# the agent says it succeeded; a failed try leaves the last good ones in place.
# With --video, the run is not marked again: choose a video from a run that
# passed. The GIF must come out under 5 MB, so the README stays quick to load;
# if it does not, try a smaller --width or --fps with --video.
#
# Exit codes: 0 when the recordings are made; 1 when the run did not pass (run
# it again: the agent does not take the same path every time); 2 when something
# could not run at all, or the GIF came out too big.

set -euo pipefail

# The commands to run. Overridable so the tests can put stand-ins in their place.
WAA="${WAA:-waa}"
ASCIINEMA="${ASCIINEMA:-asciinema}"
PYTHON="${PYTHON:-python}"
# ffmpeg comes from the imageio-ffmpeg package unless one is named here.
FFMPEG="${FFMPEG:-}"

# The largest GIF the README should carry, in bytes.
MAX_GIF_BYTES=5000000

task_file="benchmarks/tasks/iras-tax-on-first-80k.yaml"
speed="1"
width="800"
fps="5"
video=""
profile=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --task-file) task_file="${2:?--task-file needs a file}"; shift 2 ;;
    --profile) profile="${2:?--profile needs a profile name}"; shift 2 ;;
    --speed) speed="${2:?--speed needs a number}"; shift 2 ;;
    --width) width="${2:?--width needs a number of pixels}"; shift 2 ;;
    --fps) fps="${2:?--fps needs a number}"; shift 2 ;;
    --video) video="${2:?--video needs a video file}"; shift 2 ;;
    -h|--help) awk 'NR > 1 && /^#/ { sub(/^# ?/, ""); print; next } NR > 1 { exit }' "$0"; exit 0 ;;
    *) echo "demo: unknown argument: $1" >&2; exit 2 ;;
  esac
done

# require_waa: see helpers.sh.
source "$(dirname "${BASH_SOURCE[0]}")/helpers.sh"

# Everything is made in a scratch folder first and moved into docs/ only once
# both files are good, so a failure part-way never leaves a half-made pair.
scratch="$(mktemp -d)"
trap 'rm -rf "$scratch"' EXIT

# Find ffmpeg before anything is spent, so a missing extra is found first.
if [[ -z "$FFMPEG" ]]; then
  if ! FFMPEG="$("$PYTHON" -c 'import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())' 2> /dev/null)"; then
    echo "demo: cannot find ffmpeg. Run this as: uv run --extra demo scripts/demo.sh" >&2
    exit 2
  fi
fi

# Records the terminal while the agent runs the task, and finds the run's video.
# Sets `video` to the video file. Exits the script if the run did not finish.
record_run() {
  require_waa
  if ! command -v "$ASCIINEMA" > /dev/null 2>&1; then
    echo "demo: cannot find asciinema. Run this as: uv run --extra demo scripts/demo.sh" >&2
    exit 2
  fi
  if [[ ! -f "$task_file" ]]; then
    echo "demo: no task file at $task_file" >&2
    exit 2
  fi

  # The task's words and start page, read from its file, so the recording
  # always shows the task exactly as the benchmark asks it.
  local details task_id start_url instruction
  if ! details="$("$PYTHON" - "$task_file" << 'EOF'
import sys

import yaml

task = yaml.safe_load(open(sys.argv[1], encoding="utf-8"))
print(task["id"])
print(task["start_url"])
print(" ".join(task["instruction"].split()))
EOF
  )"; then
    echo "demo: cannot read the task's id, start_url and instruction from $task_file" >&2
    exit 2
  fi
  {
    read -r task_id
    read -r start_url
    read -r instruction
  } <<< "$details"

  # What asciinema records: the command, as a person would type it, then the
  # run itself. Written to a file so the task's words reach `waa` exactly as
  # they are, whatever quotes or dollar signs they hold.
  # A profile is named only when one was asked for; otherwise the configured
  # one is used, and the command shows nothing about it.
  local profile_args=()
  [[ -n "$profile" ]] && profile_args=(--profile "$profile")
  local shown="waa run --cold --headed --record-video ${profile_args[*]:+${profile_args[*]} }"
  shown+="--start-url ${start_url} \"${instruction}\""
  {
    printf 'printf "%%s\\n\\n" %q\n' "\$ ${shown}"
    printf '%q ' "$WAA" run --cold --headed --record-video "${profile_args[@]}" \
      --start-url "$start_url" "$instruction"
    printf '\necho $? > %q\n' "$scratch/exit"
  } > "$scratch/record.sh"

  # Only a video newer than this marker can belong to this run.
  touch "$scratch/started"
  echo "== recording ${task_id}: ${shown}" >&2
  # --idle-time-limit shortens long waits for the AI when the cast is played.
  "$ASCIINEMA" rec --overwrite --quiet --cols 120 --rows 32 --idle-time-limit 2 \
    --title "web-action-agent: ${task_id}" -c "bash $scratch/record.sh" "$scratch/demo.cast"

  local code
  code="$(cat "$scratch/exit" 2> /dev/null || echo 2)"
  if [[ "$code" == "1" ]]; then
    echo "demo: the run did not finish the task, so the recordings were not" \
      "replaced. Run it again: the agent does not take the same path every time." >&2
    exit 1
  elif [[ "$code" != "0" ]]; then
    echo "demo: the run could not start (exit ${code}). Read the message above." >&2
    exit 2
  fi

  # The first page's video is the one that shows the run from its start; a
  # tab the agent opened later gets a video of its own.
  video="$(find runs -path '*/video/*.webm' -newer "$scratch/started" -printf '%T@ %p\n' \
    2> /dev/null | sort -n | head -n 1 | cut -d ' ' -f 2-)"
  if [[ -z "$video" ]]; then
    echo "demo: the run finished, but left no video under runs/." >&2
    exit 2
  fi

  # The agent saying "success" is not enough: it can be sure of a wrong answer.
  # So the run is marked with the task's own benchmark checks, exactly as a
  # benchmark trial is, and only a pass is recorded. The mark is saved in the
  # run's folder as verdict.json.
  local run_folder mark status reason
  run_folder="$(dirname "$(dirname "$video")")"
  if ! mark="$("$PYTHON" - "$task_file" "$run_folder" << 'EOF'
import sys
from pathlib import Path

from web_action_agent.benchmark import load_task
from web_action_agent.eval_harness import mark_trial, write_verdict

folder = Path(sys.argv[2])
verdict = mark_trial(load_task(Path(sys.argv[1])), folder, judge=None)
write_verdict(folder, verdict)
print(verdict.status)
print(verdict.reason)
EOF
  )"; then
    echo "demo: could not mark the run in ${run_folder}." >&2
    exit 2
  fi
  {
    read -r status
    read -r reason
  } <<< "$mark"
  if [[ "$status" != "pass" ]]; then
    echo "demo: the agent reported success, but the task's checks marked the run" \
      "'${status}' (${reason}), so the recordings were not replaced. Run it again." >&2
    exit 1
  fi
  echo "== the task's checks passed the run: ${reason}" >&2

  # Kept with the run as well, so that if the GIF needs another try, --video
  # can make it again from this run without losing the terminal recording.
  cp "$scratch/demo.cast" "$run_folder/demo.cast"
}

if [[ -z "$video" ]]; then
  record_run
elif [[ ! -f "$video" ]]; then
  echo "demo: no video at $video" >&2
  exit 2
else
  # A run this script recorded keeps its terminal recording beside its video
  # folder; the pair goes into docs/ together.
  saved_cast="$(dirname "$(dirname "$video")")/demo.cast"
  [[ -f "$saved_cast" ]] && cp "$saved_cast" "$scratch/demo.cast"
fi

# The GIF, built by one chain of ffmpeg filters, in order:
#   setpts=PTS/speed      play faster than the run did (1 for real time)
#   fps                   keep this many frames a second
#   mpdecimate=max=30     drop frames that repeat the one before, but never more
#                         than 30 in a row: while the page stands still waiting
#                         for the AI, one frame in 31 is kept, so a wait plays
#                         about 30 times faster, while clicks play at full speed
#   setpts=N/fps/TB       space the frames that are left evenly again
#   scale, palette...     shrink it, with a palette made from the video's own
#                         colours, which keeps text sharp and the file small
echo "== making the GIF from ${video}" >&2
"$FFMPEG" -loglevel error -y -i "$video" \
  -vf "setpts=PTS/${speed},fps=${fps},mpdecimate=max=30,setpts=N/${fps}/TB,scale=${width}:-1:flags=lanczos,split[a][b];[a]palettegen=stats_mode=diff[p];[b][p]paletteuse=dither=bayer:bayer_scale=5:diff_mode=rectangle" \
  -loop 0 "$scratch/demo.gif"

size="$(wc -c < "$scratch/demo.gif")"
if ((size > MAX_GIF_BYTES)); then
  echo "demo: the GIF came out at ${size} bytes, over the ${MAX_GIF_BYTES} limit." \
    "Try again with a smaller --width or --fps, using: --video ${video}" >&2
  exit 2
fi

mkdir -p docs
mv "$scratch/demo.gif" docs/demo.gif
echo "== docs/demo.gif: ${size} bytes" >&2
if [[ -f "$scratch/demo.cast" ]]; then
  mv "$scratch/demo.cast" docs/demo.cast
  echo "== docs/demo.cast: $(wc -c < docs/demo.cast) bytes" >&2
fi
