#!/usr/bin/env bash
# The endpoint-modes check, start to finish.
#
# Does the agent reach a model through configuration alone? This runs one
# benchmark task three ways, one after another, with nothing changed between
# them but the endpoint profile:
#
#   openrouter   an aggregator, which routes to a host it picks (pinned here)
#   nvidia       the model maker's own service, build.nvidia.com
#   local        a vLLM server you run yourself, at http://127.0.0.1:8000/v1
#
#   waa eval --cold --task <task> --profile <profile> --out runs/modes-<profile>
#
# The results land in results/endpoint_modes.md, with a check, from what each
# sweep recorded, that every arm ran the same commit with nothing uncommitted.
#
# Usage (from the project folder; `uv run` puts the project's `waa` on the path):
#   uv run scripts/endpoint_modes.sh [--profiles "A B C"] [--task ID] [--trials N]
#                                    [--attempts N]
#
#   --profiles  the profiles to run, in one quoted list
#               (default: "openrouter nvidia local")
#   --task      the benchmark task to run (default: cpf-employee-share, which
#               is marked by plain code, so no AI judge is needed)
#   --trials    trials of the task in each arm (default: 1)
#   --attempts  tries per arm before giving up (default: 3)
#
# Each profile needs its key in .env (WAA_OPENROUTER_API_KEY, WAA_NVIDIA_API_KEY;
# the local server needs none), and the local profile needs a vLLM server
# listening on port 8000, serving the model under the name the profile uses.
# The full-precision model needs an 80 GB GPU (A100 or H100); with vLLM 0.12
# or later and the reasoning parser downloaded from the model's Hugging Face
# page:
#
#   vllm serve nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 --trust-remote-code \
#     --reasoning-parser-plugin nano_v3_reasoning_parser.py \
#     --reasoning-parser nano_v3 --max-model-len 32768 --port 8000 \
#     --served-model-name nvidia/nemotron-3-nano-30b-a3b
#
# The server can be on another machine, such as a rented GPU, reached through
# an SSH tunnel that makes it appear at port 8000 here; the profile stays as it
# is. A profile whose server cannot be reached stops the script there.
#
# Commit every change to the code first: the script refuses to start while the
# agent's code, settings, tasks or dependencies have uncommitted changes.
#
# Exit codes: 0 when every arm is complete; 1 when an arm stopped early (a
# ceiling was reached, or the AI service was too busy to start) -- run it again
# or read the message; 2 when a command could not run at all, or the code has
# uncommitted changes.

set -euo pipefail

# The commands to run. Overridable so the tests can put stand-ins in their place.
WAA="${WAA:-waa}"
GIT="${GIT:-git}"

profiles="openrouter nvidia local"
task="cpf-employee-share"
trials="1"
attempts="3"
# Seconds between tries of an arm. Overridable so the tests need not wait.
retry_wait="${RETRY_WAIT_S:-60}"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --profiles) profiles="${2:?--profiles needs a list of names}"; shift 2 ;;
    --task) task="${2:?--task needs a task id}"; shift 2 ;;
    --trials) trials="${2:?--trials needs a number}"; shift 2 ;;
    --attempts) attempts="${2:?--attempts needs a number}"; shift 2 ;;
    -h|--help) awk 'NR > 1 && /^#/ { sub(/^# ?/, ""); print; next } NR > 1 { exit }' "$0"; exit 0 ;;
    *) echo "endpoint_modes: unknown argument: $1" >&2; exit 2 ;;
  esac
done

# require_waa, require_committed, step: see helpers.sh.
source "$(dirname "${BASH_SOURCE[0]}")/helpers.sh"
require_waa
require_committed

# One arm per profile, one after another.
for profile in $profiles; do
  step "$attempts" "arm ${profile}" \
    "$WAA" eval --cold --trials "$trials" --task "$task" --profile "$profile" \
    --out "runs/modes-${profile}"
done

echo "== done: results/endpoint_modes.md" >&2
