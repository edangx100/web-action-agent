#!/usr/bin/env bash
# The multi-model comparison, start to finish.
#
# Is the agent any good only on the model it was built with? This runs the
# same benchmark on several models, one after another, with nothing changed
# between them but the endpoint profile:
#
#   waa eval --cold --profile <profile> --out runs/models-<profile>
#
# for each profile in turn. That also tests the claim that the profile is the
# only place a model is named: if any code had to change for a model, the arms
# could not all run on one commit. Every arm is cold (an empty skill library),
# so the arms compare models, not what each has learnt.
#
# The results land in results/multi_model.md, with a check, from what each
# sweep recorded, that every arm ran the same commit with nothing uncommitted.
#
# Usage (from the project folder; `uv run` puts the project's `waa` on the path):
#   uv run scripts/multi_model.sh [--profiles "A B C"] [--trials N] [--attempts N]
#
#   --profiles  the profiles to compare, in one quoted list, primary first
#               (default: "openrouter openrouter-qwen openrouter-mistral")
#   --trials    trials per task in each arm (default: 2, fewer than the
#               headline sweeps; the table states each arm's N)
#   --attempts  tries per arm before giving up (default: 3); see below
#
# This spends real money and visits real websites. Each arm has its own
# ceiling (default $15 and 6 hours, see WAA_BUDGET_USD / WAA_BUDGET_HOURS).
#
# Commit every change to the code first: the script refuses to start while the
# agent's code, settings, tasks or dependencies have uncommitted changes,
# because the results could then not show that the arms ran the same code.
#
# An arm that ends unfinished -- some trials stopped by an AI-service error --
# is tried again after a pause (a minute; RETRY_WAIT_S changes it), up to
# --attempts times. Safe to run again: each arm resumes where it stopped.
#
# Exit codes: 0 when every arm is complete; 1 when an arm stopped early (a
# ceiling was reached) -- run it again or read the message; 2 when a command
# could not run at all, or the code has uncommitted changes.

set -euo pipefail

# The commands to run. Overridable so the tests can put stand-ins in their place.
WAA="${WAA:-waa}"
GIT="${GIT:-git}"

profiles="openrouter openrouter-qwen openrouter-mistral"
trials="2"
attempts="3"
# Seconds between tries of an arm. Overridable so the tests need not wait.
retry_wait="${RETRY_WAIT_S:-60}"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --profiles) profiles="${2:?--profiles needs a list of names}"; shift 2 ;;
    --trials) trials="${2:?--trials needs a number}"; shift 2 ;;
    --attempts) attempts="${2:?--attempts needs a number}"; shift 2 ;;
    -h|--help) sed -n '2,40p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "multi_model: unknown argument: $1" >&2; exit 2 ;;
  esac
done

if ! command -v "$WAA" > /dev/null 2>&1; then
  echo "multi_model: cannot find the '$WAA' command. Run this from the project" \
    "folder as: uv run scripts/multi_model.sh ... (or set WAA to the command's path)." >&2
  exit 2
fi

# Refuse uncommitted changes before any money is spent. The paths are the ones
# the harness checks when it records a sweep's code version: the results
# tables each arm rewrites are left out, or the second arm would never start.
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if changes="$("$GIT" -C "$repo" status --porcelain --untracked-files=no \
    -- src benchmarks pyproject.toml uv.lock 2>/dev/null)"; then
  if [[ -n "$changes" ]]; then
    echo "multi_model: commit these changes first, so every arm runs one known commit:" >&2
    echo "$changes" >&2
    exit 2
  fi
else
  echo "multi_model: warning: not a git checkout, so the results cannot show" \
    "that every arm ran the same code." >&2
fi

# step: runs one step, with retries (see step.sh).
source "$(dirname "${BASH_SOURCE[0]}")/step.sh"

# One arm per profile, one after another, so each arm has the websites to itself.
for profile in $profiles; do
  step "$attempts" "arm ${profile}" \
    "$WAA" eval --cold --trials "$trials" --profile "$profile" --out "runs/models-${profile}"
done

echo "== done: results/multi_model.md" >&2
