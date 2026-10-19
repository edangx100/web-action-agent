#!/usr/bin/env bash
# The cold-versus-warm experiment, start to finish.
#
# Does learning skills make the agent better? This runs the benchmark twice on
# one model and compares:
#
#   1. cold  - the whole suite with an EMPTY skill library
#   2. learn - turn the cold pass's passing trials into skills, each one
#              checked on a live page before it is kept (waa induce --all)
#   3. warm  - the whole suite again, with the library from step 2
#
# The results land in results/cold_vs_warm.md, beside the other tables.
#
# Usage (from the project folder; `uv run` puts the project's `waa` on the path):
#   uv run scripts/cold_vs_warm.sh [--profile NAME] [--trials N] [--attempts N]
#
#   --profile   the endpoint profile to use for both passes (default: openrouter)
#   --trials    trials per task in each pass (default: 3)
#   --attempts  tries per pass before giving up (default: 3); see below
#
# This spends real money and visits real websites. Each pass has its own
# ceiling (default $15 and 6 hours, see WAA_BUDGET_USD / WAA_BUDGET_HOURS), and
# learning makes a few more model calls on top.
#
# A pass that ends unfinished -- some trials stopped by an AI-service error,
# which are set aside and run again -- is tried again after a pause (a minute;
# RETRY_WAIT_S changes it), up to --attempts times. Finished trials are kept,
# so each try only does what is left.
#
# Safe to run again. Each pass resumes where it stopped, and learning is
# skipped once the warm pass has started, because the warm pass keeps the copy
# of the library it started with.
#
# Exit codes: 0 when both passes are complete; 1 when a step stopped early (a
# ceiling was reached, or no skill survived its check) -- run it again or read
# the message; 2 when a command could not run at all.

set -euo pipefail

# The command to run. Overridable so the tests can put a stand-in in its place.
WAA="${WAA:-waa}"

profile="openrouter"
trials="3"
attempts="3"
# Seconds between tries of a pass. Overridable so the tests need not wait.
retry_wait="${RETRY_WAIT_S:-60}"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --profile) profile="${2:?--profile needs a name}"; shift 2 ;;
    --trials) trials="${2:?--trials needs a number}"; shift 2 ;;
    --attempts) attempts="${2:?--attempts needs a number}"; shift 2 ;;
    -h|--help) sed -n '2,36p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "cold_vs_warm: unknown argument: $1" >&2; exit 2 ;;
  esac
done

# Said before anything starts: run outside the project's environment, `waa`
# is simply not on the path, and the shell's own "command not found" does not
# say how to fix it.
if ! command -v "$WAA" > /dev/null 2>&1; then
  echo "cold_vs_warm: cannot find the '$WAA' command. Run this from the project" \
    "folder as: uv run scripts/cold_vs_warm.sh ... (or set WAA to the command's path)." >&2
  exit 2
fi

# Where each pass keeps its runs. Must match the names `waa eval` picks.
cold_dir="runs/eval-${profile}-cold"
warm_dir="runs/eval-${profile}-warm"

# step: runs one step, with retries (see step.sh).
source "$(dirname "${BASH_SOURCE[0]}")/step.sh"

# 1. Cold: an empty library, by construction.
step "$attempts" "cold pass" "$WAA" eval --cold --trials "$trials" --profile "$profile"

# 2. Learn, unless the warm pass has already started: from then on its library
#    is fixed, and learning again would only spend money.
if [[ -e "$warm_dir" ]]; then
  echo "== learning: skipped, the warm pass has already started in ${warm_dir}" >&2
else
  step 1 "learning" "$WAA" induce --all "$cold_dir"
fi

# 3. Warm: the harness refuses to start if the library has no skill for the
#    suite's sites, so a forgotten or failed step 2 cannot pass for a result.
step "$attempts" "warm pass" "$WAA" eval --warm --trials "$trials" --profile "$profile"

echo "== done: results/cold_vs_warm.md" >&2
