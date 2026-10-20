#!/usr/bin/env bash
# DOM only versus DOM plus screenshots, start to finish.
#
# What do screenshots actually buy? The agent reads each page as text -- the
# numbered list of things to click, and the page's words -- and adds an
# annotated screenshot only when a cheap rule says the text may be missing
# something. This runs the benchmark twice on the same planner model:
#
#   openrouter        the usual setting: a screenshot when the rule calls for one
#   openrouter-text   the same profile with no screenshot model, so no image
#                     can be sent at all
#
#   waa eval --cold --profile <profile> --out runs/modality-<profile>
#
# The results land in results/dom_vs_dual.md: success rate, tokens, time and
# cost for each arm, the change between them, and a check, from each call's own
# record, that no request in the text-only arm carried an image.
#
# Usage (from the project folder; `uv run` puts the project's `waa` on the path):
#   uv run scripts/dom_vs_dual.sh [--profiles "A B"] [--trials N] [--attempts N]
#
#   --profiles  the profiles to run, in one quoted list
#               (default: "openrouter openrouter-text")
#   --trials    trials per task in each arm (default: 2)
#   --attempts  tries per arm before giving up (default: 3)
#
# This spends real money and visits real websites. Each arm has its own
# ceiling (default $15 and 6 hours, see WAA_BUDGET_USD / WAA_BUDGET_HOURS).
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

profiles="openrouter openrouter-text"
trials="2"
attempts="3"
# Seconds between tries of an arm. Overridable so the tests need not wait.
retry_wait="${RETRY_WAIT_S:-60}"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --profiles) profiles="${2:?--profiles needs a list of names}"; shift 2 ;;
    --trials) trials="${2:?--trials needs a number}"; shift 2 ;;
    --attempts) attempts="${2:?--attempts needs a number}"; shift 2 ;;
    -h|--help) awk 'NR > 1 && /^#/ { sub(/^# ?/, ""); print; next } NR > 1 { exit }' "$0"; exit 0 ;;
    *) echo "dom_vs_dual: unknown argument: $1" >&2; exit 2 ;;
  esac
done

# require_waa, require_committed, step: see helpers.sh.
source "$(dirname "${BASH_SOURCE[0]}")/helpers.sh"
require_waa
require_committed

# One arm per profile, one after another, so each arm has the websites to itself.
for profile in $profiles; do
  step "$attempts" "arm ${profile}" \
    "$WAA" eval --cold --trials "$trials" --profile "$profile" --out "runs/modality-${profile}"
done

echo "== done: results/dom_vs_dual.md" >&2
