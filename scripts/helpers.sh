# shellcheck shell=bash
# Helpers shared by the experiment scripts. Sourced, not run:
#
#   source "$(dirname "${BASH_SOURCE[0]}")/helpers.sh"
#
#   require_waa         stop unless the `waa` command can be found
#   require_committed   stop unless the code is committed
#   step                run one step, with retries
#
# Expects the sourcing script to set `WAA` (the command to run), and, for
# `step`, `retry_wait` (the seconds to pause between tries); `require_committed`
# also reads `GIT`, defaulting to git. Messages start with the script's own name.

# Stops the script unless the `waa` command can be found. Said before anything
# starts: run outside the project's environment, `waa` is simply not on the
# path, and the shell's own "command not found" does not say how to fix it.
require_waa() {
  if ! command -v "$WAA" > /dev/null 2>&1; then
    echo "${0##*/}: cannot find the '$WAA' command. Run this from the project" \
      "folder as: uv run scripts/${0##*/} ... (or set WAA to the command's path)." >&2
    exit 2
  fi
}

# Stops the script if the code has uncommitted changes, before any money is
# spent: a run must be on one known commit for its results to show which code
# ran. The paths are the ones the harness checks when it records a sweep's
# code version; the results tables every sweep rewrites are left out, or the
# second sweep would never start. Outside a git checkout it only warns.
require_committed() {
  local repo changes
  repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
  if changes="$("${GIT:-git}" -C "$repo" status --porcelain --untracked-files=no \
      -- src benchmarks pyproject.toml uv.lock 2>/dev/null)"; then
    if [[ -n "$changes" ]]; then
      echo "${0##*/}: commit these changes first, so every arm runs one known commit:" >&2
      echo "$changes" >&2
      exit 2
    fi
  else
    echo "${0##*/}: warning: not a git checkout, so the results cannot show" \
      "that every arm ran the same code." >&2
  fi
}

# Runs one step, and stops the script if it did not finish.
#   $1 - how many tries it gets (1 for no retrying)
#   $2 - what the step is, for the message
#   $@ - the command
# A try that ends with exit 1 (did not finish) is followed by another, after a
# pause, until the tries run out. Any other failure stops at once: a typo or a
# bad setting fails the same way every time.
step() {
  local tries="$1" what="$2"
  shift 2
  local try code
  for ((try = 1; try <= tries; try++)); do
    local label="$what"
    ((tries > 1)) && label="${what} (try ${try} of ${tries})"
    echo "== ${label}: $*" >&2
    code=0
    "$@" || code=$?
    if [[ $code -eq 0 ]]; then
      return 0
    elif [[ $code -ne 1 ]]; then
      echo "${0##*/}: ${what} could not run (exit ${code})." >&2
      exit "$code"
    elif [[ $try -lt $tries ]]; then
      echo "${0##*/}: ${what} did not finish; trying again in ${retry_wait}s" \
        "(finished trials are kept)." >&2
      sleep "$retry_wait"
    fi
  done
  echo "${0##*/}: ${what} did not finish (exit 1). Read the message above;" \
    "if a ceiling stopped it, or the AI service is down, run this script again later." >&2
  exit 1
}
