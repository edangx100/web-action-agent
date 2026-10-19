# shellcheck shell=bash
# One step of an experiment script, with retries. Sourced, not run:
#
#   source "$(dirname "${BASH_SOURCE[0]}")/step.sh"
#
# Expects the sourcing script to set `retry_wait`, the seconds to pause
# between tries.

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
