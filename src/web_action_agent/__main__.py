"""Enables `python -m web_action_agent`, the no-install entry point.

A bare task string is treated as `run <task>`, so
`python -m web_action_agent "<task>"` works without naming a subcommand.
Only multi-word arguments are taken as tasks — a single unrecognized word is a
mistyped subcommand, and reporting it as such beats silently running it as a
task.
"""

import sys

from .cli import COMMANDS, main


def _looks_like_a_task(arg: str) -> bool:
    return arg not in COMMANDS and not arg.startswith("-") and len(arg.split()) > 1


if __name__ == "__main__":
    if len(sys.argv) > 1 and _looks_like_a_task(sys.argv[1]):
        sys.argv.insert(1, "run")
    raise SystemExit(main())
