"""Command-line entry point.

Only the command surface is defined here; the agent loop is not implemented yet.
"""

import sys

USAGE = """usage: waa <command> [args]

commands:
  run "<task>"          run one natural-language task end to end
  eval                  run the benchmark suite with layered verdicts
  induce <run-folder>   distill successful traces into candidate skills
  skills                list, show, or retire library skills
  judge <runs-dir>      produce verdicts and the agreement report
"""

COMMANDS = {"run", "eval", "induce", "skills", "judge"}


def main() -> int:
    argv = sys.argv[1:]
    if not argv or argv[0] in {"-h", "--help"}:
        print(USAGE)
        return 0

    command = argv[0]
    if command not in COMMANDS:
        print(f"waa: unknown command {command!r}\n", file=sys.stderr)
        print(USAGE, file=sys.stderr)
        return 2

    print(f"waa: {command!r} is not implemented yet", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
