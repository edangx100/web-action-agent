"""Command-line entry point.

Only the command surface is defined here; the agent loop is not implemented yet.

One thing it does do already is print, before any command runs, what this
installation is actually able to do with the keys it has been given — see
:mod:`web_action_agent.capabilities`. It goes to standard error so that piping
a command's output somewhere does not swallow it, and ``--help`` prints without
it, because asking what the commands are must work on a checkout with no
configuration at all.
"""

import sys

from web_action_agent.capabilities import report_capabilities
from web_action_agent.config import ConfigError, load_config, resolve_env

USAGE = """usage: waa <command> [args]

commands:
  run "<task>"          run one natural-language task end to end
  eval                  run the benchmark suite with layered verdicts
  induce <run-folder>   distill successful traces into candidate skills
  skills                list, show, or retire library skills
  judge <runs-dir>      produce verdicts and the agreement report
"""

COMMANDS = {"run", "eval", "induce", "skills", "judge"}

#: Exit code for a request the program cannot honour as asked -- an unknown
#: command, or a configuration that does not resolve.
EXIT_MISUSE = 2


def main() -> int:
    argv = sys.argv[1:]
    if not argv or argv[0] in {"-h", "--help"}:
        print(USAGE)
        return 0

    command = argv[0]
    if command not in COMMANDS:
        print(f"waa: unknown command {command!r}\n", file=sys.stderr)
        print(USAGE, file=sys.stderr)
        return EXIT_MISUSE

    try:
        env = resolve_env()
        config = load_config(env=env)
    except ConfigError as error:
        print(f"waa: {error}", file=sys.stderr)
        return EXIT_MISUSE

    print(report_capabilities(config, env=env).render(), file=sys.stderr)

    print(f"waa: {command!r} is not implemented yet", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
