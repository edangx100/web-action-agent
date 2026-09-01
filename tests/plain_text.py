"""Reading captured output the way a person reads a terminal, not a terminal.

The command line renders its help through Rich, which styles the text --
colour, bold, dim rules -- by wrapping fragments in ANSI escape sequences. On a
screen those are invisible and the output reads as prose. In a captured string
they are characters, and they land *inside* words:

    '\\x1b[1;36m-\\x1b[0m\\x1b[1;36m-start\\x1b[0m\\x1b[1;36m-url\\x1b[0m'

Rich styles each hyphen-separated piece on its own, so ``--start-url`` arrives
as three fragments and the whole flag is no longer anywhere in the string --
nor is ``--start``, though ``-url`` is.

So a test asserting that ``--start-url`` appears in the help fails, on output
that a reader would say plainly contains it.

The other direction is the one to watch, because it is silent. A test
asserting that something is *absent* passes for the wrong reason if styling
splits the string it looks for, and goes on passing while the thing it was
written to catch is happening.

Today's forbidden strings -- ``Args:`` and ``Raises:``, which must never leak
from a docstring into a user's terminal -- are safe, because Rich wraps whole
prose words rather than splitting them, and only takes option-shaped tokens
apart. That is luck rather than design: the day one of those assertions is
written about a flag name, it goes quiet instead of red. Stripping first
removes the distinction, so no assertion here depends on which kind of string
it happens to name.

Whether the styling is emitted at all depends on the environment: Rich prints
plain text when nothing is watching, and turns colour on when it detects a
continuous-integration runner. That is a sensible default -- CI logs render
colour -- but it means the same assertion passes on a laptop and fails on a
build server, which is the least useful way for a test to fail.

The fix is to assert on what the words say rather than on how they are drawn.
Every test that looks at command-line output puts it through :func:`plain`
first.
"""

from __future__ import annotations

import re

ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;:?]*[ -/]*[@-~]")
"""re.Pattern[str]: One ANSI control sequence.

Matches the ``CSI`` form, which is everything Rich emits when styling text:
an escape, ``[``, some parameter bytes, and a final letter -- ``\\x1b[0m`` to
reset, ``\\x1b[1;36m`` for bold cyan.
"""


def plain(text: str) -> str:
    """Strip terminal styling, leaving the words.

    Args:
        text: Output captured from a command, styled or not.

    Returns:
        str: The same text with every ANSI control sequence removed. Text that
        was never styled passes through unchanged, so this is safe to apply
        everywhere rather than only where colour is expected.
    """
    return ANSI_ESCAPE.sub("", text)
