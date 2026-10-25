"""Every link on the front page leads somewhere.

The README is what a visitor reads first, and most of what it claims it backs
with a link: the example run's folder, each results table, the screenshots in
the failure analysis. A link that leads nowhere turns a claim back into an
assertion, and nothing else would notice: GitHub renders a broken link exactly
like a working one.

So these tests read the README the way a browser would, find every link to a
file in this repository, and check each one: the file or folder exists, it is
not one git ignores (which would exist here but be missing from every clone),
and a link to a section (``#results``) names a heading that is really there.
The checker is tested on a small page of its own as well, so a checker that
silently found no links could not pass.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path
from urllib.parse import unquote, urlparse

import pytest

REPO = Path(__file__).resolve().parent.parent
"""Path: The repository's top folder."""

README = REPO / "README.md"
"""Path: The front page."""

EXAMPLE_README = REPO / "runs" / "example" / "README.md"
"""Path: The committed example run's own page, linked from the front page."""

#: Fenced code blocks (``` or ~~~), whose contents are shown, not followed.
FENCE = re.compile(r"^(```|~~~).*?^\1[^\n]*$", re.MULTILINE | re.DOTALL)

#: Inline code, such as `runs/<run>`, which is also shown rather than followed.
INLINE_CODE = re.compile(r"`[^`\n]*`")

#: A Markdown link or image: [text](target) or ![alt](target), with an
#: optional "title" after the target.
MARKDOWN_LINK = re.compile(r"!?\[[^\]]*\]\(\s*<?([^)\s>]+)>?(?:\s+\"[^\"]*\")?\s*\)")

#: A link or image written as HTML: href="..." or src="...".
HTML_LINK = re.compile(r"\b(?:href|src)=\"([^\"]+)\"")

#: A Markdown heading: one to six #, a space, then the text.
HEADING = re.compile(r"^#{1,6}\s+(.+?)\s*#*\s*$", re.MULTILINE)


def visible_text(markdown: str) -> str:
    """Remove the parts of a page that are shown as code rather than read.

    Args:
        markdown: The page's source.

    Returns:
        str: The source without code blocks or inline code.
    """
    return INLINE_CODE.sub("", FENCE.sub("", markdown))


def links_in(markdown: str) -> list[str]:
    """Find every link and image target on a page, in order.

    Args:
        markdown: The page's source.

    Returns:
        list[str]: Each target as written, such as ``results/README.md`` or
        ``#demo``.
    """
    text = visible_text(markdown)
    return MARKDOWN_LINK.findall(text) + HTML_LINK.findall(text)


def is_local(target: str) -> bool:
    """Whether a link points into this repository rather than to a website.

    Args:
        target: A link target as written.

    Returns:
        bool: False for web addresses and e-mail links.
    """
    return not urlparse(target).scheme


def slug(heading: str) -> str:
    """Turn a heading into the anchor GitHub gives it.

    GitHub lowercases the text, drops punctuation (keeping letters, digits,
    spaces, hyphens and underscores), and turns each space into a hyphen:
    "1. Does learning skills help?" becomes ``1-does-learning-skills-help``.

    Args:
        heading: The heading's text, without the leading #.

    Returns:
        str: The anchor.
    """
    return re.sub(r"[^\w\- ]", "", heading.strip().lower()).replace(" ", "-")


def anchors_in(markdown: str) -> set[str]:
    """Every anchor a page's headings give it.

    A heading that repeats an earlier one gets ``-1``, ``-2``... after it, as on
    GitHub, so a link can reach each of them.

    Args:
        markdown: The page's source.

    Returns:
        set[str]: The anchors.
    """
    seen: dict[str, int] = {}
    anchors = set()
    for heading in HEADING.findall(FENCE.sub("", markdown)):
        base = slug(heading)
        count = seen.get(base, 0)
        anchors.add(base if count == 0 else f"{base}-{count}")
        seen[base] = count + 1
    return anchors


def kept_by_git(path: Path) -> bool:
    """Whether a clone of this repository would have this file or folder.

    A file is kept unless the ignore rules exclude it. A folder is kept only if
    at least one file inside it is: the rules here keep every folder under
    ``runs/`` and ignore the files in them, so a run folder can exist on this
    machine and still arrive empty in a clone.

    Args:
        path: A file or folder inside the repository.

    Returns:
        bool: True if a clone would have it, with something in it.
    """
    files = [path] if path.is_file() else [p for p in path.rglob("*") if p.is_file()]
    names = [p.relative_to(REPO).as_posix() for p in files]
    if not names:
        return False
    # One question to git for every file at once. --no-index asks about the
    # rules alone, so a file not yet committed is judged like a committed one.
    ignored = subprocess.run(
        ["git", "check-ignore", "--no-index", "--stdin"],
        input="\n".join(names),
        cwd=REPO,
        capture_output=True,
        text=True,
        check=False,
    ).stdout.splitlines()
    return len(ignored) < len(names)


def problems(page: Path) -> list[str]:
    """Check every local link on a page, and say what is wrong with any.

    Args:
        page: A Markdown file in this repository.

    Returns:
        list[str]: One line per broken link, naming the link and the fault.
        Empty when every link works.
    """
    found = []
    for target in links_in(page.read_text(encoding="utf-8")):
        if not is_local(target):
            continue
        path_part, _, anchor = target.partition("#")
        # A link with no path is to a section of the same page.
        destination = (page.parent / unquote(path_part)).resolve() if path_part else page
        if not destination.exists():
            found.append(f"{target}: no such file or folder")
            continue
        if REPO in destination.parents and not kept_by_git(destination):
            found.append(f"{target}: git ignores it, so a clone would not have it")
            continue
        is_page = destination.suffix == ".md"
        if anchor and is_page and anchor not in anchors_in(destination.read_text(encoding="utf-8")):
            found.append(f"{target}: no heading with the anchor #{anchor}")
    return found


# --- The checker itself ---------------------------------------------------------------


@pytest.mark.unit
def test_the_checker_finds_every_kind_of_link() -> None:
    page = """
[a table](results/a.md) and ![a picture](docs/b.png "title")
<img src="docs/c.png"> <a href="#d">d</a> [site](https://example.com)
```bash
[not a link](inside/code.md)
```
`[not this either](inline/code.md)`
"""

    # Markdown links first, then HTML ones; nothing from inside code.
    assert links_in(page) == [
        "results/a.md",
        "docs/b.png",
        "https://example.com",
        "docs/c.png",
        "#d",
    ]


@pytest.mark.unit
def test_headings_become_githubs_anchors() -> None:
    page = "# Demo\n## 1. Does learning skills help?\n## Results\n## Results\n"

    assert anchors_in(page) == {"demo", "1-does-learning-skills-help", "results", "results-1"}


@pytest.mark.unit
def test_a_broken_link_is_reported(tmp_path: Path) -> None:
    (tmp_path / "here.md").write_text("x")
    page = tmp_path / "page.md"
    page.write_text(
        "# Top\n[ok](here.md) [ok too](#top) [gone](missing.md) [no section](#nowhere)\n"
        "[web](https://example.com)\n"
    )

    assert problems(page) == [
        "missing.md: no such file or folder",
        "#nowhere: no heading with the anchor #nowhere",
    ]


@pytest.mark.unit
def test_a_link_to_a_folder_git_ignores_is_reported(tmp_path: Path) -> None:
    # A run folder other than the example: here on this machine, empty in a
    # clone. Made for the test inside runs/, where git ignores files, and
    # removed afterwards.
    local_run = REPO / "runs" / "_readme_link_check"
    local_run.mkdir(parents=True)
    try:
        (local_run / "run.json").write_text("{}")
        page = tmp_path / "page.md"
        page.write_text(
            f"[a local run]({local_run}/) [the example]({REPO / 'runs' / 'example'}/)\n"
        )

        assert problems(page) == [
            f"{local_run}/: git ignores it, so a clone would not have it",
        ]
    finally:
        shutil.rmtree(local_run)


# --- The README -------------------------------------------------------------------------


@pytest.mark.unit
def test_every_link_in_the_readme_works() -> None:
    assert problems(README) == []


@pytest.mark.unit
def test_every_link_in_the_example_runs_page_works() -> None:
    # The front page sends readers here, and this page links back to it.
    assert problems(EXAMPLE_README) == []


@pytest.mark.unit
@pytest.mark.parametrize(
    "target",
    [
        "runs/example/",
        "results/README.md",
        "results/cold_vs_warm.md",
        "results/multi_model.md",
        "results/endpoint_modes.md",
        "results/dom_vs_dual.md",
    ],
)
def test_the_readme_links_the_example_run_and_every_results_table(target: str) -> None:
    # Each headline number is backed by a link to the table it came from, and
    # the example run by a link to the folder itself.
    assert target in links_in(README.read_text(encoding="utf-8"))


@pytest.mark.unit
def test_the_readme_links_its_own_sections() -> None:
    # The contents line at the top is only useful if it is there and works.
    targets = links_in(README.read_text(encoding="utf-8"))

    assert {"#demo", "#results-at-a-glance", "#quick-start", "#why-tasks-fail"} <= set(targets)
