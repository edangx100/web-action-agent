"""The library: a folder of skill files, and the few things done to it.

One directory, one JSON file per skill, named after the skill. That is the whole
storage design, and it is a decision rather than an omission -- the spec rules
out a database, and the reason is that the library is meant to be *looked at*. A
skill is the most interesting artefact this project produces: it is what the
agent learned. Somebody should be able to open the folder, read a file, disagree
with it, and delete it, without a tool and without this program running.

**Writing is validated, always.** Every skill goes through its own schema on the
way to disk, so a file in the library is a file that parses -- and in
particular, a skill with no ``verification`` block cannot be written at all. The
alternative is a library that accepts anything and discovers the problem during
a replay, on a page, halfway through somebody's benchmark.

**Writing is atomic.** A skill is written to a temporary file and moved into
place, so an interrupted write leaves either the old file or the new one and
never half of either. This matters more than it looks: the inducer rewrites the
library, and a crash during that pass should not be able to leave a skill file
that is valid JSON up to the point the power went out.

**Reading is strict, and says which file was wrong.** A library with one corrupt
file reports that file rather than failing vaguely or skipping it. Skipping is
the dangerous option: a skill that silently disappears from the library looks
exactly like a skill that was never induced, and the run that should have used
it just costs more for no visible reason.

**What this module does not do.** It does not decide which skills are relevant
to a task -- that is retrieval, and it reads from here. It does not decide when
a skill has stopped working -- that is curation, and it writes through here. The
store knows how to keep files and nothing about what is in them beyond the
schema.
"""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Iterator
from pathlib import Path

from pydantic import ValidationError

from .skill_schema import Skill

#: Where a library lives when nobody says otherwise. Relative, so it sits beside
#: the run folders in whatever directory the agent is being used from, and
#: git-ignored, because the working library is a local cache the inducer
#: rewrites -- the curated one that gets published is committed separately.
DEFAULT_SKILLS_ROOT = Path("skills")

#: What a skill file is called. The name is the filename, which is why the
#: schema restricts what a name may contain: a library should be the same shape
#: on every machine, and not every filesystem agrees about the rest of Unicode.
SKILL_SUFFIX = ".json"


class SkillStoreError(Exception):
    """A skill could not be read or written.

    Covers a folder that cannot be created, a file that is not JSON, and a file
    that is JSON but not a skill. All three are faults in the library rather
    than in the task being attempted, which is why they are one exception: a
    caller can do nothing about any of them except say which file.
    """


class SkillStore:
    """One folder of skills, and the operations a library needs.

    Satisfies the structural type the validator's gate asks of a skill library
    -- a :meth:`find` returning something with a name, a domain, an active flag
    and its parameter names -- without either component importing the other.
    That is deliberate on the gate's side and worth preserving on this one: the
    two are built and changed separately.

    Attributes:
        root: The directory the skills live in. Created on the first write
            rather than on construction, so that pointing at a library and
            finding it empty does not leave an empty folder behind.
    """

    def __init__(self, root: Path | None = None) -> None:
        """Point a store at a folder.

        Args:
            root: Where the skills live. Defaults to
                :data:`DEFAULT_SKILLS_ROOT`. A folder that does not exist is
                a legitimate and common state -- it is what an agent that has
                learnt nothing yet looks like -- so it is not created or
                complained about here.
        """
        self.root = root if root is not None else DEFAULT_SKILLS_ROOT

    def path_for(self, name: str) -> Path:
        """Say which file a skill by this name lives in.

        Args:
            name: The skill's name.

        Returns:
            Path: Its file, whether or not it exists.
        """
        return self.root / f"{name}{SKILL_SUFFIX}"

    def find(self, name: str) -> Skill | None:
        """Look one skill up by name.

        The question the validator's gate asks, and the reason its answer
        includes retired skills: "that skill was retired" and "there is no such
        skill" are different problems and the planner can act on the difference.
        Deciding what to do about a retired one is the gate's business, not the
        library's.

        Args:
            name: The name the planner used.

        Returns:
            Skill | None: The skill, active or retired, or None if the library
            has never heard of it.

        Raises:
            SkillStoreError: If the file exists but cannot be read as a skill.
                Distinguished from "not found" on purpose: a corrupt skill is
                something to fix, and reporting it as absent would hide it
                behind a run that merely costs more than it should.
        """
        path = self.path_for(name)
        if not path.is_file():
            return None
        return self._read(path)

    def save(self, skill: Skill) -> Path:
        """Write one skill into the library, replacing any skill of that name.

        Args:
            skill: The skill to store. Already valid by construction -- it
                could not have been built otherwise -- which is what makes "no
                skill without a verification block" a property of the library
                rather than a check somebody has to remember.

        Returns:
            Path: The file it was written to.

        Raises:
            SkillStoreError: If the folder or the file cannot be written.
        """
        try:
            self.root.mkdir(parents=True, exist_ok=True)
        except OSError as error:
            raise SkillStoreError(
                f"cannot create the skill library at {self.root}: {error}"
            ) from error

        path = self.path_for(skill.name)
        text = json.dumps(skill.to_dict(), indent=2, ensure_ascii=False) + "\n"
        self._write_atomically(path, text)
        return path

    def names(self) -> tuple[str, ...]:
        """List every skill the library holds, by name.

        Returns:
            tuple[str, ...]: The names, sorted. Sorted rather than left in
            whatever order the filesystem offers, so that two machines listing
            the same library agree -- and so a committed listing does not
            reshuffle itself in a diff.
        """
        if not self.root.is_dir():
            return ()
        return tuple(sorted(path.stem for path in self.root.glob(f"*{SKILL_SUFFIX}")))

    def load_all(self) -> tuple[Skill, ...]:
        """Read every skill in the library.

        Retired skills are included. What to leave out of a prompt is
        retrieval's decision, and a store that pre-filtered would make the
        library's contents depend on which caller was asking.

        Returns:
            tuple[Skill, ...]: Every skill, in name order.

        Raises:
            SkillStoreError: If any file cannot be read as a skill, naming the
                file. One bad file fails the read rather than being skipped: a
                skill that quietly vanishes from a library is indistinguishable
                from one that was never learnt.
        """
        return tuple(self._read(self.path_for(name)) for name in self.names())

    def __iter__(self) -> Iterator[Skill]:
        """Iterate over every skill in the library.

        Yields:
            Skill: Each skill, in name order.
        """
        yield from self.load_all()

    def __len__(self) -> int:
        """Returns: int: How many skills the library holds, retired included."""
        return len(self.names())

    def delete(self, name: str) -> bool:
        """Remove a skill from the library entirely.

        Rarely the right thing, and separate from retirement for that reason:
        retiring a skill keeps it and its statistics, which are the evidence for
        why it stopped being used. Deleting is for a skill that should never
        have been written -- one induced from a run that turned out to be wrong.

        Args:
            name: The skill to remove.

        Returns:
            bool: True if a file was removed, False if there was none. A
            missing skill is not an error to delete: the caller wanted it gone
            and it is gone.

        Raises:
            SkillStoreError: If the file exists and cannot be removed.
        """
        path = self.path_for(name)
        try:
            path.unlink()
        except FileNotFoundError:
            return False
        except OSError as error:
            raise SkillStoreError(f"cannot remove {path}: {error}") from error
        return True

    def _read(self, path: Path) -> Skill:
        """Read one skill file.

        Args:
            path: The file to read.

        Returns:
            Skill: What it holds.

        Raises:
            SkillStoreError: If it cannot be read, is not JSON, or is JSON that
                is not a skill. The message names the file, because a library
                is a folder and "a skill is invalid" does not say which one.
        """
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except OSError as error:
            raise SkillStoreError(f"cannot read {path}: {error}") from error
        except json.JSONDecodeError as error:
            raise SkillStoreError(f"{path} is not valid JSON: {error}") from error

        try:
            skill = Skill.from_dict(data)
        except ValidationError as error:
            raise SkillStoreError(f"{path} is not a valid skill: {error}") from error

        if skill.name != path.stem:
            # The filename is how `find` locates a skill, so a file whose
            # contents name a different skill is unreachable by the name it
            # claims -- and would be found under a name it does not answer to.
            raise SkillStoreError(
                f"{path} holds a skill called {skill.name!r}; a skill's file must be its name"
            )
        return skill

    def _write_atomically(self, path: Path, text: str) -> None:
        """Put text at a path without ever leaving it half-written.

        Written to a temporary file in the same directory and then moved onto
        the target. A move within one filesystem is atomic, so a reader sees
        either the old file or the new one -- never a truncated skill, which
        would be a file that parses as JSON right up to where the writer
        stopped.

        Args:
            path: Where the file should end up.
            text: What to write.

        Raises:
            SkillStoreError: If the write fails. The temporary file is removed
                first, so a failed save does not leave litter beside the
                library.
        """
        temporary: str | None = None
        try:
            # Same directory as the target, because `os.replace` is only atomic
            # within one filesystem and the system temp directory is often on
            # another one. Inside the `try` because making the temporary file
            # is itself a write, and fails for the same reasons the real one
            # would -- a read-only folder, a full disk.
            descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                handle.write(text)
                handle.flush()
                # Committed to the disk before the move, so that the file the
                # move puts in place is a complete one even if the machine
                # stops immediately afterwards.
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        except OSError as error:
            if temporary is not None:
                Path(temporary).unlink(missing_ok=True)
            raise SkillStoreError(f"cannot write {path}: {error}") from error
