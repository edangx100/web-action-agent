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

**Curation lives here, because this is where the numbers change.** A skill that
has stopped working is a fact about its record -- so many replays, so few that
worked -- and the only moment that record changes is when a replay is written
down. Deciding retirement anywhere else would mean something, somewhere, having
to remember to look.

**Retirement only goes one way, and that is the design rather than a
simplification.** A retired skill is left out of what the planner is offered, so
it is never replayed, so its success rate can never recover. The library can
forget and cannot spontaneously remember. What replaces a retired skill is not
the skill getting another chance; it is the agent working the route out by hand
on its next run, and that run being learnt from in the usual way. A site that
changed gets a new recipe, not a rehabilitated old one.

**One writer at a time is assumed, and that assumption has an edge.** Recording
a replay reads the skill from disk, adds one, and writes it back. Two processes
doing that to the same skill at the same moment can lose an update: both read
"five uses", both write "six". A benchmark running several workers over one
library is exactly that situation.

A lost count only moves a retirement slightly, which is tolerable. The case that
is not is a worker writing back a copy it read *before* another worker retired
the skill, which would quietly bring a retired skill back into service. Nothing
here prevents that today. The fix is a lock around the read-and-write, and the
right moment to add one is when a harness that actually runs workers in parallel
exists, because its shape decides whether the lock belongs here or around the
whole pass. Until then this is a known edge rather than a surprise.

**What this module does not do.** It does not decide which skills are relevant
to a task -- that is retrieval, and it reads from here. The store knows how to
keep files, and the one judgment it makes about their contents is the arithmetic
of whether a skill has earned its place.
"""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from pydantic import ValidationError

from .config import Config
from .skill_schema import Skill, SkillStats
from .trace import utc_now

#: Where a library lives when nobody says otherwise. Relative, so it sits beside
#: the run folders in whatever directory the agent is being used from, and
#: git-ignored, because the working library is a local cache the inducer
#: rewrites -- the curated one that gets published is committed separately.
DEFAULT_SKILLS_ROOT = Path("skills")

#: What a skill file is called. The name is the filename, which is why the
#: schema restricts what a name may contain: a library should be the same shape
#: on every machine, and not every filesystem agrees about the rest of Unicode.
SKILL_SUFFIX = ".json"


@dataclass(frozen=True)
class RetirementPolicy:
    """When a skill has failed often enough to be taken out of service.

    A tiny amount of arithmetic, kept as its own record for two reasons. It can
    be reasoned about without a folder or a browser anywhere near it, which is
    what makes the edges -- a skill with no uses, a skill at exactly the
    threshold -- cheap to pin down. And the numbers belong to the run's
    configuration rather than to this code, because the right values depend on
    how flaky the sites being visited are, and that is a thing an experiment
    should be able to change without a code change.

    Attributes:
        rate: The success rate a skill must keep. Below this it is retired.
        min_sample: Replays required before retirement may trigger at all. The
            reason it exists: a brand-new skill that fails once is at 0%, and
            without a floor on the sample the first unlucky replay would retire
            a skill that works.
    """

    rate: float = 0.70
    min_sample: int = 5

    @classmethod
    def from_config(cls, config: Config) -> RetirementPolicy:
        """Build the policy a run is configured for.

        Args:
            config: The run's resolved settings.

        Returns:
            RetirementPolicy: The configured thresholds.
        """
        return cls(rate=config.skill_retirement_rate, min_sample=config.skill_retirement_min_sample)

    def verdict_on(self, stats: SkillStats) -> str | None:
        """Say whether a record has fallen below the bar, and why.

        Args:
            stats: The skill's record so far, including the replay just
                written down.

        Returns:
            str | None: The reason to retire, written for a person reading the
            library months later, or None to leave the skill in service. The
            sentence carries the numbers because "retired for a low success
            rate" is not something anybody can check, and "3 of 11 replays
            worked" is.
        """
        if stats.uses < self.min_sample:
            # Not enough evidence yet. A skill is innocent until it has had a
            # fair number of chances to be guilty.
            return None
        achieved = stats.success_rate
        if achieved is None or achieved >= self.rate:
            return None
        return (
            f"{stats.successes} of {stats.uses} replays worked "
            f"({achieved:.0%}), below the {self.rate:.0%} the library requires"
        )


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

    def __init__(self, root: Path | None = None, *, policy: RetirementPolicy | None = None) -> None:
        """Point a store at a folder.

        Args:
            root: Where the skills live. Defaults to
                :data:`DEFAULT_SKILLS_ROOT`. A folder that does not exist is
                a legitimate and common state -- it is what an agent that has
                learnt nothing yet looks like -- so it is not created or
                complained about here.
            policy: When a skill has failed often enough to be retired. Held by
                the store rather than passed to each call, so that every caller
                recording a replay applies the same bar without having to
                remember to -- a library curated by two different rules
                depending on who was writing would be a library whose
                retirements mean nothing.
        """
        self.root = root if root is not None else DEFAULT_SKILLS_ROOT
        self.policy = policy or RetirementPolicy()

    @classmethod
    def from_config(cls, config: Config, root: Path | None = None) -> SkillStore:
        """Build a store curated by the run's configured thresholds.

        Args:
            config: The run's resolved settings.
            root: Where the skills live.

        Returns:
            SkillStore: A store that retires by the configured rule.
        """
        return cls(root, policy=RetirementPolicy.from_config(config))

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

    def record_use(self, name: str, *, succeeded: bool) -> Skill:
        """Write down one replay of a skill, and retire it if it has earned that.

        The only moment a skill's record changes, and therefore the only moment
        retirement is decided. Reading from disk rather than taking a caller's
        copy is deliberate: a run may have been holding that skill since before
        the replay, and writing a stale copy back would undo whatever else had
        been recorded in between.

        Args:
            name: The skill that was replayed.
            succeeded: Whether the replay ran every step. A replay that handed
                control back partway is a failure here even though the run may
                have gone on to succeed -- what is being counted is whether the
                *recipe* worked, not whether the task did.

        Returns:
            Skill: The skill as it now stands, retired or not.

        Raises:
            SkillStoreError: If the library has no such skill, or it cannot be
                read or written. A replay of a skill that is not there means
                something upstream is inconsistent, and silently creating one
                would hide it.
        """
        skill = self.find(name)
        if skill is None:
            raise SkillStoreError(f"cannot record a use of {name!r}: the library has no such skill")

        stats = SkillStats(
            uses=skill.stats.uses + 1, successes=skill.stats.successes + (1 if succeeded else 0)
        )
        updated = skill.model_copy(update={"stats": stats})

        # Only a skill still in service can be retired. One already retired
        # keeps counting -- its record is the evidence for why it went -- but
        # its retirement reason and date stay as they were, because the day it
        # was taken out of service is a fact and not a running total.
        reason = self.policy.verdict_on(stats) if skill.active else None
        if reason is not None:
            updated = updated.model_copy(
                update={"active": False, "retired_reason": reason, "retired_at": utc_now()}
            )

        self.save(updated)
        return updated

    def retire(self, name: str, reason: str) -> Skill:
        """Take a skill out of service by hand.

        Separate from the automatic rule, and for a different situation: a
        person who can see something the arithmetic cannot. A skill that works
        perfectly and does the wrong thing has a flawless record and should
        still go.

        Args:
            name: The skill to retire.
            reason: Why, for whoever reads the library later. Recorded exactly
                as given rather than dressed up, because a human reason is
                usually better than anything this code could write.

        Returns:
            Skill: The retired skill.

        Raises:
            SkillStoreError: If the library has no such skill.
        """
        skill = self.find(name)
        if skill is None:
            raise SkillStoreError(f"cannot retire {name!r}: the library has no such skill")
        retired = skill.model_copy(
            update={"active": False, "retired_reason": reason, "retired_at": utc_now()}
        )
        self.save(retired)
        return retired

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
