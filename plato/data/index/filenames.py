"""Turn image filenames into structured records.

The mapping filename -> well is deterministic but experiment-specific, so it is
expressed as a named-group regex in the config rather than in code.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from ...wells import Well, WellParseError, parse_well

REQUIRED_GROUPS = {"well"}
# `field` = position within a well (NIS "Point" index, or site/FOV).
# `seq`   = acquisition sequence / timepoint index.
# `point` = the point NAME, kept because NIS repeats the well there and a
#           mismatch between it and the Well token is a real acquisition bug.
KNOWN_GROUPS = {"plate", "well", "field", "channel", "z", "seq", "point"}


@dataclass(slots=True)
class ImageRecord:
    """One image file, after filename parsing. No plate map info yet."""

    image_id: str  # path relative to the image root; stable primary key
    path: Path
    plate: str
    well: Well
    field: str | None
    channel: str | None
    z: str | None
    seq: str | None
    point: str | None
    mtime: float
    size_bytes: int


@dataclass(slots=True)
class UnparsedFile:
    path: Path
    reason: str


def compile_pattern(pattern: str) -> re.Pattern[str]:
    """Compile and sanity-check the filename pattern."""
    try:
        compiled = re.compile(pattern)
    except re.error as exc:
        raise ValueError(f"invalid filename pattern: {exc}") from exc
    groups = set(compiled.groupindex)
    missing = REQUIRED_GROUPS - groups
    if missing:
        raise ValueError(
            f"filename pattern must capture {sorted(REQUIRED_GROUPS)}; missing {sorted(missing)}"
        )
    unknown = groups - KNOWN_GROUPS
    if unknown:
        raise ValueError(
            f"filename pattern captures unknown groups {sorted(unknown)}; "
            f"allowed: {sorted(KNOWN_GROUPS)}"
        )
    return compiled


def iter_image_files(root: Path, glob: str) -> Iterator[Path]:
    """Yield candidate image files, sorted for reproducible ordering."""
    if not root.is_dir():
        raise FileNotFoundError(f"image directory does not exist: {root}")
    yield from sorted(p for p in root.glob(glob) if p.is_file())


def parse_files(
    root: Path,
    glob: str,
    pattern: str,
    *,
    plate_format: int | None = 96,
    default_plate: str = "Plate1",
    match_on: str = "name",
    channel_aliases: dict[str, str] | None = None,
) -> tuple[list[ImageRecord], list[UnparsedFile]]:
    """Parse every matching file under ``root``.

    Args:
        match_on: ``"name"`` matches the pattern against the file name only;
            ``"relpath"`` matches against the path relative to ``root``, which
            lets you capture the plate from a parent directory.
        channel_aliases: maps raw channel captures to short display names.
            Microscope channel names are often long ("Cam-DIA DIC Master
            Screening"); the raw value stays in the filename, only the indexed
            label is shortened.

    Returns:
        ``(records, unparsed)``. Unparsed files are returned rather than raised
        so the caller can report all of them at once.
    """
    if match_on not in {"name", "relpath"}:
        raise ValueError(f"match_on must be 'name' or 'relpath', got {match_on!r}")
    aliases = channel_aliases or {}
    compiled = compile_pattern(pattern)
    records: list[ImageRecord] = []
    unparsed: list[UnparsedFile] = []

    for path in iter_image_files(root, glob):
        subject = (
            path.name if match_on == "name" else path.relative_to(root).as_posix()
        )
        match = compiled.match(subject)
        if match is None:
            unparsed.append(UnparsedFile(path, "filename does not match pattern"))
            continue
        groups = match.groupdict()
        try:
            well = parse_well(groups["well"], plate_format)
        except WellParseError as exc:
            unparsed.append(UnparsedFile(path, str(exc)))
            continue
        channel = groups.get("channel")
        if channel is not None:
            channel = aliases.get(channel, channel)
        stat = path.stat()
        records.append(
            ImageRecord(
                image_id=path.relative_to(root).as_posix(),
                path=path,
                plate=(groups.get("plate") or default_plate).strip(),
                well=well,
                field=groups.get("field"),
                channel=channel,
                z=groups.get("z"),
                seq=groups.get("seq"),
                point=groups.get("point"),
                mtime=stat.st_mtime,
                size_bytes=stat.st_size,
            )
        )
    return records, unparsed
