from __future__ import annotations

from datetime import datetime
from pathlib import Path


def dated_audio_dir(
    root: Path,
    timestamp_ns: int | None = None,
    subdir: str | None = None,
) -> Path:
    dt = local_datetime(timestamp_ns)
    path = root / f"{dt.year:04d}" / f"{dt.month:02d}" / f"{dt.day:02d}"
    if subdir:
        path = path / subdir
    return path


def local_datetime(timestamp_ns: int | None = None) -> datetime:
    if timestamp_ns is None:
        return datetime.now().astimezone()
    return datetime.fromtimestamp(timestamp_ns / 1_000_000_000).astimezone()


def timestamped_audio_filename(
    role: str,
    suffix: str,
    timestamp_ns: int | None = None,
    extra: str | None = None,
) -> str:
    dt = local_datetime(timestamp_ns)
    parts = [dt.strftime("%H-%M-%S-%f"), role]
    if extra:
        parts.append(extra)
    return f"{'_'.join(parts)}{suffix}"


def dated_audio_path(
    root: Path,
    role: str,
    suffix: str = ".wav",
    timestamp_ns: int | None = None,
    extra: str | None = None,
    subdir: str | None = None,
) -> Path:
    return dated_audio_dir(root, timestamp_ns, subdir) / timestamped_audio_filename(
        role=role,
        suffix=suffix,
        timestamp_ns=timestamp_ns,
        extra=extra,
    )
