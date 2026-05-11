from __future__ import annotations

import mimetypes
import platform
import re
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Optional

from wheatley.audio.log_paths import dated_audio_path
from wheatley.config import Config
from wheatley.tools.registry import ToolResult


def take_photo(cfg: Config, args: Dict[str, object]) -> ToolResult:
    del args
    output_path = dated_audio_path(
        Path(cfg.runtime.data_dir) / "photos",
        "photo",
        suffix=".jpg",
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.unlink(missing_ok=True)
    short_side = _bounded_int(cfg.tools.photo_short_side, 160, 1080, 640)
    quality = _bounded_int(cfg.tools.photo_quality, 30, 95, 75)
    timeout = max(1.0, float(cfg.tools.photo_timeout_seconds or 8.0))

    command = _photo_command(cfg, output_path, short_side, quality)
    if not command:
        return ToolResult(
            name="take_photo",
            ok=False,
            content={
                "error": "camera_command_not_available",
                "hint": (
                    "Install imagesnap, fswebcam, libcamera-still/rpicam-still, "
                    "or ffmpeg, or configure tools.photo_command."
                ),
            },
        )

    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            shell=False,
            timeout=timeout,
        )
    except FileNotFoundError as exc:
        return ToolResult(
            name="take_photo",
            ok=False,
            content={"error": "camera_command_not_found", "command": command[0]},
        )
    except subprocess.TimeoutExpired as exc:
        return ToolResult(
            name="take_photo",
            ok=False,
            content={
                "error": "camera_timeout",
                "command": command[0],
                "timeout_seconds": timeout,
                "stderr": (exc.stderr or "")[-800:],
            },
        )

    if completed.returncode != 0:
        return ToolResult(
            name="take_photo",
            ok=False,
            content={
                "error": "camera_command_failed",
                "command": command[0],
                "returncode": completed.returncode,
                "stderr": completed.stderr[-800:],
            },
        )
    if not output_path.exists() or output_path.stat().st_size <= 0:
        return ToolResult(
            name="take_photo",
            ok=False,
            content={
                "error": "camera_output_missing",
                "command": command[0],
                "path": str(output_path),
                "stderr": completed.stderr[-800:],
            },
        )

    _resize_photo_if_possible(output_path, short_side, quality)
    dimensions = _photo_dimensions(output_path)
    return ToolResult(
        name="take_photo",
        ok=True,
        content={
            "path": str(output_path),
            "mime_type": mimetypes.guess_type(str(output_path))[0] or "image/jpeg",
            "bytes": output_path.stat().st_size,
            "width": dimensions[0],
            "height": dimensions[1],
            "short_side": short_side,
            "quality": quality,
            "command": command[0],
            "stderr": completed.stderr[-800:],
        },
    )


def _photo_command(
    cfg: Config,
    output_path: Path,
    short_side: int,
    quality: int,
) -> Optional[List[str]]:
    if cfg.tools.photo_command:
        return [
            part.format(
                output=str(output_path),
                short_side=short_side,
                quality=quality,
            )
            for part in cfg.tools.photo_command
        ]
    return _auto_photo_command(output_path, short_side, quality)


def _auto_photo_command(
    output_path: Path,
    short_side: int,
    quality: int,
) -> Optional[List[str]]:
    landscape_size = _landscape_source_size(short_side)
    if shutil.which("imagesnap"):
        return ["imagesnap", "-w", "1", str(output_path)]
    if shutil.which("fswebcam"):
        return [
            "fswebcam",
            "-r",
            landscape_size,
            "--jpeg",
            str(quality),
            "--no-banner",
            str(output_path),
        ]
    if shutil.which("libcamera-still"):
        return _rpicam_command("libcamera-still", output_path, short_side, quality)
    if shutil.which("rpicam-still"):
        return _rpicam_command("rpicam-still", output_path, short_side, quality)
    if shutil.which("ffmpeg"):
        system = platform.system().lower()
        if system == "darwin":
            camera = _macos_avfoundation_camera()
            source_size = _macos_source_size(camera)
            return [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-f",
                "avfoundation",
                "-pixel_format",
                "uyvy422",
                "-framerate",
                "30",
                "-video_size",
                source_size,
                "-i",
                f"{camera[0]}:none",
                "-ss",
                "2",
                "-frames:v",
                "1",
                "-vf",
                _short_side_scale_filter(short_side),
                str(output_path),
            ]
        if system == "linux" and Path("/dev/video0").exists():
            return [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-f",
                "v4l2",
                "-video_size",
                landscape_size,
                "-i",
                "/dev/video0",
                "-frames:v",
                "1",
                "-vf",
                _short_side_scale_filter(short_side),
                str(output_path),
            ]
    return None


def _macos_avfoundation_camera() -> tuple[int, str]:
    output = ""
    try:
        completed = subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-f",
                "avfoundation",
                "-list_devices",
                "true",
                "-i",
                "",
            ],
            capture_output=True,
            text=True,
            shell=False,
            timeout=4,
        )
        output = (completed.stderr or "") + "\n" + (completed.stdout or "")
    except Exception:
        return (0, "FaceTime HD Camera")

    cameras = _parse_avfoundation_video_devices(output)
    for index, name in cameras:
        lowered = name.lower()
        if (
            "iphone" in lowered
            and "camera" in lowered
            and "desk view" not in lowered
        ):
            return (index, name)
    for index, name in cameras:
        lowered = name.lower()
        if "screen" not in lowered and "desk view" not in lowered:
            return (index, name)
    return (0, "FaceTime HD Camera")


def _parse_avfoundation_video_devices(output: str) -> List[tuple[int, str]]:
    devices: List[tuple[int, str]] = []
    in_video_section = False
    for line in output.splitlines():
        if "AVFoundation video devices:" in line:
            in_video_section = True
            continue
        if "AVFoundation audio devices:" in line:
            in_video_section = False
            continue
        if not in_video_section:
            continue
        match = re.search(r"\[(\d+)\]\s+(.+)$", line)
        if match:
            devices.append((int(match.group(1)), match.group(2).strip()))
    return devices


def _macos_source_size(camera: tuple[int, str]) -> str:
    name = camera[1].lower()
    if "iphone" in name:
        return "1920x1440"
    return "1080x1920"


def _landscape_source_size(short_side: int) -> str:
    if short_side <= 480:
        return "640x480"
    if short_side <= 720:
        return "1280x720"
    return "1920x1080"


def _short_side_scale_filter(short_side: int) -> str:
    return (
        f"scale='if(lt(iw,ih),{short_side},-2)':"
        f"'if(lt(iw,ih),-2,{short_side})'"
    )


def _rpicam_command(
    binary: str,
    output_path: Path,
    short_side: int,
    quality: int,
) -> List[str]:
    width, height = _landscape_source_size(short_side).split("x", 1)
    return [
        binary,
        "--width",
        width,
        "--height",
        height,
        "--quality",
        str(quality),
        "-o",
        str(output_path),
        "--nopreview",
        "--timeout",
        "1000",
    ]


def _resize_photo_if_possible(
    output_path: Path,
    short_side: int,
    quality: int,
) -> None:
    temp_path = output_path.with_name(output_path.stem + "_small.jpg")
    if shutil.which("ffmpeg"):
        command = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(output_path),
            "-vf",
            _short_side_scale_filter(short_side),
            "-frames:v",
            "1",
            "-q:v",
            "5",
            str(temp_path),
        ]
    elif shutil.which("sips"):
        width, height = _photo_dimensions(output_path)
        if width is None or height is None:
            return
        resize_flag = "--resampleWidth" if width <= height else "--resampleHeight"
        command = [
            "sips",
            resize_flag,
            str(short_side),
            "--setProperty",
            "format",
            "jpeg",
            "--setProperty",
            "formatOptions",
            str(quality),
            str(output_path),
            "--out",
            str(temp_path),
        ]
    else:
        return

    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            shell=False,
            timeout=8,
        )
    except Exception:
        return
    if completed.returncode == 0 and temp_path.exists() and temp_path.stat().st_size > 0:
        temp_path.replace(output_path)
    elif temp_path.exists():
        temp_path.unlink(missing_ok=True)


def _photo_dimensions(path: Path) -> tuple[int | None, int | None]:
    if not shutil.which("sips"):
        return (None, None)
    try:
        completed = subprocess.run(
            ["sips", "-g", "pixelWidth", "-g", "pixelHeight", str(path)],
            capture_output=True,
            text=True,
            shell=False,
            timeout=4,
        )
    except Exception:
        return (None, None)
    if completed.returncode != 0:
        return (None, None)
    width = _metadata_value(completed.stdout, "pixelWidth")
    height = _metadata_value(completed.stdout, "pixelHeight")
    return (width, height)


def _metadata_value(text: str, key: str) -> int | None:
    match = re.search(rf"\b{re.escape(key)}:\s*(\d+)", text)
    if not match:
        return None
    return int(match.group(1))


def _bounded_int(value: int, minimum: int, maximum: int, fallback: int) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return fallback
    return min(max(number, minimum), maximum)
