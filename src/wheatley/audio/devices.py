from __future__ import annotations

from typing import Any, Dict, List, Optional

from wheatley.config import AudioConfig


_AUTO_INPUT_KEYWORDS = [
    "bluetooth",
    "headset",
    "hands-free",
    "handsfree",
    "airpods",
    "buds",
    "jabra",
    "plantronics",
    "poly",
    "bose",
    "beats",
    "logitech",
    "steelseries",
    "wh-",
    "wf-",
    "srs",
]

_BUILT_IN_INPUT_KEYWORDS = [
    "built-in",
    "built in",
    "macbook",
    "mac mini",
    "studio display",
]


def input_stream_device_kwargs(cfg: AudioConfig, sd) -> Dict[str, Any]:
    device = resolve_input_device(cfg, sd)
    if device is None:
        return {}
    return {"device": device}


def resolve_input_device(cfg: AudioConfig, sd) -> Optional[int]:
    if cfg.input_device_index is not None:
        return int(cfg.input_device_index)
    name = (cfg.input_device_name or "").strip()
    if not name:
        return _auto_input_device(cfg, sd.query_devices())
    return _find_input_device_by_name(name, sd.query_devices())


def list_audio_devices() -> List[Dict[str, Any]]:
    try:
        import sounddevice as sd
    except ImportError:
        return []
    return normalize_sounddevice_devices(sd.query_devices())


def normalize_sounddevice_devices(raw_devices) -> List[Dict[str, Any]]:
    devices = []
    for index, device in enumerate(raw_devices):
        devices.append(
            {
                "index": index,
                "name": str(device.get("name", "")),
                "max_input_channels": int(device.get("max_input_channels", 0) or 0),
                "max_output_channels": int(device.get("max_output_channels", 0) or 0),
                "default_samplerate": float(device.get("default_samplerate", 0.0) or 0.0),
            }
        )
    return devices


def _find_input_device_by_name(name: str, raw_devices) -> int:
    devices = normalize_sounddevice_devices(raw_devices)
    device = _match_input_device(name, devices)
    if device is not None:
        return int(device["index"])
    available = ", ".join(
        f"{item['index']}:{item['name']}" for item in _input_devices(devices)
    ) or "none"
    raise RuntimeError(
        f"Configured audio.input_device_name '{name}' was not found. "
        f"Available input devices: {available}"
    )


def _auto_input_device(cfg: AudioConfig, raw_devices) -> Optional[int]:
    mode = (cfg.input_device_mode or "default").strip().lower()
    if mode in {"", "default", "system"}:
        return None
    if mode not in {"auto", "prefer_headset", "prefer_bluetooth"}:
        raise RuntimeError(
            "Invalid audio.input_device_mode "
            f"'{cfg.input_device_mode}'. Use 'default' or 'auto'."
        )
    devices = normalize_sounddevice_devices(raw_devices)
    for preferred in cfg.input_device_preferred_names:
        device = _match_input_device(str(preferred), devices)
        if device is not None:
            return int(device["index"])
    for device in _input_devices(devices):
        name = device["name"].casefold()
        if any(keyword in name for keyword in _AUTO_INPUT_KEYWORDS) and not any(
            keyword in name for keyword in _BUILT_IN_INPUT_KEYWORDS
        ):
            return int(device["index"])
    return None


def _match_input_device(name: str, devices) -> Optional[Dict[str, Any]]:
    needle = name.casefold()
    input_devices = _input_devices(devices)
    exact = [
        item
        for item in input_devices
        if item["name"].casefold() == needle
    ]
    if exact:
        return int(exact[0]["index"])
    partial = [
        item
        for item in input_devices
        if needle in item["name"].casefold()
    ]
    if partial:
        return partial[0]
    return None


def _input_devices(devices):
    return [item for item in devices if item["max_input_channels"] > 0]
