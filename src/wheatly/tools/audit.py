from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

from wheatly.tools.registry import ToolCall, ToolResult


def log_tool_event(
    path: str,
    call: ToolCall,
    result: ToolResult,
    *,
    source: str,
    duration_seconds: float,
    call_index: Optional[int] = None,
) -> None:
    if not path:
        return
    record = {
        "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
        "source": source,
        "tool": call.name,
        "arguments": call.arguments or {},
        "result": asdict(result),
        "duration_seconds": round(duration_seconds, 4),
    }
    if call_index is not None:
        record["call_index"] = call_index
    try:
        log_path = Path(path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=True, default=str) + "\n")
    except OSError:
        pass
