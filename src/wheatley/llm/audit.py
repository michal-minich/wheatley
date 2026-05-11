from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from wheatley.llm.base import LLMMessage, LLMResponse


def log_system_llm_event(
    path: str,
    *,
    purpose: str,
    mode: str,
    model_name: str,
    messages: List[LLMMessage],
    duration_seconds: float,
    response: Optional[LLMResponse] = None,
    error: Optional[BaseException] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    if not path:
        return
    record: Dict[str, Any] = {
        "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
        "purpose": purpose,
        "mode": mode,
        "model_name": model_name,
        "messages": [message.to_dict() for message in messages],
        "duration_seconds": round(duration_seconds, 4),
        "ok": error is None,
    }
    if metadata:
        record["metadata"] = metadata
    if response is not None:
        record["response"] = {
            "content": response.content,
            "raw": response.raw,
        }
    if error is not None:
        record["error"] = {
            "type": type(error).__name__,
            "message": str(error),
        }
    try:
        log_path = Path(path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=True, default=str) + "\n")
    except OSError:
        pass
