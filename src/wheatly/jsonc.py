from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


def load_jsonc(path: Path) -> Any:
    try:
        text = path.read_text(encoding="utf-8")
        return json.loads(strip_jsonc(text))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON/JSONC in {path}: {exc}") from exc


def loads_jsonc(text: str, source: str = "<string>") -> Any:
    try:
        return json.loads(strip_jsonc(text))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON/JSONC in {source}: {exc}") from exc


def strip_jsonc(text: str) -> str:
    text = _strip_comments(text)
    return re.sub(r",(\s*[}\]])", r"\1", text)


def _strip_comments(text: str) -> str:
    output: list[str] = []
    in_string = False
    escape = False
    in_line_comment = False
    in_block_comment = False
    index = 0
    while index < len(text):
        char = text[index]
        next_char = text[index + 1] if index + 1 < len(text) else ""

        if in_line_comment:
            if char == "\n":
                in_line_comment = False
                output.append(char)
            index += 1
            continue

        if in_block_comment:
            if char == "*" and next_char == "/":
                in_block_comment = False
                index += 2
            else:
                if char == "\n":
                    output.append("\n")
                index += 1
            continue

        if in_string:
            output.append(char)
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            index += 1
            continue

        if char == '"':
            in_string = True
            output.append(char)
            index += 1
            continue

        if char == "/" and next_char == "/":
            in_line_comment = True
            index += 2
            continue

        if char == "/" and next_char == "*":
            in_block_comment = True
            index += 2
            continue

        output.append(char)
        index += 1
    return "".join(output)
