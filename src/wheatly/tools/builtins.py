from __future__ import annotations

import json
import ast
import math
import operator
import platform
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Dict

from wheatly.config import Config
from wheatly.language import language_status_payload, set_language_state
from wheatly.prompting import load_tool_overrides
from wheatly.tools.registry import ToolRegistry, ToolResult, ToolSpec

STARTED_AT = time.time()


def build_registry(cfg: Config) -> ToolRegistry:
    registry = ToolRegistry()

    registry.register(
        ToolSpec(
            name="get_time",
            description="Return local date, time, timezone and uptime.",
            parameters={"type": "object", "properties": {}},
        ),
        lambda args: _get_time(),
    )
    registry.register(
        ToolSpec(
            name="robot_status",
            description="Return basic runtime, platform and robot state.",
            parameters={"type": "object", "properties": {}},
        ),
        lambda args: _robot_status(cfg),
    )
    registry.register(
        ToolSpec(
            name="set_eye_expression",
            description="Set the robot eye expression state.",
            parameters={
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "enum": [
                            "neutral",
                            "happy",
                            "confused",
                            "thinking",
                            "worried",
                            "excited",
                        ],
                    }
                },
                "required": ["expression"],
            },
        ),
        lambda args: _set_eye_expression(cfg, args),
    )
    registry.register(
        ToolSpec(
            name="calculator",
            description="Evaluate a safe math expression with functions like sqrt, sin, gcd and round.",
            parameters={
                "type": "object",
                "properties": {
                    "expression": {"type": "string"},
                    "round_digits": {"type": "integer"},
                },
                "required": ["expression"],
            },
        ),
        lambda args: _calculator(args),
    )
    registry.register(
        ToolSpec(
            name="remember",
            description="Persist a short user-provided memory for future chats.",
            parameters={
                "type": "object",
                "properties": {
                    "memory": {"type": "string"},
                },
                "required": ["memory"],
            },
        ),
        lambda args: _remember(cfg, args),
    )
    registry.register(
        ToolSpec(
            name="set_language",
            description="Switch the active conversation language between English and Slovak.",
            parameters={
                "type": "object",
                "properties": {
                    "language": {
                        "type": "string",
                        "enum": sorted(cfg.language.languages),
                    }
                },
                "required": ["language"],
            },
        ),
        lambda args: _set_language(cfg, args),
    )
    registry.register(
        ToolSpec(
            name="take_photo",
            description="Capture a photo if a configured safe camera command exists.",
            parameters={"type": "object", "properties": {}},
        ),
        lambda args: _take_photo(cfg),
    )
    registry.register(
        ToolSpec(
            name="run_safe_cli_tool",
            description="Run one whitelisted local command by name.",
            parameters={
                "type": "object",
                "properties": {
                    "command": {"type": "string"},
                    "args": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["command"],
            },
        ),
        lambda args: _run_safe_cli_tool(cfg, args),
    )

    for name, override in load_tool_overrides(cfg.prompts.tools_path).items():
        registry.update_spec(
            name,
            description=override.get("description", ""),
            instructions=override.get("instructions", ""),
        )

    return registry


def _get_time() -> ToolResult:
    now = datetime.now().astimezone()
    return ToolResult(
        name="get_time",
        ok=True,
        content={
            "iso": now.isoformat(timespec="seconds"),
            "timezone": now.tzname(),
            "uptime_seconds": round(time.time() - STARTED_AT, 3),
        },
    )


def _robot_status(cfg: Config) -> ToolResult:
    state_path = Path(cfg.runtime.state_dir) / "eye.json"
    eye = {}
    if state_path.exists():
        try:
            eye = json.loads(state_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            eye = {"error": "invalid_eye_state"}
    return ToolResult(
        name="robot_status",
        ok=True,
        content={
            "agent": cfg.agent.name,
            "platform": platform.platform(),
            "python": platform.python_version(),
            "llm_backend": cfg.llm.backend,
            "stt_backend": cfg.stt.backend,
            "tts_backend": cfg.resolved_tts_backend(),
            "language": language_status_payload(cfg),
            "eye": eye or {"expression": "neutral"},
        },
    )


def _set_eye_expression(cfg: Config, args: Dict[str, object]) -> ToolResult:
    expression = str(args.get("expression", "neutral"))
    allowed = {"neutral", "happy", "confused", "thinking", "worried", "excited"}
    if expression not in allowed:
        return ToolResult(
            name="set_eye_expression",
            ok=False,
            content={"error": "invalid_expression", "allowed": sorted(allowed)},
        )
    Path(cfg.runtime.state_dir).mkdir(parents=True, exist_ok=True)
    state = {"expression": expression, "updated_at": datetime.now().isoformat()}
    (Path(cfg.runtime.state_dir) / "eye.json").write_text(
        json.dumps(state, indent=2), encoding="utf-8"
    )
    return ToolResult(name="set_eye_expression", ok=True, content=state)


def _calculator(args: Dict[str, object]) -> ToolResult:
    expression = str(args.get("expression", "")).strip()
    if not expression:
        return ToolResult(name="calculator", ok=False, content={"error": "empty_expression"})
    try:
        value = _eval_math(expression)
        round_digits = args.get("round_digits")
        if round_digits is not None:
            digits = int(round_digits)
            value = round(value, digits)
            result_display = f"{value:.{digits}f}"
        else:
            result_display = str(value)
    except Exception as exc:
        return ToolResult(
            name="calculator",
            ok=False,
            content={"expression": expression, "error": str(exc)},
        )
    return ToolResult(
        name="calculator",
        ok=True,
        content={"expression": expression, "result": value, "result_display": result_display},
    )


def _remember(cfg: Config, args: Dict[str, object]) -> ToolResult:
    memory = str(args.get("memory", "")).strip()
    if not memory:
        return ToolResult(name="remember", ok=False, content={"error": "empty_memory"})
    path = Path(cfg.prompts.memory_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text("# Wheatly Memory\n\n", encoding="utf-8")
    timestamp = datetime.now().astimezone().isoformat(timespec="seconds")
    with path.open("a", encoding="utf-8") as handle:
        handle.write(f"- {timestamp}: {memory}\n")
    return ToolResult(
        name="remember",
        ok=True,
        content={"memory": memory, "path": str(path)},
    )


def _set_language(cfg: Config, args: Dict[str, object]) -> ToolResult:
    ok, content = set_language_state(cfg, str(args.get("language", "")))
    return ToolResult(name="set_language", ok=ok, content=content)


def _take_photo(cfg: Config) -> ToolResult:
    if not cfg.tools.photo_command:
        return ToolResult(
            name="take_photo",
            ok=False,
            content={"error": "photo_command_not_configured"},
        )
    output_path = Path(cfg.runtime.data_dir) / "photo_latest.jpg"
    command = [part.format(output=str(output_path)) for part in cfg.tools.photo_command]
    completed = subprocess.run(
        command, capture_output=True, text=True, shell=False, timeout=8
    )
    return ToolResult(
        name="take_photo",
        ok=completed.returncode == 0,
        content={
            "path": str(output_path),
            "returncode": completed.returncode,
            "stderr": completed.stderr[-800:],
        },
    )


def _run_safe_cli_tool(cfg: Config, args: Dict[str, object]) -> ToolResult:
    command_name = str(args.get("command", ""))
    base = cfg.tools.allowed_commands.get(command_name)
    if not base:
        return ToolResult(
            name="run_safe_cli_tool",
            ok=False,
            content={
                "error": "command_not_allowed",
                "allowed": sorted(cfg.tools.allowed_commands),
            },
        )
    extra = args.get("args", [])
    if not isinstance(extra, list):
        extra = []
    command = base + [str(part) for part in extra]
    completed = subprocess.run(
        command, capture_output=True, text=True, shell=False, timeout=5
    )
    return ToolResult(
        name="run_safe_cli_tool",
        ok=completed.returncode == 0,
        content={
            "command": command_name,
            "returncode": completed.returncode,
            "stdout": completed.stdout[-2000:],
            "stderr": completed.stderr[-1000:],
        },
    )


_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

_UNARY_OPS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}

_NAMES = {
    "pi": math.pi,
    "e": math.e,
    "tau": math.tau,
}

_FUNCS = {
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "sum": sum,
    "sqrt": math.sqrt,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "asin": math.asin,
    "acos": math.acos,
    "atan": math.atan,
    "atan2": math.atan2,
    "degrees": math.degrees,
    "radians": math.radians,
    "log": math.log,
    "log10": math.log10,
    "log2": math.log2,
    "exp": math.exp,
    "floor": math.floor,
    "ceil": math.ceil,
    "trunc": math.trunc,
    "factorial": math.factorial,
    "gcd": math.gcd,
    "lcm": math.lcm,
    "pow": pow,
}


def _eval_math(expression: str) -> int | float:
    tree = ast.parse(expression, mode="eval")
    return _eval_node(tree.body)


def _eval_node(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        left = _eval_node(node.left)
        right = _eval_node(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > 100:
            raise ValueError("exponent too large")
        return _BIN_OPS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_eval_node(node.operand))
    if isinstance(node, ast.Name) and node.id in _NAMES:
        return _NAMES[node.id]
    if isinstance(node, ast.List) or isinstance(node, ast.Tuple):
        return [_eval_node(item) for item in node.elts]
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        func = _FUNCS.get(node.func.id)
        if not func:
            raise ValueError(f"function not allowed: {node.func.id}")
        if node.keywords:
            raise ValueError("keyword arguments are not allowed")
        args = [_eval_node(arg) for arg in node.args]
        return func(*args)
    raise ValueError(f"unsupported expression: {ast.dump(node, include_attributes=False)}")
