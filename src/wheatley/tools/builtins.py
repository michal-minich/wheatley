from __future__ import annotations

import ast
import json
import math
import operator
import platform
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Dict

from wheatley.config import Config
from wheatley.language import language_status_payload
from wheatley.prompting import load_tool_overrides_from_config
from wheatley.tools.photo import take_photo
from wheatley.tools.python_interpreter import python_interpreter
from wheatley.tools.registry import ToolRegistry, ToolResult, ToolSpec
from wheatley.tools.web import web_search, web_search_available

STARTED_AT = time.time()


def build_registry(cfg: Config) -> ToolRegistry:
    registry = ToolRegistry()
    overrides = load_tool_overrides_from_config(cfg)

    def register(spec: ToolSpec, handler) -> None:
        if cfg.tools.is_tool_enabled(spec.name):
            registry.register(spec, handler)

    def tool_description(name: str, fallback: str) -> str:
        override = overrides.get(name, {})
        description = str(override.get("description", "")).strip()
        return description or fallback

    register(
        ToolSpec(
            name="get_time",
            description=tool_description("get_time", "Return local date/time details."),
            parameters={"type": "object", "properties": {}},
        ),
        lambda args: _get_time(),
    )
    register(
        ToolSpec(
            name="system_status",
            description=tool_description("system_status", "Return system runtime state."),
            parameters={"type": "object", "properties": {}},
        ),
        lambda args: _system_status(cfg),
    )
    register(
        ToolSpec(
            name="set_eye_expression",
            description=tool_description(
                "set_eye_expression",
                "Set assistant eye expression.",
            ),
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
    register(
        ToolSpec(
            name="calculator",
            description=tool_description("calculator", "Evaluate a safe math expression."),
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
    register(
        ToolSpec(
            name="remember",
            description=tool_description("remember", "Persist a short memory."),
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
    register(
        ToolSpec(
            name="take_photo",
            description=tool_description(
                "take_photo",
                "Capture a small camera photo and attach it to vision-capable LLMs.",
            ),
            parameters={"type": "object", "properties": {}},
        ),
        lambda args: take_photo(cfg, args),
    )
    if cfg.tools.allowed_commands:
        register(
            ToolSpec(
                name="run_safe_cli_tool",
                description=tool_description("run_safe_cli_tool", "Run a whitelisted local command."),
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
    if cfg.tools.is_tool_enabled("web_search") and web_search_available(cfg):
        register(
            ToolSpec(
                name="web_search",
                description=tool_description("web_search", "Search the public web."),
                parameters={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "max_results": {"type": "integer"},
                    },
                    "required": ["query"],
                },
            ),
            lambda args: web_search(cfg, args),
        )
    register(
        ToolSpec(
            name="python_interpreter",
            description=tool_description(
                "python_interpreter",
                "Run a bounded Python scratchpad.",
            ),
            parameters={
                "type": "object",
                "properties": {
                    "code": {"type": "string"},
                    "input": {"type": "object"},
                },
                "required": ["code"],
            },
        ),
        lambda args: python_interpreter(cfg, args),
    )
    for name, override in overrides.items():
        registry.update_spec(
            name,
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
        },
    )


def _system_status(cfg: Config) -> ToolResult:
    state_path = Path(cfg.runtime.state_dir) / "eye.json"
    eye = {}
    if state_path.exists():
        try:
            eye = json.loads(state_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            eye = {"error": "invalid_eye_state"}
    return ToolResult(
        name="system_status",
        ok=True,
        content={
            "platform": platform.platform(),
            "python": platform.python_version(),
            "llm_backend": cfg.llm.backend,
            "stt_backend": cfg.stt.backend,
            "tts_backend": cfg.resolved_tts_backend(),
            "uptime_seconds": round(time.time() - STARTED_AT, 3),
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
        path.write_text("# Wheatley Memory\n\n", encoding="utf-8")
    timestamp = datetime.now().astimezone().isoformat(timespec="seconds")
    with path.open("a", encoding="utf-8") as handle:
        handle.write(f"- {timestamp}: {memory}\n")
    return ToolResult(
        name="remember",
        ok=True,
        content={"memory": memory, "path": str(path)},
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


SAFE_BINARY_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

SAFE_UNARY_OPERATORS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}

SAFE_NAMES = {
    "pi": math.pi,
    "e": math.e,
    "tau": math.tau,
}

SAFE_FUNCTIONS = {
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
    if isinstance(node, ast.BinOp) and type(node.op) in SAFE_BINARY_OPERATORS:
        left = _eval_node(node.left)
        right = _eval_node(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > 100:
            raise ValueError("exponent too large")
        return SAFE_BINARY_OPERATORS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in SAFE_UNARY_OPERATORS:
        return SAFE_UNARY_OPERATORS[type(node.op)](_eval_node(node.operand))
    if isinstance(node, ast.Name) and node.id in SAFE_NAMES:
        return SAFE_NAMES[node.id]
    if isinstance(node, ast.List) or isinstance(node, ast.Tuple):
        return [_eval_node(item) for item in node.elts]
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        func = SAFE_FUNCTIONS.get(node.func.id)
        if not func:
            raise ValueError(f"function not allowed: {node.func.id}")
        if node.keywords:
            raise ValueError("keyword arguments are not allowed")
        args = [_eval_node(arg) for arg in node.args]
        return func(*args)
    raise ValueError(f"unsupported expression: {ast.dump(node, include_attributes=False)}")
