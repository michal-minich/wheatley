from __future__ import annotations

import ast
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, List

from wheatley.config import Config
from wheatley.tools.registry import ToolResult


RUNNER = r"""
from __future__ import annotations

import ast
import contextlib
import io
import json
import os
import socket
import subprocess
import sys
import traceback
from pathlib import Path

try:
    import resource
except ImportError:
    resource = None


FORBIDDEN_NODES = (
    ast.Import,
    ast.ImportFrom,
    ast.ClassDef,
    ast.AsyncFunctionDef,
    ast.Await,
    ast.AsyncFor,
    ast.AsyncWith,
    ast.With,
    ast.Global,
    ast.Nonlocal,
)

FORBIDDEN_NAMES = {
    "__import__",
    "breakpoint",
    "builtins",
    "compile",
    "ctypes",
    "delattr",
    "dir",
    "eval",
    "exec",
    "getattr",
    "globals",
    "help",
    "importlib",
    "inspect",
    "locals",
    "memoryview",
    "multiprocessing",
    "object",
    "open",
    "os",
    "Path",
    "pathlib",
    "pkgutil",
    "resource",
    "runpy",
    "setattr",
    "shutil",
    "signal",
    "site",
    "socket",
    "subprocess",
    "super",
    "sys",
    "tempfile",
    "threading",
    "type",
    "vars",
}

FORBIDDEN_ATTRIBUTES = {
    "chmod",
    "chown",
    "execv",
    "execve",
    "fork",
    "kill",
    "mkdir",
    "makedirs",
    "open",
    "popen",
    "remove",
    "rename",
    "replace",
    "rmdir",
    "rmtree",
    "spawnl",
    "spawnle",
    "spawnlp",
    "spawnlpe",
    "spawnv",
    "spawnve",
    "spawnvp",
    "spawnvpe",
    "system",
    "unlink",
    "write",
    "writelines",
    "write_text",
    "write_bytes",
}


def _blocked(*args, **kwargs):
    raise RuntimeError("operation blocked in python_interpreter")


def _block_runtime_escape_hatches():
    socket.socket = _blocked
    socket.create_connection = _blocked
    subprocess.Popen = _blocked
    subprocess.run = _blocked
    for name in (
        "system",
        "popen",
        "fork",
        "spawnl",
        "spawnle",
        "spawnlp",
        "spawnlpe",
        "spawnv",
        "spawnve",
        "spawnvp",
        "spawnvpe",
        "execv",
        "execve",
        "remove",
        "unlink",
        "rename",
        "replace",
        "rmdir",
        "mkdir",
        "makedirs",
        "chmod",
        "chown",
    ):
        if hasattr(os, name):
            setattr(os, name, _blocked)


def _apply_resource_limits(memory_limit_mb, file_size_limit_mb):
    warnings = []
    if resource is None:
        return ["resource module unavailable"]
    limits = (
        ("RLIMIT_AS", int(memory_limit_mb) * 1024 * 1024),
        ("RLIMIT_DATA", int(memory_limit_mb) * 1024 * 1024),
        ("RLIMIT_FSIZE", int(file_size_limit_mb) * 1024 * 1024),
    )
    for name, value in limits:
        if value <= 0 or not hasattr(resource, name):
            continue
        limit = getattr(resource, name)
        try:
            soft, hard = resource.getrlimit(limit)
            del soft
            if hard not in (-1, resource.RLIM_INFINITY) and hard > 0:
                value = min(value, hard)
            resource.setrlimit(limit, (value, hard))
        except (OSError, ValueError) as exc:
            if (
                sys.platform == "darwin"
                and name in {"RLIMIT_AS", "RLIMIT_DATA"}
                and "current limit exceeds maximum limit" in str(exc)
            ):
                continue
            warnings.append(f"{name}: {exc}")
    return warnings


def _validate_user_code(code):
    try:
        tree = ast.parse(code, mode="exec")
    except SyntaxError as exc:
        return f"syntax_error: {exc}"
    for node in ast.walk(tree):
        if isinstance(node, FORBIDDEN_NODES):
            return f"blocked_node: {type(node).__name__}"
        if isinstance(node, ast.Name):
            if node.id.startswith("__") or node.id in FORBIDDEN_NAMES:
                return f"blocked_name: {node.id}"
        if isinstance(node, ast.Attribute):
            if node.attr.startswith("__") or node.attr in FORBIDDEN_ATTRIBUTES:
                return f"blocked_attribute: {node.attr}"
    return ""


def _is_relative_to(path, root):
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _safe_relative_path(path):
    if not isinstance(path, str) or not path.strip():
        raise ValueError("path must be a non-empty string")
    candidate = Path(path)
    if candidate.is_absolute():
        raise ValueError("absolute paths are not allowed")
    if ".." in candidate.parts:
        raise ValueError("parent directory paths are not allowed")
    return candidate


def _make_file_helpers(read_roots):
    roots = [Path(item).resolve(strict=False) for item in read_roots]

    def _resolve_read_file(path):
        relative = _safe_relative_path(path)
        for root in roots:
            try:
                candidate = (root / relative).resolve(strict=True)
            except OSError:
                continue
            if _is_relative_to(candidate, root) and candidate.is_file():
                return candidate
        raise FileNotFoundError(str(path))

    def read_text(path):
        return _resolve_read_file(path).read_text(encoding="utf-8")

    def read_json(path):
        return json.loads(read_text(path))

    def list_files(pattern="**/*"):
        relative = _safe_relative_path(pattern)
        matches = []
        seen = set()
        for root in roots:
            if not root.exists():
                continue
            for item in root.glob(relative.as_posix()):
                try:
                    resolved = item.resolve(strict=True)
                except OSError:
                    continue
                if not _is_relative_to(resolved, root) or not resolved.is_file():
                    continue
                label = resolved.relative_to(root).as_posix()
                if label not in seen:
                    matches.append(label)
                    seen.add(label)
                if len(matches) >= 500:
                    return sorted(matches)
        return sorted(matches)

    return read_text, read_json, list_files


def _safe_json(value, max_chars, depth=0):
    if depth > 6:
        return repr(value)[:500]
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return value[:max_chars]
    if isinstance(value, dict):
        output = {}
        for index, (key, item) in enumerate(value.items()):
            if index >= 200:
                output["..."] = "truncated"
                break
            output[str(key)[:200]] = _safe_json(item, max_chars, depth + 1)
        return output
    if isinstance(value, (list, tuple)):
        output = [_safe_json(item, max_chars, depth + 1) for item in value[:200]]
        if len(value) > 200:
            output.append("... truncated")
        return output
    if isinstance(value, set):
        items = sorted((repr(item) for item in value))[:200]
        if len(value) > 200:
            items.append("... truncated")
        return items
    isoformat = getattr(value, "isoformat", None)
    if callable(isoformat):
        try:
            return isoformat()
        except Exception:
            pass
    return repr(value)[:max_chars]


def _trim(text, limit):
    if len(text) <= limit:
        return text
    return text[:limit] + "\n... truncated"


def _safe_builtins(include_import):
    allowed = {
        "abs": abs,
        "all": all,
        "any": any,
        "bool": bool,
        "bytes": bytes,
        "dict": dict,
        "enumerate": enumerate,
        "Exception": Exception,
        "filter": filter,
        "float": float,
        "format": format,
        "frozenset": frozenset,
        "IndexError": IndexError,
        "int": int,
        "isinstance": isinstance,
        "issubclass": issubclass,
        "KeyError": KeyError,
        "len": len,
        "list": list,
        "map": map,
        "max": max,
        "min": min,
        "next": next,
        "print": print,
        "range": range,
        "repr": repr,
        "reversed": reversed,
        "round": round,
        "RuntimeError": RuntimeError,
        "set": set,
        "slice": slice,
        "sorted": sorted,
        "str": str,
        "sum": sum,
        "tuple": tuple,
        "TypeError": TypeError,
        "ValueError": ValueError,
        "ZeroDivisionError": ZeroDivisionError,
        "zip": zip,
    }
    if include_import:
        allowed["__build_class__"] = __build_class__
        allowed["__import__"] = __import__
        allowed["object"] = object
        allowed["type"] = type
    return allowed


def main():
    payload = json.loads(sys.stdin.read() or "{}")
    code = str(payload.get("code", ""))
    preamble = str(payload.get("preamble", ""))
    structured_input = payload.get("input")
    read_roots = payload.get("read_roots") or []
    max_stdout = int(payload.get("max_stdout_chars", 4000))
    max_stderr = int(payload.get("max_stderr_chars", 2000))
    max_result = int(payload.get("max_result_chars", 8000))
    memory_limit_mb = int(payload.get("memory_limit_mb", 256))
    file_size_limit_mb = int(payload.get("file_size_limit_mb", 1))

    blocked = _validate_user_code(code)
    if blocked:
        print(json.dumps({"ok": False, "error": blocked, "phase": "validation"}))
        return

    resource_warnings = _apply_resource_limits(memory_limit_mb, file_size_limit_mb)
    _block_runtime_escape_hatches()
    read_text, read_json, list_files = _make_file_helpers(read_roots)
    stdout = io.StringIO()
    stderr = io.StringIO()
    missing_result = object()
    globals_dict = {
        "__builtins__": _safe_builtins(include_import=True),
        "__name__": "__wheatley_python_interpreter__",
        "input": structured_input,
        "result": missing_result,
        "read_text": read_text,
        "read_json": read_json,
        "list_files": list_files,
    }
    try:
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            if preamble:
                exec(compile(preamble, "<python_preamble.py>", "exec"), globals_dict)
    except Exception as exc:
        print(
            json.dumps(
                {
                    "ok": False,
                    "phase": "preamble",
                    "error": str(exc),
                    "error_type": type(exc).__name__,
                    "traceback": _trim(traceback.format_exc(), max_stderr),
                    "stdout": _trim(stdout.getvalue(), max_stdout),
                    "stderr": _trim(stderr.getvalue(), max_stderr),
                    "resource_warnings": resource_warnings,
                }
            )
        )
        return

    try:
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            globals_dict["__builtins__"] = _safe_builtins(include_import=False)
            exec(compile(code, "<python_interpreter>", "exec"), globals_dict)
    except Exception as exc:
        print(
            json.dumps(
                {
                    "ok": False,
                    "phase": "code",
                    "error": str(exc),
                    "error_type": type(exc).__name__,
                    "traceback": _trim(traceback.format_exc(), max_stderr),
                    "stdout": _trim(stdout.getvalue(), max_stdout),
                    "stderr": _trim(stderr.getvalue(), max_stderr),
                    "resource_warnings": resource_warnings,
                }
            )
        )
        return

    if globals_dict.get("result") is missing_result:
        print(
            json.dumps(
                {
                    "ok": False,
                    "phase": "code",
                    "error": "missing_result",
                    "stdout": _trim(stdout.getvalue(), max_stdout),
                    "stderr": _trim(stderr.getvalue(), max_stderr),
                    "resource_warnings": resource_warnings,
                }
            )
        )
        return

    safe_result = _safe_json(globals_dict.get("result"), max_result)
    result_text = json.dumps(safe_result, ensure_ascii=True)
    truncated = len(result_text) > max_result
    if truncated:
        safe_result = result_text[:max_result] + "... truncated"
    print(
        json.dumps(
            {
                "ok": True,
                "result": safe_result,
                "result_truncated": truncated,
                "stdout": _trim(stdout.getvalue(), max_stdout),
                "stderr": _trim(stderr.getvalue(), max_stderr),
                "resource_warnings": resource_warnings,
            },
            ensure_ascii=True,
        )
    )


if __name__ == "__main__":
    main()
"""


FORBIDDEN_NODES = (
    ast.Import,
    ast.ImportFrom,
    ast.ClassDef,
    ast.AsyncFunctionDef,
    ast.Await,
    ast.AsyncFor,
    ast.AsyncWith,
    ast.With,
    ast.Global,
    ast.Nonlocal,
)

FORBIDDEN_NAMES = {
    "__import__",
    "breakpoint",
    "builtins",
    "compile",
    "ctypes",
    "delattr",
    "dir",
    "eval",
    "exec",
    "getattr",
    "globals",
    "help",
    "importlib",
    "inspect",
    "locals",
    "memoryview",
    "multiprocessing",
    "object",
    "open",
    "os",
    "Path",
    "pathlib",
    "pkgutil",
    "resource",
    "runpy",
    "setattr",
    "shutil",
    "signal",
    "site",
    "socket",
    "subprocess",
    "super",
    "sys",
    "tempfile",
    "threading",
    "type",
    "vars",
}

FORBIDDEN_ATTRIBUTES = {
    "chmod",
    "chown",
    "execv",
    "execve",
    "fork",
    "kill",
    "mkdir",
    "makedirs",
    "open",
    "popen",
    "remove",
    "rename",
    "replace",
    "rmdir",
    "rmtree",
    "spawnl",
    "spawnle",
    "spawnlp",
    "spawnlpe",
    "spawnv",
    "spawnve",
    "spawnvp",
    "spawnvpe",
    "system",
    "unlink",
    "write",
    "writelines",
    "write_text",
    "write_bytes",
}


def python_interpreter(cfg: Config, args: Dict[str, object]) -> ToolResult:
    code = str(args.get("code", "")).strip()
    if not code:
        return ToolResult(
            name="python_interpreter",
            ok=False,
            content={"error": "empty_code"},
        )

    blocked = _validate_user_code(code)
    if blocked:
        return ToolResult(
            name="python_interpreter",
            ok=False,
            content={"error": blocked, "phase": "validation"},
        )

    structured_input = args.get("input")
    if structured_input is None:
        structured_input = {}
    try:
        json.dumps(structured_input)
    except (TypeError, ValueError) as exc:
        return ToolResult(
            name="python_interpreter",
            ok=False,
            content={"error": f"input_not_json_serializable: {exc}"},
        )

    timeout = max(0.1, float(cfg.tools.python_interpreter_timeout_seconds or 30.0))
    read_roots = _resolve_read_roots(cfg)
    preamble_path = Path(cfg.profile_dir) / "python_preamble.py"
    try:
        preamble = preamble_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        preamble = ""
    except OSError as exc:
        return ToolResult(
            name="python_interpreter",
            ok=False,
            content={"error": f"preamble_read_failed: {exc}"},
        )

    payload = {
        "code": code,
        "input": structured_input,
        "preamble": preamble,
        "read_roots": [str(path) for path in read_roots],
        "max_stdout_chars": cfg.tools.python_interpreter_max_stdout_chars,
        "max_stderr_chars": cfg.tools.python_interpreter_max_stderr_chars,
        "max_result_chars": cfg.tools.python_interpreter_max_result_chars,
        "memory_limit_mb": cfg.tools.python_interpreter_memory_limit_mb,
        "file_size_limit_mb": cfg.tools.python_interpreter_file_size_limit_mb,
    }
    started_at = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="wheatley-python-") as tmp:
        try:
            completed = subprocess.run(
                [sys.executable, "-I", "-c", RUNNER],
                input=json.dumps(payload),
                capture_output=True,
                text=True,
                cwd=tmp,
                env={"PYTHONIOENCODING": "utf-8"},
                timeout=timeout,
            )
        except subprocess.TimeoutExpired as exc:
            return ToolResult(
                name="python_interpreter",
                ok=False,
                content={
                    "error": "timeout",
                    "timeout_seconds": timeout,
                    "stdout": _trim(
                        exc.stdout or "",
                        cfg.tools.python_interpreter_max_stdout_chars,
                    ),
                    "stderr": _trim(
                        exc.stderr or "",
                        cfg.tools.python_interpreter_max_stderr_chars,
                    ),
                    "duration_seconds": round(time.perf_counter() - started_at, 3),
                },
            )

    duration = round(time.perf_counter() - started_at, 3)
    if completed.returncode != 0:
        return ToolResult(
            name="python_interpreter",
            ok=False,
            content={
                "error": "runner_failed",
                "returncode": completed.returncode,
                "stdout": _trim(
                    completed.stdout,
                    cfg.tools.python_interpreter_max_stdout_chars,
                ),
                "stderr": _trim(
                    completed.stderr,
                    cfg.tools.python_interpreter_max_stderr_chars,
                ),
                "duration_seconds": duration,
            },
        )

    try:
        child = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        return ToolResult(
            name="python_interpreter",
            ok=False,
            content={
                "error": f"invalid_runner_output: {exc}",
                "stdout": _trim(
                    completed.stdout,
                    cfg.tools.python_interpreter_max_stdout_chars,
                ),
                "stderr": _trim(
                    completed.stderr,
                    cfg.tools.python_interpreter_max_stderr_chars,
                ),
                "duration_seconds": duration,
            },
        )

    content = {
        key: child.get(key)
        for key in (
            "result",
            "result_truncated",
            "stdout",
            "stderr",
            "error",
            "error_type",
            "phase",
            "traceback",
            "resource_warnings",
        )
        if key in child
    }
    content["duration_seconds"] = duration
    return ToolResult(
        name="python_interpreter",
        ok=bool(child.get("ok")),
        content=content,
    )


def _validate_user_code(code: str) -> str:
    try:
        tree = ast.parse(code, mode="exec")
    except SyntaxError as exc:
        return f"syntax_error: {exc}"
    for node in ast.walk(tree):
        if isinstance(node, FORBIDDEN_NODES):
            return f"blocked_node: {type(node).__name__}"
        if isinstance(node, ast.Name):
            if node.id.startswith("__") or node.id in FORBIDDEN_NAMES:
                return f"blocked_name: {node.id}"
        if isinstance(node, ast.Attribute):
            if node.attr.startswith("__") or node.attr in FORBIDDEN_ATTRIBUTES:
                return f"blocked_attribute: {node.attr}"
    return ""


def _resolve_read_roots(cfg: Config) -> List[Path]:
    roots = cfg.tools.python_interpreter_read_roots or ["files"]
    profile = Path(cfg.profile_dir)
    resolved = []
    for root in roots:
        root_path = Path(str(root))
        if not root_path.is_absolute():
            root_path = profile / root_path
        resolved.append(root_path.resolve(strict=False))
    return resolved


def _trim(text: str, limit: int) -> str:
    text = text or ""
    limit = max(0, int(limit))
    if len(text) <= limit:
        return text
    return text[:limit] + "\n... truncated"
