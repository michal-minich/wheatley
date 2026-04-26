from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List


@dataclass
class ToolCall:
    name: str
    arguments: Dict[str, Any]


@dataclass
class ToolResult:
    name: str
    ok: bool
    content: Dict[str, Any]


@dataclass
class ToolSpec:
    name: str
    description: str
    parameters: Dict[str, Any]


ToolHandler = Callable[[Dict[str, Any]], ToolResult]


class ToolRegistry:
    def __init__(self) -> None:
        self._specs: Dict[str, ToolSpec] = {}
        self._handlers: Dict[str, ToolHandler] = {}

    def register(self, spec: ToolSpec, handler: ToolHandler) -> None:
        self._specs[spec.name] = spec
        self._handlers[spec.name] = handler

    def specs(self) -> List[ToolSpec]:
        return list(self._specs.values())

    def update_spec(
        self,
        name: str,
        description: str = "",
        instructions: str = "",
    ) -> None:
        spec = self._specs.get(name)
        if not spec:
            return
        if description:
            spec.description = description
        if instructions:
            spec.description = f"{spec.description} Instructions: {instructions}"

    def execute(self, call: ToolCall) -> ToolResult:
        handler = self._handlers.get(call.name)
        if not handler:
            return ToolResult(
                name=call.name,
                ok=False,
                content={"error": "unknown_tool", "available": sorted(self._handlers)},
            )
        try:
            return handler(call.arguments or {})
        except Exception as exc:  # Tool errors must not crash the agent loop.
            return ToolResult(name=call.name, ok=False, content={"error": str(exc)})
