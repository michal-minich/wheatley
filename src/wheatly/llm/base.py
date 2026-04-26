from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional


@dataclass
class LLMMessage:
    role: str
    content: str

    def to_dict(self) -> Dict[str, str]:
        return {"role": self.role, "content": self.content}


@dataclass
class LLMResponse:
    content: str
    raw: Optional[dict] = None


class LLMBackend:
    def complete(self, messages: List[LLMMessage]) -> LLMResponse:
        raise NotImplementedError

    def stream_complete(self, messages: List[LLMMessage]) -> Iterator[str]:
        yield self.complete(messages).content
