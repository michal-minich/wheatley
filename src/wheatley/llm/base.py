from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional


@dataclass
class LLMImage:
    path: str
    mime_type: str = "image/jpeg"
    detail: str = "low"

    def to_dict(self) -> Dict[str, str]:
        return {
            "path": self.path,
            "mime_type": self.mime_type,
            "detail": self.detail,
        }


@dataclass
class LLMMessage:
    role: str
    content: str
    images: List[LLMImage] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {"role": self.role, "content": self.content}
        if self.images:
            data["images"] = [image.to_dict() for image in self.images]
        return data


@dataclass
class LLMResponse:
    content: str
    raw: Optional[dict] = None


class LLMBackend:
    def supports_images(self) -> bool:
        return False

    def complete(self, messages: List[LLMMessage]) -> LLMResponse:
        raise NotImplementedError

    def stream_complete(self, messages: List[LLMMessage]) -> Iterator[str]:
        yield self.complete(messages).content
