from __future__ import annotations

import ipaddress
import json
import os
import re
import socket
import urllib.error
import urllib.parse
import urllib.request
from html.parser import HTMLParser
from typing import Any, Dict, Iterable, List

from wheatly.config import Config
from wheatly.tools.registry import ToolResult


def web_search(cfg: Config, args: Dict[str, Any]) -> ToolResult:
    query = str(args.get("query", "")).strip()
    if not query:
        return ToolResult(name="web_search", ok=False, content={"error": "empty_query"})
    if len(query) > 400:
        return ToolResult(name="web_search", ok=False, content={"error": "query_too_long"})

    provider = cfg.tools.web_search_provider.lower().strip()
    limit = _bounded_int(args.get("max_results"), cfg.tools.web_search_max_results, 1, 10)
    try:
        if provider == "brave":
            content = _search_brave(cfg, query, limit)
        elif provider == "searxng":
            content = _search_searxng(cfg, query, limit)
        elif provider == "tavily":
            content = _search_tavily(cfg, query, limit)
        else:
            return ToolResult(
                name="web_search",
                ok=False,
                content={"error": "unsupported_provider", "provider": provider},
            )
    except Exception as exc:
        return ToolResult(name="web_search", ok=False, content={"error": str(exc)})
    return ToolResult(name="web_search", ok=True, content=content)


def fetch_url(cfg: Config, args: Dict[str, Any]) -> ToolResult:
    url = str(args.get("url", "")).strip()
    if not url:
        return ToolResult(name="fetch_url", ok=False, content={"error": "empty_url"})
    try:
        normalized = _validate_fetch_url(cfg, url)
        raw, final_url, content_type = _download_url(cfg, normalized)
        text = _decode_bytes(raw)
        if _is_html(content_type, text):
            extracted = _html_to_markdownish(text, final_url)
            format_name = "markdown"
        elif _is_text(content_type):
            extracted = _clean_text(text)
            format_name = "text"
        else:
            return ToolResult(
                name="fetch_url",
                ok=False,
                content={
                    "url": normalized,
                    "final_url": final_url,
                    "content_type": content_type,
                    "error": "unsupported_content_type",
                },
            )
        truncated = len(extracted) > cfg.tools.web_fetch_max_chars
        extracted = extracted[: cfg.tools.web_fetch_max_chars].rstrip()
        return ToolResult(
            name="fetch_url",
            ok=True,
            content={
                "url": normalized,
                "final_url": final_url,
                "content_type": content_type,
                "format": format_name,
                "truncated": truncated,
                "text": extracted,
            },
        )
    except Exception as exc:
        return ToolResult(name="fetch_url", ok=False, content={"url": url, "error": str(exc)})


def _search_brave(cfg: Config, query: str, limit: int) -> Dict[str, Any]:
    key = _api_key(cfg)
    params = urllib.parse.urlencode(
        {
            "q": query,
            "count": str(limit),
            "safesearch": "moderate",
            "extra_snippets": "true",
        }
    )
    raw = _get_json(
        f"https://api.search.brave.com/res/v1/web/search?{params}",
        headers={
            "Accept": "application/json",
            "X-Subscription-Token": key,
        },
        timeout=cfg.tools.web_search_timeout_seconds,
    )
    results = []
    for item in raw.get("web", {}).get("results", [])[:limit]:
        results.append(
            {
                "title": _clean_text(str(item.get("title", ""))),
                "url": str(item.get("url", "")),
                "snippet": _clean_text(str(item.get("description", ""))),
                "extra_snippets": [
                    _clean_text(str(part))
                    for part in item.get("extra_snippets", [])[:3]
                    if str(part).strip()
                ],
            }
        )
    return {"provider": "brave", "query": query, "results": results}


def _search_searxng(cfg: Config, query: str, limit: int) -> Dict[str, Any]:
    if not cfg.tools.web_search_endpoint:
        raise RuntimeError("web_search_endpoint is required for searxng")
    base = cfg.tools.web_search_endpoint.rstrip("/")
    endpoint = base if base.endswith("/search") else f"{base}/search"
    params = urllib.parse.urlencode(
        {
            "q": query,
            "format": "json",
            "safesearch": "1",
        }
    )
    raw = _get_json(
        f"{endpoint}?{params}",
        headers={"Accept": "application/json"},
        timeout=cfg.tools.web_search_timeout_seconds,
    )
    results = []
    for item in raw.get("results", [])[:limit]:
        results.append(
            {
                "title": _clean_text(str(item.get("title", ""))),
                "url": str(item.get("url", "")),
                "snippet": _clean_text(str(item.get("content", ""))),
                "engine": item.get("engine") or item.get("engines"),
            }
        )
    return {"provider": "searxng", "query": query, "results": results}


def _search_tavily(cfg: Config, query: str, limit: int) -> Dict[str, Any]:
    key = _api_key(cfg)
    payload = {
        "query": query,
        "max_results": limit,
        "include_answer": False,
        "include_raw_content": False,
    }
    raw = _post_json(
        "https://api.tavily.com/search",
        payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {key}",
        },
        timeout=cfg.tools.web_search_timeout_seconds,
    )
    results = []
    for item in raw.get("results", [])[:limit]:
        results.append(
            {
                "title": _clean_text(str(item.get("title", ""))),
                "url": str(item.get("url", "")),
                "snippet": _clean_text(str(item.get("content", ""))),
                "score": item.get("score"),
            }
        )
    return {"provider": "tavily", "query": query, "results": results}


def _api_key(cfg: Config) -> str:
    env_name = cfg.tools.web_search_api_key_env
    key = os.environ.get(env_name, "").strip()
    if not key:
        raise RuntimeError(f"missing API key env var: {env_name}")
    return key


def _download_url(cfg: Config, url: str) -> tuple[bytes, str, str]:
    req = urllib.request.Request(
        url,
        headers={
            "Accept": "text/html, text/plain;q=0.9, */*;q=0.1",
            "User-Agent": cfg.tools.web_fetch_user_agent,
        },
        method="GET",
    )
    opener = urllib.request.build_opener(_SafeRedirectHandler(cfg))
    try:
        with opener.open(req, timeout=cfg.tools.web_fetch_timeout_seconds) as resp:
            content_type = resp.headers.get("Content-Type", "")
            max_bytes = max(1, cfg.tools.web_fetch_max_bytes)
            raw = resp.read(max_bytes + 1)
            if len(raw) > max_bytes:
                raw = raw[:max_bytes]
            return raw, resp.geturl(), content_type
    except urllib.error.URLError as exc:
        raise RuntimeError(f"fetch failed: {exc}") from exc


class _SafeRedirectHandler(urllib.request.HTTPRedirectHandler):
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # type: ignore[no-untyped-def]
        _validate_fetch_url(self.cfg, newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _validate_fetch_url(cfg: Config, url: str) -> str:
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("only http and https URLs are allowed")
    if not parsed.hostname:
        raise ValueError("URL host is required")
    if not cfg.tools.web_fetch_allow_private_networks:
        _reject_private_host(parsed.hostname, parsed.port or (443 if parsed.scheme == "https" else 80))
    return urllib.parse.urlunparse(parsed)


def _reject_private_host(hostname: str, port: int) -> None:
    try:
        infos = socket.getaddrinfo(hostname, port, type=socket.SOCK_STREAM)
    except socket.gaierror as exc:
        raise ValueError(f"cannot resolve host: {hostname}") from exc
    for info in infos:
        address = info[4][0]
        ip = ipaddress.ip_address(address)
        if (
            ip.is_private
            or ip.is_loopback
            or ip.is_link_local
            or ip.is_multicast
            or ip.is_reserved
            or ip.is_unspecified
        ):
            raise ValueError("private, local, reserved, and multicast addresses are blocked")


def _get_json(url: str, headers: Dict[str, str], timeout: float) -> Dict[str, Any]:
    req = urllib.request.Request(url, headers=headers, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        raise RuntimeError(f"request failed for {url}: {exc}") from exc


def _post_json(
    url: str,
    payload: Dict[str, Any],
    headers: Dict[str, str],
    timeout: float,
) -> Dict[str, Any]:
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        raise RuntimeError(f"request failed for {url}: {exc}") from exc


def _bounded_int(value: Any, default: int, lower: int, upper: int) -> int:
    try:
        number = int(value if value is not None else default)
    except (TypeError, ValueError):
        number = default
    return max(lower, min(upper, number))


def _decode_bytes(raw: bytes) -> str:
    for encoding in ("utf-8", "windows-1252", "latin-1"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def _is_html(content_type: str, text: str) -> bool:
    lowered = content_type.lower()
    return "text/html" in lowered or "<html" in text[:500].lower()


def _is_text(content_type: str) -> bool:
    lowered = content_type.lower()
    return lowered.startswith("text/") or "application/json" in lowered


def _html_to_markdownish(html: str, base_url: str) -> str:
    parser = _ReadableHTMLParser(base_url)
    parser.feed(html)
    parser.close()
    return _clean_text_blocks(parser.text())


def _clean_text(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def _clean_text_blocks(value: str) -> str:
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in value.splitlines()]
    kept = [line for line in lines if line]
    blocks: List[str] = []
    for line in kept:
        if blocks and line == blocks[-1]:
            continue
        blocks.append(line)
    return "\n\n".join(blocks)


class _ReadableHTMLParser(HTMLParser):
    _SKIP_TAGS = {"script", "style", "noscript", "svg", "canvas", "form", "button", "template"}
    _BLOCK_TAGS = {
        "article",
        "aside",
        "blockquote",
        "br",
        "div",
        "footer",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "header",
        "li",
        "main",
        "nav",
        "ol",
        "p",
        "pre",
        "section",
        "table",
        "tr",
        "ul",
    }

    def __init__(self, base_url: str) -> None:
        super().__init__(convert_charrefs=True)
        self.base_url = base_url
        self.parts: List[str] = []
        self.skip_stack: List[str] = []
        self.link_stack: List[str] = []

    def handle_starttag(self, tag: str, attrs: Iterable[tuple[str, str | None]]) -> None:
        if tag in self._SKIP_TAGS:
            self.skip_stack.append(tag)
            return
        if self.skip_stack:
            return
        if tag in self._BLOCK_TAGS:
            self._newline()
        if tag in {"h1", "h2", "h3"}:
            self.parts.append("#" * int(tag[1]) + " ")
        if tag == "li":
            self.parts.append("- ")
        if tag == "a":
            href = dict(attrs).get("href")
            if href:
                self.link_stack.append(urllib.parse.urljoin(self.base_url, href))

    def handle_endtag(self, tag: str) -> None:
        if self.skip_stack:
            if tag == self.skip_stack[-1]:
                self.skip_stack.pop()
            return
        if tag == "a" and self.link_stack:
            href = self.link_stack.pop()
            if href.startswith(("http://", "https://")):
                self.parts.append(f" ({href})")
        if tag in self._BLOCK_TAGS:
            self._newline()

    def handle_data(self, data: str) -> None:
        if self.skip_stack:
            return
        text = _clean_text(data)
        if text:
            self.parts.append(text + " ")

    def text(self) -> str:
        return "".join(self.parts)

    def _newline(self) -> None:
        if self.parts and not self.parts[-1].endswith("\n"):
            self.parts.append("\n")
