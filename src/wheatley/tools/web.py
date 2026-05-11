from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Dict

from wheatley.config import Config
from wheatley.tools.registry import ToolResult

BRAVE_API_KEY_ENV = "BRAVE_SEARCH_API_KEY"
WEB_SEARCH_PROBE_URL = "https://api.search.brave.com/"
WEB_SEARCH_PROBE_MAX_TIMEOUT_SECONDS = 1.0


def web_search_available(cfg: Config) -> bool:
    try:
        _brave_api_key(cfg)
    except RuntimeError:
        return False
    timeout = min(
        max(float(cfg.tools.web_search_timeout_seconds or 0.0), 0.1),
        WEB_SEARCH_PROBE_MAX_TIMEOUT_SECONDS,
    )
    return internet_available(WEB_SEARCH_PROBE_URL, timeout=timeout)


def internet_available(url: str = WEB_SEARCH_PROBE_URL, timeout: float = 1.0) -> bool:
    request = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(request, timeout=timeout):
            return True
    except urllib.error.HTTPError:
        return True
    except (OSError, TimeoutError, urllib.error.URLError):
        return False


def web_search(cfg: Config, args: Dict[str, Any]) -> ToolResult:
    query = str(args.get("query", "")).strip()
    if not query:
        return ToolResult(name="web_search", ok=False, content={"error": "empty_query"})
    if len(query) > 400:
        return ToolResult(name="web_search", ok=False, content={"error": "query_too_long"})

    limit = _bounded_int(args.get("max_results"), cfg.tools.web_search_max_results, 1, 10)
    try:
        content = _search_brave(cfg, query, limit)
    except Exception as exc:
        return ToolResult(name="web_search", ok=False, content={"error": str(exc)})
    return ToolResult(name="web_search", ok=True, content=content)


def _search_brave(cfg: Config, query: str, limit: int) -> Dict[str, Any]:
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
            "X-Subscription-Token": _brave_api_key(cfg),
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


def _brave_api_key(cfg: Config) -> str:
    del cfg
    key = os.environ.get(BRAVE_API_KEY_ENV, "").strip()
    if not key:
        raise RuntimeError(f"missing API key env var: {BRAVE_API_KEY_ENV}")
    return key


def _get_json(url: str, headers: Dict[str, str], timeout: float) -> Dict[str, Any]:
    req = urllib.request.Request(url, headers=headers, method="GET")
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


def _clean_text(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()
