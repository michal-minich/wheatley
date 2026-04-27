from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable, Dict, List, Optional

from wheatley.config import Config
from wheatley.llm.base import LLMBackend, LLMMessage


AUTO_MEMORY_FILENAME = "auto_memory.md"
UPDATE_INSTRUCTIONS_FILENAME = "memory_update.md"
CONSOLIDATE_INSTRUCTIONS_FILENAME = "memory_consolidate.md"
LEGACY_BUILDER_FILENAME = "memory_builder.md"
STATE_PATH = Path("runtime/state/memory_state.json")
CANDIDATES_PATH = Path("runtime/state/memory_candidates.jsonl")

SECTION_TITLES = {
    "stable_user_facts": "Stable User Facts",
    "preferences": "Preferences",
    "current_projects": "Current Projects",
    "recent_context": "Recent Context",
}

FACT_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "been",
    "being",
    "for",
    "from",
    "had",
    "has",
    "have",
    "in",
    "is",
    "of",
    "on",
    "or",
    "recently",
    "currently",
    "seems",
    "that",
    "the",
    "to",
    "user",
    "was",
    "were",
    "with",
}


@dataclass
class MemoryRefreshResult:
    updated: bool = False
    consolidated: bool = False
    processed_turns: int = 0


@dataclass
class _MemoryState:
    last_processed_offset: int = 0
    last_processed_timestamp: Optional[str] = None
    last_incremental_update_at: Optional[str] = None
    last_full_rewrite_at: Optional[str] = None
    loaded: bool = False


@dataclass
class _TurnReadResult:
    turns: List[dict]
    offset: int
    has_more: bool = False


@dataclass
class _Patch:
    sections: Dict[str, List[str]] = field(default_factory=dict)
    candidates: List[dict] = field(default_factory=list)


def refresh_auto_memory(
    cfg: Config,
    llm: LLMBackend,
    mode: str,
    notify: Optional[Callable[[str], None]] = None,
) -> MemoryRefreshResult:
    if not cfg.memory.auto_enabled:
        return MemoryRefreshResult()

    state = _load_state(cfg)
    result = MemoryRefreshResult()
    include_assistant = _include_assistant_text(cfg, mode)
    now = _now()
    state_dirty = _bootstrap_incremental_offset(cfg, state, mode)
    read_result = _read_new_turns(
        Path(cfg.runtime.turn_log),
        state.last_processed_offset,
        include_assistant=include_assistant,
        max_turns=cfg.memory.max_turns_per_update,
        since_timestamp=_incremental_since_timestamp(state),
    )

    if read_result.turns:
        _notify(cfg, notify, "update_start")
        try:
            patch = _extract_incremental_patch(cfg, llm, read_result.turns, mode)
            if _apply_incremental_patch(cfg, patch, now):
                result.updated = True
            state.last_processed_offset = read_result.offset
            state.last_processed_timestamp = str(read_result.turns[-1].get("timestamp", ""))
            state.last_incremental_update_at = now
            _save_state(cfg, state)
            result.processed_turns = len(read_result.turns)
            _notify(cfg, notify, "update_done")
        except Exception:
            return result
    elif read_result.offset != state.last_processed_offset:
        state.last_processed_offset = read_result.offset
        state.last_incremental_update_at = now
        state_dirty = True

    if _full_rewrite_due(cfg, mode, state, now):
        _notify(cfg, notify, "consolidate_start")
        try:
            if _run_full_rewrite(cfg, llm, mode, now):
                result.consolidated = True
            state.last_full_rewrite_at = now
            _save_state(cfg, state)
            _notify(cfg, notify, "consolidate_done")
        except Exception:
            if state_dirty:
                _save_state(cfg, state)
            return result

    if state_dirty:
        _save_state(cfg, state)
    return result


def auto_memory_path(cfg: Config) -> Path:
    return Path(cfg.profile_dir) / AUTO_MEMORY_FILENAME


def memory_update_instructions_path(cfg: Config) -> Path:
    return Path(cfg.profile_dir) / UPDATE_INSTRUCTIONS_FILENAME


def memory_consolidate_instructions_path(cfg: Config) -> Path:
    return Path(cfg.profile_dir) / CONSOLIDATE_INSTRUCTIONS_FILENAME


def memory_builder_path(cfg: Config) -> Path:
    return Path(cfg.profile_dir) / LEGACY_BUILDER_FILENAME


def memory_state_path(cfg: Config) -> Path:
    return Path(cfg.profile_dir) / STATE_PATH


def memory_candidates_path(cfg: Config) -> Path:
    return Path(cfg.profile_dir) / CANDIDATES_PATH


def _include_assistant_text(cfg: Config, mode: str) -> bool:
    if mode == "online":
        return cfg.memory.include_assistant_text_online
    return cfg.memory.include_assistant_text_offline


def _read_new_turns(
    path: Path,
    offset: int,
    include_assistant: bool,
    max_turns: int,
    since_timestamp: Optional[str] = None,
) -> _TurnReadResult:
    if not path.exists() or max_turns <= 0:
        return _TurnReadResult([], offset)
    size = path.stat().st_size
    if offset < 0 or offset > size:
        offset = 0
    turns: List[dict] = []
    has_more = False
    since = _parse_datetime(since_timestamp or "")
    with path.open("rb") as handle:
        handle.seek(offset)
        while len(turns) < max_turns:
            raw = handle.readline()
            if not raw:
                break
            item = _parse_turn(raw, include_assistant)
            if item:
                timestamp = _parse_datetime(str(item.get("timestamp", "")))
                if since is not None and (timestamp is None or timestamp <= since):
                    continue
                turns.append(item)
        offset = handle.tell()
        has_more = bool(handle.readline())
    return _TurnReadResult(turns, offset, has_more)


def _read_recent_turns(cfg: Config, include_assistant: bool) -> List[dict]:
    path = Path(cfg.runtime.turn_log)
    if not path.exists():
        return []
    cutoff = datetime.now().astimezone() - timedelta(
        days=max(1, cfg.memory.full_rewrite_recent_days)
    )
    turns: List[dict] = []
    try:
        with path.open("rb") as handle:
            for raw in handle:
                item = _parse_turn(raw, include_assistant)
                if not item:
                    continue
                timestamp = _parse_datetime(str(item.get("timestamp", "")))
                if timestamp is None or timestamp >= cutoff:
                    turns.append(item)
    except OSError:
        return []
    return turns[-cfg.memory.max_turns_per_update :]


def _parse_turn(raw: bytes, include_assistant: bool) -> dict:
    try:
        record = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return {}
    user_text = str(record.get("user_text", "")).strip()
    if not user_text:
        return {}
    item = {
        "timestamp": str(record.get("timestamp", "")),
        "user_text": user_text,
    }
    if include_assistant:
        assistant_text = str(record.get("assistant_text", "")).strip()
        if assistant_text:
            item["assistant_text"] = assistant_text
    return item


def _extract_incremental_patch(
    cfg: Config,
    llm: LLMBackend,
    turns: List[dict],
    mode: str,
) -> _Patch:
    prompt = _build_incremental_prompt(cfg, turns, mode)
    response = llm.complete(
        [
            LLMMessage("system", _memory_system_prompt(cfg)),
            LLMMessage("user", prompt),
        ]
    )
    return _patch_from_payload(_load_json_payload(response.content))


def _run_full_rewrite(
    cfg: Config,
    llm: LLMBackend,
    mode: str,
    now: str,
) -> bool:
    include_assistant = _include_assistant_text(cfg, mode)
    payload = {
        "current_auto_memory": _read_text(auto_memory_path(cfg)),
        "memory_candidates": _read_candidates(cfg),
        "recent_turns": _read_recent_turns(cfg, include_assistant),
        "limits": _limits_payload(cfg),
        "now": now,
        "mode": mode,
    }
    response = llm.complete(
        [
            LLMMessage("system", _memory_system_prompt(cfg)),
            LLMMessage(
                "user",
                _consolidate_instructions(cfg)
                + "\n\nConsolidate the conversation-derived memory. "
                + "Merge duplicate and overlapping facts into one best bullet. "
                + "Return JSON only with either an auto_memory_md string or "
                + "section arrays named stable_user_facts, preferences, "
                + "current_projects, and recent_context.\n\n"
                + json.dumps(payload, ensure_ascii=True),
            ),
        ]
    )
    raw = _load_json_payload(response.content)
    markdown = str(raw.get("auto_memory_md", "")).strip() if isinstance(raw, dict) else ""
    if not markdown:
        patch = _patch_from_payload(raw)
        if not any(patch.sections.values()):
            return False
        markdown = _render_auto_memory(patch.sections, now, cfg)
    else:
        markdown = _normalize_auto_memory_markdown(markdown, now, cfg)
    return _write_if_changed(auto_memory_path(cfg), markdown.rstrip() + "\n")


def _apply_incremental_patch(cfg: Config, patch: _Patch, now: str) -> bool:
    if not any(patch.sections.values()) and not patch.candidates:
        return False
    current = _parse_auto_memory(_read_text(auto_memory_path(cfg)))
    existing_memory_facts = _all_section_values(current)
    for key, values in patch.sections.items():
        if key not in SECTION_TITLES:
            continue
        current.setdefault(key, [])
        for value in values:
            if _contains_similar_fact(_all_section_values(current), value):
                continue
            _append_unique(current[key], value)
    _enforce_section_limits(current, cfg)
    markdown = _render_auto_memory(current, now, cfg)
    changed = _write_if_changed(auto_memory_path(cfg), markdown)
    _append_candidates(cfg, patch, now, existing_memory_facts)
    return changed


def _patch_from_payload(raw) -> _Patch:
    if not isinstance(raw, dict):
        return _Patch()
    sections: Dict[str, List[str]] = {}
    for key in SECTION_TITLES:
        sections[key] = _string_list(raw.get(key, []))
    candidates = raw.get("candidates", [])
    if not isinstance(candidates, list):
        candidates = []
    normalized_candidates = []
    for item in candidates:
        if isinstance(item, dict):
            fact = str(item.get("fact", "")).strip()
            if fact:
                normalized_candidates.append({**item, "fact": fact})
        elif isinstance(item, str) and item.strip():
            normalized_candidates.append({"fact": item.strip()})
    if not normalized_candidates:
        for key, values in sections.items():
            for value in values:
                normalized_candidates.append({"category": key, "fact": value})
    return _Patch(sections=sections, candidates=normalized_candidates)


def _build_incremental_prompt(cfg: Config, turns: List[dict], mode: str) -> str:
    payload = {
        "current_auto_memory": _read_text(auto_memory_path(cfg)),
        "existing_candidate_facts": [
            str(item.get("fact", ""))
            for item in _read_candidates(cfg)[-cfg.memory.max_candidates_for_rewrite :]
        ],
        "new_turns": turns,
        "limits": _limits_payload(cfg),
        "mode": mode,
        "now": _now(),
    }
    return (
        _update_instructions(cfg)
        + "\n\nExtract only useful additions from the new turns. "
        + "Do not restate, paraphrase, or slightly rename facts that already "
        + "appear in current_auto_memory or existing_candidate_facts. "
        + "Do not turn vague one-off utterances into speculative memory. "
        + "Return JSON only with arrays named stable_user_facts, preferences, "
        + "current_projects, recent_context, and optionally candidates. "
        + "Use empty arrays when nothing should be remembered.\n\n"
        + json.dumps(payload, ensure_ascii=True)
    )


def _memory_system_prompt(cfg: Config) -> str:
    return (
        "You update a compact profile memory for a local voice assistant. "
        "Be conservative. Do not invent facts. Prefer short bullets. "
        f"Keep the total memory under about {cfg.memory.max_total_words} words. "
        "Return valid JSON only."
    )


def _update_instructions(cfg: Config) -> str:
    custom = _read_text(memory_update_instructions_path(cfg)).strip()
    if custom:
        return custom
    legacy = _read_text(memory_builder_path(cfg)).strip()
    if legacy:
        return legacy
    return (
        "# Memory Update Instructions\n"
        "- Use user text as evidence. Assistant text is context only when provided.\n"
        "- Return only genuinely new facts from the provided new turns.\n"
        "- If a fact is already present or only reworded, return nothing for it.\n"
        "- Ignore transcription noise, one-off jokes, and routine time/status questions.\n"
        "- Do not infer sensitive facts unless the user says them explicitly.\n"
    )


def _consolidate_instructions(cfg: Config) -> str:
    custom = _read_text(memory_consolidate_instructions_path(cfg)).strip()
    if custom:
        return custom
    legacy = _read_text(memory_builder_path(cfg)).strip()
    if legacy:
        return legacy
    return (
        "# Memory Consolidation Instructions\n"
        "- Rewrite the generated memory into compact, non-duplicated bullets.\n"
        "- Merge overlapping facts and keep the most recent explicit version.\n"
        "- Keep stable facts separate from preferences, active projects, and recent context.\n"
        "- Remove stale recent context and low-value trivia.\n"
        "- Do not infer sensitive facts unless the user says them explicitly.\n"
    )


def _parse_auto_memory(text: str) -> Dict[str, List[str]]:
    sections = {key: [] for key in SECTION_TITLES}
    current_key: Optional[str] = None
    title_to_key = {title.lower(): key for key, title in SECTION_TITLES.items()}
    for line in text.splitlines():
        heading = re.match(r"^##\s+(.+?)\s*$", line)
        if heading:
            current_key = title_to_key.get(heading.group(1).strip().lower())
            continue
        if current_key is None:
            continue
        bullet = re.match(r"^\s*-\s+(.+?)\s*$", line)
        if bullet:
            _append_unique(sections[current_key], bullet.group(1).strip())
    return sections


def _render_auto_memory(
    sections: Dict[str, List[str]],
    updated_at: str,
    cfg: Config,
) -> str:
    _enforce_section_limits(sections, cfg)
    lines = [
        "# Wheatley Auto Memory",
        "",
        "Generated from conversation history. Manual memories remain in memory.md.",
        f"Updated: {updated_at}",
    ]
    for key, title in SECTION_TITLES.items():
        lines.extend(["", f"## {title}"])
        values = sections.get(key, [])
        if values:
            lines.extend(f"- {value}" for value in values)
        else:
            lines.append("- None yet.")
    return _clamp_markdown("\n".join(lines).rstrip() + "\n", cfg)


def _clamp_markdown(markdown: str, cfg: Config) -> str:
    if _word_count(markdown) <= cfg.memory.max_total_words:
        return markdown
    sections = _parse_auto_memory(markdown)
    for key in ("recent_context", "preferences", "current_projects", "stable_user_facts"):
        while sections.get(key) and _word_count(
            _render_auto_memory_unclamped(sections, _now())
        ) > cfg.memory.max_total_words:
            sections[key].pop(0)
    return _render_auto_memory_unclamped(sections, _now())


def _normalize_auto_memory_markdown(markdown: str, updated_at: str, cfg: Config) -> str:
    if not markdown.startswith("#"):
        markdown = "# Wheatley Auto Memory\n\n" + markdown
    sections = _parse_auto_memory(markdown)
    if any(sections.values()):
        return _render_auto_memory(sections, updated_at, cfg)
    return _clamp_markdown(markdown, cfg)


def _render_auto_memory_unclamped(
    sections: Dict[str, List[str]], updated_at: str
) -> str:
    lines = [
        "# Wheatley Auto Memory",
        "",
        "Generated from conversation history. Manual memories remain in memory.md.",
        f"Updated: {updated_at}",
    ]
    for key, title in SECTION_TITLES.items():
        lines.extend(["", f"## {title}"])
        values = sections.get(key, [])
        lines.extend(f"- {value}" for value in values or ["None yet."])
    return "\n".join(lines).rstrip() + "\n"


def _enforce_section_limits(sections: Dict[str, List[str]], cfg: Config) -> None:
    limits = {
        "stable_user_facts": cfg.memory.max_stable_facts,
        "preferences": cfg.memory.max_preferences,
        "current_projects": cfg.memory.max_current_projects,
        "recent_context": cfg.memory.max_recent_context,
    }
    for key, limit in limits.items():
        values = _dedupe_facts(
            [
                value
                for value in sections.get(key, [])
                if value and value != "None yet."
            ]
        )
        sections[key] = values[-max(0, limit) :]


def _append_unique(values: List[str], value: str) -> bool:
    value = _clean_fact(value)
    if not value:
        return False
    if _contains_similar_fact(values, value):
        return False
    values.append(value)
    return True


def _append_candidates(
    cfg: Config,
    patch: _Patch,
    now: str,
    existing_memory_facts: List[str],
) -> bool:
    if not patch.candidates:
        return False
    existing_candidates = [str(item.get("fact", "")) for item in _read_candidates(cfg)]
    pending: List[dict] = []
    pending_facts: List[str] = []
    blocked_facts = existing_memory_facts + existing_candidates
    for item in patch.candidates:
        fact = _clean_fact(item.get("fact", ""))
        if not fact:
            continue
        if _contains_similar_fact(blocked_facts, fact):
            continue
        if _contains_similar_fact(pending_facts, fact):
            continue
        pending.append({**item, "fact": fact})
        pending_facts.append(fact)
    if not pending:
        return False
    path = memory_candidates_path(cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for item in pending:
            record = {
                "recorded_at": now,
                "category": str(item.get("category", "")),
                "fact": str(item.get("fact", "")).strip(),
                "source_timestamp": str(item.get("source_timestamp", "")),
            }
            if record["fact"]:
                handle.write(json.dumps(record, ensure_ascii=True) + "\n")
    return True


def _read_candidates(cfg: Config) -> List[dict]:
    path = memory_candidates_path(cfg)
    if not path.exists():
        return []
    records = []
    seen_facts: List[str] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue
                fact = _clean_fact(item.get("fact", "")) if isinstance(item, dict) else ""
                if fact and not _contains_similar_fact(seen_facts, fact):
                    item["fact"] = fact
                    records.append(item)
                    seen_facts.append(fact)
    except OSError:
        return []
    return records[-cfg.memory.max_candidates_for_rewrite :]


def _limits_payload(cfg: Config) -> dict:
    return {
        "max_total_words": cfg.memory.max_total_words,
        "max_stable_facts": cfg.memory.max_stable_facts,
        "max_preferences": cfg.memory.max_preferences,
        "max_current_projects": cfg.memory.max_current_projects,
        "max_recent_context": cfg.memory.max_recent_context,
    }


def _load_json_payload(text: str):
    text = text.strip()
    if not text:
        return {}
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            return {}
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return {}


def _full_rewrite_due(
    cfg: Config,
    mode: str,
    state: _MemoryState,
    now: str,
) -> bool:
    if cfg.memory.full_rewrite_requires_online and mode != "online":
        return False
    if (
        not auto_memory_path(cfg).exists()
        and not memory_candidates_path(cfg).exists()
        and not Path(cfg.runtime.turn_log).exists()
    ):
        return False
    if cfg.memory.full_rewrite_interval_days <= 0:
        return True
    last = _parse_datetime(state.last_full_rewrite_at or "")
    if last is None:
        return True
    current = _parse_datetime(now)
    if current is None:
        return False
    return current - last >= timedelta(days=cfg.memory.full_rewrite_interval_days)


def _load_state(cfg: Config) -> _MemoryState:
    path = memory_state_path(cfg)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return _MemoryState()
    return _MemoryState(
        last_processed_offset=int(data.get("last_processed_offset", 0) or 0),
        last_processed_timestamp=data.get("last_processed_timestamp"),
        last_incremental_update_at=data.get("last_incremental_update_at"),
        last_full_rewrite_at=data.get("last_full_rewrite_at"),
        loaded=True,
    )


def _incremental_since_timestamp(state: _MemoryState) -> Optional[str]:
    updated = _parse_datetime(state.last_incremental_update_at or "")
    processed = _parse_datetime(state.last_processed_timestamp or "")
    if updated is None:
        return state.last_processed_timestamp
    if processed is None:
        return state.last_incremental_update_at
    if updated - processed > timedelta(minutes=5):
        return state.last_incremental_update_at
    return state.last_processed_timestamp


def _save_state(cfg: Config, state: _MemoryState) -> None:
    path = memory_state_path(cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "last_processed_offset": state.last_processed_offset,
                "last_processed_timestamp": state.last_processed_timestamp,
                "last_incremental_update_at": state.last_incremental_update_at,
                "last_full_rewrite_at": state.last_full_rewrite_at,
            },
            indent=2,
            ensure_ascii=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _notify(cfg: Config, notify: Optional[Callable[[str], None]], kind: str) -> None:
    if not notify:
        return
    notify(_localized_notice(cfg, kind))


def _localized_notice(cfg: Config, kind: str) -> str:
    slovak = str(cfg.agent.default_response_language).lower().startswith("slovak")
    english = {
        "update_start": "wait, I'm updating my memory...",
        "update_done": "my memory was updated.",
        "consolidate_start": "wait, I'm consolidating my memory...",
        "consolidate_done": "my memory was consolidated.",
    }
    slovak_messages = {
        "update_start": "počkaj, aktualizujem si pamäť...",
        "update_done": "moja pamäť bola aktualizovaná.",
        "consolidate_start": "počkaj, konsolidujem si pamäť...",
        "consolidate_done": "moja pamäť bola skonsolidovaná.",
    }
    return (slovak_messages if slovak else english).get(kind, "")


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8") if path.exists() else ""
    except OSError:
        return ""


def _write_if_changed(path: Path, text: str) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    old = _read_text(path)
    if old == text:
        return False
    path.write_text(text, encoding="utf-8")
    return True


def _string_list(value) -> List[str]:
    if not isinstance(value, list):
        return []
    return [_clean_fact(item) for item in value if _clean_fact(item)]


def _clean_fact(value) -> str:
    return " ".join(str(value).strip().split())


def _normalize_bullet(value: str) -> str:
    text = unicodedata.normalize("NFKD", value.lower())
    text = "".join(char for char in text if not unicodedata.combining(char))
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


def _fact_tokens(value: str) -> set[str]:
    return {
        token
        for token in _normalize_bullet(value).split()
        if token and token not in FACT_STOPWORDS
    }


def _contains_similar_fact(values: List[str], candidate: str) -> bool:
    return any(_facts_similar(existing, candidate) for existing in values)


def _facts_similar(left: str, right: str) -> bool:
    left_norm = _normalize_bullet(left)
    right_norm = _normalize_bullet(right)
    if not left_norm or not right_norm:
        return False
    if left_norm == right_norm:
        return True
    shorter, longer = sorted([left_norm, right_norm], key=len)
    if len(shorter) >= 18 and shorter in longer:
        return True
    left_tokens = _fact_tokens(left)
    right_tokens = _fact_tokens(right)
    if not left_tokens or not right_tokens:
        return False
    overlap = len(left_tokens & right_tokens)
    if overlap < 2:
        return False
    union = len(left_tokens | right_tokens)
    smaller = min(len(left_tokens), len(right_tokens))
    return overlap / union >= 0.72 or (overlap >= 3 and overlap / smaller >= 0.84)


def _dedupe_facts(values: List[str]) -> List[str]:
    deduped: List[str] = []
    for value in values:
        _append_unique(deduped, value)
    return deduped


def _all_section_values(sections: Dict[str, List[str]]) -> List[str]:
    values: List[str] = []
    for key in SECTION_TITLES:
        values.extend(sections.get(key, []))
    return values


def _bootstrap_incremental_offset(cfg: Config, state: _MemoryState, mode: str) -> bool:
    if state.loaded or state.last_processed_offset > 0:
        return False
    path = Path(cfg.runtime.turn_log)
    if not path.exists():
        return False
    if not (auto_memory_path(cfg).exists() or memory_candidates_path(cfg).exists() or mode == "online"):
        return False
    state.last_processed_offset = path.stat().st_size
    state.last_processed_timestamp = _last_turn_timestamp(path)
    return True


def _last_turn_timestamp(path: Path) -> Optional[str]:
    timestamp = None
    try:
        with path.open("rb") as handle:
            for raw in handle:
                try:
                    record = json.loads(raw.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError):
                    continue
                value = str(record.get("timestamp", "")).strip()
                if value:
                    timestamp = value
    except OSError:
        return None
    return timestamp


def _word_count(text: str) -> int:
    return len(re.findall(r"\b\S+\b", text))


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _parse_datetime(value: str) -> Optional[datetime]:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.astimezone()
    return parsed
