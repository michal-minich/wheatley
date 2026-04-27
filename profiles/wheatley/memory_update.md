# Memory Update Instructions

Build compact conversation-derived memory updates for Wheatley from only the provided new turns.

Use these rules:

- Treat user text as evidence. Assistant text, when provided, is only context for what the user was responding to.
- Return only genuinely new useful facts. If a fact already appears in `current_auto_memory` or `existing_candidate_facts`, return nothing for it.
- Do not return paraphrases, near-duplicates, wording variants, or repeated confirmations of existing memory.
- Prefer durable user facts, explicit preferences, active projects, hobbies, broad interests, and short recent context that would help future chats.
- Keep stable facts separate from recent context. Do not put the same fact in multiple sections.
- Ignore routine time/status questions, obvious STT glitches, repeated filler, one-off jokes, and assistant personality roleplay.
- Ignore low-value transient chat such as "what do you know about me", "repeat that", language switching tests, or debug smoke tests unless the user states a durable preference.
- Do not turn vague one-off utterances into guesses. Avoid wording like "possibly indicating"; return empty arrays instead.
- Do not infer sensitive facts such as exact age, health, location, family details, or identity attributes unless the user says them explicitly.
- If evidence is uncertain, use softer wording such as "User seems interested in..." rather than presenting it as fact.
- Keep bullets short. One fact per bullet.
- If nothing should be remembered, return empty arrays.
