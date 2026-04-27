# Memory Consolidation Instructions

Rewrite Wheatly's generated conversation-derived memory into a compact, deduplicated profile.

Use these rules:

- Treat user text as evidence. Assistant text, when provided, is only context for what the user was responding to.
- Merge duplicate, overlapping, or contradictory bullets into the single best current wording.
- Prefer newer explicit user statements when they contradict older inferred or assistant-summarized facts.
- Keep stable facts separate from preferences, active projects, and recent context. Do not repeat the same idea across sections.
- Keep recent context short and perishable. Remove stale jokes, old smoke tests, routine time checks, and one-off debugging chatter.
- Keep useful durable facts, repeated patterns, active projects, user preferences, hobbies, broad interests, and compact recent context.
- Do not infer sensitive facts such as exact age, health, location, family details, or identity attributes unless the user says them explicitly.
- If evidence is uncertain, use softer wording such as "User seems interested in..." rather than presenting it as fact.
- Keep bullets short. One fact per bullet.
- Stay within the provided section and total memory limits.
