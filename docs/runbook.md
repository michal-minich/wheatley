# Runbook

## Profiles

Active editable files are grouped by persona in `profiles/<name>/`.

```text
config.jsonc   runtime, model, STT, TTS, tools and file paths
system.md      main system prompt
user.md        always-on user preferences
tools.jsonc    tool descriptions and tool instructions
memory.md      persistent memory injected into context
```

Examples are in `examples/profiles/`.

## Commands

```bash
./scripts/run_voice_default.sh
PYTHONPATH=src python3 -m wheatly --profile wheatly chat --stream
PYTHONPATH=src python3 -m wheatly --profile wheatly once --stream --text "what time is it?"
PYTHONPATH=src python3 -m wheatly --profile wheatly tools
PYTHONPATH=src python3 -m wheatly --profile wheatly stats
```

## Voice Commands

`Stop.`, `Stop!`, `quit`, `exit`, and `goodbye` exit the loop.

`Start a new chat.` clears conversation history. The next turn still injects `system.md`, `user.md`, `tools.jsonc`, and `memory.md`.

`Remember this: ...` writes to the active profile memory file.

`Switch to Slovak.` / `hovor po slovensky` switches STT, prompt language hint, and TTS voice to Slovak and answers `Ahoj`.

`Switch to English.` / `hovor po anglicky` switches back and answers `Hi`.

## Smart Remote Model

Per-profile remote model selection is configured in `llm.remote` inside `config.jsonc`. On each new chat the agent probes the configured OpenAI-compatible `/models` endpoint quickly. If reachable, it switches to that backend; otherwise it keeps the local `llm` backend.

## Partial Transcript

`audio.partial_transcript_enabled` controls the live `you~>` preview while recording. It is only a console preview; the final `you>` transcript is still produced from the full utterance and is the only text sent to the LLM.

## Validation

```bash
make test
make doctor
make smoke
```

`make PROFILE=other-profile doctor` runs the same target with another profile.
