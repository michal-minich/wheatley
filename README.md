# Wheatley

Offline-first talking AI foundation for a small Wheatley-style robot and other local personas.

The main runtime is a debuggable pipeline:

```text
microphone -> STT -> LLM -> tools -> streaming text -> TTS -> speaker
```

## Run

Default profile:

```bash
./scripts/start_wheatley.sh
```

Equivalent:

```bash
PYTHONPATH=src python3 -m wheatley voice
```

One text turn:

```bash
PYTHONPATH=src python3 -m wheatley once --stream --text "hello"
```

## Profiles

Everything editable for a persona lives in one folder:

```text
profiles/wheatley/
  config.jsonc
  system.md
  user.md
  tools.jsonc
  memory.md
  auto_memory.md
  memory_update.md
  memory_consolidate.md
  runtime/
```

`config.jsonc` is the main file. It contains comments next to the settings, so prefer editing it over duplicating settings in docs.

The default runtime uses `profiles/wheatley/config.jsonc`. Keep runtime choices in that config instead of passing startup flags. `memory.md` is explicit/manual memory; `auto_memory.md` is generated from the profile-local runtime log using rules in `memory_update.md` and `memory_consolidate.md`.

## Useful Commands

```bash
make test
make doctor
make tools
make stats
```

Voice commands:

- `Stop.` exits the voice loop.
- `Start a new chat.` clears conversation history but keeps profile instructions and memory.
- `Remember this: ...` appends to the active profile memory.
- `Switch language.` / `prepni jazyk` toggles between configured languages.
- Selected tools print a colored `tool>` cue and, when speech is enabled, say a short active-language cue like `Searching...` / `Hľadám...`.

Remote smart-model fallback is configured per profile in `llm.remote`.

Remote STT fallback is configured per profile in `stt`. The default profile tries
Janka Mac at `http://jankas-mac-mini.local:8765/v1` first, then falls back to the
local STT model for the active language.

Run the STT server on a stronger Mac:

```bash
PYTHONPATH=src python3 -m wheatley stt-server \
  --host 0.0.0.0 \
  --port 8765 \
  --default-model small.en \
  --model en=small.en \
  --model sk=models/whisper/whisper-large-v3-sk-ct2-int8
```

## Docs

- [Architecture](docs/architecture.md)
- [Runbook](docs/runbook.md)
- [Hardware and Models](docs/hardware-and-models.md)
- [Decision Log](docs/decisions.md)
