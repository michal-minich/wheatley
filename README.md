# Wheatly

Offline-first talking AI foundation for a small Wheatley-style robot and other local personas.

The main runtime is a debuggable pipeline:

```text
microphone -> STT -> LLM -> tools -> streaming text -> TTS -> speaker
```

## Run

Default profile:

```bash
./scripts/run_voice_default.sh
```

Equivalent:

```bash
PYTHONPATH=src python3 -m wheatly --profile wheatly voice
```

One text turn:

```bash
PYTHONPATH=src python3 -m wheatly --profile wheatly once --stream --text "hello"
```

## Profiles

Everything editable for a persona lives in one folder:

```text
profiles/wheatly/
  config.jsonc
  system.md
  user.md
  tools.jsonc
  memory.md
```

`config.jsonc` is the main file. It contains comments next to the settings, so prefer editing it over duplicating settings in docs.

Use another persona:

```bash
PYTHONPATH=src python3 -m wheatly --profile my-persona chat --stream
```

Examples live under `examples/profiles/`. Copy one into `profiles/` and edit it.

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

## Docs

- [Architecture](docs/architecture.md)
- [Runbook](docs/runbook.md)
- [Hardware and Models](docs/hardware-and-models.md)
- [Decision Log](docs/decisions.md)
