# Runbook

## Profiles

Active editable files are in `profiles/wheatly/`.

```text
config.jsonc   runtime, model, STT, TTS, tools and file paths
system.md      main system prompt
user.md        always-on user preferences
tools.jsonc    tool descriptions and tool instructions
memory.md      persistent memory injected into context
```

## Commands

```bash
./scripts/start_wheatly.sh
PYTHONPATH=src python3 -m wheatly chat --stream
PYTHONPATH=src python3 -m wheatly once --stream --text "what time is it?"
PYTHONPATH=src python3 -m wheatly tools
PYTHONPATH=src python3 -m wheatly stats
```

## Voice Commands

`Stop.`, `Stop!`, `quit`, `exit`, and `goodbye` exit the loop.

`Start a new chat.` clears conversation history. The next turn still injects `system.md`, `user.md`, `tools.jsonc`, and `memory.md`.

`Remember this: ...` writes to the active profile memory file.

`Switch to Slovak.` / `hovor po slovensky` switches STT, prompt language hint, and TTS voice to Slovak and answers `Ahoj`.

`Switch to English.` / `hovor po anglicky` switches back and answers `Hi`.

`Switch language.` / `prepni jazyk` switches to the language of the spoken command. If that is already the active language, it toggles to the previous or next configured language.

Slovak currently defaults to Edge TTS `sk-SK-LukasNeural` for a male voice. The local Piper fallback remains configured in `config.jsonc`. Edge TTS needs the optional `edge-tts` Python package and internet access.

## Smart Remote Model

Per-profile remote model selection is configured in `llm.remote` inside `config.jsonc`. On each new chat the agent probes the configured OpenAI-compatible `/models` endpoint quickly. If reachable, it switches to that backend; otherwise it keeps the local `llm` backend.

The offline model is always `llm.model`. Online LM Studio model names are language-specific under `language.languages.*.online_llm_model`.

## Slovak STT

Slovak STT uses `models/whisper/whisper-large-v3-turbo-sk-ct2-int8`, converted from `NaiveNeuron/whisper-large-v3-turbo-sk`.

Recreate it on a new machine with:

```bash
python3 -m pip install '.[stt-convert]'
mkdir -p models/whisper
ct2-transformers-converter \
  --model NaiveNeuron/whisper-large-v3-turbo-sk \
  --output_dir models/whisper/whisper-large-v3-turbo-sk-ct2-int8 \
  --quantization int8 \
  --copy_files preprocessor_config.json tokenizer_config.json vocab.json merges.txt normalizer.json added_tokens.json special_tokens_map.json generation_config.json
```

## Partial Transcript

`audio.partial_transcript_enabled` controls the live `you~>` preview while recording. If `audio.partial_transcript_use_as_final` is true and the preview is fresh enough, the final `you>` text reuses that transcript instead of running a second full STT pass.

## Validation

```bash
make test
make doctor
make smoke
```
