# Runbook

## Profiles

Active editable files are in `profiles/wheatly/`.

```text
config.jsonc   runtime, model, STT, TTS, tools and file paths
system.md      main system prompt
user.md        always-on user preferences
tools.jsonc    tool descriptions and tool instructions
memory.md      manual persistent memory injected into context
auto_memory.md generated memory from runtime/logs/turns.jsonl
memory_update.md editable rules for quick automatic memory updates
memory_consolidate.md editable rules for full automatic memory consolidation
runtime/       profile-local logs, state, and generated audio
```

## Commands

```bash
./scripts/start_wheatly.sh
PYTHONPATH=src python3 -m wheatly chat --stream
PYTHONPATH=src python3 -m wheatly once --stream --text "what time is it?"
PYTHONPATH=src python3 -m wheatly transcribe path/to/audio.wav
PYTHONPATH=src python3 -m wheatly tools
PYTHONPATH=src python3 -m wheatly stats
```

## Web Search

The default profile enables `web_search` with Brave Search and `fetch_url` for readable public pages. Put the Brave key in a local ignored `.env` file:

```bash
BRAVE_SEARCH_API_KEY=...
```

`./scripts/start_wheatly.sh` loads `.env` before starting the voice loop. One-off commands can use the same environment variable from the shell. Do not commit real API keys.

`--stream` controls whether generated text streams in the terminal. When `--speak` is enabled, speech still uses streaming TTS automatically if `tts.stream_speech` is true in the active language.

Streaming speech for Piper/Edge/external backends uses pipelined chunk preparation and playback. Tune startup continuity with:

- `tts.stream_max_inter_chunk_wait_seconds`
- `tts.stream_playback_prebuffer_chunks`
- `tts.stream_playback_prebuffer_max_wait_seconds`

Language-specific overrides are available as:

- `language.languages.<code>.tts_stream_max_inter_chunk_wait_seconds`
- `language.languages.<code>.tts_stream_playback_prebuffer_chunks`
- `language.languages.<code>.tts_stream_playback_prebuffer_max_wait_seconds`

## Voice Commands

`Stop.`, `Stop!`, `quit`, `exit`, and `goodbye` exit the loop.

`Start a new chat.` clears conversation history. The next turn still injects `system.md`, `user.md`, `tools.jsonc`, `memory.md`, and `auto_memory.md`.

`Remember this: ...` writes to the active profile memory file.

Automatic memory refresh runs at startup and at the start of a new chat when new turns or a due consolidation exist. Quick updates print/say `wait, I'm updating my memory...`; full rewrites print/say `wait, I'm consolidating my memory...`. Quick updates process only turns newer than the last memory update; full consolidation can revisit recent logs and compact candidates. Raw `profiles/wheatly/runtime/logs/turns.jsonl` is append-only and should not be deleted. Each turn row includes active LLM `model_name`.

Tool calls are audited separately in `profiles/wheatly/runtime/logs/tools.jsonl`. Each JSONL record includes the source (`direct_route`, `llm`, or `cli`), tool name, arguments, full result payload, and duration so calculator, memory, web search/fetch, and other tool issues can be replayed from the log.

Runtime audio under `profiles/wheatly/runtime/audio/` is also preserved. The app does not auto-delete recordings, generated replies, or partial STT snapshots.

If playback gets regular short pauses while speaking, check `audio.speech_interrupt_pause_tts_while_verifying` in `config.jsonc`. Keeping it `false` avoids pause spikes from false interrupt candidates.

When `remember`, `run_safe_cli_tool`, `web_search`, or `fetch_url` starts, the CLI prints a colored `tool>` line. If speech is enabled, Wheatly also says a short active-language cue such as `Searching...` or `Hľadám...`.

`Switch to Slovak.` / `hovor po slovensky` switches STT, prompt language hint, and TTS voice to Slovak and answers `Ahoj`.

`Switch to English.` / `hovor po anglicky` switches back and answers `Hi`.

`Switch language.` / `prepni jazyk` switches to the language of the spoken command. If that is already the active language, it toggles to the previous or next configured language.

Slovak currently defaults to Edge TTS `sk-SK-LukasNeural` for a male voice. The local Piper fallback remains configured in `config.jsonc`. Edge TTS needs the optional `edge-tts` Python package and internet access.

## Smart Remote Model

Per-profile remote model selection is configured in `llm.remote` inside `config.jsonc`. On each new chat the agent probes the configured OpenAI-compatible `/models` endpoint quickly. If reachable, it switches to that backend; otherwise it keeps the local `llm` backend.

The offline model is always `llm.model`. Online LM Studio model names are language-specific under `language.languages.*.online_llm_model`.

The startup/new-chat `model>` line reports both independent services. Possible English messages are:

- `using smarter online model and remote speech recognition.`
- `using smarter online model and local speech recognition.`
- `using offline model and remote speech recognition.`
- `using offline model and local speech recognition.`

Possible Slovak messages are:

- `Používam múdrejší online model a vzdialené rozpoznávanie reči.`
- `Používam múdrejší online model a lokálne rozpoznávanie reči.`
- `Používam offline model a vzdialené rozpoznávanie reči.`
- `Používam offline model a lokálne rozpoznávanie reči.`

## Remote STT Server

The robot profile can use remote STT independently from the remote LLM. `stt.backend` is `remote_fallback`, so the robot first tries `stt.remote_base_url` and then uses the local fallback backend if the server is unreachable.

Start the server on Janka Mac from this repo:

```bash
cd /Users/janka/mm/Wheatly
python3 -m venv .venv
. .venv/bin/activate
python3 -m pip install -e '.[stt]'
./scripts/start_janka_stt_server.sh
```

Health and model checks:

```bash
curl http://jankas-mac-mini.local:8765/health
curl http://jankas-mac-mini.local:8765/v1/models
```

Transcription endpoint:

```bash
curl http://jankas-mac-mini.local:8765/v1/audio/transcriptions \
  -F file=@profiles/wheatly/runtime/audio/example.wav \
  -F language=sk \
  -F model=models/whisper/whisper-large-v3-sk-ct2-int8
```

The server selects models by language when `model=default` or no model is provided. The current intended server models are:

- English: `small.en`
- Slovak primary: `models/whisper/whisper-large-v3-sk-ct2-int8`
- Robot local Slovak fallback: `models/whisper/whisper-large-v3-turbo-sk-ct2-int8`

Model files are not committed. Keep downloaded or converted weights under `models/` or the serving machine's normal Hugging Face cache.

Create the remote Slovak primary model on Janka Mac:

```bash
python3 -m pip install '.[stt-convert]'
mkdir -p models/whisper
ct2-transformers-converter \
  --model NaiveNeuron/whisper-large-v3-sk \
  --output_dir models/whisper/whisper-large-v3-sk-ct2-int8 \
  --quantization int8 \
  --copy_files preprocessor_config.json tokenizer_config.json vocab.json merges.txt normalizer.json added_tokens.json special_tokens_map.json generation_config.json
```

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
