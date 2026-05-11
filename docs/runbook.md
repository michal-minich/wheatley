# Runbook

Use this after the project is installed. For first setup, start with [Local English Startup](local-english-startup.md).

## Daily Commands

Run diagnostics:

```bash
PYTHONPATH=src python3 -m wheatley doctor
```

Run all tests:

```bash
PYTHONPATH=src python3 -m unittest discover -s tests
```

Run the isolated smoke profile:

```bash
PYTHONPATH=src python3 -m wheatley --profile test once --text "what time is it?"
```

Run one real text turn:

```bash
PYTHONPATH=src python3 -m wheatley once --stream --text "hello"
```

Run text chat:

```bash
PYTHONPATH=src python3 -m wheatley chat --stream
```

Run voice:

```bash
PYTHONPATH=src python3 -m wheatley voice
```

## Editing The Assistant

Edit the main profile in `profiles/wheatley/`.

```text
config.jsonc   runtime settings, models, voice, tools
system.md      assistant behavior
user.md        user preferences
memory.md      manual persistent memory
auto_memory.md generated memory from prior turns
runtime/       logs, state, audio
```

Automatic generated memory is configured in the `memory` block. The checked-in
profiles keep `auto_enabled` on for real profiles but set
`consolidation_enabled` to false, so startup memory work only performs
incremental updates unless that switch is explicitly re-enabled.

Prefer editing `config.jsonc` directly because it contains comments beside most settings.

Use `profiles/test/` for test turns and automation checks. That profile keeps logs separate from the real assistant memory.

Piper name pronunciation fixes live in `tts.piper_pronunciation_replacements`.
The keys are Python regex patterns and the values are the text sent only to Piper;
raw Piper phoneme blocks such as `[[mˈixal]]` can force exact sounds while logs and
streaming text keep the original assistant wording.

## Voice Commands

- `Quit.` or `Exit.` stops the voice loop.
- `Start a new chat.` clears recent conversation history.
- `Remember this: ...` appends a manual memory fact.
- `Look at this.` or `Take a photo.` lets the LLM use the camera tool when appropriate.
- Loud `stop` interrupts current speech playback when interrupt monitoring is enabled.

## Logs And State

The active profile owns its runtime files.

```text
profiles/<profile>/runtime/logs/turns.jsonl
profiles/<profile>/runtime/logs/tools.jsonl
profiles/<profile>/runtime/logs/system_llm.jsonl
profiles/<profile>/runtime/state/
profiles/<profile>/runtime/audio/
```

Use these files to verify what the assistant heard, what model answered, which tools ran, and what audio was generated.

## Validation Before Handing Back Changes

Run:

```bash
PYTHONPATH=src python3 -m unittest discover -s tests
PYTHONPATH=src python3 -m wheatley doctor
PYTHONPATH=src python3 -m wheatley --profile test once --text "what time is it?"
```

If optional model, microphone, or TTS dependencies are missing, keep the unit tests and test-profile smoke check passing and report the missing dependency clearly.

## Technical Details

`--stream` controls terminal text streaming. If speech is enabled and `tts.stream_speech` is true, TTS can still stream by chunks even when terminal token streaming is off.

Runtime audio and logs are preserved. The app does not automatically clean old recordings, generated speech, partial STT snapshots, or JSONL logs.

When idle speech is disabled, voice mode waits indefinitely for the next utterance, so idle silence does not repeatedly restart listening or replay chimes. One-shot `listen` and startup resume prompts still use `audio.max_wait_seconds`. The microphone recorder keeps a short `audio.pre_roll_seconds` buffer before VAD starts so the STT input preserves first syllables, treats `audio.max_utterance_seconds` as a soft budget that still waits for endpoint silence, then trims trailing silence while keeping at least `audio.trailing_silence_keep_seconds` and up to the endpoint silence tail.

If `idle_speech.enabled` is true, continuous voice mode uses randomized pre-speech timeouts so the assistant can say a short idle remark after real silence. The checked-in profiles use `idle_speech.interval_seconds: 100.0` with `random_min_multiplier: 1.0` and `random_max_multiplier: 5.0`, so idle remarks happen after roughly 100-500 seconds of silence. The instruction text lives in hardcoded profile-relative `idle.md`. Empty captures are idle-neutral when live preview recognized no text; any non-empty final transcript is handled as a user turn.

To use a headset microphone on machines with several audio devices, run `PYTHONPATH=src python3 -m wheatley audio-devices` to inspect PortAudio names. Leave `audio.input_device_mode` as `"default"` for the operating-system default input, or set it to `"auto"` to prefer headset/Bluetooth-style input devices and fall back to the default input when none is available. Add partial names such as `"USB Headset"` or `"Wireless Headset"` to `audio.input_device_preferred_names` to control auto-mode priority. `audio.input_device_name` and `audio.input_device_index` are exact overrides; index takes priority over name.

Tool audit records include the source, tool name, arguments, result payload, and duration. This is the best place to debug calculator, memory, Python interpreter, or configured-tool behavior.

Camera captures are written under `profiles/<profile>/runtime/photos/YYYY/MM/DD/`, using the same dated folder shape as audio logs. The tool result records the exact path in `tools.jsonl`. The tool uses `tools.photo_command` when configured, otherwise it tries common local capture commands such as `imagesnap`, `fswebcam`, `libcamera-still`, `rpicam-still`, or `ffmpeg`. `tools.photo_short_side` keeps the image small while preserving the camera's returned aspect ratio. Captured photos are attached to LLM requests only when the active model name looks vision-capable; text-only defaults such as `qwen3.5:4b` get photo metadata only.

For `python_interpreter`, trusted imports and helper functions live in `profiles/<profile>/python_preamble.py`. Model-written code should not include imports and can only read files through helpers such as `read_text`, `read_json`, and `list_files` under configured read roots, defaulting to `profiles/<profile>/files`. Interpreter code must assign the final answer to `result`; failures report a `phase` of `validation`, `preamble`, or `code`.
