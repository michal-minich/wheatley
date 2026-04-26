# Runbook

## 1. Smoke Test Without Models

```bash
PYTHONPATH=src python3 -m wheatly doctor
PYTHONPATH=src python3 -m unittest discover -s tests
PYTHONPATH=src python3 -m wheatly once --text "hello"
PYTHONPATH=src python3 -m wheatly once --stream --text "hello"
PYTHONPATH=src python3 -m wheatly once --text "what time is it?"
PYTHONPATH=src python3 -m wheatly bench --repeat 3 --text "Give me a short status update."
```

The `echo` backend should answer and exercise tool calling without downloads.

## 2. Create Local Config

```bash
cp configs/wheatly.example.json configs/wheatly.local.json
```

Edit `configs/wheatly.local.json`. It is ignored by git.

## 3. Ollama LLM Path

Install Ollama and pull a small model:

```bash
ollama pull qwen3.5:4b
```

Set:

```json
{
  "llm": {
    "backend": "ollama",
    "model": "qwen3.5:4b",
    "base_url": "http://localhost:11434",
    "max_tokens": 80,
    "enable_thinking": false
  }
}
```

Run:

```bash
PYTHONPATH=src python3 -m wheatly once --text "give me a two sentence status"
```

## 4. OpenAI-Compatible Local Server Path

Use this for llama.cpp, vLLM, or SGLang servers.

Set:

```json
{
  "llm": {
    "backend": "openai_compat",
    "model": "Qwen/Qwen3.5-4B",
    "base_url": "http://localhost:8000",
    "api_key": "EMPTY",
    "max_tokens": 80,
    "enable_thinking": false
  }
}
```

Run:

```bash
PYTHONPATH=src python3 -m wheatly chat
```

## 5. faster-whisper STT

Install:

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -e '.[audio,stt]'
```

Set:

```json
{
  "stt": {
    "backend": "faster_whisper",
    "model": "small.en",
    "language": "en",
    "device": "cpu",
    "compute_type": "int8"
  }
}
```

Transcribe a file:

```bash
PYTHONPATH=src python3 -m wheatly transcribe path/to/audio.wav
```

Listen once through the microphone:

```bash
PYTHONPATH=src python3 -m wheatly listen
```

Run continuous voice chat:

```bash
./scripts/run_voice_default.sh
```

The voice loop prints `listening...` in green while recording and `answering...` in red after recording stops. It streams generated text and starts TTS after the first usable phrase.

Voice commands:

- `Stop.`, `Stop!`, `quit`, `exit`, or `goodbye` exits the loop.
- `Start a new chat.` clears conversation history. The next turn uses only editable instructions and persistent memory.

## 6. Piper TTS

Place Piper voices under `models/piper/`, for example:

```text
models/piper/en_GB-alan-medium.onnx
models/piper/en_GB-alan-medium.onnx.json
```

Set:

```json
{
  "tts": {
    "backend": "piper",
    "enabled": true,
    "piper_binary": "piper",
    "piper_model": "models/piper/en_GB-alan-medium.onnx",
    "filter": {
      "enabled": true,
      "ffmpeg_binary": "ffmpeg",
      "preset": "wheatley_light"
    }
  }
}
```

Test:

```bash
PYTHONPATH=src python3 -m wheatly speak "Tiny local voice path online."
```

## 6b. External TTS Wrapper

Use this path for Kokoro or any custom TTS script. The command must create the file passed as `{output}`.

```json
{
  "tts": {
    "backend": "external",
    "enabled": true,
    "external_command": [
      "python3",
      "path/to/your_kokoro_wrapper.py",
      "--text",
      "{text}",
      "--output",
      "{output}"
    ],
    "filter": {
      "enabled": true,
      "ffmpeg_binary": "ffmpeg",
      "preset": "wheatley_light"
    }
  }
}
```

## 7. Tool Configuration

Safe CLI tools are disabled by default. Add explicit commands only:

```json
{
  "tools": {
    "allowed_commands": {
      "uptime": ["/usr/bin/uptime"],
      "disk": ["/bin/df", "-h"]
    }
  }
}
```

The model can then request:

```json
{"tool_calls":[{"name":"run_safe_cli_tool","arguments":{"command":"uptime","args":[]}}]}
```

Test tools directly:

```bash
PYTHONPATH=src python3 -m wheatly tool robot_status
PYTHONPATH=src python3 -m wheatly tool set_eye_expression --args '{"expression":"thinking"}'
PYTHONPATH=src python3 -m wheatly tool calculator --args '{"expression":"sqrt(5+5)+sin(4)**3+6/7","round_digits":4}'
```

There is no internet/web-search tool in the current default. The notes-search tool is not registered.

## 7a. Editable Instructions And Memory

Edit these files directly:

```text
prompts/system.md
prompts/user.md
prompts/tools.json
memory/wheatly.md
```

`prompts/system.md` controls main behavior. `prompts/user.md` is for your always-on preferences. `prompts/tools.json` overrides tool descriptions and per-tool instructions while keeping the JSON schemas in code. `memory/wheatly.md` is injected into context automatically.

Say or type:

```text
Remember this: I prefer short direct answers.
```

The agent appends that memory to `memory/wheatly.md` and injects it into future turns. Test it directly:

```bash
PYTHONPATH=src python3 -m wheatly once --text "Remember this: I prefer short direct answers."
```

## 7b. Voice Tuning

Piper voice behavior is configured in `configs/wheatly.local.json`:

```json
{
  "tts": {
    "length_scale": 0.66,
    "noise_scale": 0.72,
    "noise_w_scale": 0.82,
    "sentence_silence": 0.02,
    "stream_speech": true,
    "stream_initial_min_words": 12,
    "stream_min_words": 24,
    "stream_max_words": 60,
    "stream_feedback_min_words": 8,
    "stream_max_initial_wait_seconds": 1.2,
    "volume": 1.05,
    "filter": {"enabled": false, "preset": "wheatley_bright"}
  }
}
```

Lower `length_scale` speaks faster. The current default is intentionally fast because Piper playback is slower than Qwen text generation on this Mac. `stream_initial_min_words` controls only the first spoken chunk, while `stream_min_words` / `stream_max_words` control later chunks. This lets the assistant start speaking sooner without making the middle of the answer too fragmented.

Adaptive streaming stores recent LLM/TTS speed in `runtime/state/latency_stats.json`. `stream_initial_min_words` is the normal first-chunk minimum, `stream_min_words` is the normal later-chunk minimum, `stream_max_words` is the hard chunk size cap, `stream_feedback_min_words` is the fallback first chunk size, and `stream_max_initial_wait_seconds` is the maximum initial wait before speaking. After that wait expires, the speaker prefers the first complete sentence; if there is no sentence yet, it speaks the feedback word chunk.

Inspect the current adaptive recommendation:

```bash
PYTHONPATH=src python3 -m wheatly stats
```

The ffmpeg voice filter is disabled by default because filtered chunk playback caused audible static tails on macOS. If chunked speech still causes audible gaps, set `stream_speech` to `false`; text will still stream, and speech will happen once per full answer.

## 8. Latency Measurement Checklist

For each hardware/model combo, record:

- STT latency from speech end to text.
- LLM first-token time if runtime exposes it.
- LLM total time for 40, 80, and 150 tokens.
- TTS first-audio time.
- End-to-end perceived response time.
- Misrecognitions from Slovak-accented English.

Use the built-in text benchmark for quick LLM loop comparisons:

```bash
PYTHONPATH=src python3 -m wheatly bench --repeat 5 --text "Answer in one short sentence: are you online?"
```
