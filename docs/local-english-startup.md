# Local English Startup

This guide is for people and agents cloning the repo and bringing up the first useful local setup. It intentionally covers only the English, local-only path:

```text
microphone -> faster-whisper -> Ollama LLM -> Piper TTS -> speaker
```

No remote STT, no remote LLM, and no Slovak mode are required for this guide.

## 1. Install System Prerequisites

You need:

- Python 3.10 or newer.
- Git.
- A working microphone and speaker.
- Ollama for the local LLM.
- FFmpeg or a platform playback tool for generated audio.

macOS:

```bash
brew install python git portaudio ffmpeg
```

Install Ollama from <https://ollama.com/download>, or use your normal package manager if you already have one.

Ubuntu/Debian PC:

```bash
sudo apt-get update
sudo apt-get install -y git python3 python3-venv python3-dev portaudio19-dev ffmpeg alsa-utils
```

Install Ollama from <https://ollama.com/download>.

Windows PC:

- Install Python 3.10+ from <https://www.python.org/downloads/>.
- Install Git from <https://git-scm.com/downloads>.
- Install Ollama from <https://ollama.com/download>.
- Install FFmpeg if you want speech playback from generated audio.

The shell commands below are written for macOS/Linux. On Windows, use the same steps with PowerShell path syntax.

## 2. Clone And Create The Python Environment

```bash
git clone REPO_URL local-talking-assistant
cd local-talking-assistant
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[audio,stt]'
python -m pip install piper-tts
```

The first STT run may download the selected Whisper model into the normal Hugging Face cache. After that, STT can run offline from cache.

## 3. Download The Local LLM

Start Ollama, then pull the model used by the default profile:

```bash
ollama pull qwen3.5:4b
ollama run qwen3.5:4b "Answer in one short sentence: are you ready?"
```

If this model is unavailable on your machine, choose another small local chat model and update `profiles/wheatley/config.jsonc` so `llm.model` matches the model name you pulled.

## 4. Download The English Piper Voice

The default English profile expects the Piper voice here:

```text
models/piper/en_GB-alan-medium.onnx
models/piper/en_GB-alan-medium.onnx.json
```

Download both files:

```bash
mkdir -p models/piper
curl -L -o models/piper/en_GB-alan-medium.onnx \
  https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_GB/alan/medium/en_GB-alan-medium.onnx
curl -L -o models/piper/en_GB-alan-medium.onnx.json \
  https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_GB/alan/medium/en_GB-alan-medium.onnx.json
```

Do not commit files under `models/`.

## 5. Make The Profile Local-Only

Open `profiles/wheatley/config.jsonc` and validate these values. Set them if they differ.

```jsonc
{
  "runtime": {
    "default_language": "en"
  },
  "stt": {
    "backend": "faster_whisper",
    "model": "small",
    "language": "en",
    "device": "cpu",
    "compute_type": "int8",
    "beam_size": 1,
    "vad_filter": true,
    "condition_on_previous_text": false,
    "preview_model": "small",
    "preview_remote_model": "small",
    "preview_use_remote": false,
    "preview_beam_size": 1,
    "final_model": "distil-large-v3",
    "final_remote_model": "distil-large-v3",
    "final_use_remote": false,
    "final_beam_size": 3
  },
  "language": {
    "default": "en",
    "languages": {
      "en": {
        "stt_model": "small",
        "stt_language": "en",
        "tts_backend": "piper",
        "tts_piper_model": "models/piper/en_GB-alan-medium.onnx"
      }
    }
  },
  "llm": {
    "backend": "ollama",
    "model": "qwen3.5:4b",
    "remote": {
      "enabled": false
    }
  },
  "tts": {
    "backend": "piper",
    "enabled": true,
    "playback": true,
    "piper_binary": ".venv/bin/python",
    "piper_model": "models/piper/en_GB-alan-medium.onnx"
  },
  "tools": {
    "python_interpreter_read_roots": [
      "files"
    ]
  }
}
```

Only copy the shown fields into their existing sections. Do not replace the whole config file with this snippet.

The current voice path uses two-phase STT. Live preview uses multilingual `small` with CPU int8 and beam size 1. After the endpoint, final transcription uses `distil-large-v3` with beam size 3. Remote preview/final can be enabled separately; if a configured remote phase is unavailable, Wheatley falls back to the local preview model.

If this profile has been run before, clear any persisted language state so startup stays English:

```bash
rm -f profiles/wheatley/runtime/state/language.json
```

## 6. Validate The Environment

Run tests:

```bash
PYTHONPATH=src python3 -m unittest discover -s tests
```

Run diagnostics:

```bash
PYTHONPATH=src python3 -m wheatley doctor
```

Expected basics:

- `ollama`: true.
- `numpy`: true.
- `faster_whisper`: true.
- A playback path exists through `afplay`, `aplay`, or `ffplay`.
- `sounddevice`: true is needed for microphone voice mode.

Run the isolated echo smoke test:

```bash
PYTHONPATH=src python3 -m wheatley --profile test once --text "what time is it?"
```

Run the real local LLM text path:

```bash
PYTHONPATH=src python3 -m wheatley once --stream --text "Answer in one short sentence: what are you?"
```

Test TTS:

```bash
PYTHONPATH=src python3 -m wheatley speak "Hello. I am your local assistant."
```

## 7. Run The Assistant

Text chat:

```bash
PYTHONPATH=src python3 -m wheatley chat --stream
```

One microphone turn:

```bash
PYTHONPATH=src python3 -m wheatley listen --speak
```

Continuous voice loop:

```bash
PYTHONPATH=src python3 -m wheatley voice
```

Or:

```bash
./scripts/start_wheatley.sh
```

Voice commands:

- `Quit.` or `Exit.` stops the voice loop.
- `Start a new chat.` clears conversation history.
- `Remember this: ...` writes a manual memory fact.
- Loud `stop` interrupts current speech playback when speech interruption is enabled.

## 8. Where To Check Results

Profile runtime files are under:

```text
profiles/wheatley/runtime/
```

Useful files:

- `logs/turns.jsonl`: user text, assistant text, model name, and tool results.
- `logs/tools.jsonl`: tool requests and results.
- `audio/`: recorded user audio and generated speech.
- `state/`: runtime state such as language, eye expression, and latency stats.

For tests and smoke checks, prefer `--profile test` so experiments do not affect the main profile memory.

## Troubleshooting

If `doctor` says `sounddevice: false`, install PortAudio for your OS and reinstall Python dependencies in the venv.

If TTS fails, confirm:

```bash
. .venv/bin/activate
python -m piper --help
ls models/piper/en_GB-alan-medium.onnx*
```

If Ollama fails, confirm it is running:

```bash
ollama list
curl http://localhost:11434/api/tags
```

If the first STT run is slow, wait for the model download and conversion cache to finish. Later runs should be faster.

## Technical Notes

`VoiceAgent` applies the active language settings at startup, so English STT and TTS values under `language.languages.en` can override the top-level `stt` and `tts` fields. That is why this guide asks you to check both places.

The app does not delete runtime audio or logs automatically. Clear `profiles/wheatley/runtime/` manually only when you intentionally want to reset local run history.
