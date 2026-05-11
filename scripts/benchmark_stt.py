#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
import statistics
import time
import urllib.error
import urllib.request
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from faster_whisper import WhisperModel


DEFAULT_MODELS = [
    "Systran/faster-distil-whisper-small.en",
    "tiny.en",
    "small.en",
    "small",
]

DEFAULT_REFERENCE_MODEL = "distil-large-v3"


@dataclass
class AudioSample:
    path: Path
    duration_seconds: float
    bucket: str = ""


@dataclass
class ModelSpec:
    name: str
    beam_size: int
    language: str
    compute_type: str
    device: str

    @property
    def label(self) -> str:
        return f"{self.name} | beam={self.beam_size} | {self.device}/{self.compute_type}"


def main() -> int:
    args = parse_args()
    samples = select_samples(args)
    if not samples:
        raise SystemExit("No audio samples found. Pass --audio or --audio-root.")

    model_specs = parse_model_specs(args)
    phase_specs = parse_phase_specs(args)
    reference_spec = ModelSpec(
        name=args.reference_model,
        beam_size=args.reference_beam_size,
        language=args.language,
        compute_type=args.compute_type,
        device=args.device,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    json_path = output_dir / f"stt-benchmark-{timestamp}.json"
    csv_path = output_dir / f"stt-benchmark-{timestamp}.csv"
    md_path = Path(args.markdown)

    print(f"Samples: {len(samples)}")
    print(f"Reference: {reference_spec.label}")
    references, reference_rows = transcribe_reference(reference_spec, samples, args)

    rows: list[dict] = []
    rows.extend(reference_rows)
    if phase_specs:
        for phase, spec in phase_specs:
            rows.extend(transcribe_model(spec, samples, references, args, phase=phase))
            if args.include_remote:
                rows.extend(transcribe_remote_model(spec, samples, references, args, phase=phase))
    else:
        for spec in model_specs:
            if spec.name == reference_spec.name and spec.beam_size == reference_spec.beam_size:
                continue
            rows.extend(transcribe_model(spec, samples, references, args))
            if args.include_remote:
                rows.extend(transcribe_remote_model(spec, samples, references, args))

    payload = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "reference_model": reference_spec.__dict__,
        "models": [spec.__dict__ for spec in model_specs],
        "phase_models": [
            {"phase": phase, **spec.__dict__} for phase, spec in phase_specs
        ],
        "remote_base_url": args.remote_base_url,
        "remote_label": args.remote_label,
        "sample_count": len(samples),
        "samples": [
            {
                "path": str(sample.path),
                "duration_seconds": sample.duration_seconds,
                "bucket": sample.bucket,
            }
            for sample in samples
        ],
        "rows": rows,
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_csv(csv_path, rows)
    write_markdown(md_path, payload, rows, json_path, csv_path)
    print(f"Wrote {md_path}")
    print(f"Wrote {json_path}")
    print(f"Wrote {csv_path}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark faster-whisper STT models on saved profile audio logs."
    )
    parser.add_argument(
        "--audio-root",
        action="append",
        default=[
            "profiles/wheatley/runtime/audio",
        ],
        help="Root to scan recursively for user WAVs. Can be repeated.",
    )
    parser.add_argument(
        "--audio",
        action="append",
        default=[],
        help="Explicit audio file to include. Can be repeated.",
    )
    parser.add_argument("--max-samples", type=int, default=12)
    parser.add_argument("--min-duration", type=float, default=0.4)
    parser.add_argument("--max-duration", type=float, default=18.0)
    parser.add_argument(
        "--duration-bucket",
        action="append",
        default=[],
        help=(
            "Stratified sample bucket as label:min_seconds:max_seconds:count. "
            "Can be repeated. Overrides random --max-samples selection."
        ),
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--model",
        action="append",
        default=[],
        help=(
            "Model to test. Optional suffix format: model@beam. "
            "Can be repeated. Defaults to current Systran, tiny.en, small.en, small."
        ),
    )
    parser.add_argument("--reference-model", default=DEFAULT_REFERENCE_MODEL)
    parser.add_argument("--reference-beam-size", type=int, default=3)
    parser.add_argument("--beam-size", type=int, default=1)
    parser.add_argument("--language", default="en")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--compute-type", default="int8")
    parser.add_argument("--no-vad-filter", action="store_true")
    parser.add_argument("--condition-on-previous-text", action="store_true")
    parser.add_argument(
        "--phase-model",
        action="append",
        default=[],
        help=(
            "Two-phase scenario entry as phase:model@beam, for example "
            "preview:small@1 or final:distil-large-v3@3. Can be repeated."
        ),
    )
    parser.add_argument("--remote-base-url", default="")
    parser.add_argument("--remote-label", default="remote")
    parser.add_argument("--include-remote", action="store_true")
    parser.add_argument("--remote-timeout", type=float, default=180.0)
    parser.add_argument("--output-dir", default="docs/benchmarks")
    parser.add_argument("--markdown", default="docs/stt-benchmark-results.md")
    return parser.parse_args()


def parse_model_specs(args: argparse.Namespace) -> list[ModelSpec]:
    raw_models = args.model or DEFAULT_MODELS
    specs = []
    for raw in raw_models:
        name, beam_size = parse_model(raw, args.beam_size)
        specs.append(
            ModelSpec(
                name=name,
                beam_size=beam_size,
                language=args.language,
                compute_type=args.compute_type,
                device=args.device,
            )
        )
    return specs


def parse_phase_specs(args: argparse.Namespace) -> list[tuple[str, ModelSpec]]:
    specs = []
    for raw in args.phase_model:
        if ":" not in raw:
            raise SystemExit("--phase-model must use phase:model@beam format")
        phase, model = raw.split(":", 1)
        name, beam_size = parse_model(model, args.beam_size)
        specs.append(
            (
                phase.strip(),
                ModelSpec(
                    name=name,
                    beam_size=beam_size,
                    language=args.language,
                    compute_type=args.compute_type,
                    device=args.device,
                ),
            )
        )
    return specs


def parse_model(raw: str, default_beam_size: int) -> tuple[str, int]:
    if "@" not in raw:
        return raw, default_beam_size
    name, beam = raw.rsplit("@", 1)
    return name, int(beam)


def select_samples(args: argparse.Namespace) -> list[AudioSample]:
    paths: list[Path] = []
    paths.extend(Path(item) for item in args.audio)
    for root in args.audio_root:
        root_path = Path(root)
        if root_path.exists():
            paths.extend(root_path.rglob("*user*.wav"))
    unique = sorted({path.resolve() for path in paths if path.exists()})
    samples = []
    for path in unique:
        if not is_endpoint_user_wav(path):
            continue
        try:
            duration = wav_duration(path)
        except Exception:
            continue
        if args.min_duration <= duration <= args.max_duration:
            samples.append(AudioSample(path=path, duration_seconds=duration))
    buckets = parse_duration_buckets(args.duration_bucket)
    if buckets:
        return select_bucketed_samples(samples, buckets, args.seed)
    random.Random(args.seed).shuffle(samples)
    return samples[: args.max_samples]


def parse_duration_buckets(items: list[str]) -> list[tuple[str, float, float, int]]:
    buckets = []
    for item in items:
        parts = item.split(":")
        if len(parts) != 4:
            raise SystemExit(
                "--duration-bucket must be label:min_seconds:max_seconds:count"
            )
        label, minimum, maximum, count = parts
        buckets.append((label, float(minimum), float(maximum), int(count)))
    return buckets


def select_bucketed_samples(
    samples: list[AudioSample],
    buckets: list[tuple[str, float, float, int]],
    seed: int,
) -> list[AudioSample]:
    rng = random.Random(seed)
    selected: list[AudioSample] = []
    used: set[Path] = set()
    for label, minimum, maximum, count in buckets:
        candidates = [
            sample
            for sample in samples
            if sample.path not in used and minimum <= sample.duration_seconds < maximum
        ]
        rng.shuffle(candidates)
        for sample in candidates[:count]:
            sample.bucket = label
            selected.append(sample)
            used.add(sample.path)
    return selected


def is_endpoint_user_wav(path: Path) -> bool:
    name = path.name
    parts = set(path.parts)
    if "partials" in parts or "interrupts" in parts or "resume" in parts:
        return False
    if "user_partial" in name:
        return False
    return bool(re.search(r"_user(?:_\d+)?\.wav$", name))


def wav_duration(path: Path) -> float:
    with wave.open(str(path), "rb") as handle:
        frames = handle.getnframes()
        rate = handle.getframerate()
        return frames / float(rate or 1)


def transcribe_reference(
    spec: ModelSpec,
    samples: list[AudioSample],
    args: argparse.Namespace,
) -> tuple[dict[str, str], list[dict]]:
    print(f"Loading reference {spec.label}")
    loaded_at = time.perf_counter()
    model = WhisperModel(spec.name, device=spec.device, compute_type=spec.compute_type)
    load_seconds = time.perf_counter() - loaded_at
    references = {}
    rows = []
    for sample in samples:
        result = run_transcription(model, spec, sample, args)
        result.update(
            {
                "provider": "local",
                "phase": "reference",
                "is_reference": True,
                "reference_text": result["text"],
                "scoreable": bool(tokenize_words(result["text"])),
                "wer": 0.0,
                "cer": 0.0,
                "text_similarity": 1.0,
                "model_load_seconds": load_seconds,
            }
        )
        references[str(sample.path)] = result["text"]
        rows.append(result)
        print(f"  ref {sample.path.name}: {result['wall_seconds']:.3f}s")
    return references, rows


def transcribe_model(
    spec: ModelSpec,
    samples: list[AudioSample],
    references: dict[str, str],
    args: argparse.Namespace,
    phase: str = "model",
) -> list[dict]:
    print(f"Loading {spec.label}")
    loaded_at = time.perf_counter()
    model = WhisperModel(spec.name, device=spec.device, compute_type=spec.compute_type)
    load_seconds = time.perf_counter() - loaded_at
    rows = []
    for sample in samples:
        result = run_transcription(model, spec, sample, args)
        reference_text = references[str(sample.path)]
        scoreable = bool(tokenize_words(reference_text))
        wer = word_error_rate(reference_text, result["text"]) if scoreable else math.nan
        cer = char_error_rate(reference_text, result["text"]) if scoreable else math.nan
        result.update(
            {
                "provider": "local",
                "phase": phase,
                "is_reference": False,
                "reference_text": reference_text,
                "scoreable": scoreable,
                "wer": wer,
                "cer": cer,
                "text_similarity": max(0.0, 1.0 - cer) if scoreable else math.nan,
                "model_load_seconds": load_seconds,
            }
        )
        rows.append(result)
        print(
            f"  {sample.path.name}: {result['wall_seconds']:.3f}s, "
            f"rtf={result['rtf']:.2f}, wer={fmt(wer) or 'n/a'}"
        )
    return rows


def transcribe_remote_model(
    spec: ModelSpec,
    samples: list[AudioSample],
    references: dict[str, str],
    args: argparse.Namespace,
    phase: str = "model",
) -> list[dict]:
    print(f"Remote {args.remote_label} {phase}: {spec.name} | beam={spec.beam_size}")
    rows = []
    for sample in samples:
        result = run_remote_transcription(spec, sample, args)
        reference_text = references[str(sample.path)]
        scoreable = bool(tokenize_words(reference_text))
        wer = word_error_rate(reference_text, result["text"]) if scoreable else math.nan
        cer = char_error_rate(reference_text, result["text"]) if scoreable else math.nan
        result.update(
            {
                "provider": args.remote_label,
                "phase": phase,
                "is_reference": False,
                "reference_text": reference_text,
                "scoreable": scoreable,
                "wer": wer,
                "cer": cer,
                "text_similarity": max(0.0, 1.0 - cer) if scoreable else math.nan,
                "model_load_seconds": 0.0,
            }
        )
        rows.append(result)
        print(
            f"  {sample.path.name}: {result['wall_seconds']:.3f}s, "
            f"rtf={result['rtf']:.2f}, wer={fmt(wer) or 'n/a'}"
        )
    return rows


def run_remote_transcription(
    spec: ModelSpec,
    sample: AudioSample,
    args: argparse.Namespace,
) -> dict:
    endpoint = remote_stt_endpoint(args.remote_base_url)
    fields = {
        "model": spec.name,
        "language": spec.language,
        "response_format": "json",
        "beam_size": str(spec.beam_size),
    }
    body, content_type = multipart_body(sample.path, fields)
    request = urllib.request.Request(
        endpoint,
        data=body,
        headers={"Content-Type": content_type, "Accept": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=args.remote_timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"remote STT failed: HTTP {exc.code}: {detail}") from exc
    wall_seconds = time.perf_counter() - started
    text = str(payload.get("text", "")).strip()
    words = tokenize_words(text)
    return {
        "model": spec.name,
        "beam_size": spec.beam_size,
        "device": "remote",
        "compute_type": spec.compute_type,
        "language": spec.language,
        "audio_path": str(sample.path),
        "audio_duration_seconds": round(sample.duration_seconds, 4),
        "bucket": sample.bucket,
        "wall_seconds": round(wall_seconds, 4),
        "server_wall_seconds": payload.get("wall_seconds"),
        "rtf": round(wall_seconds / max(sample.duration_seconds, 0.001), 4),
        "words": len(words),
        "words_per_second": round(len(words) / max(wall_seconds, 0.001), 4),
        "detected_language": payload.get("language"),
        "text": text,
    }


def run_transcription(
    model: WhisperModel,
    spec: ModelSpec,
    sample: AudioSample,
    args: argparse.Namespace,
) -> dict:
    started = time.perf_counter()
    segments, info = model.transcribe(
        str(sample.path),
        language=spec.language,
        task="transcribe",
        beam_size=spec.beam_size,
        repetition_penalty=1.15,
        no_repeat_ngram_size=3,
        max_new_tokens=160,
        vad_filter=not args.no_vad_filter,
        condition_on_previous_text=args.condition_on_previous_text,
    )
    text = " ".join(segment.text.strip() for segment in segments).strip()
    wall_seconds = time.perf_counter() - started
    words = tokenize_words(text)
    return {
        "provider": "local",
        "phase": "model",
        "model": spec.name,
        "beam_size": spec.beam_size,
        "device": spec.device,
        "compute_type": spec.compute_type,
        "language": spec.language,
        "audio_path": str(sample.path),
        "audio_duration_seconds": round(sample.duration_seconds, 4),
        "bucket": sample.bucket,
        "wall_seconds": round(wall_seconds, 4),
        "server_wall_seconds": None,
        "rtf": round(wall_seconds / max(sample.duration_seconds, 0.001), 4),
        "words": len(words),
        "words_per_second": round(len(words) / max(wall_seconds, 0.001), 4),
        "detected_language": getattr(info, "language", None),
        "text": text,
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    fieldnames = [
        "provider",
        "phase",
        "model",
        "beam_size",
        "is_reference",
        "audio_path",
        "audio_duration_seconds",
        "bucket",
        "wall_seconds",
        "server_wall_seconds",
        "rtf",
        "words",
        "words_per_second",
        "wer",
        "cer",
        "text_similarity",
        "scoreable",
        "text",
        "reference_text",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(
    path: Path,
    payload: dict,
    rows: list[dict],
    json_path: Path,
    csv_path: Path,
) -> None:
    groups = group_by_model(rows)
    lines = [
        "# STT Benchmark Results",
        "",
        f"Generated: `{payload['created_at']}`",
        "",
        f"Reference model: `{payload['reference_model']['name']}` "
        f"beam `{payload['reference_model']['beam_size']}`.",
        "",
        f"Samples: `{payload['sample_count']}` user WAV files from profile runtime logs.",
        "",
        f"Raw JSON: `{json_path}`",
        "",
        f"Raw CSV: `{csv_path}`",
        "",
        "## Summary",
        "",
        "| Provider | Phase | Model | Beam | Mean wall s | Median wall s | Mean RTF | Mean WER vs reference | Mean CER | Words/s |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for key, items in sorted(groups.items()):
        provider, phase, model, beam = key
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{provider}`",
                    f"`{phase}`",
                    f"`{model}`",
                    str(beam),
                    fmt(mean(item["wall_seconds"] for item in items)),
                    fmt(median(item["wall_seconds"] for item in items)),
                    fmt(mean(item["rtf"] for item in items)),
                    fmt(mean_finite(item["wer"] for item in items)),
                    fmt(mean_finite(item["cer"] for item in items)),
                    fmt(mean(item["words_per_second"] for item in items)),
                ]
            )
            + " |"
        )
    simulation_rows = two_phase_simulation_rows(rows)
    if simulation_rows:
        lines.extend(
            [
                "",
                "## Two-Phase Simulation",
                "",
                "| Provider | Preview model | Final model | Mean preview s | Mean final s | Mean total s | Mean total RTF |",
                "| --- | --- | --- | ---: | ---: | ---: | ---: |",
            ]
        )
        for item in simulation_rows:
            lines.append(
                "| "
                + " | ".join(
                    [
                        f"`{item['provider']}`",
                        f"`{item['preview_model']}`",
                        f"`{item['final_model']}`",
                        fmt(item["mean_preview_seconds"]),
                        fmt(item["mean_final_seconds"]),
                        fmt(item["mean_total_seconds"]),
                        fmt(item["mean_total_rtf"]),
                    ]
                )
                + " |"
            )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- WER/CER are measured against the configured reference model transcript, not human labels.",
            "- Empty-reference samples count for timing but are excluded from WER/CER averages.",
            "- RTF is wall transcription seconds divided by audio duration; lower is better.",
            "- `text_similarity` in raw outputs is `1 - CER` after simple normalization.",
            "- This benchmark uses saved endpoint WAVs, so it measures final transcription latency, not live partial cadence.",
            "",
            "## Sample Details",
            "",
            "Sample-level details are intentionally omitted from the Markdown summary because they can include private runtime paths, timestamps, and real utterance text. Raw JSON/CSV outputs are local artifacts and should not be committed.",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def group_by_model(rows: Iterable[dict]) -> dict[tuple[str, str, str, int], list[dict]]:
    groups: dict[tuple[str, str, str, int], list[dict]] = {}
    for row in rows:
        groups.setdefault(
            (
                row.get("provider", "local"),
                row.get("phase", "model"),
                row["model"],
                row["beam_size"],
            ),
            [],
        ).append(row)
    return groups


def two_phase_simulation_rows(rows: list[dict]) -> list[dict]:
    providers = sorted({row.get("provider", "local") for row in rows})
    result = []
    for provider in providers:
        preview_rows = [
            row for row in rows if row.get("provider") == provider and row.get("phase") == "preview"
        ]
        final_rows = [
            row for row in rows if row.get("provider") == provider and row.get("phase") == "final"
        ]
        if not preview_rows or not final_rows:
            continue
        final_by_path = {row["audio_path"]: row for row in final_rows}
        totals = []
        rtfs = []
        previews = []
        finals = []
        for preview in preview_rows:
            final = final_by_path.get(preview["audio_path"])
            if not final:
                continue
            total = preview["wall_seconds"] + final["wall_seconds"]
            duration = max(float(preview["audio_duration_seconds"]), 0.001)
            previews.append(preview["wall_seconds"])
            finals.append(final["wall_seconds"])
            totals.append(total)
            rtfs.append(total / duration)
        if not totals:
            continue
        result.append(
            {
                "provider": provider,
                "preview_model": preview_rows[0]["model"],
                "final_model": final_rows[0]["model"],
                "mean_preview_seconds": mean(previews),
                "mean_final_seconds": mean(finals),
                "mean_total_seconds": mean(totals),
                "mean_total_rtf": mean(rtfs),
            }
        )
    return result


def remote_stt_endpoint(base_url: str) -> str:
    base = base_url.rstrip("/")
    if base.endswith("/audio/transcriptions"):
        return base
    if base.endswith("/v1"):
        return f"{base}/audio/transcriptions"
    return f"{base}/v1/audio/transcriptions"


def multipart_body(audio_path: Path, fields: dict[str, str]) -> tuple[bytes, str]:
    boundary = f"wheatley-bench-{time.time_ns()}"
    parts: list[bytes] = []
    for name, value in fields.items():
        parts.extend(
            [
                f"--{boundary}\r\n".encode("ascii"),
                f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode("ascii"),
                str(value).encode("utf-8"),
                b"\r\n",
            ]
        )
    parts.extend(
        [
            f"--{boundary}\r\n".encode("ascii"),
            (
                'Content-Disposition: form-data; name="file"; '
                f'filename="{audio_path.name}"\r\n'
            ).encode("utf-8"),
            b"Content-Type: audio/wav\r\n\r\n",
            audio_path.read_bytes(),
            b"\r\n",
            f"--{boundary}--\r\n".encode("ascii"),
        ]
    )
    return b"".join(parts), f"multipart/form-data; boundary={boundary}"


def mean(values: Iterable[float]) -> float:
    data = list(values)
    return statistics.fmean(data) if data else math.nan


def mean_finite(values: Iterable[float]) -> float:
    data = [value for value in values if isinstance(value, (int, float)) and math.isfinite(value)]
    return statistics.fmean(data) if data else math.nan


def median(values: Iterable[float]) -> float:
    data = list(values)
    return statistics.median(data) if data else math.nan


def fmt(value: float) -> str:
    if math.isnan(value):
        return ""
    return f"{value:.3f}"


def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize_words(text: str) -> list[str]:
    normalized = normalize_text(text)
    return normalized.split() if normalized else []


def word_error_rate(reference: str, hypothesis: str) -> float:
    ref = tokenize_words(reference)
    hyp = tokenize_words(hypothesis)
    if not ref:
        return 0.0 if not hyp else 1.0
    return edit_distance(ref, hyp) / len(ref)


def char_error_rate(reference: str, hypothesis: str) -> float:
    ref = list(normalize_text(reference).replace(" ", ""))
    hyp = list(normalize_text(hypothesis).replace(" ", ""))
    if not ref:
        return 0.0 if not hyp else 1.0
    return edit_distance(ref, hyp) / len(ref)


def edit_distance(left: list[str], right: list[str]) -> int:
    previous = list(range(len(right) + 1))
    for i, left_item in enumerate(left, start=1):
        current = [i]
        for j, right_item in enumerate(right, start=1):
            current.append(
                min(
                    previous[j] + 1,
                    current[j - 1] + 1,
                    previous[j - 1] + (left_item != right_item),
                )
            )
        previous = current
    return previous[-1]


if __name__ == "__main__":
    raise SystemExit(main())
