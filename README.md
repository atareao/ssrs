# SSRS — Speech-to-Text CLI

[![CI](https://github.com/atareao/ssrs/actions/workflows/ci.yml/badge.svg)](https://github.com/atareao/ssrs/actions/workflows/ci.yml)
[![Latest Release](https://img.shields.io/github/v/release/atareao/ssrs?label=release)](https://github.com/atareao/ssrs/releases)
[![License: MIT](https://img.shields.io/crates/l/ssrs)](https://github.com/atareao/ssrs/blob/main/LICENSE)
[![crates.io](https://img.shields.io/crates/v/ssrs)](https://crates.io/crates/ssrs)

Fast, offline speech-to-text transcription powered by [Whisper](https://openai.com/research/whisper) and [Candle](https://github.com/huggingface/candle) (Rust ML inference engine). Run state-of-the-art models locally on CPU or CUDA with support for multiple audio formats and output formats.

---

## Features

- **Local inference** — no cloud API calls, all processing happens on your machine
- **Multiple Whisper model sizes** — from `tiny` (fast) to `large` (accurate), via HuggingFace Hub
- **Audio format support** — MP3, WAV, FLAC, OGG/Vorbis
- **Output formats** — plain text, SRT subtitles, or structured JSON
- **Model management** — search, download, list, and remove models from HuggingFace Hub
- **Device support** — CPU and CUDA (GPU) acceleration
- **Language detection** — auto-detect or specify ISO 639-1 language codes
- **Advanced decoding** — configurable temperature and beam size
- **Long audio** — automatic chunking of audio longer than 30 seconds

---

## Installation

### From crates.io

```sh
cargo install ssrs
```

### From source

```sh
git clone https://github.com/atareao/ssrs.git
cd ssrs
cargo build --release
# binary is at target/release/ssrs
```

### Download pre-built binaries

See the [latest release](https://github.com/atareao/ssrs/releases) for Linux (x86_64 / aarch64), macOS (aarch64), and Windows (x86_64) binaries.

---

## Quick Start

### 1. Download a model

```sh
ssrs models download openai/whisper-tiny
```

### 2. Transcribe an audio file

```sh
ssrs transcribe -f audio.mp3
```

With options:

```sh
ssrs transcribe -f audio.mp3 \
  -m openai/whisper-base \
  -d cpu \
  -l en \
  -o srt \
  -O output.srt
```

---

## CLI Reference

### Transcription

```
ssrs transcribe [OPTIONS] --file <FILE>
```

| Flag | Description | Default |
|------|-------------|---------|
| `-f, --file` | Path to audio file (MP3, WAV, FLAC, OGG) | required |
| `-m, --model` | HuggingFace model ID | `openai/whisper-tiny` |
| `-d, --device` | Device: `cpu` or `cuda` | `cpu` |
| `-l, --language` | Language code (e.g. `en`, `es`) or `auto` | auto-detect |
| `-o, --output` | Output format: `txt`, `srt`, `json` | `txt` |
| `-O, --output-file` | Write output to file instead of stdout | stdout |
| `--initial-prompt` | Guide transcription (capitalization, punctuation, terms) | none |
| `--temperature` | Sampling temperature (0.0 = greedy) | `0.0` |
| `--beam-size` | Beam size for decoding (1 = greedy) | `1` |

### Model Management

```sh
# Search available Whisper models on HuggingFace
ssrs models search [-q <query>] [-l <limit>]

# List locally cached models
ssrs models list

# Download a model
ssrs models download <model_id>

# Remove a cached model
ssrs models remove <model_id>
```

---

## Supported Models

SSRS supports any Whisper model available on HuggingFace Hub that includes compatible weights:

- **Safetensors** — full precision (`model.safetensors`)
- **GGUF** — quantized (`model.gguf`)

Recommended model IDs:

| Model | Size | Speed | Accuracy |
|-------|------|-------|----------|
| `openai/whisper-tiny` | ~75 MB | Fastest | Good |
| `openai/whisper-base` | ~140 MB | Fast | Better |
| `openai/whisper-small` | ~480 MB | Moderate | High |
| `openai/whisper-medium` | ~1.5 GB | Slow | Higher |
| `openai/whisper-large-v3` | ~3 GB | Slowest | Highest |

---

## Output Formats

### Plain text (`-o txt`)
```
Hello, this is a test of the transcription system.
```

### SRT subtitles (`-o srt`)
```
1
00:00:00,000 --> 00:00:04,000
Hello, this is a test of the transcription system.
```

### JSON (`-o json`)
```json
{
  "text": "Hello, this is a test of the transcription system.",
  "segments": [
    { "start": 0.0, "end": 4.0, "text": "Hello, this is a test of the transcription system." }
  ],
  "model": "openai/whisper-tiny",
  "language": "en",
  "duration_secs": 4.0
}
```

---

## Model Cache

Models are cached at `~/.cache/ssrs/<model_id>/` (with `/` replaced by `__`).

---

## Development

```sh
# Build
cargo build

# Run tests
cargo test

# Lint
cargo clippy --all-targets --all-features -- -D warnings

# Format check
cargo fmt --check
```

### Tech Stack

- **[Candle](https://github.com/huggingface/candle)** — Rust ML inference (CPU/GPU)
- **[clap](https://github.com/clap-rs/clap)** — CLI argument parsing
- **[symphonia](https://github.com/pdeljanov/Symphonia)** — Audio decoding
- **[hf-hub](https://github.com/huggingface/hf-hub)** — HuggingFace Hub client
- **[tokenizers](https://github.com/huggingface/tokenizers)** — Whisper token encoding/decoding

---

## License

MIT — see [LICENSE](LICENSE)

---

## Acknowledgments

- [OpenAI Whisper](https://github.com/openai/whisper) — the underlying speech recognition model
- [HuggingFace Candle](https://github.com/huggingface/candle) — the Rust ML inference framework
- [HuggingFace Hub](https://huggingface.co) — model hosting
