# SSRS — Speech-to-Text CLI

Terminal tool for audio transcription using Rust, `clap` (CLI), `candle` (ML inference), and HuggingFace model hub.

## Build & Run

```sh
cargo build            # debug build
cargo build --release  # optimized (required for usable inference speed)
cargo run -- <args>    # run with CLI args
```

## CLI

```
ssrs transcribe -f <audio> -m openai/whisper-tiny [-d cpu|cuda] [-l en] [-o txt|srt|json]
ssrs models search [-q <query>] [-l <limit>]
ssrs models list
ssrs models download <model_id>
ssrs models remove <model_id>
```

## Key Dependencies

- **clap** — CLI argument parsing with derive macros
- **candle-core / candle-nn / candle-transformers** — ML inference (CPU/GPU)
- **hf-hub** — download models from HuggingFace Hub (sync API via `ureq`)
- **reqwest** (blocking) — HuggingFace REST API for model search
- **symphonia** — audio decoding (MP3, WAV, FLAC, OGG)
- **tokenizers** — HuggingFace tokenizer for Whisper token encoding/decoding

## Architecture

```
src/
├── main.rs     # entrypoint, CLI dispatch, subcommand handlers
├── cli.rs      # clap CLI definition (Cli, Command, ModelsAction, etc.)
├── hf.rs       # HuggingFace REST API client (search models by pipeline_tag)
├── cache.rs    # local model manifest (~/.cache/ssrs/models.json)
├── model.rs    # model loading: safetensors (VarBuilder::from_mmaped_safetensors)
│               #   and GGUF (VarBuilder::from_gguf) auto-detected per directory
├── whisper.rs  # Whisper inference: encode, decode, transcribe (chunking)
└── audio.rs    # audio loading (symphonia), mono conversion, resample to 16kHz
```

## Model Management

- Models cached at `~/.cache/ssrs/<model_id>/` (with `/` → `__` escaping)
- Manifest at `~/.cache/ssrs/models.json` tracks downloaded models, variant, and timestamp
- Auto-detects variant: if `model.gguf` exists → GGUF, else → safetensors
- Downloads `config.json`, `tokenizer.json`, `mel_filters.npz`, and weights on `models download`

## Inference Pipeline

1. Load audio via symphonia (MP3, WAV, FLAC, OGG) → PCM samples → mono → resample to 16kHz
2. Split audio into 30s chunks (480000 samples) for long audio
3. Compute mel spectrogram via `candle_transformers::models::whisper::audio::pcm_to_mel`
4. Truncate mel to N_FRAMES (3000) to fit encoder positional embedding
5. Encode mel spectrogram through Whisper encoder
6. Decode token-by-token (greedy) through Whisper decoder with KV cache
7. Decode token IDs → text via HuggingFace `tokenizers`
8. Output as plain text, SRT (per-segment timestamps), or JSON (with segments array)

## Model Format Support

- **Safetensors** — full precision, loaded via `VarBuilder::from_mmaped_safetensors` (zero-copy mmap)
- **GGUF** — quantized, loaded via `VarBuilder::from_gguf`
- **NOT supported**: CTranslate2 / faster-whisper format (candle has no CT2 loader)

## Conventions

- Use `thiserror` for library errors, `anyhow` for binary error propagation
- Prefer `&str` / `&[u8]` in function signatures; own only when necessary
- Run `cargo clippy --all-targets --all-features -- -D warnings` before committing
- Run `cargo fmt --check` to verify formatting
- Use `#[expect(clippy::lint)]` over `#[allow(...)]` with justification
