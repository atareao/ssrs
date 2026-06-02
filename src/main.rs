mod audio;
mod cache;
mod cli;
mod hf;
mod model;
mod whisper;

use std::io::Write;

use anyhow::Result;
use clap::Parser;
use cli::{Cli, Command, Device as CliDevice, ModelsAction, OutputFormat};

fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Command::Transcribe {
            file,
            model: model_id,
            device,
            language,
            output,
            output_file,
            initial_prompt,
            temperature,
            beam_size,
        } => cmd_transcribe(
            &file,
            &model_id,
            device,
            language.as_deref(),
            output,
            output_file.as_deref(),
            initial_prompt.as_deref(),
            temperature,
            beam_size,
        ),
        Command::Models { action } => match action {
            ModelsAction::Search { query, limit } => cmd_models_search(query.as_deref(), limit),
            ModelsAction::List => cmd_models_list(),
            ModelsAction::Download { model_id } => cmd_models_download(&model_id),
            ModelsAction::Remove { model_id } => cmd_models_remove(&model_id),
        },
    }
}

fn cli_device(d: CliDevice) -> Result<candle_core::Device> {
    match d {
        CliDevice::Cpu => Ok(candle_core::Device::Cpu),
        CliDevice::Cuda => candle_core::Device::new_cuda(0).map_err(|_| {
            anyhow::anyhow!(
                "CUDA not available. Install candle with CUDA support or use --device cpu."
            )
        }),
    }
}

#[expect(clippy::too_many_arguments)]
fn cmd_transcribe(
    file: &str,
    model_id: &str,
    device: CliDevice,
    language: Option<&str>,
    output: OutputFormat,
    output_file: Option<&str>,
    initial_prompt: Option<&str>,
    temperature: f64,
    beam_size: usize,
) -> Result<()> {
    let dev = cli_device(device)?;

    eprintln!("Loading model '{model_id}'...");
    let mut loaded = model::load_model(model_id, &dev)?;

    eprintln!("Loading audio '{file}'...");
    let audio_data = audio::load_audio(std::path::Path::new(file))?;
    let samples = audio::to_mono(&audio_data.samples, audio_data.channels);
    let samples = audio::resample_to_16k(&samples, audio_data.sample_rate);
    let duration = audio::duration_secs(&samples, 16000);

    let options = whisper::TranscribeOptions {
        language,
        initial_prompt,
        temperature,
        beam_size,
    };
    let segments = whisper::transcribe(&mut loaded, &samples, &options)?;

    let output_str = match output {
        OutputFormat::Txt => {
            let mut text = String::new();
            for seg in &segments {
                let trimmed = seg.text.trim();
                if !trimmed.is_empty() {
                    text.push_str(trimmed);
                    text.push(' ');
                }
            }
            text
        }
        OutputFormat::Srt => {
            let mut srt = String::new();
            for (i, seg) in segments.iter().enumerate() {
                let start = format_srt_time(seg.start_sec);
                let end = format_srt_time(seg.end_sec);
                srt.push_str(&format!(
                    "{}\n{} --> {}\n{}\n\n",
                    i + 1,
                    start,
                    end,
                    seg.text.trim()
                ));
            }
            srt
        }
        OutputFormat::Json => {
            let json_segments: Vec<_> = segments
                .iter()
                .map(|s| {
                    serde_json::json!({
                        "start": s.start_sec,
                        "end": s.end_sec,
                        "text": s.text.trim(),
                    })
                })
                .collect();
            let full_text = segments
                .iter()
                .map(|s| s.text.trim())
                .filter(|t| !t.is_empty())
                .collect::<Vec<_>>()
                .join(" ");
            let json = serde_json::json!({
                "text": full_text,
                "segments": json_segments,
                "model": model_id,
                "language": language.unwrap_or("auto"),
                "duration_secs": duration,
            });
            json.to_string()
        }
    };

    if let Some(path) = output_file {
        let mut file = std::fs::File::create(path)?;
        file.write_all(output_str.as_bytes())?;
        eprintln!("Output written to {path}");
    } else {
        print!("{output_str}");
    }

    Ok(())
}

fn format_srt_time(secs: f64) -> String {
    let total_ms = (secs * 1000.0) as u64;
    let ms = total_ms % 1000;
    let total_s = total_ms / 1000;
    let s = total_s % 60;
    let total_m = total_s / 60;
    let m = total_m % 60;
    let h = total_m / 60;
    format!("{h:02}:{m:02}:{s:02},{ms:03}")
}

fn cmd_models_search(query: Option<&str>, limit: usize) -> Result<()> {
    let models = hf::search_models(query, limit)?;
    if models.is_empty() {
        println!("No models found.");
        return Ok(());
    }
    println!("{:<50} {:<12} {:<8}", "Model ID", "Downloads", "Likes");
    println!("{}", "-".repeat(72));
    for m in &models {
        println!("{:<50} {:<12} {:<8}", m.id, m.downloads, m.likes);
    }
    Ok(())
}

fn cmd_models_list() -> Result<()> {
    let models = cache::list_models()?;
    if models.is_empty() {
        println!("No models cached. Use `ssrs models download <model_id>` to download one.");
        return Ok(());
    }
    println!("{:<50} {:<15} {:<30}", "Model ID", "Variant", "Downloaded");
    println!("{}", "-".repeat(97));
    for m in &models {
        println!(
            "{:<50} {:<15} {:<30}",
            m.model_id,
            format!("{:?}", m.variant),
            m.downloaded_at
        );
    }
    Ok(())
}

fn cmd_models_download(model_id: &str) -> Result<()> {
    let api = hf_hub::api::sync::Api::new()?;
    let repo = api.repo(hf_hub::Repo::with_revision(
        model_id.to_string(),
        hf_hub::RepoType::Model,
        "main".to_string(),
    ));

    let dir = cache::model_cache_dir(model_id);
    std::fs::create_dir_all(&dir)?;

    let gguf_path = repo.get("model.gguf");
    let st_path = repo.get("model.safetensors");
    let has_gguf = gguf_path.is_ok();
    let has_safetensors = st_path.is_ok();

    if !has_gguf && !has_safetensors {
        return Err(anyhow::anyhow!(
            "No compatible weights found in '{model_id}'. Expected model.safetensors or model.gguf."
        ));
    }

    println!("Downloading config.json...");
    let config_path = repo.get("config.json")?;
    std::fs::copy(&config_path, dir.join("config.json"))?;

    println!("Downloading tokenizer.json...");
    let tok_path = repo.get("tokenizer.json")?;
    std::fs::copy(&tok_path, dir.join("tokenizer.json"))?;

    println!("Downloading mel_filters.npz...");
    if let Ok(filters_path) = repo.get("mel_filters.npz") {
        std::fs::copy(&filters_path, dir.join("mel_filters.npz"))?;
    } else {
        println!("Warning: mel_filters.npz not found, transcription may fail.");
    }

    let variant = if has_gguf {
        println!("Downloading model.gguf...");
        let weights = gguf_path.unwrap();
        std::fs::copy(&weights, dir.join("model.gguf"))?;
        cache::ModelVariant::Gguf
    } else {
        println!("Downloading model.safetensors...");
        let weights = st_path.unwrap();
        std::fs::copy(&weights, dir.join("model.safetensors"))?;
        cache::ModelVariant::Safetensors
    };

    cache::register_model(model_id, variant)?;
    println!("Model '{model_id}' downloaded to {dir:?}");
    Ok(())
}

fn cmd_models_remove(model_id: &str) -> Result<()> {
    if cache::remove_model(model_id)? {
        println!("Model '{model_id}' removed.");
    } else {
        println!("Model '{model_id}' not found in cache.");
    }
    Ok(())
}
