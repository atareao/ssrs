use std::path::Path;

use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use candle_transformers::models::whisper;

use crate::cache::{self, ModelVariant};

pub struct LoadedModel {
    pub model: WhisperModel,
    pub tokenizer: tokenizers::Tokenizer,
    pub config: whisper::Config,
    pub filters: Vec<f32>,
    pub device: Device,
}

pub enum WhisperModel {
    Normal(whisper::model::Whisper),
    Quantized(whisper::quantized_model::Whisper),
}

fn detect_variant(model_dir: &Path) -> ModelVariant {
    if model_dir.join("model.gguf").exists() {
        ModelVariant::Gguf
    } else {
        ModelVariant::Safetensors
    }
}

pub fn load_model(model_id: &str, device: &Device) -> Result<LoadedModel> {
    let entry = cache::get_model(model_id)?.ok_or_else(|| {
        anyhow::anyhow!(
            "Model '{model_id}' not found. Run `ssrs models download {model_id}` first."
        )
    })?;

    let model_dir = &entry.path;
    let variant = detect_variant(model_dir);

    let config_path = model_dir.join("config.json");
    let config_str = std::fs::read_to_string(&config_path)?;
    let config: whisper::Config = serde_json::from_str(&config_str)?;

    let tokenizer_path = model_dir.join("tokenizer.json");
    let tokenizer = tokenizers::Tokenizer::from_file(&tokenizer_path)
        .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {e}"))?;

    let filters_path = model_dir.join("mel_filters.npz");
    let filters = load_filters(&filters_path, &config)?;

    let dtype = DType::F32;

    let model = match variant {
        ModelVariant::Safetensors => {
            let weights_path = model_dir.join("model.safetensors");
            // SAFETY: The safetensors file is downloaded from a trusted HuggingFace repo
            // and is memory-mapped read-only. The file must remain on disk and unmodified
            // for the duration of the VarBuilder's lifetime.
            let vb = unsafe {
                candle_nn::VarBuilder::from_mmaped_safetensors(&[weights_path], dtype, device)?
            };
            WhisperModel::Normal(whisper::model::Whisper::load(&vb, config.clone())?)
        }
        ModelVariant::Gguf => {
            let weights_path = model_dir.join("model.gguf");
            let vb = candle_transformers::quantized_var_builder::VarBuilder::from_gguf(
                &weights_path,
                device,
            )?;
            WhisperModel::Quantized(whisper::quantized_model::Whisper::load(
                &vb,
                config.clone(),
            )?)
        }
    };

    Ok(LoadedModel {
        model,
        tokenizer,
        config,
        filters,
        device: device.clone(),
    })
}

fn load_filters(path: &Path, config: &whisper::Config) -> Result<Vec<f32>> {
    // Try mel_filters.npz first (if present in model dir)
    if path.exists() {
        let tensors = Tensor::read_npz(path)?;
        let mel_filter = tensors
            .into_iter()
            .find(|(name, _)| name == "mel_filters")
            .map(|(_, t)| t)
            .ok_or_else(|| anyhow::anyhow!("No mel filters found in {}", path.display()))?;

        let filters: Vec<f32> = mel_filter.to_vec1()?;
        return Ok(filters);
    }

    // Use embedded mel filters (from candle-examples, compatible with unnormalized FFT)
    let mel_bytes: &[u8] = include_bytes!("../melfilters.bytes");
    let filters: Vec<f32> = mel_bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect();
    let expected = config.num_mel_bins * (1 + whisper::N_FFT / 2);
    if filters.len() != expected {
        return Err(anyhow::anyhow!(
            "Embedded mel filter size mismatch: expected {expected}, got {}",
            filters.len()
        ));
    }
    Ok(filters)
}
