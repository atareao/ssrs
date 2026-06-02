use anyhow::Result;
use candle_core::{DType, Tensor};
use candle_nn::ops::{log_softmax, softmax};
use candle_transformers::models::whisper::{self, N_FRAMES, audio as whisper_audio};

use crate::model::{LoadedModel, WhisperModel};

const MAX_DECODE_STEPS: usize = 448;
const CHUNK_SAMPLES: usize = 480000; // 30s at 16kHz

pub struct Segment {
    pub start_sec: f64,
    pub end_sec: f64,
    pub text: String,
}

pub struct TranscribeOptions<'a> {
    pub language: Option<&'a str>,
    pub initial_prompt: Option<&'a str>,
    pub temperature: f64,
    pub beam_size: usize,
}

pub fn encode(
    model: &mut WhisperModel,
    audio: &[f32],
    config: &whisper::Config,
    filters: &[f32],
    device: &candle_core::Device,
) -> Result<Tensor> {
    let mel = whisper_audio::pcm_to_mel(config, audio, filters);
    let n_mels = config.num_mel_bins;
    let n_len_full = mel.len() / n_mels;
    let n_len = n_len_full.min(N_FRAMES);
    let mel = Tensor::new(mel.as_slice(), device)?
        .reshape((n_mels, n_len_full))?
        .narrow(1, 0, n_len)?
        .unsqueeze(0)?;

    match model {
        WhisperModel::Normal(m) => Ok(m.encoder.forward(&mel, false)?),
        WhisperModel::Quantized(m) => Ok(m.encoder.forward(&mel, false)?),
    }
}

fn build_prompt(
    tokenizer: &tokenizers::Tokenizer,
    lang: &str,
    initial_prompt: Option<&str>,
) -> Result<Vec<u32>> {
    let sot = tokenizer
        .token_to_id(whisper::SOT_TOKEN)
        .ok_or_else(|| anyhow::anyhow!("Missing SOT token"))?;
    let transcribe = tokenizer
        .token_to_id(whisper::TRANSCRIBE_TOKEN)
        .ok_or_else(|| anyhow::anyhow!("Missing TRANSCRIBE token"))?;
    let no_timestamps = tokenizer
        .token_to_id(whisper::NO_TIMESTAMPS_TOKEN)
        .ok_or_else(|| anyhow::anyhow!("Missing NO_TIMESTAMPS token"))?;

    let lang_token = format!("<|{lang}|>");
    let lang_id = tokenizer
        .token_to_id(&lang_token)
        .ok_or_else(|| anyhow::anyhow!("Language '{lang}' not supported"))?;

    let mut prompt = Vec::new();

    if let Some(text) = initial_prompt {
        let encoded = tokenizer
            .encode(text, true)
            .map_err(|e| anyhow::anyhow!("Failed to encode initial prompt: {e}"))?;
        prompt.extend_from_slice(encoded.get_ids());
    }

    prompt.push(sot);
    prompt.push(lang_id);
    prompt.push(transcribe);
    prompt.push(no_timestamps);

    Ok(prompt)
}

pub fn decode(
    model: &mut WhisperModel,
    encoder_output: &Tensor,
    tokenizer: &tokenizers::Tokenizer,
    device: &candle_core::Device,
    options: &TranscribeOptions<'_>,
) -> Result<String> {
    let lang = options.language.unwrap_or("en");
    let prompt = build_prompt(tokenizer, lang, options.initial_prompt)?;
    let eot = tokenizer
        .token_to_id(whisper::EOT_TOKEN)
        .ok_or_else(|| anyhow::anyhow!("Missing EOT token"))?;

    if options.beam_size > 1 {
        return beam_search(
            model,
            encoder_output,
            tokenizer,
            device,
            &prompt,
            eot,
            options.beam_size,
        );
    }

    let mut prompt = prompt;
    let mut generated = Vec::new();

    for step in 0..MAX_DECODE_STEPS {
        let logits = forward_decoder(model, &prompt, encoder_output, step == 0, device)?;
        let logits = logits.squeeze(0)?;
        let seq_len = logits.dim(0)?;
        let last_logits = logits.narrow(0, seq_len - 1, 1)?.squeeze(0)?;
        let next_token = sample_token(&last_logits, options.temperature)?;
        if next_token == eot {
            break;
        }
        generated.push(next_token);
        prompt.push(next_token);
    }

    tokenizer
        .decode(&generated, true)
        .map_err(|e| anyhow::anyhow!("Tokenizer decode error: {e}"))
}

fn forward_decoder(
    model: &mut WhisperModel,
    prompt: &[u32],
    encoder_output: &Tensor,
    flush: bool,
    device: &candle_core::Device,
) -> Result<Tensor> {
    let tokens_tensor = Tensor::new(prompt, device)?.unsqueeze(0)?;
    match model {
        WhisperModel::Normal(m) => {
            let output = m.decoder.forward(&tokens_tensor, encoder_output, flush)?;
            Ok(m.decoder.final_linear(&output)?)
        }
        WhisperModel::Quantized(m) => {
            let output = m.decoder.forward(&tokens_tensor, encoder_output, flush)?;
            Ok(m.decoder.final_linear(&output)?)
        }
    }
}

fn sample_token(logits: &Tensor, temperature: f64) -> Result<u32> {
    let logits = logits.to_dtype(DType::F32)?;
    if temperature <= 0.0 {
        let ids = logits.argmax(candle_core::D::Minus1)?;
        return Ok(ids.to_scalar::<u32>()?);
    }
    let logits = logits.broadcast_div(&Tensor::new(temperature as f32, logits.device())?)?;
    let probs = softmax(&logits, candle_core::D::Minus1)?;
    let probs_vec: Vec<f32> = probs.to_vec1()?;
    let token = weighted_sample(&probs_vec);
    Ok(token)
}

fn weighted_sample(probs: &[f32]) -> u32 {
    let mut rng_state: u64 = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos() as u64;
    // Simple LCG random float in [0, 1)
    rng_state = rng_state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    let r = (rng_state >> 40) as f32 / (1u64 << 24) as f32;

    let mut cumulative = 0.0;
    for (i, &p) in probs.iter().enumerate() {
        cumulative += p;
        if r <= cumulative {
            return i as u32;
        }
    }
    (probs.len() as u32).saturating_sub(1)
}

fn beam_search(
    model: &mut WhisperModel,
    encoder_output: &Tensor,
    tokenizer: &tokenizers::Tokenizer,
    device: &candle_core::Device,
    prompt: &[u32],
    eot: u32,
    beam_size: usize,
) -> Result<String> {
    #[derive(Clone)]
    struct Beam {
        tokens: Vec<u32>,
        score: f64,
        finished: bool,
    }

    let mut beams = vec![Beam {
        tokens: prompt.to_vec(),
        score: 0.0,
        finished: false,
    }];

    for step in 0..MAX_DECODE_STEPS {
        let mut all_candidates: Vec<Beam> = Vec::new();

        for beam in &beams {
            if beam.finished {
                continue;
            }
            let logits = forward_decoder(model, &beam.tokens, encoder_output, step == 0, device)?;
            let logits = logits.squeeze(0)?;
            let seq_len = logits.dim(0)?;
            let last_logits = logits.narrow(0, seq_len - 1, 1)?.squeeze(0)?;
            let last_logits_f32 = last_logits.to_dtype(DType::F32)?;
            let log_probs = log_softmax(&last_logits_f32, candle_core::D::Minus1)?;
            let log_probs_vec: Vec<f64> = log_probs
                .to_vec1::<f32>()?
                .into_iter()
                .map(|x| x as f64)
                .collect();

            let candidates = top_k_indices_float(&log_probs_vec, beam_size * 2);
            for (token, score) in candidates {
                let mut new_tokens = beam.tokens.clone();
                new_tokens.push(token);
                let new_score = beam.score + score;
                if token == eot {
                    all_candidates.push(Beam {
                        tokens: new_tokens,
                        score: new_score,
                        finished: true,
                    });
                } else {
                    all_candidates.push(Beam {
                        tokens: new_tokens,
                        score: new_score,
                        finished: false,
                    });
                }
            }
        }

        all_candidates.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        all_candidates.truncate(beam_size);
        beams = all_candidates;

        if beams.is_empty() || beams.iter().all(|b| b.finished) {
            break;
        }
    }

    let mut finished_beams = beams;
    finished_beams.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let best = finished_beams
        .iter()
        .find(|b| b.finished)
        .or_else(|| finished_beams.first())
        .ok_or_else(|| anyhow::anyhow!("No valid transcription produced"))?;

    let generated = &best.tokens[prompt.len()..];
    tokenizer
        .decode(generated, true)
        .map_err(|e| anyhow::anyhow!("Tokenizer decode error: {e}"))
}

fn top_k_indices_float(slice: &[f64], k: usize) -> Vec<(u32, f64)> {
    let mut indexed: Vec<(u32, f64)> = slice
        .iter()
        .enumerate()
        .map(|(i, &v)| (i as u32, v))
        .collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    indexed.truncate(k);
    indexed
}

pub fn detect_language(
    model: &mut WhisperModel,
    encoder_output: &Tensor,
    tokenizer: &tokenizers::Tokenizer,
    device: &candle_core::Device,
) -> Result<String> {
    let sot = tokenizer
        .token_to_id(whisper::SOT_TOKEN)
        .ok_or_else(|| anyhow::anyhow!("Missing SOT token"))?;

    let prompt = vec![sot];
    let logits = forward_decoder(model, &prompt, encoder_output, true, device)?;
    let logits = logits.squeeze(0)?.squeeze(0)?.to_dtype(DType::F32)?;
    let log_probs = log_softmax(&logits, candle_core::D::Minus1)?;
    let log_probs_vec: Vec<f32> = log_probs.to_vec1()?;

    let vocab_size = tokenizer.get_vocab_size(true);
    let mut best_lang = "en".to_string();
    let mut best_score = f32::NEG_INFINITY;

    for i in 0..vocab_size as u32 {
        let token_str = tokenizer.id_to_token(i).unwrap_or_default();
        if !token_str.starts_with("<|") || !token_str.ends_with("|>") {
            continue;
        }
        let lang = &token_str[2..token_str.len() - 2];
        if lang.len() != 2 {
            continue;
        }
        let score = log_probs_vec[i as usize];
        if score > best_score {
            best_score = score;
            best_lang = lang.to_string();
        }
    }

    eprintln!("Detected language: {best_lang}");
    Ok(best_lang)
}

pub fn transcribe(
    loaded: &mut LoadedModel,
    audio: &[f32],
    options: &TranscribeOptions<'_>,
) -> Result<Vec<Segment>> {
    let total_samples = audio.len();
    let sample_rate = 16000u32;
    let mut segments = Vec::new();

    let mut resolved_language = options.language.map(|s| s.to_string());

    if total_samples <= CHUNK_SAMPLES {
        eprintln!("Transcribing...");
        let enc = encode(
            &mut loaded.model,
            audio,
            &loaded.config,
            &loaded.filters,
            &loaded.device,
        )?;

        if resolved_language.as_deref() == Some("auto") {
            let lang = detect_language(&mut loaded.model, &enc, &loaded.tokenizer, &loaded.device)?;
            resolved_language = Some(lang);
        }

        let lang_ref = resolved_language.as_deref();
        let chunk_options = TranscribeOptions {
            language: lang_ref,
            initial_prompt: options.initial_prompt,
            temperature: options.temperature,
            beam_size: options.beam_size,
        };
        let text = decode(
            &mut loaded.model,
            &enc,
            &loaded.tokenizer,
            &loaded.device,
            &chunk_options,
        )?;
        let duration = total_samples as f64 / sample_rate as f64;
        segments.push(Segment {
            start_sec: 0.0,
            end_sec: duration,
            text,
        });
    } else {
        let total_chunks = total_samples.div_ceil(CHUNK_SAMPLES);
        for chunk_idx in 0..total_chunks {
            let start = chunk_idx * CHUNK_SAMPLES;
            let end = std::cmp::min(start + CHUNK_SAMPLES, total_samples);
            let chunk = &audio[start..end];

            eprintln!("Transcribing chunk {}/{}...", chunk_idx + 1, total_chunks);

            let enc = encode(
                &mut loaded.model,
                chunk,
                &loaded.config,
                &loaded.filters,
                &loaded.device,
            )?;

            if resolved_language.as_deref() == Some("auto") && chunk_idx == 0 {
                let lang =
                    detect_language(&mut loaded.model, &enc, &loaded.tokenizer, &loaded.device)?;
                resolved_language = Some(lang);
            }

            let lang_ref = resolved_language.as_deref();
            let chunk_options = TranscribeOptions {
                language: lang_ref,
                initial_prompt: options.initial_prompt,
                temperature: options.temperature,
                beam_size: options.beam_size,
            };
            let text = decode(
                &mut loaded.model,
                &enc,
                &loaded.tokenizer,
                &loaded.device,
                &chunk_options,
            )?;

            let start_sec = start as f64 / sample_rate as f64;
            let end_sec = end as f64 / sample_rate as f64;
            segments.push(Segment {
                start_sec,
                end_sec,
                text,
            });
        }
    }

    Ok(segments)
}
