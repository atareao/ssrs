use std::fs::File;
use std::path::Path;

use anyhow::{Context, Result};
use symphonia::core::codecs::{CodecParameters, audio::AudioDecoderOptions};
use symphonia::core::formats::{FormatOptions, TrackType, probe::Hint};
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;

pub struct AudioData {
    pub samples: Vec<f32>,
    pub sample_rate: u32,
    pub channels: u16,
}

pub fn load_audio(path: &Path) -> Result<AudioData> {
    let file = File::open(path)
        .with_context(|| format!("Failed to open audio file: {}", path.display()))?;
    let mss = MediaSourceStream::new(Box::new(file), Default::default());

    let mut hint = Hint::new();
    if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
        hint.with_extension(ext);
    }

    let meta_opts: MetadataOptions = Default::default();
    let fmt_opts: FormatOptions = Default::default();

    let mut format = symphonia::default::get_probe()
        .probe(&hint, mss, fmt_opts, meta_opts)
        .context("Unsupported audio format")?;

    let track = format
        .default_track(TrackType::Audio)
        .context("No supported audio track found")?;

    let codec_params = match &track.codec_params {
        Some(CodecParameters::Audio(p)) => p,
        _ => anyhow::bail!("Track has no audio codec parameters"),
    };

    let sample_rate = codec_params.sample_rate.unwrap_or(16000) as u32;
    let channels = codec_params
        .channels
        .as_ref()
        .map(|c| c.count() as u16)
        .unwrap_or(1);

    let dec_opts: AudioDecoderOptions = Default::default();
    let mut decoder = symphonia::default::get_codecs()
        .make_audio_decoder(codec_params, &dec_opts)
        .context("Failed to create decoder")?;

    let track_id = track.id;
    let mut all_samples: Vec<f32> = Vec::new();

    loop {
        let packet = match format.next_packet() {
            Ok(Some(p)) => p,
            Ok(None) => break,
            Err(symphonia::core::errors::Error::IoError(ref e))
                if e.kind() == std::io::ErrorKind::UnexpectedEof =>
            {
                break;
            }
            Err(e) => return Err(e.into()),
        };

        if packet.track_id != track_id {
            continue;
        }

        match decoder.decode(&packet) {
            Ok(audio_buf) => {
                let mut samples = Vec::new();
                audio_buf.copy_to_vec_interleaved(&mut samples);
                all_samples.extend_from_slice(&samples);
            }
            Err(symphonia::core::errors::Error::DecodeError(_)) => continue,
            Err(e) => return Err(e.into()),
        }
    }

    if all_samples.is_empty() {
        anyhow::bail!("No audio samples decoded from {}", path.display());
    }

    Ok(AudioData {
        samples: all_samples,
        sample_rate,
        channels,
    })
}

pub fn resample_to_16k(samples: &[f32], original_rate: u32) -> Vec<f32> {
    if original_rate == 16000 {
        return samples.to_vec();
    }
    let ratio = original_rate as f64 / 16000.0;
    let new_len = (samples.len() as f64 / ratio) as usize;
    (0..new_len)
        .map(|i| {
            let pos = i as f64 * ratio;
            let idx = pos as usize;
            let frac = pos - idx as f64;
            let a = samples.get(idx).copied().unwrap_or(0.0);
            let b = samples.get(idx + 1).copied().unwrap_or(a);
            a * (1.0 - frac) as f32 + b * frac as f32
        })
        .collect()
}

pub fn to_mono(samples: &[f32], channels: u16) -> Vec<f32> {
    if channels == 1 {
        return samples.to_vec();
    }
    samples
        .chunks(channels as usize)
        .map(|frame| frame.iter().sum::<f32>() / channels as f32)
        .collect()
}

pub fn duration_secs(samples: &[f32], sample_rate: u32) -> f64 {
    samples.len() as f64 / sample_rate as f64
}
