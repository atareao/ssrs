use clap::{Parser, Subcommand, ValueEnum};

#[derive(Parser)]
#[command(name = "ssrs", about = "Speech-to-text CLI using Whisper models")]
pub struct Cli {
    #[command(subcommand)]
    pub command: Command,
}

#[derive(Subcommand)]
pub enum Command {
    /// Transcribe an audio file
    Transcribe {
        /// Path to the audio file (MP3, WAV, FLAC, OGG supported)
        #[arg(short, long)]
        file: String,

        /// Model to use (HuggingFace model ID, e.g. "openai/whisper-tiny")
        #[arg(short, long, default_value = "openai/whisper-tiny")]
        model: String,

        /// Device to run inference on
        #[arg(short, long, default_value = "cpu")]
        device: Device,

        /// Language (ISO 639-1 code, e.g. "en", "es"). Use "auto" for detection.
        #[arg(short, long)]
        language: Option<String>,

        /// Output format
        #[arg(short, long, default_value = "txt")]
        output: OutputFormat,

        /// Write output to file instead of stdout
        #[arg(short = 'O', long)]
        output_file: Option<String>,

        /// Initial prompt to guide transcription (capitalization, punctuation, domain terms)
        #[arg(long)]
        initial_prompt: Option<String>,

        /// Sampling temperature (0.0 = greedy, higher = more random)
        #[arg(long, default_value = "0.0")]
        temperature: f64,

        /// Beam size for decoding (1 = greedy, >1 = beam search)
        #[arg(long, default_value_t = 1)]
        beam_size: usize,
    },

    /// Manage models
    Models {
        #[command(subcommand)]
        action: ModelsAction,
    },
}

#[derive(Subcommand)]
pub enum ModelsAction {
    /// Search available models on HuggingFace
    Search {
        /// Search query
        #[arg(short, long)]
        query: Option<String>,

        /// Maximum results to return
        #[arg(short, long, default_value = "10")]
        limit: usize,
    },

    /// List locally cached models
    List,

    /// Download a model from HuggingFace
    Download {
        /// HuggingFace model ID (e.g. "openai/whisper-tiny")
        model_id: String,
    },

    /// Remove a locally cached model
    Remove {
        /// HuggingFace model ID
        model_id: String,
    },
}

#[derive(Clone, ValueEnum)]
pub enum Device {
    Cpu,
    Cuda,
}

#[derive(Clone, ValueEnum)]
pub enum OutputFormat {
    Txt,
    Srt,
    Json,
}
