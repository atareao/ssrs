use std::sync::LazyLock;

use serde::Deserialize;

const HF_API_BASE: &str = "https://huggingface.co/api/models";

static CLIENT: LazyLock<reqwest::blocking::Client> = LazyLock::new(|| {
    reqwest::blocking::Client::builder()
        .timeout(std::time::Duration::from_secs(30))
        .build()
        .expect("Failed to create HTTP client")
});

#[derive(Debug, Deserialize)]
pub struct ModelInfo {
    pub id: String,
    #[serde(default)]
    pub downloads: u64,
    #[serde(default)]
    pub likes: u64,
}

pub fn search_models(query: Option<&str>, limit: usize) -> anyhow::Result<Vec<ModelInfo>> {
    let mut url = format!(
        "{HF_API_BASE}?pipeline_tag=automatic-speech-recognition&sort=downloads&direction=-1&limit={limit}"
    );
    if let Some(q) = query {
        url.push_str(&format!("&search={q}"));
    }
    let resp: Vec<ModelInfo> = CLIENT
        .get(&url)
        .send()
        .map_err(|e| anyhow::anyhow!("HuggingFace API request failed: {e}"))?
        .json()
        .map_err(|e| anyhow::anyhow!("Failed to parse API response: {e}"))?;
    Ok(resp)
}
