use std::collections::BTreeMap;
use std::fs;
use std::path::PathBuf;

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelEntry {
    pub model_id: String,
    pub variant: ModelVariant,
    pub downloaded_at: String,
    pub path: PathBuf,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ModelVariant {
    Safetensors,
    Gguf,
}

fn manifest_dir() -> PathBuf {
    let base = dirs().unwrap_or_else(|| PathBuf::from("."));
    base.join("models")
}

fn manifest_path() -> PathBuf {
    manifest_dir().join("models.json")
}

fn cache_base() -> PathBuf {
    let base = dirs().unwrap_or_else(|| PathBuf::from("."));
    base.join("models")
}

fn dirs() -> Option<PathBuf> {
    let home = std::env::var("HOME").ok().map(PathBuf::from)?;
    Some(home.join(".cache").join("ssrs"))
}

fn load_manifest() -> anyhow::Result<BTreeMap<String, ModelEntry>> {
    let path = manifest_path();
    if !path.exists() {
        return Ok(BTreeMap::new());
    }
    let content = fs::read_to_string(&path)
        .map_err(|e| anyhow::anyhow!("Failed to read manifest at {}: {e}", path.display()))?;
    serde_json::from_str(&content)
        .map_err(|e| anyhow::anyhow!("Corrupted manifest at {}: {e}", path.display()))
}

fn save_manifest(manifest: &BTreeMap<String, ModelEntry>) -> anyhow::Result<()> {
    let dir = manifest_dir();
    fs::create_dir_all(&dir)?;
    let json = serde_json::to_string_pretty(manifest)?;
    fs::write(manifest_path(), json)?;
    Ok(())
}

pub fn model_cache_dir(model_id: &str) -> PathBuf {
    cache_base().join(model_id.replace('/', "__"))
}

pub fn register_model(model_id: &str, variant: ModelVariant) -> anyhow::Result<()> {
    let mut manifest = load_manifest()?;
    let path = model_cache_dir(model_id);
    manifest.insert(
        model_id.to_string(),
        ModelEntry {
            model_id: model_id.to_string(),
            variant,
            downloaded_at: human_now(),
            path,
        },
    );
    save_manifest(&manifest)
}

pub fn list_models() -> anyhow::Result<Vec<ModelEntry>> {
    Ok(load_manifest()?.into_values().collect())
}

pub fn get_model(model_id: &str) -> anyhow::Result<Option<ModelEntry>> {
    Ok(load_manifest()?.get(model_id).cloned())
}

pub fn remove_model(model_id: &str) -> anyhow::Result<bool> {
    let mut manifest = load_manifest()?;
    if let Some(entry) = manifest.remove(model_id) {
        if entry.path.exists() {
            fs::remove_dir_all(&entry.path)?;
        }
        save_manifest(&manifest)?;
        Ok(true)
    } else {
        Ok(false)
    }
}

fn human_now() -> String {
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    let days = secs / 86400;
    let time = secs % 86400;
    let h = time / 3600;
    let m = (time % 3600) / 60;
    let s = time % 60;
    format!("{days}d {h:02}:{m:02}:{s:02}")
}
