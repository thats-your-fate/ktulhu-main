use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use serde::Serialize;
use tokio::{fs::File, io::AsyncWriteExt};
use uuid::Uuid;

#[derive(Clone)]
pub struct StorageService {
    root: PathBuf,
}

#[derive(Debug, Clone, Serialize)]
pub struct StoredFile {
    pub id: String,
    pub filename: String,
    pub relative_path: String,
    pub size: usize,
}

impl StorageService {
    pub async fn new(root: impl AsRef<Path>) -> Result<Self> {
        let root = root.as_ref().to_path_buf();
        tokio::fs::create_dir_all(&root)
            .await
            .with_context(|| format!("Failed to create storage dir at {}", root.display()))?;

        Ok(Self { root })
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    pub async fn save(&self, bytes: &[u8], original_name: Option<&str>) -> Result<StoredFile> {
        let ext = original_name
            .and_then(|name| Path::new(name).extension().and_then(|e| e.to_str()))
            .and_then(clean_extension);

        let id = Uuid::new_v4().to_string();
        let filename = match ext {
            Some(ext) if !ext.is_empty() => format!("{id}.{ext}"),
            _ => id.clone(),
        };

        let path = self.root.join(&filename);

        let mut file = File::create(&path)
            .await
            .with_context(|| format!("Failed to create file {}", path.display()))?;

        file.write_all(bytes)
            .await
            .with_context(|| format!("Failed to write file {}", path.display()))?;

        Ok(StoredFile {
            id,
            filename: filename.clone(),
            relative_path: self
                .root
                .file_name()
                .and_then(|s| s.to_str())
                .map(|dir| format!("{dir}/{filename}"))
                .unwrap_or_else(|| filename.clone()),
            size: bytes.len(),
        })
    }
}

fn clean_extension(ext: &str) -> Option<String> {
    let filtered: String = ext.chars().filter(|c| c.is_ascii_alphanumeric()).collect();

    if filtered.is_empty() {
        None
    } else {
        Some(filtered.to_lowercase())
    }
}
