use crate::manager::ModelManager;
use anyhow::Result;

pub async fn classify_scope(models: &ModelManager, text: &str) -> Result<String> {
    let (label, _) = models.roberta.classify(text, &["narrow", "broad"]).await?;
    Ok(label)
}
