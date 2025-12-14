use anyhow::Result;

use crate::{manager::ModelManager, prompts};

pub async fn classify_intent(models: &ModelManager, text: &str) -> Result<(String, f32)> {
    models.roberta.classify(text, prompts::INTENT_LABELS).await
}
