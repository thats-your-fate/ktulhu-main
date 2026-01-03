use crate::{
    inference::roberta_classifier::logits_argmax,
    manager::ModelManager,
};
use anyhow::Result;

pub async fn classify_scope(models: &ModelManager, text: &str) -> Result<String> {
    let logits = models.roberta.classify(text)?;
    let (domain_idx, _) = logits_argmax(&logits.domain)?;
    let label = if domain_idx == 2 { "narrow" } else { "broad" };
    Ok(label.to_string())
}
