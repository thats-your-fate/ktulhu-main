use crate::{inference::intent_router::logits_argmax, manager::ModelManager};
use anyhow::Result;

pub fn classify_scope(models: &ModelManager, text: &str) -> Result<String> {
    let logits = models.intent_router.classify(text)?;
    let (domain_idx, _) = logits_argmax(&logits.domain)?;
    let label = if domain_idx == 2 { "narrow" } else { "broad" };
    Ok(label.to_string())
}
