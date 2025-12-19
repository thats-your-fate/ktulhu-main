use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};

use crate::{
    conversation::{sanitize_chatml_text, strip_chatml_markers, trim_partial_chatml},
    manager::ModelManager,
};
use anyhow::{anyhow, Result};
use tracing::info;

pub async fn run_hidden_completion(
    models: &ModelManager,
    prompt: String,
    cancel: Arc<AtomicBool>,
) -> Result<String> {
    if cancel.load(Ordering::SeqCst) {
        return Err(anyhow!("cancelled"));
    }

    info!(
        stop_sequences = "<|im_end|>",
        max_new_tokens = 4096,
        "hidden_completion starting"
    );

    let output = models
        .mistral
        .generate_completion(prompt, cancel.clone())
        .await?;

    let cleaned = strip_chatml_markers(&output);
    let trimmed = trim_partial_chatml(&cleaned);
    let preview: String = trimmed.chars().take(200).collect();
    info!(
        trimmed_len = trimmed.len(),
        stop_sequence_applied = (cleaned.len() != trimmed.len()),
        preview = preview,
        "hidden_completion postprocess"
    );

    Ok(trimmed.trim().to_string())
}

pub fn inject_hidden_block(base_prompt: &str, hidden_text: &str) -> String {
    const ASSISTANT_MARKER: &str = "<|im_start|>assistant\n";
    let sanitized_hidden = sanitize_chatml_text(hidden_text.trim());
    let hidden_len = sanitized_hidden.len();
    if let Some(idx) = base_prompt.rfind(ASSISTANT_MARKER) {
        let mut prompt = String::with_capacity(base_prompt.len() + hidden_len + 32);
        prompt.push_str(&base_prompt[..idx]);
        prompt.push_str("<|im_start|>system\n");
        prompt.push_str(&sanitized_hidden);
        prompt.push('\n');
        prompt.push_str("<|im_end|>\n");
        prompt.push_str(&base_prompt[idx..]);
        prompt
    } else {
        let mut fallback = base_prompt.to_string();
        fallback.push_str("\n<|im_start|>system\n");
        fallback.push_str(&sanitized_hidden);
        fallback.push_str("\n<|im_end|>\n");
        fallback
    }
}
