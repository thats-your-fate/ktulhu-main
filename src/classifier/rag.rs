use anyhow::Result;

use crate::manager::ModelManager;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RagDecision {
    NoRag,
    RagModelOnly,
    RagInternet,
}

impl RagDecision {
    pub fn as_tag(&self) -> &'static str {
        match self {
            RagDecision::NoRag => "no_rag",
            RagDecision::RagModelOnly => "rag_required:model",
            RagDecision::RagInternet => "rag_required:internet",
        }
    }
}

pub async fn classify_rag(models: &ModelManager, text: &str) -> Result<RagDecision> {
    // 1. Hard override
    if let Some(decision) = requires_forced_rag(text) {
        return Ok(decision);
    }

    // 2. ML intent classification
    let (label, _) = models
        .roberta
        .classify(
            text,
            &[
                "general_knowledge",
                "industrial_process",
                "medical_question",
                "scholarly_question",
                "recent_or_dynamic",
            ],
        )
        .await?;

    // 3. Policy mapping
    let decision = match label.as_str() {
        "general_knowledge" => RagDecision::NoRag,
        "industrial_process" => RagDecision::RagModelOnly,
        "medical_question" => RagDecision::RagModelOnly,
        "scholarly_question" => RagDecision::RagModelOnly,
        "recent_or_dynamic" => RagDecision::RagInternet,
        _ => RagDecision::RagModelOnly, // safe fallback
    };

    Ok(decision)
}

fn requires_forced_rag(text: &str) -> Option<RagDecision> {
    let normalized = text.to_lowercase();

    const INTERNET_KEYWORDS: &[&str] = &[
        "today",
        "yesterday",
        "this week",
        "latest",
        "news",
        "stock",
        "price",
        "market",
        "update",
        "current events",
    ];

    if INTERNET_KEYWORDS.iter().any(|kw| normalized.contains(kw)) {
        return Some(RagDecision::RagInternet);
    }

    const MODEL_KEYWORDS: &[&str] = &[
        "whitepaper",
        "thesis",
        "paper",
        "study",
        "report",
        "documentation",
        "manual",
    ];

    if MODEL_KEYWORDS.iter().any(|kw| normalized.contains(kw)) {
        return Some(RagDecision::RagModelOnly);
    }

    None
}
