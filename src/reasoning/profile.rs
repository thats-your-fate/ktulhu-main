use crate::manager::ModelManager;
use serde::{Deserialize, Serialize};
use tracing::{info, warn};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ReasoningProfile {
    General,
    RegulatedTaxLegal,
    FormalLogic,
    ConstraintPuzzle,
    MathWordProblem,
    AlgorithmicCode,
    Planning,
    ArgumentCritique,
    RiddleMetaphor,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReasoningSubtype {
    FormalLogic,
    ConstraintPuzzle,
    RiddleMetaphor,
    MathWordProblem,
    GeneralReasoning,
}

const SUBTYPE_LABELS: &[&str] = &[
    "formal logic proof question",
    "logic puzzle with constraints (switches, doors, rooms, lamps)",
    "riddle / metaphor question",
    "math word problem",
    "general reasoning question",
];

impl From<ReasoningSubtype> for ReasoningProfile {
    fn from(value: ReasoningSubtype) -> Self {
        match value {
            ReasoningSubtype::FormalLogic => ReasoningProfile::FormalLogic,
            ReasoningSubtype::ConstraintPuzzle => ReasoningProfile::ConstraintPuzzle,
            ReasoningSubtype::RiddleMetaphor => ReasoningProfile::RiddleMetaphor,
            ReasoningSubtype::MathWordProblem => ReasoningProfile::MathWordProblem,
            ReasoningSubtype::GeneralReasoning => ReasoningProfile::General,
        }
    }
}

pub async fn select_reasoning_profile(
    models: &ModelManager,
    text: &str,
    language: Option<&str>,
    intent: &str,
) -> ReasoningProfile {
    if intent == "regulated_tax_legal" {
        return log_and_return(
            "regulated intent override",
            ReasoningProfile::RegulatedTaxLegal,
        );
    }

    let trimmed = text.trim();
    let should_subtype = matches!(
        intent,
        "reasoning_logical"
            | "opinion_casual"
            | "opinion_reflective"
            | "chat_casual"
            | "culture_context"
    ) && trimmed.len() > 25
        && trimmed.contains('?');

    if should_subtype {
        let subtype = classify_reasoning_subtype_roberta(models, text, language).await;
        return log_and_return("roberta reasoning subtype", subtype.into());
    }

    log_and_return("intent mapping", profile_from_intent(intent))
}

pub fn profile_from_intent(intent: &str) -> ReasoningProfile {
    match intent {
        "task_short" => ReasoningProfile::General,
        "advice_practical" => ReasoningProfile::General,
        "opinion_reflective" => ReasoningProfile::General,
        "opinion_casual" => ReasoningProfile::General,
        "culture_context" => ReasoningProfile::General,
        "reasoning_logical" => ReasoningProfile::General,
        "chat_casual" => ReasoningProfile::General,
        _ => ReasoningProfile::General,
    }
}

async fn classify_reasoning_subtype_roberta(
    models: &ModelManager,
    text: &str,
    language: Option<&str>,
) -> ReasoningSubtype {
    // RoBERTa operates on English labels; language routed text still works because the
    // embedding comparison is language agnostic enough for subtype clustering.
    let labels = SUBTYPE_LABELS;
    let classify_text = format!("Task: classify reasoning subtype.\n\n{text}");
    match models.roberta.classify(&classify_text, labels).await {
        Ok((label, score)) => {
            let threshold = subtype_accept_threshold(language);
            if score < threshold {
                return ReasoningSubtype::GeneralReasoning;
            }
            match label.as_str() {
                "formal logic proof question" => ReasoningSubtype::FormalLogic,
                "logic puzzle with constraints (switches, doors, rooms, lamps)" => {
                    ReasoningSubtype::ConstraintPuzzle
                }
                "riddle / metaphor question" => ReasoningSubtype::RiddleMetaphor,
                "math word problem" => ReasoningSubtype::MathWordProblem,
                _ => ReasoningSubtype::GeneralReasoning,
            }
        }
        Err(err) => {
            warn!(
                error = ?err,
                language = language.unwrap_or(""),
                "reasoning subtype classification failed"
            );
            ReasoningSubtype::GeneralReasoning
        }
    }
}

fn subtype_accept_threshold(language: Option<&str>) -> f32 {
    match language.and_then(|lang| lang.split(|c| c == '-' || c == '_').next()) {
        Some("ru") => 0.40,
        Some("es") => 0.45,
        Some("pt") => 0.45,
        _ => 0.50,
    }
}

fn log_and_return(reason: &str, profile: ReasoningProfile) -> ReasoningProfile {
    info!(
        reasoning_profile = ?profile,
        profile_reason = reason,
        "reasoning profile selected"
    );
    profile
}
