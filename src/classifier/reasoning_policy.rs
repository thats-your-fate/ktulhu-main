#[derive(Debug, Clone, Copy)]
pub enum ReasoningMode {
    None,
    AnalyzeThenAnswer,
    DecomposeThenAnswer,
}

use crate::classifier::routing::ReasoningProfile;

pub fn select_reasoning_mode(
    detected_profile: ReasoningProfile,
    intent: &str,
    confidence: f32,
    text: &str,
) -> ReasoningMode {
    let trimmed = text.trim();

    if trimmed.is_empty() {
        return ReasoningMode::None;
    }

    if intent == "regulated_tax_legal" {
        return ReasoningMode::None;
    }

    let lower = trimmed.to_lowercase();
    if matches!(lower.as_str(), "hi" | "hello" | "hey" | "yo" | "sup") {
        return ReasoningMode::None;
    }

    if matches!(
        detected_profile,
        ReasoningProfile::FormalLogic
            | ReasoningProfile::ConstraintPuzzle
            | ReasoningProfile::MathWordProblem
            | ReasoningProfile::RiddleMetaphor
            | ReasoningProfile::ReflectiveAnalysis
    ) {
        return ReasoningMode::AnalyzeThenAnswer;
    }

    let is_reasoning_intent = intent == "reasoning";
    let char_count = trimmed.chars().count();

    if char_count < 20 && !is_reasoning_intent {
        return ReasoningMode::None;
    }

    if !trimmed.contains('?') && !is_reasoning_intent {
        return ReasoningMode::None;
    }

    let mut mode = match intent {
        "reasoning" => ReasoningMode::AnalyzeThenAnswer,
        "advice_practical" if char_count > 120 => ReasoningMode::AnalyzeThenAnswer,
        "task_short" if confidence < 0.7 => ReasoningMode::DecomposeThenAnswer,
        _ => ReasoningMode::None,
    };

    if matches!(mode, ReasoningMode::DecomposeThenAnswer) && !allow_decomposition(trimmed) {
        mode = ReasoningMode::AnalyzeThenAnswer;
    }

    if matches!(mode, ReasoningMode::None) && is_reasoning_candidate(trimmed) {
        mode = ReasoningMode::AnalyzeThenAnswer;
    }

    mode
}

fn allow_decomposition(text: &str) -> bool {
    let char_count = text.chars().count();

    char_count > 120 && text.contains('?') && text.matches('?').count() > 1
}

fn is_reasoning_candidate(text: &str) -> bool {
    let lower = text.to_lowercase();
    if !lower.contains('?') {
        return false;
    }

    const KEYWORDS: &[&str] = &[
        "should",
        "why",
        "how",
        "option",
        "choose",
        "choice",
        "trade-off",
        "tradeoff",
        "pros and cons",
        "probability",
        "chance",
        "risk",
        "uncertainty",
        "factors",
        "influence",
    ];

    KEYWORDS.iter().any(|kw| lower.contains(kw))
}
