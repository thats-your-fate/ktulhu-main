use anyhow::Result;
use serde::{Deserialize, Serialize};

use crate::{manager::ModelManager, prompts};
use tracing::{info, warn};

const MULTI_INTENT_MIN_LEN: usize = 40;
const TASK_GATE_THRESHOLD: f32 = 0.55;
const TASK_LAYER_LABELS: &[&str] = &[
    "task_short",
    "task_explain",
    "task_technical",
    "regulated_tax_legal",
];
const CHAT_LAYER_LABELS: &[&str] = &[
    "chat_casual",
    "chat_greeting",
    "chat_reaction",
    "opinion_casual",
    "opinion_reflective",
    "culture_context",
];
const TASK_OR_CHAT_LABELS: &[&str] = &["task", "chat"];

#[derive(Debug, Clone, Serialize)]
pub struct IntentRoutingResult {
    pub intent: String,
    pub confidence: f32,
    pub multi_intent: bool,
    pub clarification_needed: bool,
    pub notes: Vec<String>,
    pub language: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_profile: Option<ReasoningProfile>,
}

impl Default for IntentRoutingResult {
    fn default() -> Self {
        Self {
            intent: prompts::default_intent().to_string(),
            confidence: 0.0,
            multi_intent: false,
            clarification_needed: false,
            notes: Vec::new(),
            language: "en".to_string(),
            reasoning_profile: None,
        }
    }
}

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
enum ReasoningSubtype {
    FormalLogic,
    ConstraintPuzzle,
    RiddleMetaphor,
    MathWordProblem,
    GeneralReasoning,
}

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

const SUBTYPE_LABELS: &[&str] = &[
    "formal logic proof question",
    "logic puzzle with constraints (switches, doors, rooms, lamps)",
    "riddle / metaphor question",
    "math word problem",
    "general reasoning question",
];

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
        return log_and_return("roberta reasoning subtype", ReasoningProfile::from(subtype));
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
        "regulated_tax_legal" => ReasoningProfile::RegulatedTaxLegal,
        _ => ReasoningProfile::General,
    }
}

async fn classify_reasoning_subtype_roberta(
    models: &ModelManager,
    text: &str,
    language: Option<&str>,
) -> ReasoningSubtype {
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

fn normalize_language(language: Option<&str>) -> String {
    language
        .and_then(|lang| lang.split(|c| c == '-' || c == '_').next())
        .unwrap_or("en")
        .to_ascii_lowercase()
}

fn map_task_label(label: &str) -> (&'static str, Option<&'static str>) {
    match label {
        "task_short" => ("task_short", None),
        "task_explain" => ("advice_practical", Some("mapped explanation → advice intent")),
        "task_technical" => ("reasoning_logical", Some("mapped technical task → reasoning intent")),
        "regulated_tax_legal" => ("regulated_tax_legal", Some("mapped regulated intent via layer2")),
        _ => ("task_short", Some("fallback to task_short")),
    }
}

fn map_chat_label(label: &str) -> (&'static str, Option<&'static str>) {
    match label {
        "opinion_reflective" => ("opinion_reflective", None),
        "opinion_casual" => ("opinion_casual", None),
        "culture_context" => ("culture_context", None),
        "chat_greeting" => ("chat_casual", Some("greeting collapsed into chat_casual")),
        "chat_reaction" => ("chat_casual", Some("reaction collapsed into chat_casual")),
        _ => ("chat_casual", None),
    }
}

pub async fn route_intent(
    models: &ModelManager,
    text: &str,
    language: Option<&str>,
) -> Result<IntentRoutingResult> {
    let trimmed = text.trim();
    let mut result = IntentRoutingResult::default();
    let resolved_language = normalize_language(language);
    result.language = resolved_language.clone();

    if trimmed.is_empty() {
        result.notes.push("empty input".to_string());
        return Ok(result);
    }

    let utterances = split_into_utterances(trimmed);
    let multi_intent = utterances
        .iter()
        .filter(|segment| segment.trim().chars().count() >= MULTI_INTENT_MIN_LEN)
        .count()
        > 1;

    if multi_intent {
        result.multi_intent = true;
        result
            .notes
            .push("multiple significant utterances detected".to_string());
        return Ok(result);
    }

    let classify_input = utterances
        .first()
        .cloned()
        .unwrap_or_else(|| trimmed.to_string());

    let mut notes = Vec::new();
    let (layer1_label, layer1_conf) = models
        .roberta
        .classify(&classify_input, TASK_OR_CHAT_LABELS)
        .await?;
    notes.push(format!(
        "layer1 task-vs-chat → {} ({:.2})",
        layer1_label, layer1_conf
    ));

    let is_task = layer1_label == "task" && layer1_conf >= TASK_GATE_THRESHOLD;

    let (intent, confidence) = if is_task {
        let (task_label, conf) = models
            .roberta
            .classify(&classify_input, TASK_LAYER_LABELS)
            .await?;
        notes.push(format!("layer2 task intent → {} ({:.2})", task_label, conf));
        let (intent, extra_note) = map_task_label(&task_label);
        if let Some(note) = extra_note {
            notes.push(note.to_string());
        }
        (intent.to_string(), conf)
    } else {
        let (chat_label, conf) = models
            .roberta
            .classify(&classify_input, CHAT_LAYER_LABELS)
            .await?;
        notes.push(format!("layer3 chat intent → {} ({:.2})", chat_label, conf));
        let (intent, extra_note) = map_chat_label(&chat_label);
        if let Some(note) = extra_note {
            notes.push(note.to_string());
        }
        (intent.to_string(), conf)
    };

    let clarification_needed = false;
    let detected_profile = select_reasoning_profile(
        models,
        text,
        Some(resolved_language.as_str()),
        intent.as_str(),
    )
    .await;

    result.intent = intent;
    result.confidence = confidence;
    result.multi_intent = false;
    result.clarification_needed = clarification_needed;
    result.notes = notes;
    result.reasoning_profile = Some(detected_profile);

    Ok(result)
}

fn split_into_utterances(text: &str) -> Vec<String> {
    let mut segments = Vec::new();
    let mut buffer = String::new();
    let mut newline_streak = 0usize;

    for ch in text.chars() {
        buffer.push(ch);

        if ch == '\n' {
            newline_streak += 1;
            if newline_streak >= 2 {
                push_segment(&mut buffer, &mut segments);
                newline_streak = 0;
            }
            continue;
        }

        newline_streak = 0;

        if ch == '?' || ch == '!' {
            push_segment(&mut buffer, &mut segments);
        }
    }

    push_segment(&mut buffer, &mut segments);

    if segments.is_empty() && !text.trim().is_empty() {
        segments.push(text.trim().to_string());
    }

    segments
}

fn push_segment(buffer: &mut String, segments: &mut Vec<String>) {
    let trimmed = buffer.trim();
    if !trimmed.is_empty() {
        segments.push(trimmed.to_string());
    }
    buffer.clear();
}
