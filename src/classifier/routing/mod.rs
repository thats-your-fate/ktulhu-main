mod chat_layer;
mod labels;
mod layer1;
mod task_layer;

use anyhow::Result;
use serde::{Deserialize, Serialize};

use crate::{manager::ModelManager, prompts};

use chat_layer::run_chat_layer;
use labels::routing_labels;
use layer1::run_layer1;
use task_layer::run_task_layer;

const MULTI_INTENT_MIN_LEN: usize = 40;

#[derive(Debug, Clone, Copy, Serialize, PartialEq)]
pub enum IntentKind {
    ChatCasual,
    Task,
    Reasoning,
}

#[derive(Debug, Clone, Copy, Serialize)]
pub enum RoutingPath {
    EmptyInput,
    MultiIntent,
    TaskLayer,
    ChatLayer,
}

#[derive(Debug, Clone, Serialize)]
pub struct IntentRoutingResult {
    pub intent: String,
    pub confidence: f32,
    pub multi_intent: bool,
    pub clarification_needed: bool,
    pub notes: Vec<String>,
    pub language: String,
    pub intent_kind: IntentKind,
    pub prompt_key: String,
    pub path: RoutingPath,
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
            intent_kind: IntentKind::ChatCasual,
            prompt_key: prompts::default_intent().to_string(),
            path: RoutingPath::ChatLayer,
            reasoning_profile: None,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ReasoningProfile {
    General,
    ReflectiveAnalysis,
    RegulatedTaxLegal,
    FormalLogic,
    ConstraintPuzzle,
    MathWordProblem,
    AlgorithmicCode,
    Planning,
    ArgumentCritique,
    RiddleMetaphor,
}

pub fn select_reasoning_profile(
    _text: &str,
    _language: Option<&str>,
    intent: &str,
    _intent_kind: IntentKind,
) -> ReasoningProfile {
    profile_from_intent(intent)
}

pub fn profile_from_intent(intent: &str) -> ReasoningProfile {
    if intent == "regulated_tax_legal" {
        ReasoningProfile::RegulatedTaxLegal
    } else {
        ReasoningProfile::General
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
    let label_set = routing_labels(language);

    if trimmed.is_empty() {
        result.notes.push("empty input".into());
        result.path = RoutingPath::EmptyInput;
        result.prompt_key =
            prompts::resolved_prompt_key(result.intent.as_str(), result.reasoning_profile);
        return Ok(result);
    }

    let mut notes = Vec::new();

    if trimmed.chars().count() <= 20 {
        result.intent = "chat_casual".into();
        result.intent_kind = IntentKind::ChatCasual;
        result.confidence = 1.0;
        result.path = RoutingPath::ChatLayer;
        notes.push("short message → chat_casual".into());
        result.notes = notes;
        result.prompt_key =
            prompts::resolved_prompt_key(result.intent.as_str(), result.reasoning_profile);
        return Ok(result);
    }

    let utterances = split_into_utterances(trimmed);
    let multi_intent = utterances
        .iter()
        .filter(|u| u.chars().count() >= MULTI_INTENT_MIN_LEN)
        .count()
        > 1;

    if multi_intent {
        result.multi_intent = true;
        notes.push("multiple significant utterances detected".into());
        result.path = RoutingPath::MultiIntent;
        result.notes = notes;
        result.prompt_key =
            prompts::resolved_prompt_key(result.intent.as_str(), result.reasoning_profile);
        return Ok(result);
    }

    let classify_input = utterances
        .first()
        .cloned()
        .unwrap_or_else(|| trimmed.to_string());

    let layer1_decision = run_layer1(models, &classify_input, label_set).await?;
    notes.push(format!(
        "layer1 {label} ({conf:.2}) → {display}",
        label = layer1_decision.label.as_str(),
        conf = layer1_decision.confidence,
        display = layer1_decision.display.as_str()
    ));

    let mut force_chat_layer = false;
    if !layer1_decision.actionable {
        notes.push("low actionability → chat_casual".into());
        result.reasoning_profile = None;
        force_chat_layer = true;
    }

    let (mut intent, confidence, mut intent_kind, path) =
        if !force_chat_layer && layer1_decision.enter_task_layer() {
            let task_decision = run_task_layer(models, &classify_input, label_set).await?;
            notes.push(format!(
                "layer2 {label} ({conf:.2}) → {display}",
                label = task_decision.label.as_str(),
                conf = task_decision.confidence,
                display = task_decision.display.as_str()
            ));
            if let Some(n) = task_decision.mapping_note {
                notes.push(n.to_string());
            }
            (
                task_decision.intent,
                task_decision.confidence,
                task_decision.intent_kind,
                RoutingPath::TaskLayer,
            )
        } else {
            let chat_decision = run_chat_layer(models, &classify_input, label_set).await?;
            notes.push(format!(
                "layer3 {label} ({conf:.2}) → {display}",
                label = chat_decision.display.as_str(),
                conf = chat_decision.confidence,
                display = chat_decision.display.as_str()
            ));
            (
                chat_decision.intent,
                chat_decision.confidence,
                chat_decision.intent_kind,
                RoutingPath::ChatLayer,
            )
        };

    let mut reasoning_profile = select_reasoning_profile(
        text,
        Some(resolved_language.as_str()),
        intent.as_str(),
        intent_kind,
    );
    let prompt_key = prompts::resolved_prompt_key(intent.as_str(), Some(reasoning_profile));
    notes.push(format!("routing path resolved → {:?}", path));
    notes.push(format!("prompt slot resolved → {}", prompt_key));

    result.intent = intent;
    result.confidence = confidence;
    result.notes = notes;
    result.reasoning_profile = Some(reasoning_profile);
    result.prompt_key = prompt_key;
    result.intent_kind = intent_kind;
    result.path = path;

    Ok(result)
}

fn normalize_language(language: Option<&str>) -> String {
    language
        .and_then(|lang| lang.split(|c| c == '-' || c == '_').next())
        .unwrap_or("en")
        .to_ascii_lowercase()
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

pub fn is_chat_followup_intent(intent: &str) -> bool {
    matches!(
        intent,
        "chat_casual"
            | "task_short"
            | "advice_practical"
            | "opinion_reflective"
            | "opinion_casual"
            | "culture_context"
    )
}
