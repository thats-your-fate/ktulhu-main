use anyhow::Result;
use once_cell::sync::Lazy;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::{manager::ModelManager, prompts};
use tracing::{info, warn};

const MULTI_INTENT_MIN_LEN: usize = 40;
const TASK_GATE_THRESHOLD: f32 = 0.55;

const TASK_OR_CHAT_LABELS: &[&str] = &["task", "chat"];
const TASK_LAYER_LABELS: &[&str] = &[
    "task_short",
    "task_explain",
    "task_technical",
    "regulated_tax_legal",
];
const CHAT_STYLE_LABELS: &[&str] = &["chat_casual", "chat_greeting", "chat_reaction"];

#[derive(Deserialize)]
struct RoutingLabelFile {
    layer1: HashMap<String, String>,
    task_types: HashMap<String, String>,
    chat_styles: HashMap<String, String>,
}

struct RoutingLabelSet {
    layer1: HashMap<String, String>,
    task_types: HashMap<String, String>,
    chat_styles: HashMap<String, String>,
}

impl RoutingLabelSet {
    fn layer1_display(&self, key: &str) -> String {
        self.layer1
            .get(key)
            .cloned()
            .unwrap_or_else(|| key.to_string())
    }

    fn task_display(&self, key: &str) -> String {
        self.task_types
            .get(key)
            .cloned()
            .unwrap_or_else(|| key.to_string())
    }

    fn chat_display(&self, key: &str) -> String {
        self.chat_styles
            .get(key)
            .cloned()
            .unwrap_or_else(|| key.to_string())
    }
}

macro_rules! routing_labels_file {
    ($lang:literal) => {
        include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/lang/",
            $lang,
            "/routing_labels.json"
        ))
    };
}

static EN_ROUTING_LABELS: Lazy<RoutingLabelSet> =
    Lazy::new(|| load_routing_labels(routing_labels_file!("en")));
static ES_ROUTING_LABELS: Lazy<RoutingLabelSet> =
    Lazy::new(|| load_routing_labels(routing_labels_file!("es")));
static RU_ROUTING_LABELS: Lazy<RoutingLabelSet> =
    Lazy::new(|| load_routing_labels(routing_labels_file!("ru")));
static PT_ROUTING_LABELS: Lazy<RoutingLabelSet> =
    Lazy::new(|| load_routing_labels(routing_labels_file!("pt")));

fn load_routing_labels(raw: &str) -> RoutingLabelSet {
    let parsed: RoutingLabelFile =
        serde_json::from_str(raw).expect("invalid routing label config");
    RoutingLabelSet {
        layer1: parsed.layer1,
        task_types: parsed.task_types,
        chat_styles: parsed.chat_styles,
    }
}

fn routing_labels(language: Option<&str>) -> &'static RoutingLabelSet {
    let normalized = language
        .and_then(|lang| lang.split(|c| c == '-' || c == '_').next())
        .unwrap_or("en")
        .to_ascii_lowercase();

    match normalized.as_str() {
        "es" => &ES_ROUTING_LABELS,
        "ru" => &RU_ROUTING_LABELS,
        "pt" => &PT_ROUTING_LABELS,
        _ => &EN_ROUTING_LABELS,
    }
}


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
        "task_explain" => ("advice_practical", Some("task_explain → advice_practical")),
        "task_technical" => ("reasoning_logical", Some("task_technical → reasoning_logical")),
        "regulated_tax_legal" => ("regulated_tax_legal", None),
        _ => ("task_short", Some("fallback → task_short")),
    }
}

fn map_chat_label(label: &str) -> (&'static str, Option<&'static str>) {
    match label {
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
    let localized_labels = routing_labels(language);

    if trimmed.is_empty() {
        result.notes.push("empty input".into());
        return Ok(result);
    }

    // ------------------------------------------------------------
    // Multi-intent detection (unchanged)
    // ------------------------------------------------------------
    let utterances = split_into_utterances(trimmed);
    let multi_intent = utterances
        .iter()
        .filter(|u| u.chars().count() >= MULTI_INTENT_MIN_LEN)
        .count() > 1;

    if multi_intent {
        result.multi_intent = true;
        result.notes.push("multiple significant utterances detected".into());
        return Ok(result);
    }

    let classify_input = utterances
        .first()
        .cloned()
        .unwrap_or_else(|| trimmed.to_string());

    let mut notes = Vec::new();

    // ------------------------------------------------------------
    // LAYER 1 — TASK vs CHAT (ML-critical)
    // ------------------------------------------------------------
    let (layer1_label, layer1_conf) = models
        .roberta
        .classify(&classify_input, TASK_OR_CHAT_LABELS)
        .await?;

    let layer1_display = localized_labels.layer1_display(&layer1_label);
    info!(
        layer = "task_gate",
        label = layer1_label.as_str(),
        label_display = layer1_display.as_str(),
        confidence = layer1_conf,
        language = resolved_language.as_str(),
        "intent routing layer resolved"
    );
    notes.push(format!(
        "layer1 {label} ({conf:.2}) → {display}",
        label = layer1_label,
        conf = layer1_conf,
        display = layer1_display
    ));

    let is_task = layer1_label == "task" && layer1_conf >= TASK_GATE_THRESHOLD;

    // ------------------------------------------------------------
    // LAYER 2 — TASK KIND (only if task)
    // ------------------------------------------------------------
    let (intent, confidence) = if is_task {
        let (task_label, conf) = models
            .roberta
            .classify(&classify_input, TASK_LAYER_LABELS)
            .await?;

        let display_text = localized_labels.task_display(&task_label);
        info!(
            layer = "task_kind",
            label = task_label.as_str(),
            label_display = display_text.as_str(),
            confidence = conf,
            language = resolved_language.as_str(),
            "intent routing layer resolved"
        );
        notes.push(format!(
            "layer2 {label} ({conf:.2}) → {display}",
            label = task_label,
            conf = conf,
            display = display_text
        ));

        let (intent, note) = map_task_label(&task_label);
        if let Some(n) = note {
            notes.push(n.to_string());
        }

        (intent.to_string(), conf)
    } else {
        // --------------------------------------------------------
        // LAYER 3 — CHAT STYLE (lightweight)
        // --------------------------------------------------------
        let (chat_label, conf) = models
            .roberta
            .classify(&classify_input, CHAT_STYLE_LABELS)
            .await?;

        let display_text = localized_labels.chat_display(&chat_label);
        info!(
            layer = "chat_style",
            label = chat_label.as_str(),
            label_display = display_text.as_str(),
            confidence = conf,
            language = resolved_language.as_str(),
            "intent routing layer resolved"
        );
        notes.push(format!(
            "layer3 {label} ({conf:.2}) → {display}",
            label = chat_label,
            conf = conf,
            display = display_text
        ));

        let (intent, note) = map_chat_label(&chat_label);
        if let Some(n) = note {
            notes.push(n.to_string());
        }

        (intent.to_string(), conf)
    };

    // ------------------------------------------------------------
    // Reasoning profile (unchanged, downstream concern)
    // ------------------------------------------------------------
    let reasoning_profile = select_reasoning_profile(
        models,
        text,
        Some(resolved_language.as_str()),
        intent.as_str(),
    )
    .await;

    info!(
        final_intent = intent.as_str(),
        confidence = confidence,
        multi_intent = result.multi_intent,
        language = resolved_language.as_str(),
        "intent routing completed"
    );

    result.intent = intent;
    result.confidence = confidence;
    result.notes = notes;
    result.reasoning_profile = Some(reasoning_profile);

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
