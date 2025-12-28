use anyhow::Result;
use once_cell::sync::Lazy;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::{manager::ModelManager, prompts};
use tracing::info;

const MULTI_INTENT_MIN_LEN: usize = 40;
const TASK_GATE_THRESHOLD: f32 = 0.55;

const TASK_OR_CHAT_LABELS: &[&str] = &["task", "chat"];
const TASK_LAYER_LABELS: &[&str] = &[
    "task_short",
    "task_explain",
    "task_technical",
    "regulated_tax_legal",
];
const CHAT_STYLE_LABELS: &[&str] = &["social_chat", "content_chat"];

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


#[derive(Debug, Clone, Copy, Serialize)]
pub enum IntentKind {
    ChatCasual,
    Task,
    Reasoning,
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

struct ReasoningSignal {
    language: &'static str,
    keywords: &'static [&'static str],
    profile: ReasoningProfile,
}

const REFLECTIVE_EN: &[&str] = &["what factors", "before making", "consider", "decision"];
const REFLECTIVE_ES: &[&str] =
    &["qué factores", "antes de tomar", "por qué", "analizar", "decisión"];
const REFLECTIVE_PT: &[&str] =
    &["quais aspectos", "antes de tomar", "ponderar", "explique", "decisão"];

static REASONING_SIGNALS: &[ReasoningSignal] = &[
    ReasoningSignal {
        language: "en",
        keywords: REFLECTIVE_EN,
        profile: ReasoningProfile::ReflectiveAnalysis,
    },
    ReasoningSignal {
        language: "es",
        keywords: REFLECTIVE_ES,
        profile: ReasoningProfile::ReflectiveAnalysis,
    },
    ReasoningSignal {
        language: "pt",
        keywords: REFLECTIVE_PT,
        profile: ReasoningProfile::ReflectiveAnalysis,
    },
];

pub fn select_reasoning_profile(
    text: &str,
    language: Option<&str>,
    intent: &str,
    intent_kind: IntentKind,
) -> ReasoningProfile {
    if intent == "regulated_tax_legal" {
        return log_and_return(
            "regulated intent override",
            ReasoningProfile::RegulatedTaxLegal,
        );
    }

    if matches!(intent_kind, IntentKind::Reasoning) {
        let profile = infer_reasoning_profile(text, language);
        return log_and_return("reasoning intent mapping", profile);
    }

    log_and_return("intent mapping", profile_from_intent(intent))
}

pub fn profile_from_intent(intent: &str) -> ReasoningProfile {
    if intent == "regulated_tax_legal" {
        ReasoningProfile::RegulatedTaxLegal
    } else {
        ReasoningProfile::General
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

fn detect_reasoning_signal(text: &str, language: &str) -> Option<ReasoningProfile> {
    let lower = text.to_lowercase();
    REASONING_SIGNALS
        .iter()
        .filter(|signal| signal.language == language || signal.language == "any")
        .find(|signal| signal.keywords.iter().any(|kw| lower.contains(kw)))
        .map(|signal| signal.profile)
}

fn normalize_language(language: Option<&str>) -> String {
    language
        .and_then(|lang| lang.split(|c| c == '-' || c == '_').next())
        .unwrap_or("en")
        .to_ascii_lowercase()
}

fn map_task_label(
    label: &str,
) -> (&'static str, Option<&'static str>, IntentKind) {
    match label {
        "task_short" => ("task_short", None, IntentKind::Task),
        "task_explain" => (
            "advice_practical",
            Some("task_explain → advice_practical"),
            IntentKind::Task,
        ),
        "task_technical" => (
            "reasoning",
            Some("task_technical → reasoning intent"),
            IntentKind::Reasoning,
        ),
        "regulated_tax_legal" => ("regulated_tax_legal", None, IntentKind::Task),
        _ => ("task_short", Some("fallback → task_short"), IntentKind::Task),
    }
}

fn map_chat_label(label: &str) -> (&'static str, Option<&'static str>, IntentKind) {
    match label {
        "social_chat" => ("chat_casual", None, IntentKind::ChatCasual),
        "content_chat" => (
            "reasoning",
            Some("content_chat → reasoning intent"),
            IntentKind::Reasoning,
        ),
        _ => ("chat_casual", Some("fallback → chat_casual"), IntentKind::ChatCasual),
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

    if let Some(signal_profile) =
        detect_reasoning_signal(trimmed, resolved_language.as_str())
    {
        result.intent = "reasoning".to_string();
        result.intent_kind = IntentKind::Reasoning;
        result.confidence = 1.0;
        result.reasoning_profile = Some(signal_profile);
        result
            .notes
            .push("reasoning override: keyword signal detected".into());
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
    let intent_kind: IntentKind;

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

        let (intent, note, mapped_kind) = map_task_label(&task_label);
        if let Some(n) = note {
            notes.push(n.to_string());
        }
        intent_kind = mapped_kind;

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

        let (intent, note, mapped_kind) = map_chat_label(&chat_label);
        if let Some(n) = note {
            notes.push(n.to_string());
        }
        intent_kind = mapped_kind;
        (intent.to_string(), conf)
    };

    let reasoning_profile = select_reasoning_profile(
        text,
        Some(resolved_language.as_str()),
        intent.as_str(),
        intent_kind,
    );

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
    result.intent_kind = intent_kind;

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

fn infer_reasoning_profile(text: &str, _language: Option<&str>) -> ReasoningProfile {
    let lower = text.to_lowercase();

    if detect_formal_constraints(&lower) {
        return ReasoningProfile::ConstraintPuzzle;
    }
    if detect_math_tokens(&lower) {
        return ReasoningProfile::MathWordProblem;
    }
    if detect_code_tokens(&lower) {
        return ReasoningProfile::AlgorithmicCode;
    }
    if detect_policy_terms(&lower) {
        return ReasoningProfile::RegulatedTaxLegal;
    }

    ReasoningProfile::ReflectiveAnalysis
}

fn detect_formal_constraints(text: &str) -> bool {
    const FORMAL_KEYWORDS: &[&str] = &[
        "sudoku",
        "logic grid",
        "truth table",
        "if and only if",
        "premise",
        "deduce",
        "switch",
        "door",
        "lamp",
        "constraint",
    ];
    FORMAL_KEYWORDS.iter().any(|kw| text.contains(kw))
}

fn detect_math_tokens(text: &str) -> bool {
    const MATH_KEYWORDS: &[&str] = &[
        "sum",
        "total",
        "ratio",
        "perimeter",
        "probability",
        "algebra",
        "equation",
        "percentage",
        "porcentaje",
        "porcentagem",
        "promedio",
    ];
    let digit_count = text.chars().filter(|c| c.is_ascii_digit()).count();
    let has_equation = text.contains('+') || text.contains('-') || text.contains('=') || digit_count > 6;
    has_equation || MATH_KEYWORDS.iter().any(|kw| text.contains(kw))
}

fn detect_code_tokens(text: &str) -> bool {
    const CODE_KEYWORDS: &[&str] = &[
        "fn ",
        "def ",
        "class ",
        "console.",
        "return ",
        "public ",
        "private ",
        "async ",
        "await ",
        "```",
    ];
    CODE_KEYWORDS.iter().any(|kw| text.contains(kw))
}

fn detect_policy_terms(text: &str) -> bool {
    const POLICY_KEYWORDS: &[&str] = &[
        "tax",
        "irs",
        "regulation",
        "compliance",
        "policy",
        "legal",
        "contract",
        "gdpr",
        "lei",
        "legislación",
    ];
    POLICY_KEYWORDS.iter().any(|kw| text.contains(kw))
}
