use anyhow::Result;
use candle_core::Tensor;
use serde::Serialize;
use tracing::info;

use crate::{
    inference::roberta_classifier::logits_argmax,
    manager::ModelManager,
    prompts,
};

const SPEECH_ACT_LABELS: &[&str] = &["SOCIAL", "SHARING", "ASKING", "DIRECTING", "OTHER"];
const DOMAIN_LABELS: &[&str] = &["technical", "personal", "social", "legal", "other"];
const EXPECTATION_LABELS: &[&str] = &["NONE", "INFO", "ADVICE", "ACTION", "OTHER"];

#[derive(Debug, Clone, Copy, Serialize, PartialEq, Eq)]
pub enum IntentKind {
    ChatCasual,
    Task,
    Reasoning,
}

#[derive(Debug, Clone, Copy, Serialize, PartialEq, Eq)]
pub enum RoutingPath {
    EmptyInput,
    ChatLayer,
    TaskLayer,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
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

#[derive(Debug, Clone, Serialize)]
pub struct LabelDecision {
    pub label: String,
    pub confidence: f32,
}

impl LabelDecision {
    fn new(label: impl Into<String>, confidence: f32) -> Self {
        Self {
            label: label.into(),
            confidence,
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct IntentRoutingResult {
    pub language: String,
    pub speech_act: LabelDecision,
    pub domain: LabelDecision,
    pub expectation: LabelDecision,
    pub final_intent_kind: IntentKind,
    pub routing_path: RoutingPath,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_profile: Option<ReasoningProfile>,
    pub prompt_key: String,
    pub notes: Vec<String>,
}

impl Default for IntentRoutingResult {
    fn default() -> Self {
        Self {
            language: "en".to_string(),
            speech_act: LabelDecision::new("SOCIAL", 1.0),
            domain: LabelDecision::new("chat", 1.0),
            expectation: LabelDecision::new("NONE", 1.0),
            final_intent_kind: IntentKind::ChatCasual,
            routing_path: RoutingPath::ChatLayer,
            reasoning_profile: None,
            prompt_key: prompts::default_intent().to_string(),
            notes: vec!["default routing result".into()],
        }
    }
}

pub async fn route_intent(
    models: &ModelManager,
    text: &str,
    language_hint: Option<&str>,
) -> Result<IntentRoutingResult> {
    let trimmed = text.trim();
    let mut result = IntentRoutingResult::default();
    result.language = normalize_language(language_hint);
    result.notes.clear();

    if trimmed.is_empty() {
        result.routing_path = RoutingPath::EmptyInput;
        result.notes
            .push("empty input → default chat behavior".into());
        info!(?result, "routing pipeline result");
        return Ok(result);
    }

    let utterances = split_into_utterances(trimmed);
    if has_multi_intent(&utterances) {
        result
            .notes
            .push("multiple significant utterances detected".into());
    }
    let classify_input = utterances
        .first()
        .cloned()
        .unwrap_or_else(|| trimmed.to_string());

    let (gate_idx, gate_conf) = models.phatic_gate.classify(&classify_input)?;
    let gate_label = match gate_idx {
        0 => "CONTENTFUL",
        1 => "PHATIC",
        _ => "UNKNOWN",
    };
    result
        .notes
        .push(format!("phatic_gate={gate_label} ({gate_conf:.2})"));
    if gate_label == "PHATIC" && gate_conf > 0.8 {
        result.speech_act = LabelDecision::new("SOCIAL", gate_conf);
        result.domain = LabelDecision::new("social", gate_conf);
        result.expectation = LabelDecision::new("NONE", gate_conf);
        result.final_intent_kind = IntentKind::ChatCasual;
        result.routing_path = RoutingPath::ChatLayer;
        result.prompt_key = prompts::default_intent().to_string();
        info!(?result, "routing pipeline result");
        return Ok(result);
    }

    let logits = models.roberta.classify(&classify_input)?;
    let mut speech_act = decode_head(&logits.speech_act, SPEECH_ACT_LABELS)?;
    speech_act.label = speech_act.label.to_ascii_uppercase();
    let mut domain = decode_head(&logits.domain, DOMAIN_LABELS)?;
    domain.label = domain.label.to_ascii_lowercase();
    let mut expectation = decode_head(&logits.expectation, EXPECTATION_LABELS)?;
    expectation.label = expectation.label.to_ascii_uppercase();

    result.notes.push(format!(
        "speech_act={} ({:.2})",
        speech_act.label.as_str(),
        speech_act.confidence
    ));
    result.notes.push(format!(
        "domain={} ({:.2})",
        domain.label.as_str(),
        domain.confidence
    ));
    result.notes.push(format!(
        "expectation={} ({:.2})",
        expectation.label.as_str(),
        expectation.confidence
    ));

    let (final_kind, routing_path, prompt_stub, mut routing_notes) =
        resolve_routing(&speech_act.label, &expectation.label, &domain.label);
    result.notes.append(&mut routing_notes);

    let reasoning_profile = if routing_path == RoutingPath::TaskLayer {
        Some(select_reasoning_profile(
            text,
            Some(result.language.as_str()),
            prompt_stub,
            final_kind,
        ))
    } else {
        None
    };

    let mut prompt_key = prompts::resolved_prompt_key(prompt_stub, reasoning_profile);
    if domain.label == "technical"
        && prompt_key != "reasoning"
        && prompt_key != "advice_practical"
    {
        result
            .notes
            .push("domain=technical → forcing reasoning prompt".into());
        prompt_key = "reasoning".to_string();
    }

    result.speech_act = speech_act;
    result.domain = domain;
    result.expectation = expectation;
    result.final_intent_kind = final_kind;
    result.routing_path = routing_path;
    result.reasoning_profile = reasoning_profile;
    result.prompt_key = prompt_key;

    info!(?result, "routing pipeline result");
    Ok(result)
}

fn decode_head(logits: &Tensor, labels: &[&str]) -> Result<LabelDecision> {
    let (idx, conf) = logits_argmax(logits)?;
    let label = labels
        .get(idx)
        .copied()
        .unwrap_or("unknown");
    Ok(LabelDecision::new(label, conf))
}

fn resolve_routing(
    speech_act: &str,
    expectation: &str,
    domain: &str,
) -> (IntentKind, RoutingPath, &'static str, Vec<String>) {
    let mut notes = Vec::new();
    if speech_act == "DIRECTING" && expectation == "ADVICE" {
        notes.push("DIRECTING + ADVICE expectation → task escalation".into());
        let intent = intent_from_domain(domain);
        let intent_kind = if intent == "reasoning" {
            IntentKind::Reasoning
        } else {
            IntentKind::Task
        };
        return (intent_kind, RoutingPath::TaskLayer, intent, notes);
    }

    if speech_act == "DIRECTING" && expectation != "ADVICE" && domain == "technical" {
        notes.push("DIRECTING + technical domain → reasoning depth".into());
        return (
            IntentKind::ChatCasual,
            RoutingPath::ChatLayer,
            "reasoning",
            notes,
        );
    }

    if speech_act == "COLLABORATIVE" {
        notes.push("COLLABORATIVE/INITIATING → rapport handling".into());
        let prompt = intent_from_domain(domain);
        return (IntentKind::ChatCasual, RoutingPath::ChatLayer, prompt, notes);
    }

    notes.push("chat-first routing applied".into());
    let prompt = intent_from_domain(domain);
    (IntentKind::ChatCasual, RoutingPath::ChatLayer, prompt, notes)
}

fn intent_from_domain(domain: &str) -> &'static str {
    match domain {
        "technical" => "reasoning",
        "legal" => "advice_practical",
        "personal" => "opinion_reflective",
        "social" => "chat_casual",
        _ => "chat_casual",
    }
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

fn has_multi_intent(utterances: &[String]) -> bool {
    const MULTI_INTENT_MIN_LEN: usize = 40;
    utterances
        .iter()
        .filter(|u| u.chars().count() >= MULTI_INTENT_MIN_LEN)
        .count()
        > 1
}

fn select_reasoning_profile(
    _text: &str,
    _language: Option<&str>,
    intent: &str,
    _intent_kind: IntentKind,
) -> ReasoningProfile {
    profile_from_intent(intent)
}

fn profile_from_intent(intent: &str) -> ReasoningProfile {
    match intent {
        "reasoning" => ReasoningProfile::General,
        "regulated_tax_legal" => ReasoningProfile::RegulatedTaxLegal,
        "opinion_reflective" => ReasoningProfile::ReflectiveAnalysis,
        _ => ReasoningProfile::General,
    }
}
