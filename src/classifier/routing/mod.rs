use anyhow::{anyhow, Result};
use serde::Serialize;
use tracing::{debug, info};

use crate::{manager::ModelManager, prompts};

const SUPPORT_INTENT_THRESHOLD: f32 = 0.3;
const PHATIC_LABELS: &[&str] = &["SMALL_TALK", "CONTENTFUL"];
const SPEECH_ACT_LABELS: &[&str] = &["SOCIAL", "ASKING", "DIRECTING", "EXPRESSING", "SHARING"];
const DOMAIN_LABELS: &[&str] = &[
    "technical",
    "general",
    "personal",
    "professional",
    "social",
    "legal",
    "other",
];
const EXPECTATION_LABELS: &[&str] = &["NONE", "INFO", "ADVICE", "ACTION", "OTHER"];
const SUPPORT_LABELS: &[&str] = &["NO_SUPPORT", "SUPPORT"];

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
pub struct HeadPrediction {
    pub label: String,
    pub score: f32,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub distribution: Vec<f32>,
}

impl HeadPrediction {
    fn new(label: impl Into<String>, score: f32) -> Self {
        Self {
            label: label.into(),
            score,
            distribution: Vec::new(),
        }
    }

    fn with_distribution(label: impl Into<String>, score: f32, dist: Vec<f32>) -> Self {
        Self {
            label: label.into(),
            score,
            distribution: dist,
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct IntentRoutingResult {
    pub language: String,
    pub speech_act: HeadPrediction,
    pub domain: HeadPrediction,
    pub expectation: HeadPrediction,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub phatic: Option<HeadPrediction>,
    pub final_intent_kind: IntentKind,
    pub routing_path: RoutingPath,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_profile: Option<ReasoningProfile>,
    pub prompt_key: String,
    pub notes: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub support: Option<HeadPrediction>,
    pub support_intent: bool,
}

impl Default for IntentRoutingResult {
    fn default() -> Self {
        Self {
            language: "en".to_string(),
            speech_act: HeadPrediction::new("SOCIAL", 1.0),
            domain: HeadPrediction::new("chat", 1.0),
            expectation: HeadPrediction::new("NONE", 1.0),
            phatic: None,
            final_intent_kind: IntentKind::ChatCasual,
            routing_path: RoutingPath::ChatLayer,
            reasoning_profile: None,
            prompt_key: prompts::default_intent().to_string(),
            notes: vec!["default routing result".into()],
            support: None,
            support_intent: false,
        }
    }
}

pub fn route_intent(
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
        result
            .notes
            .push("empty input → default chat behavior".into());
        log_prompt_selection(&result);
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

    let logits = models.intent_router.classify(&classify_input)?;

    let mut speech_act = decode_head(&logits.speech_act, SPEECH_ACT_LABELS)?;
    speech_act.label = speech_act.label.to_ascii_uppercase();
    let mut domain = decode_head(&logits.domain, DOMAIN_LABELS)?;
    domain.label = domain.label.to_ascii_lowercase();
    let mut expectation = decode_head(&logits.expectation, EXPECTATION_LABELS)?;
    expectation.label = expectation.label.to_ascii_uppercase();
    let phatic_prediction = logits
        .phatic
        .as_ref()
        .and_then(|values| decode_head(values, PHATIC_LABELS).ok());

    let (support_pred, support_on) = decode_support(
        logits.support.as_deref(),
        SUPPORT_LABELS,
        SUPPORT_INTENT_THRESHOLD,
    )?;
    result.support = support_pred.clone();
    result.support_intent = support_on;
    if support_on {
        if let Some(pred) = &support_pred {
            result
                .notes
                .push(format!("support_intent=ON ({:.2})", pred.score));
        } else {
            result.notes.push("support_intent=ON".into());
        }
    }

    if speech_act.label == "DIRECTING"
        && expectation.label == "ADVICE"
        && matches!(domain.label.as_str(), "personal" | "social")
    {
        speech_act.label = "EXPRESSING".to_string();
        result
            .notes
            .push("DIRECTING+ADVICE in personal/social → reinterpreting as EXPRESSING".into());
    }

    result.notes.push(format!(
        "speech_act={} ({:.2})",
        speech_act.label.as_str(),
        speech_act.score
    ));
    result.notes.push(format!(
        "domain={} ({:.2})",
        domain.label.as_str(),
        domain.score
    ));
    result.notes.push(format!(
        "expectation={} ({:.2})",
        expectation.label.as_str(),
        expectation.score
    ));

    log_head_predictions(
        &speech_act,
        &domain,
        &expectation,
        phatic_prediction.as_ref(),
        support_pred.as_ref(),
    );
    result.phatic = phatic_prediction.clone();

    if result.support_intent {
        result
            .notes
            .push("support intent override → support_reflective prompt".into());
        result.speech_act = speech_act;
        result.domain = domain;
        result.expectation = expectation;
        result.final_intent_kind = IntentKind::ChatCasual;
        result.routing_path = RoutingPath::ChatLayer;
        result.reasoning_profile = None;
        result.prompt_key = "support_reflective".to_string();
        log_prompt_selection(&result);
        return Ok(result);
    }

    let preference_hint = speech_act.label == "EXPRESSING"
        && domain.label == "personal"
        && mentions_preference_topics(trimmed);

    let support_label = result.support.as_ref().map(|p| p.label.as_str());
    let support_is_no_support = matches!(support_label, Some("NO_SUPPORT"));

    let (final_kind, routing_path, prompt_stub, mut routing_notes) = resolve_routing(
        &speech_act.label,
        &expectation.label,
        &domain.label,
        preference_hint,
        support_is_no_support,
    );
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
    let expectation_forces_reasoning =
        matches!(expectation.label.as_str(), "INFO" | "ADVICE" | "ACTION");
    if domain.label == "technical"
        && prompt_key != "reasoning"
        && prompt_key != "advice_practical"
        && expectation_forces_reasoning
    {
        if result.support_intent {
            result
                .notes
                .push("support intent active → empathetic-first response".into());
        } else {
            result
                .notes
                .push("domain=technical → forcing reasoning prompt".into());
            prompt_key = "reasoning".to_string();
        }
    }

    if result.support_intent {
        prompt_key = "support_reflective".to_string();
    }

    result.speech_act = speech_act;
    result.domain = domain;
    result.expectation = expectation;
    result.final_intent_kind = final_kind;
    result.routing_path = routing_path;
    result.reasoning_profile = reasoning_profile;
    result.prompt_key = prompt_key;

    log_prompt_selection(&result);
    Ok(result)
}

fn decode_head(logits: &[f32], labels: &[&str]) -> Result<HeadPrediction> {
    if logits.is_empty() {
        return Err(anyhow!("empty logits tensor"));
    }
    let probs = softmax(logits);
    let (idx, _) = logits
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        .ok_or_else(|| anyhow!("empty logits tensor"))?;
    let label = labels.get(idx).copied().unwrap_or("unknown");
    let score = probs.get(idx).copied().unwrap_or(0.0);
    Ok(HeadPrediction::with_distribution(label, score, probs))
}

fn decode_support(
    logits: Option<&[f32]>,
    labels: &[&str],
    threshold: f32,
) -> Result<(Option<HeadPrediction>, bool)> {
    let Some(logits) = logits else {
        return Ok((None, false));
    };
    if logits.len() != 2 {
        return Ok((None, false));
    }
    let probs = softmax2([logits[0], logits[1]]);

    let (idx, prob) = if probs[0] >= probs[1] {
        (0, probs[0])
    } else {
        (1, probs[1])
    };

    let label = labels.get(idx).copied().unwrap_or("unknown");

    let prediction = HeadPrediction {
        label: label.to_string(),
        score: prob,
        distribution: vec![probs[0], probs[1]],
    };
    debug!(
        "support head: label={} score={:.3} dist={:?}",
        prediction.label, prediction.score, prediction.distribution
    );

    let support_on = label == "SUPPORT" && prob >= threshold;

    Ok((Some(prediction), support_on))
}

fn softmax(values: &[f32]) -> Vec<f32> {
    if values.is_empty() {
        return Vec::new();
    }
    let max = values
        .iter()
        .copied()
        .fold(f32::NEG_INFINITY, |a, b| a.max(b));
    let mut exps = Vec::with_capacity(values.len());
    let mut sum = 0.0f32;
    for &v in values {
        let val = (v - max).exp();
        sum += val;
        exps.push(val);
    }
    if sum == 0.0 {
        return vec![0.0; values.len()];
    }
    exps.into_iter().map(|v| v / sum).collect()
}

fn softmax2(logits: [f32; 2]) -> [f32; 2] {
    let m = logits[0].max(logits[1]);
    let e0 = (logits[0] - m).exp();
    let e1 = (logits[1] - m).exp();
    let s = e0 + e1;
    if s == 0.0 {
        [0.0, 0.0]
    } else {
        [e0 / s, e1 / s]
    }
}

fn log_head_predictions(
    speech: &HeadPrediction,
    domain: &HeadPrediction,
    expectation: &HeadPrediction,
    phatic: Option<&HeadPrediction>,
    support: Option<&HeadPrediction>,
) {
    let mut lines = vec![
        format!(
            "  speech_act: {} ({:.3})",
            speech.label.as_str(),
            speech.score
        ),
        format!("  domain: {} ({:.3})", domain.label.as_str(), domain.score),
        format!(
            "  expectation: {} ({:.3})",
            expectation.label.as_str(),
            expectation.score
        ),
    ];

    if let Some(pred) = phatic {
        lines.push(format!(
            "  phatic: {} ({:.3})",
            pred.label.as_str(),
            pred.score
        ));
    }

    if let Some(pred) = support {
        lines.push(format!(
            "  support: {} ({:.3})",
            pred.label.as_str(),
            pred.score
        ));
    }

    if !lines.is_empty() {
        info!("intent classifier predictions:\n{}", lines.join("\n"));
    }
}

fn log_prompt_selection(result: &IntentRoutingResult) {
    info!(
        "intent router prompt → {} (intent={:?}, path={:?})",
        result.prompt_key, result.final_intent_kind, result.routing_path
    );
}

fn resolve_routing(
    speech_act: &str,
    expectation: &str,
    domain: &str,
    preference_hint: bool,
    support_is_no_support: bool,
) -> (IntentKind, RoutingPath, &'static str, Vec<String>) {
    let mut notes = Vec::new();
    if speech_act == "EXPRESSING"
        && domain == "personal"
        && expectation == "ADVICE"
        && support_is_no_support
    {
        notes.push("personal expressive advice without support need → chat_narrative".into());
        return (
            IntentKind::ChatCasual,
            RoutingPath::ChatLayer,
            "chat_narrative",
            notes,
        );
    }

    if speech_act == "EXPRESSING" && domain == "personal" && preference_hint {
        notes.push("personal preference topic detected → opinion_casual prompt".into());
        return (
            IntentKind::ChatCasual,
            RoutingPath::ChatLayer,
            "opinion_casual",
            notes,
        );
    }

    if expectation == "NONE" && domain == "personal" {
        if speech_act == "DIRECTING" {
            notes.push("personal narrative detected → chat_narrative prompt".into());
            return (
                IntentKind::ChatCasual,
                RoutingPath::ChatLayer,
                "chat_narrative",
                notes,
            );
        }

        if speech_act == "EXPRESSING" {
            notes.push("personal reflection detected → chat_narrative prompt".into());
            return (
                IntentKind::ChatCasual,
                RoutingPath::ChatLayer,
                "chat_narrative",
                notes,
            );
        }
    }

    if speech_act == "EXPRESSING" {
        notes.push("EXPRESSING speech act → chat layer".into());
        let prompt = if expectation == "NONE" && domain == "technical" {
            notes.push("EXPRESSING + expectation NONE → reflective technical chat".into());
            "chat_technical_reflective"
        } else {
            intent_from_domain(domain)
        };
        return (
            IntentKind::ChatCasual,
            RoutingPath::ChatLayer,
            prompt,
            notes,
        );
    }

    if speech_act == "DIRECTING"
        && matches!(expectation, "INFO" | "ADVICE")
        && matches!(domain, "technical" | "legal")
    {
        notes.push("DIRECTING + info/advice (technical/legal) → task escalation".into());
        let intent = intent_from_domain(domain);
        let intent_kind = if intent == "reasoning" {
            IntentKind::Reasoning
        } else {
            IntentKind::Task
        };
        return (intent_kind, RoutingPath::TaskLayer, intent, notes);
    }

    if speech_act == "ASKING" && domain != "social" {
        notes.push("ASKING intent outside social → task escalation".into());
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
        return (
            IntentKind::ChatCasual,
            RoutingPath::ChatLayer,
            prompt,
            notes,
        );
    }

    notes.push("chat-first routing applied".into());
    let prompt = intent_from_domain(domain);
    (
        IntentKind::ChatCasual,
        RoutingPath::ChatLayer,
        prompt,
        notes,
    )
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

fn mentions_preference_topics(text: &str) -> bool {
    const PREFERENCE_TOKENS: &[&str] = &[
        "book",
        "books",
        "novel",
        "novels",
        "movie",
        "movies",
        "film",
        "films",
        "music",
        "song",
        "songs",
        "album",
        "albums",
        "artist",
        "artists",
        "band",
        "bands",
        "tv",
        "show",
        "shows",
        "series",
        "podcast",
        "podcasts",
        "game",
        "games",
        "food",
        "foods",
        "cuisine",
        "taste",
        "tastes",
        "flavor",
        "flavour",
        "favorite",
        "favorites",
        "favourite",
        "favourites",
        "preference",
        "preferences",
        "genre",
        "genres",
    ];
    text.to_ascii_lowercase()
        .split(|c: char| !c.is_ascii_alphabetic())
        .filter(|token| !token.is_empty())
        .any(|token| PREFERENCE_TOKENS.contains(&token))
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn expressing_none_technical_stays_in_chat_layer() {
        let (intent_kind, routing_path, prompt, _) =
            resolve_routing("EXPRESSING", "NONE", "technical", false, false);
        assert_eq!(intent_kind, IntentKind::ChatCasual);
        assert_eq!(routing_path, RoutingPath::ChatLayer);
        assert_eq!(prompt, "chat_technical_reflective");
    }

    #[test]
    fn expressing_personal_none_routes_to_chat_narrative() {
        let (_, _, prompt, _) = resolve_routing("EXPRESSING", "NONE", "personal", false, false);
        assert_eq!(prompt, "chat_narrative");
    }

    #[test]
    fn expressing_personal_preferences_route_to_opinion_casual() {
        let (_, _, prompt, _) = resolve_routing("EXPRESSING", "NONE", "personal", true, false);
        assert_eq!(prompt, "opinion_casual");
    }

    #[test]
    fn expressing_personal_advice_without_support_goes_to_chat_narrative() {
        let (_, _, prompt, _) = resolve_routing("EXPRESSING", "ADVICE", "personal", false, true);
        assert_eq!(prompt, "chat_narrative");
    }
}
