use anyhow::Result;
use once_cell::sync::Lazy;
use serde::{Deserialize, Serialize};

use crate::{
    manager::ModelManager,
    prompts,
    reasoning::{select_reasoning_profile, ReasoningProfile},
};

const INTENT_CONFIDENCE_THRESHOLD: f32 = 0.65;
const INTENT_CONFIDENCE_THRESHOLD_REASONING: f32 = 0.45;
const CLARIFICATION_THRESHOLD: f32 = 0.5;
const MULTI_INTENT_MIN_LEN: usize = 40;

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

#[derive(Debug)]
struct LanguageKeywordSet {
    culture_keywords: Vec<String>,
    task_prefixes: Vec<String>,
    task_keywords: Vec<String>,
    advice_patterns: Vec<String>,
    emotional_keywords: Vec<String>,
    casual_markers: Vec<String>,
    reasoning_markers: Vec<String>,
    constraint_markers: Vec<String>,
    ambiguity_stopwords: Vec<String>,
}

#[derive(Deserialize)]
struct TaskCommandConfig {
    prefixes: Vec<String>,
    keywords: Vec<String>,
}

macro_rules! lang_file {
    ($lang:literal, $file:literal) => {
        include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/lang/",
            $lang,
            "/",
            $file
        ))
    };
}

static EN_KEYWORDS: Lazy<LanguageKeywordSet> = Lazy::new(|| load_language_keywords("en"));
static ES_KEYWORDS: Lazy<LanguageKeywordSet> = Lazy::new(|| load_language_keywords("es"));
static RU_KEYWORDS: Lazy<LanguageKeywordSet> = Lazy::new(|| load_language_keywords("ru"));
static PT_KEYWORDS: Lazy<LanguageKeywordSet> = Lazy::new(|| load_language_keywords("pt"));

fn load_language_keywords(lang: &str) -> LanguageKeywordSet {
    macro_rules! load_lang {
        ($lang:literal) => {{
            let task = parse_task_command(lang_file!($lang, "starts_with_task_command.json"));
            LanguageKeywordSet {
                culture_keywords: parse_string_list(lang_file!(
                    $lang,
                    "contains_culture_keywords.json"
                )),
                task_prefixes: task.prefixes,
                task_keywords: task.keywords,
                advice_patterns: parse_string_list(lang_file!(
                    $lang,
                    "contains_advice_patterns.json"
                )),
                emotional_keywords: parse_string_list(lang_file!($lang, "keywords.json")),
                casual_markers: parse_string_list(lang_file!($lang, "casual_markers.json")),
                reasoning_markers: parse_string_list(lang_file!(
                    $lang,
                    "reasoning_patterns.json"
                )),
                constraint_markers: parse_string_list(lang_file!(
                    $lang,
                    "constraint_patterns.json"
                )),
                ambiguity_stopwords: parse_string_list(lang_file!(
                    $lang,
                    "ambiguity_stopwords.json"
                )),
            }
        }};
    }

    match lang {
        "es" => load_lang!("es"),
        "ru" => load_lang!("ru"),
        "pt" => load_lang!("pt"),
        _ => load_lang!("en"),
    }
}

fn parse_string_list(raw: &str) -> Vec<String> {
    serde_json::from_str(raw).expect("invalid keyword list")
}

fn parse_task_command(raw: &str) -> TaskCommandConfig {
    serde_json::from_str(raw).expect("invalid task command config")
}

fn language_keywords(language: Option<&str>) -> (&'static LanguageKeywordSet, &'static str) {
    let normalized = language
        .and_then(|lang| lang.split(|c| c == '-' || c == '_').next())
        .unwrap_or("en")
        .to_ascii_lowercase();

    match normalized.as_str() {
        "es" => (&ES_KEYWORDS, "es"),
        "ru" => (&RU_KEYWORDS, "ru"),
        "pt" => (&PT_KEYWORDS, "pt"),
        _ => (&EN_KEYWORDS, "en"),
    }
}

fn intent_language_threshold(language: Option<&str>) -> f32 {
    match language.and_then(|lang| lang.split(|c| c == '-' || c == '_').next()) {
        Some("ru") => 0.40,
        Some("es") => 0.45,
        Some("pt") => 0.45,
        _ => 0.50,
    }
}

pub async fn route_intent(
    models: &ModelManager,
    text: &str,
    language: Option<&str>,
) -> Result<IntentRoutingResult> {
    let utterances = split_into_utterances(text);
    let multi_intent = utterances
        .iter()
        .filter(|segment| segment.trim().chars().count() >= MULTI_INTENT_MIN_LEN)
        .count()
        > 1;
    let (language_pack, resolved_language) = language_keywords(language);

    if multi_intent {
        let mut result = IntentRoutingResult::default();
        result.multi_intent = true;
        result
            .notes
            .push("multiple significant utterances detected".to_string());
        result.language = resolved_language.to_string();
        return Ok(result);
    }

    let classification_target = utterances
        .first()
        .cloned()
        .unwrap_or_else(|| text.to_string());

    let cleaned_target = clean_intent_text(&classification_target, language_pack);
    let classify_input = if cleaned_target.trim().is_empty() {
        classification_target.as_str()
    } else {
        cleaned_target.as_str()
    };

    let (mut intent, mut confidence) =
        crate::classifier::intent::classify_intent(models, classify_input).await?;

    let mut notes = Vec::new();
    let lower_text = text.to_lowercase();

    if is_emotional_but_topicless(text, language_pack) {
        notes.push("emotional but topicless input".to_string());
        intent = "chat_casual".to_string();
        confidence = confidence.max(0.7);
    } else if is_ambiguous_request(text, language_pack) {
        notes.push("ambiguous short request detected".to_string());
        intent = prompts::default_intent().to_string();
    } else if let Some(override_intent) =
        apply_intent_rules(text, intent.as_str(), language_pack, confidence, &mut notes)
    {
        if override_intent != intent {
            notes.push(format!(
                "rule override: {} → {}",
                intent.as_str(),
                override_intent.as_str()
            ));
            intent = override_intent;
        }
    }

    if starts_with_task_command(&lower_text, language_pack) && intent != "task_short" {
        notes.push("hard task override".to_string());
        intent = "task_short".to_string();
    }

    if intent == "chat_casual" && !is_chat_casual_allowed(text, language_pack) {
        let forced = select_fallback_intent(text, intent.as_str(), language_pack);
        if forced != intent {
            notes.push(format!(
                "chat_casual disallowed; forcing {}",
                forced.as_str()
            ));
            intent = forced;
        }
    }

    let mut confidence_threshold = if intent == "reasoning_logical" {
        INTENT_CONFIDENCE_THRESHOLD_REASONING
    } else {
        INTENT_CONFIDENCE_THRESHOLD
    };
    let language_threshold = intent_language_threshold(Some(resolved_language));
    if language_threshold < confidence_threshold {
        confidence_threshold = language_threshold;
    }

    if confidence < confidence_threshold {
        let fallback = select_fallback_intent(text, intent.as_str(), language_pack);
        if fallback != intent && intent != "reasoning_logical" {
            notes.push(format!(
                "confidence {:.2} below threshold; fallback {} → {}",
                confidence,
                intent.as_str(),
                fallback.as_str()
            ));
            intent = fallback;
        } else {
            notes.push(format!(
                "confidence {:.2} below threshold; keeping intent {}",
                confidence,
                intent.as_str()
            ));
        }
    }

    let clarification_needed = false;
    let detected_profile = select_reasoning_profile(
        models,
        text,
        Some(resolved_language),
        intent.as_str(),
    )
    .await;

    Ok(IntentRoutingResult {
        intent,
        confidence,
        multi_intent: false,
        clarification_needed,
        notes,
        language: resolved_language.to_string(),
        reasoning_profile: Some(detected_profile),
    })
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

fn clean_intent_text(text: &str, keywords: &LanguageKeywordSet) -> String {
    let lower = text.trim().to_lowercase();
    let mut tokens: Vec<&str> = lower.split_whitespace().collect();

    while let Some(token) = tokens.first() {
        if is_casual_marker(token, keywords) {
            tokens.remove(0);
        } else {
            break;
        }
    }

    let recombined = tokens.join(" ");
    strip_emojis(&recombined).trim().to_string()
}

fn is_casual_marker(token: &str, keywords: &LanguageKeywordSet) -> bool {
    keywords.casual_markers.iter().any(|marker| marker == token)
}

fn is_ambiguous_request(text: &str, keywords: &LanguageKeywordSet) -> bool {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return true;
    }

    if trimmed.chars().count() > 40 {
        return false;
    }

    let lower = trimmed.to_lowercase();

    if lower.contains('?') {
        return false;
    }

    let informative_tokens = lower
        .split_whitespace()
        .filter(|token| {
            !keywords
                .ambiguity_stopwords
                .iter()
                .any(|stop| stop == token)
        })
        .count();

    informative_tokens <= 2
}

fn is_informal_opinion(text: &str, keywords: &LanguageKeywordSet) -> bool {
    let lower = text.to_lowercase();
    if lower.chars().count() > 160 {
        return false;
    }

    let has_casual_marker = keywords
        .casual_markers
        .iter()
        .any(|marker| lower.contains(marker));

    has_casual_marker || has_emotional_language(&lower, keywords)
}

fn is_short_opinion_question(text: &str) -> bool {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return false;
    }

    trimmed.chars().count() < 120 && trimmed.contains('?')
}

fn is_emotional_but_topicless(text: &str, keywords: &LanguageKeywordSet) -> bool {
    let lower = text.to_lowercase();
    if lower.chars().count() >= 80 {
        return false;
    }

    has_emotional_language(&lower, keywords)
        && !contains_culture_keywords(&lower, keywords)
        && !contains_reasoning_patterns(&lower, keywords)
        && !contains_advice_patterns(&lower, keywords)
        && !starts_with_task_command(&lower, keywords)
}

fn strip_emojis(text: &str) -> String {
    text.chars().filter(|c| !is_emoji(*c)).collect::<String>()
}

fn is_emoji(c: char) -> bool {
    matches!(
        c,
        '\u{1F300}'..='\u{1FAFF}'
            | '\u{1F1E6}'..='\u{1F1FF}'
            | '\u{1F900}'..='\u{1F9FF}'
            | '\u{2600}'..='\u{27BF}'
            | '\u{1F000}'..='\u{1F02F}'
    )
}

fn apply_intent_rules(
    text: &str,
    current_intent: &str,
    keywords: &LanguageKeywordSet,
    confidence: f32,
    notes: &mut Vec<String>,
) -> Option<String> {
    let lower = text.to_lowercase();

    // culture (soft)
    let culture_detected = contains_culture_keywords(&lower, keywords);
    if culture_detected {
        notes.push("culture keyword detected".to_string());
        if confidence < 0.6 && matches!(current_intent, "opinion_reflective" | "task_short") {
            return Some("culture_context".to_string());
        }
    }

    if contains_advice_patterns(&lower, keywords) {
        return Some("advice_practical".to_string());
    }

    if contains_reasoning_patterns(&lower, keywords) && is_reasoning_question(text) {
        notes.push("reasoning intent detected".to_string());
        return Some("reasoning_logical".to_string());
    }

    if starts_with_task_command(&lower, keywords) {
        return Some("task_short".to_string());
    }

    if current_intent == "opinion_reflective"
        && (is_informal_opinion(text, keywords) || is_short_opinion_question(text))
    {
        notes.push("opinion tone set to casual".to_string());
        return Some("opinion_casual".to_string());
    }

    None
}

fn select_fallback_intent(
    text: &str,
    current_intent: &str,
    keywords: &LanguageKeywordSet,
) -> String {
    let lower = text.to_lowercase();

    if starts_with_task_command(&lower, keywords) {
        return "task_short".to_string();
    }

    if contains_advice_patterns(&lower, keywords) {
        return "advice_practical".to_string();
    }

    if contains_reasoning_patterns(&lower, keywords) {
        return "reasoning_logical".to_string();
    }

    if current_intent == "task_short"
        || current_intent == "advice_practical"
        || current_intent == "opinion_reflective"
        || current_intent == "opinion_casual"
        || current_intent == "reasoning_logical"
    {
        return current_intent.to_string();
    }

    if has_emotional_language(&lower, keywords) && is_chat_casual_allowed(text, keywords) {
        return "chat_casual".to_string();
    }

    "task_short".to_string()
}

fn contains_culture_keywords(text: &str, keywords: &LanguageKeywordSet) -> bool {
    keywords.culture_keywords.iter().any(|kw| text.contains(kw))
}

fn starts_with_task_command(text: &str, keywords: &LanguageKeywordSet) -> bool {
    if contains_reasoning_patterns(text, keywords) {
        return false;
    }

    let trimmed = text.trim_start();
    keywords
        .task_prefixes
        .iter()
        .any(|p| trimmed.starts_with(p))
        || keywords.task_keywords.iter().any(|kw| text.contains(kw))
}

fn contains_advice_patterns(text: &str, keywords: &LanguageKeywordSet) -> bool {
    keywords
        .advice_patterns
        .iter()
        .any(|phrase| text.contains(phrase))
}

fn contains_reasoning_patterns(text: &str, keywords: &LanguageKeywordSet) -> bool {
    if keywords
        .reasoning_markers
        .iter()
        .any(|phrase| text.contains(phrase))
    {
        return true;
    }

    contains_constraint_reasoning(text, keywords)
}

fn is_reasoning_question(text: &str) -> bool {
    let lower = text.trim_start().to_lowercase();
    lower.contains('?')
        || lower.starts_with("why ")
        || lower.starts_with("how ")
        || lower.starts_with("what can you conclude")
        || lower.starts_with("which of")
}

fn contains_constraint_reasoning(text: &str, keywords: &LanguageKeywordSet) -> bool {
    keywords
        .constraint_markers
        .iter()
        .any(|marker| text.contains(marker))
}

fn has_emotional_language(text: &str, keywords: &LanguageKeywordSet) -> bool {
    keywords
        .emotional_keywords
        .iter()
        .any(|kw| text.contains(kw))
}

fn is_chat_casual_allowed(text: &str, keywords: &LanguageKeywordSet) -> bool {
    let lower = text.to_lowercase();
    let has_casual = keywords.casual_markers.iter().any(|marker| lower.contains(marker));
    let is_short = lower.chars().count() <= 60;

    (has_emotional_language(&lower, keywords) || has_casual || is_short)
        && !starts_with_task_command(&lower, keywords)
        && !contains_advice_patterns(&lower, keywords)
        && !contains_reasoning_patterns(&lower, keywords)
}
