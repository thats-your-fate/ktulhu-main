use crate::classifier::routing::ReasoningProfile;
use once_cell::sync::Lazy;
use serde::Deserialize;
use std::collections::HashMap;

const DEFAULT_INTENT: &str = "chat_casual";
const CHAT_LAYER_ENGAGEMENT_HINT: &str =
    "Always be engaged in conversation, ask follow-up questions, and seek clarifications when needed.";

#[derive(Deserialize)]
struct PromptFile {
    default: String,
    prompts: HashMap<String, String>,
}

struct LanguagePromptSet {
    default_prompt: String,
    prompts: HashMap<String, String>,
}

macro_rules! prompt_file {
    ($lang:literal) => {
        include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/lang/",
            $lang,
            "/prompts.json"
        ))
    };
}

static EN_PROMPTS: Lazy<LanguagePromptSet> = Lazy::new(|| load_prompt_set(prompt_file!("en")));
static ES_PROMPTS: Lazy<LanguagePromptSet> = Lazy::new(|| load_prompt_set(prompt_file!("es")));
static RU_PROMPTS: Lazy<LanguagePromptSet> = Lazy::new(|| load_prompt_set(prompt_file!("ru")));
static PT_PROMPTS: Lazy<LanguagePromptSet> = Lazy::new(|| load_prompt_set(prompt_file!("pt")));

fn load_prompt_set(raw: &str) -> LanguagePromptSet {
    let parsed: PromptFile = serde_json::from_str(raw).expect("invalid prompt config");
    LanguagePromptSet {
        default_prompt: parsed.default,
        prompts: parsed.prompts,
    }
}

fn language_prompts(language: Option<&str>) -> &'static LanguagePromptSet {
    let normalized = language
        .and_then(|lang| lang.split(|c| c == '-' || c == '_').next())
        .unwrap_or("en")
        .to_ascii_lowercase();

    match normalized.as_str() {
        "es" => &ES_PROMPTS,
        "ru" => &RU_PROMPTS,
        "pt" => &PT_PROMPTS,
        _ => &EN_PROMPTS,
    }
}

pub fn default_intent() -> &'static str {
    DEFAULT_INTENT
}

fn reasoning_prompt_override(profile: ReasoningProfile) -> Option<&'static str> {
    match profile {
        ReasoningProfile::RegulatedTaxLegal => Some("reasoning_regulated"),
        ReasoningProfile::RiddleMetaphor => Some("reasoning_riddle"),
        ReasoningProfile::FormalLogic => Some("reasoning_formal_logic"),
        ReasoningProfile::ReflectiveAnalysis => Some("reasoning_reflective_metaphor"),
        _ => None,
    }
}

pub fn prompt_for_intent(intent: &str, language: Option<&str>) -> String {
    let set = language_prompts(language);
    set.prompts
        .get(intent)
        .cloned()
        .or_else(|| set.prompts.get(DEFAULT_INTENT).cloned())
        .unwrap_or_else(|| set.default_prompt.clone())
}

pub fn resolved_prompt_key(intent: &str, profile: Option<ReasoningProfile>) -> String {
    profile
        .and_then(reasoning_prompt_override)
        .unwrap_or(intent)
        .to_string()
}

pub fn chat_layer_engagement_hint() -> &'static str {
    CHAT_LAYER_ENGAGEMENT_HINT
}
