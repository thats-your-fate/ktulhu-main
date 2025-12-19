use once_cell::sync::Lazy;
use serde::Deserialize;
use std::collections::HashMap;


// NOTE:
// Intent prompts are language-localized to improve
// tone, clarity, and reasoning alignment.
// Core reasoning logic remains language-agnostic.

pub const INTENT_LABELS: &[&str] = &[
    "chat_casual",
    "task_short",
    "advice_practical",
    "opinion_reflective",
    "opinion_casual",
    "culture_context",
    "reasoning_logical",
];

const DEFAULT_INTENT: &str = "chat_casual";

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

pub fn prompt_for_intent(intent: &str, language: Option<&str>) -> String {
    let set = language_prompts(language);
    set.prompts
        .get(intent)
        .cloned()
        .or_else(|| set.prompts.get(DEFAULT_INTENT).cloned())
        .unwrap_or_else(|| set.default_prompt.clone())
}
