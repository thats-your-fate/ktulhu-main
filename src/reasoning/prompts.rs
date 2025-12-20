use crate::classifier::routing::ReasoningProfile;
use once_cell::sync::Lazy;
use serde::Deserialize;
use std::collections::HashMap;

#[derive(Deserialize)]
struct ReasoningPromptFile {
    analysis: HashMap<String, String>,
    final_rules: HashMap<String, String>,
    decomposition: String,
    validation: String,
    response_constraints: ResponseConstraintSet,
}

#[derive(Deserialize)]
struct ResponseConstraintSet {
    strict: String,
    light: String,
}

struct LanguageReasoningPrompts {
    analysis: HashMap<String, String>,
    final_rules: HashMap<String, String>,
    decomposition: String,
    validation: String,
    response_constraints_strict: String,
    response_constraints_light: String,
}

macro_rules! reasoning_file {
    ($lang:literal) => {
        include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/lang/",
            $lang,
            "/reasoning_prompts.json"
        ))
    };
}

static EN_REASONING_PROMPTS: Lazy<LanguageReasoningPrompts> =
    Lazy::new(|| load_reasoning_prompts(reasoning_file!("en")));
static ES_REASONING_PROMPTS: Lazy<LanguageReasoningPrompts> =
    Lazy::new(|| load_reasoning_prompts(reasoning_file!("es")));
static RU_REASONING_PROMPTS: Lazy<LanguageReasoningPrompts> =
    Lazy::new(|| load_reasoning_prompts(reasoning_file!("ru")));
static PT_REASONING_PROMPTS: Lazy<LanguageReasoningPrompts> =
    Lazy::new(|| load_reasoning_prompts(reasoning_file!("pt")));

fn load_reasoning_prompts(raw: &str) -> LanguageReasoningPrompts {
    let parsed: ReasoningPromptFile =
        serde_json::from_str(raw).expect("invalid reasoning prompt configuration");
    LanguageReasoningPrompts {
        analysis: parsed.analysis,
        final_rules: parsed.final_rules,
        decomposition: parsed.decomposition,
        validation: parsed.validation,
        response_constraints_strict: parsed.response_constraints.strict,
        response_constraints_light: parsed.response_constraints.light,
    }
}

fn prompts_for_language(language: Option<&str>) -> &'static LanguageReasoningPrompts {
    let normalized = language
        .and_then(|lang| lang.split(|c| c == '-' || c == '_').next())
        .unwrap_or("en")
        .to_ascii_lowercase();

    match normalized.as_str() {
        "es" => &ES_REASONING_PROMPTS,
        "ru" => &RU_REASONING_PROMPTS,
        "pt" => &PT_REASONING_PROMPTS,
        _ => &EN_REASONING_PROMPTS,
    }
}

fn profile_key(profile: ReasoningProfile) -> &'static str {
    match profile {
        ReasoningProfile::General => "general",
        ReasoningProfile::RegulatedTaxLegal => "regulated_tax_legal",
        ReasoningProfile::FormalLogic => "formal_logic",
        ReasoningProfile::ConstraintPuzzle => "constraint_puzzle",
        ReasoningProfile::MathWordProblem => "math_word_problem",
        ReasoningProfile::AlgorithmicCode => "algorithmic_code",
        ReasoningProfile::Planning => "planning",
        ReasoningProfile::ArgumentCritique => "argument_critique",
        ReasoningProfile::RiddleMetaphor => "riddle_metaphor",
    }
}

pub fn analysis_system_prompt(
    profile: ReasoningProfile,
    language: Option<&str>,
) -> &'static str {
    let prompts = prompts_for_language(language);
    let key = profile_key(profile);
    prompts
        .analysis
        .get(key)
        .or_else(|| prompts.analysis.get("general"))
        .map(|s| s.as_str())
        .unwrap_or("")
}

pub fn final_response_rules(profile: ReasoningProfile, language: Option<&str>) -> &'static str {
    let prompts = prompts_for_language(language);
    let key = profile_key(profile);
    prompts
        .final_rules
        .get(key)
        .or_else(|| prompts.final_rules.get("general"))
        .map(|s| s.as_str())
        .unwrap_or("")
}

pub fn decomposition_system_prompt(language: Option<&str>) -> &'static str {
    prompts_for_language(language).decomposition.as_str()
}

pub fn validation_system_prompt(language: Option<&str>) -> &'static str {
    prompts_for_language(language).validation.as_str()
}

pub fn response_behavior_constraints_strict(language: Option<&str>) -> &'static str {
    prompts_for_language(language)
        .response_constraints_strict
        .as_str()
}

pub fn response_behavior_constraints_light(language: Option<&str>) -> &'static str {
    prompts_for_language(language)
        .response_constraints_light
        .as_str()
}
