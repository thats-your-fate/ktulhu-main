use crate::classifier::routing::ReasoningProfile;
use once_cell::sync::Lazy;
use serde::Deserialize;
use std::collections::HashMap;
use tracing::debug;

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

pub fn prompt_for_intent(intent: &str, language: Option<&str>) -> String {
    let set = language_prompts(language);
    set.prompts
        .get(intent)
        .cloned()
        .or_else(|| set.prompts.get(DEFAULT_INTENT).cloned())
        .unwrap_or_else(|| set.default_prompt.clone())
}

pub fn resolved_prompt_key(intent: &str, profile: Option<ReasoningProfile>) -> String {
    if matches!(
        profile,
        Some(
            ReasoningProfile::RegulatedTaxLegal
                | ReasoningProfile::RiddleMetaphor
                | ReasoningProfile::FormalLogic
                | ReasoningProfile::ReflectiveAnalysis
        )
    ) {
        "reasoning".to_string()
    } else {
        intent.to_string()
    }
}

pub fn chat_layer_engagement_hint() -> &'static str {
    CHAT_LAYER_ENGAGEMENT_HINT
}

#[derive(Clone, Copy, Debug)]
pub enum Tone {
    Casual,
    Supportive,
    Neutral,
    Formal,
}

#[derive(Clone, Copy, Debug)]
pub enum Depth {
    Shallow,
    Medium,
    Deep,
}

#[derive(Clone, Copy, Debug)]
pub enum Initiative {
    Reactive,
    Suggestive,
    Proactive,
}

#[derive(Clone, Copy, Debug)]
pub enum Constraint {
    ExplainSteps,
}

#[derive(Clone, Debug)]
pub struct PromptPlan {
    pub base_prompt: String,
    pub tone: Tone,
    pub depth: Depth,
    pub initiative: Initiative,
    pub constraints: Vec<Constraint>,
}

pub fn build_prompt_plan(routing: &crate::classifier::routing::IntentRoutingResult) -> PromptPlan {
    let mut base_prompt = routing.prompt_key.clone();

    if routing.support_intent {
        return PromptPlan {
            base_prompt,
            tone: Tone::Supportive,
            depth: Depth::Shallow,
            initiative: Initiative::Suggestive,
            constraints: Vec::new(),
        };
    }

    // Domain overrides ensure technical content always lands in safe prompts.
    match routing.domain.label.as_str() {
        "technical"
            if matches!(
                routing.expectation.label.as_str(),
                "INFO" | "ADVICE" | "ACTION"
            ) =>
        {
            base_prompt = "reasoning".into()
        }
        "legal" => base_prompt = "advice_practical".into(),
        _ => {}
    }

    // Tone is primarily influenced by the speech act.
    let is_statement_like =
        routing.expectation.label == "NONE" && routing.speech_act.label == "DIRECTING";
    let tone = if is_statement_like {
        Tone::Casual
    } else {
        match routing.speech_act.label.as_str() {
            "DIRECTING" => Tone::Supportive,
            "ASKING" => Tone::Neutral,
            "COLLABORATIVE" | "SOCIAL" | "SHARING" => Tone::Casual,
            _ => Tone::Casual,
        }
    };

    // Depth aligns with expectation and domain.
    let depth = if routing.domain.label == "technical" {
        Depth::Deep
    } else {
        match routing.expectation.label.as_str() {
            "ADVICE" => Depth::Medium,
            "INFO" => Depth::Medium,
            _ => Depth::Shallow,
        }
    };

    let initiative = match routing.speech_act.label.as_str() {
        "DIRECTING" => Initiative::Suggestive,
        "COLLABORATIVE" => Initiative::Proactive,
        "ASKING" => Initiative::Reactive,
        _ => Initiative::Reactive,
    };

    let mut constraints = Vec::new();
    if routing.domain.label == "technical"
        || base_prompt.as_str() == "reasoning"
        || matches!(
            routing.final_intent_kind,
            crate::classifier::routing::IntentKind::Reasoning
        )
    {
        constraints.push(Constraint::ExplainSteps);
    }

    PromptPlan {
        base_prompt,
        tone,
        depth,
        initiative,
        constraints,
    }
}

pub fn render_prompt(plan: &PromptPlan, language: Option<&str>) -> String {
    let mut prompt = prompt_for_intent(plan.base_prompt.as_str(), language);

    match plan.tone {
        Tone::Supportive => prompt.push_str("\nBe supportive and encouraging."),
        Tone::Formal => prompt.push_str("\nUse precise, professional language."),
        Tone::Neutral => prompt.push_str("\nMaintain an even, balanced tone."),
        Tone::Casual => prompt.push_str("\nKeep the tone relaxed and friendly."),
    }

    match plan.depth {
        Depth::Deep => prompt.push_str("\nProvide detailed reasoning and context."),
        Depth::Medium => prompt.push_str("\nOffer a moderate level of detail."),
        Depth::Shallow => prompt.push_str("\nKeep responses concise."),
    }

    match plan.initiative {
        Initiative::Proactive => prompt.push_str("\nOffer proactive suggestions when appropriate."),
        Initiative::Suggestive => prompt.push_str("\nSuggest next steps when it helps."),
        Initiative::Reactive => prompt.push_str("\nRespond directly to the user input."),
    }

    for constraint in &plan.constraints {
        match constraint {
            Constraint::ExplainSteps => prompt.push_str("\nExplain your reasoning step by step."),
        }
    }

    debug!(
        base = plan.base_prompt,
        tone = ?plan.tone,
        depth = ?plan.depth,
        initiative = ?plan.initiative,
        constraints = ?plan.constraints,
        "rendered prompt plan"
    );

    prompt
}
