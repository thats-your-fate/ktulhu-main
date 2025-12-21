mod executor;
mod planner;
mod prompts;

pub use crate::classifier::routing::{
    profile_from_intent, select_reasoning_profile, ReasoningProfile,
};

use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};

use crate::{classifier::reasoning_policy::ReasoningMode, manager::ModelManager};
use anyhow::{anyhow, Result};

use executor::{inject_hidden_block, run_hidden_completion};
use planner::{
    analysis_hidden_instruction, build_analysis_prompt, build_decomposition_prompt,
    build_validation_prompt, decomposition_hidden_instruction,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReasoningStage {
    None,
    AnalyzeThenAnswer,
    DecomposeThenAnswer,
    ValidationFailed,
}

impl ReasoningStage {
    pub fn as_str(&self) -> &'static str {
        match self {
            ReasoningStage::None => "none",
            ReasoningStage::AnalyzeThenAnswer => "analyze_then_answer",
            ReasoningStage::DecomposeThenAnswer => "decompose_then_answer",
            ReasoningStage::ValidationFailed => "validation_failed",
        }
    }
}

#[derive(Debug)]
pub struct ReasoningResult {
    pub final_prompt: String,
    pub meta: ReasoningMeta,
    pub debug: ReasoningDebugInfo,
}

impl ReasoningResult {
    pub fn fallback(base_prompt: &str) -> Self {
        Self {
            final_prompt: base_prompt.to_string(),
            meta: ReasoningMeta {
                steps: 0,
                stage: ReasoningStage::None,
            },
            debug: ReasoningDebugInfo::default(),
        }
    }
}

#[derive(Debug, Default)]
pub struct ReasoningDebugInfo {
    pub intermediate_prompt: Option<String>,
    pub intermediate_output: Option<String>,
    pub hidden_block: Option<String>,
    pub validation_prompt: Option<String>,
    pub validation_output: Option<String>,
}

#[derive(Debug)]
pub struct ReasoningMeta {
    pub steps: usize,
    pub stage: ReasoningStage,
}

pub async fn run_reasoning(
    models: &ModelManager,
    mode: ReasoningMode,
    user_text: &str,
    language: Option<&str>,
    base_prompt: &str,
    profile: ReasoningProfile,
    cancel: Arc<AtomicBool>,
) -> Result<ReasoningResult> {
    match mode {
        ReasoningMode::None => Ok(ReasoningResult {
            final_prompt: base_prompt.to_string(),
            meta: ReasoningMeta {
                steps: 0,
                stage: ReasoningStage::None,
            },
            debug: ReasoningDebugInfo::default(),
        }),
        ReasoningMode::AnalyzeThenAnswer => {
            analyze_then_answer(models, user_text, base_prompt, profile, language, cancel).await
        }
        ReasoningMode::DecomposeThenAnswer => {
            decompose_then_answer(models, user_text, base_prompt, profile, language, cancel).await
        }
    }
}

async fn analyze_then_answer(
    models: &ModelManager,
    user_text: &str,
    base_prompt: &str,
    profile: ReasoningProfile,
    language: Option<&str>,
    cancel: Arc<AtomicBool>,
) -> Result<ReasoningResult> {
    if cancel.load(Ordering::SeqCst) {
        return Err(anyhow!("cancelled"));
    }

    let analysis_prompt = build_analysis_prompt(user_text, profile, language);
    let analysis = run_hidden_completion(models, analysis_prompt.clone(), cancel.clone()).await?;
    if cancel.load(Ordering::SeqCst) {
        return Err(anyhow!("cancelled"));
    }

    let mut validation_prompt_opt: Option<String> = None;
    let mut validation_output_opt: Option<String> = None;
    let mut validation_ran = false;

    if matches!(
        profile,
        ReasoningProfile::ConstraintPuzzle | ReasoningProfile::FormalLogic
    ) {
        let validation_prompt = build_validation_prompt(user_text, &analysis, language);
        let validation =
            run_hidden_completion(models, validation_prompt.clone(), cancel.clone()).await?;
        let trimmed = validation.trim();
        let normalized_verdict = trimmed.to_uppercase();
        let is_ok = normalized_verdict == "OK";
        let validation_clean = trimmed.to_string();
        validation_prompt_opt = Some(validation_prompt.clone());
        validation_output_opt = Some(validation_clean.clone());
        validation_ran = true;

        if !is_ok {
            return Ok(ReasoningResult {
                final_prompt: base_prompt.to_string(),
                meta: ReasoningMeta {
                    steps: 2,
                    stage: ReasoningStage::ValidationFailed,
                },
                debug: ReasoningDebugInfo {
                    intermediate_prompt: Some(analysis_prompt),
                    intermediate_output: Some(analysis),
                    hidden_block: None,
                    validation_prompt: Some(validation_prompt),
                    validation_output: Some(validation_clean),
                },
            });
        }
    }

    let reasoning_steps = if validation_ran { 2 } else { 1 };

    let hidden = analysis_hidden_instruction(
        &analysis,
        validation_output_opt.as_deref(),
        profile,
        reasoning_steps,
        language,
    );
    let final_prompt = inject_hidden_block(base_prompt, &hidden);

    Ok(ReasoningResult {
        final_prompt,
        meta: ReasoningMeta {
            steps: reasoning_steps,
            stage: ReasoningStage::AnalyzeThenAnswer,
        },
        debug: ReasoningDebugInfo {
            intermediate_prompt: Some(analysis_prompt),
            intermediate_output: Some(analysis),
            hidden_block: Some(hidden),
            validation_prompt: validation_prompt_opt,
            validation_output: validation_output_opt,
        },
    })
}

async fn decompose_then_answer(
    models: &ModelManager,
    user_text: &str,
    base_prompt: &str,
    profile: ReasoningProfile,
    language: Option<&str>,
    cancel: Arc<AtomicBool>,
) -> Result<ReasoningResult> {
    if cancel.load(Ordering::SeqCst) {
        return Err(anyhow!("cancelled"));
    }

    let decomposition_prompt = build_decomposition_prompt(user_text, language);
    let sub_questions =
        run_hidden_completion(models, decomposition_prompt.clone(), cancel.clone()).await?;
    let hidden = decomposition_hidden_instruction(&sub_questions, profile, language);
    let final_prompt = inject_hidden_block(base_prompt, &hidden);

    Ok(ReasoningResult {
        final_prompt,
        meta: ReasoningMeta {
            steps: 1,
            stage: ReasoningStage::DecomposeThenAnswer,
        },
        debug: ReasoningDebugInfo {
            intermediate_prompt: Some(decomposition_prompt),
            intermediate_output: Some(sub_questions),
            hidden_block: Some(hidden),
            validation_prompt: None,
            validation_output: None,
        },
    })
}
