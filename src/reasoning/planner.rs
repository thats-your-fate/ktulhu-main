use super::prompts::{
    analysis_system_prompt, decomposition_system_prompt, final_response_rules,
    response_behavior_constraints_light, response_behavior_constraints_strict,
    validation_system_prompt,
};
use crate::classifier::routing::ReasoningProfile;
use crate::conversation::sanitize_chatml_text;

fn build_hidden_chatml_prompt(system_text: &str, user_text: &str) -> String {
    let mut prompt = String::new();
    prompt.push_str("<|im_start|>system\n");
    prompt.push_str(&sanitize_chatml_text(system_text));
    prompt.push_str("\n<|im_end|>\n");
    prompt.push_str("<|im_start|>user\n");
    prompt.push_str(&sanitize_chatml_text(user_text));
    prompt.push_str("\n<|im_end|>\n");
    prompt.push_str("<|im_start|>assistant\n");
    prompt
}

pub fn build_analysis_prompt(
    user_text: &str,
    profile: ReasoningProfile,
    language: Option<&str>,
) -> String {
    let template = analysis_system_prompt(profile, language);
    build_hidden_chatml_prompt(template, user_text)
}

pub fn build_validation_prompt(
    user_text: &str,
    analysis: &str,
    language: Option<&str>,
) -> String {
    let user_section = format!(
        "USER:\n{}\n\nANALYSIS:\n{}",
        user_text.trim(),
        analysis.trim()
    );
    build_hidden_chatml_prompt(validation_system_prompt(language), &user_section)
}

pub fn build_decomposition_prompt(user_text: &str, language: Option<&str>) -> String {
    build_hidden_chatml_prompt(decomposition_system_prompt(language), user_text)
}

pub fn analysis_hidden_instruction(
    analysis: &str,
    validation: Option<&str>,
    profile: ReasoningProfile,
    reasoning_steps: usize,
    language: Option<&str>,
) -> String {
    tracing::info!(
        reasoning_profile = ?profile,
        reasoning_steps,
        "analysis hidden instruction profile selected"
    );

    if reasoning_steps == 0 || analysis.trim().is_empty() {
        return String::new();
    }

    let final_rules = final_response_rules(profile, language);

    if matches!(profile, ReasoningProfile::RiddleMetaphor) {
        let mut block = String::new();
        block.push_str("[INTERNAL_REASONING]\n");
        block.push_str(analysis.trim());
        block.push_str("\n\n");
        block.push_str(&final_rules);
        block.push('\n');
        return block;
    }

    let constraints = if reasoning_steps > 0 {
        response_behavior_constraints_strict(language)
    } else {
        response_behavior_constraints_light(language)
    };
    let mut block = String::new();
    block.push_str("[INTERNAL_ANALYSIS]\n");
    block.push_str(analysis.trim());
    block.push_str("\n\n");

    if let Some(validation_text) = validation {
        let verdict = validation_text.trim().to_uppercase();
        if verdict == "INSUFFICIENT" || verdict == "INCONSISTENT" {
            block.push_str("[VALIDATION_WARNING]\n");
            block.push_str("Analysis may be insufficient or inconsistent.\n\n");
        }
    }

    block.push_str(constraints);
    block.push('\n');
    block.push_str(final_rules);
    block.push('\n');
    block
}

pub fn decomposition_hidden_instruction(
    plan: &str,
    profile: ReasoningProfile,
    language: Option<&str>,
) -> String {
    let final_rules = final_response_rules(profile, language);
    let constraints = response_behavior_constraints_strict(language);
    format!(
        "[INTERNAL_NOTE]\nBreak the problem into sub-steps internally before answering.\n\n{}\n\n{}\n",
        constraints,
        final_rules
    )
}
