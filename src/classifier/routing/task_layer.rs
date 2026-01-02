use anyhow::Result;

use crate::manager::ModelManager;

use super::{labels::RoutingLabelSet, IntentKind};

const TASK_LAYER_LABELS: &[&str] = &[
    "task_short",
    "task_explain",
    "task_technical",
    "regulated_tax_legal",
];

pub struct TaskLayerDecision {
    pub label: String,
    pub confidence: f32,
    pub display: String,
    pub intent: String,
    pub intent_kind: IntentKind,
    pub mapping_note: Option<&'static str>,
}

pub async fn run_task_layer(
    models: &ModelManager,
    text: &str,
    labels: &RoutingLabelSet,
) -> Result<TaskLayerDecision> {
    let (task_label, conf) = models.roberta.classify(text, TASK_LAYER_LABELS).await?;
    let display = labels.task_display(&task_label);
    let (intent, note, kind) = map_task_label(&task_label);
    Ok(TaskLayerDecision {
        label: task_label,
        confidence: conf,
        display,
        intent: intent.to_string(),
        intent_kind: kind,
        mapping_note: note,
    })
}

fn map_task_label(label: &str) -> (&'static str, Option<&'static str>, IntentKind) {
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
        _ => (
            "task_short",
            Some("fallback → task_short"),
            IntentKind::Task,
        ),
    }
}
