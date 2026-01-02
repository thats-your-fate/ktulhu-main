use anyhow::Result;

use crate::manager::ModelManager;

use super::labels::RoutingLabelSet;

const TASK_OR_CHAT_LABELS: &[&str] = &["task", "chat"];
const TASK_GATE_THRESHOLD: f32 = 0.65;
const ACTIONABILITY_MARGIN: f32 = 0.18;

pub struct Layer1Decision {
    pub label: String,
    pub confidence: f32,
    pub display: String,
    pub actionable: bool,
}

impl Layer1Decision {
    pub fn enter_task_layer(&self) -> bool {
        self.actionable && self.label == "task" && self.confidence >= TASK_GATE_THRESHOLD
    }
}

pub async fn run_layer1(
    models: &ModelManager,
    text: &str,
    labels: &RoutingLabelSet,
) -> Result<Layer1Decision> {
    let (layer_label, layer_conf) = models.roberta.classify(text, TASK_OR_CHAT_LABELS).await?;
    let (_, task_conf) = models.roberta.classify(text, &["task"]).await?;
    let (_, chat_conf) = models.roberta.classify(text, &["chat"]).await?;
    let actionable = is_actionable(task_conf, chat_conf);
    Ok(Layer1Decision {
        label: layer_label.clone(),
        confidence: layer_conf,
        display: labels.layer1_display(&layer_label),
        actionable,
    })
}

fn is_actionable(task_conf: f32, chat_conf: f32) -> bool {
    (task_conf - chat_conf).abs() >= ACTIONABILITY_MARGIN
}
