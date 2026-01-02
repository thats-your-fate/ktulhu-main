use anyhow::Result;

use crate::manager::ModelManager;

use super::{labels::RoutingLabelSet, IntentKind};

struct ChatOption {
    label: &'static str,
    intent: &'static str,
    display: &'static str,
    kind: IntentKind,
}

const CHAT_OPTIONS: &[ChatOption] = &[
    ChatOption {
        label: "You are an empathetic, upbeat companion. Respond with warmth and short friendly messages.",
        intent: "chat_casual",
        display: "chat_casual",
        kind: IntentKind::ChatCasual,
    },
    ChatOption {
        label: "Offer grounded, actionable advice with concise steps and trade-offs.",
        intent: "advice_practical",
        display: "advice_practical",
        kind: IntentKind::Task,
    },
    ChatOption {
        label: "Present balanced opinions, acknowledging uncertainty and both sides.",
        intent: "opinion_reflective",
        display: "opinion_reflective",
        kind: IntentKind::ChatCasual,
    },
    ChatOption {
        label: "Share thoughtful opinions in a conversational tone with empathy.",
        intent: "opinion_casual",
        display: "opinion_casual",
        kind: IntentKind::ChatCasual,
    },
    ChatOption {
        label: "Use culturally sensitive language, note regional differences, avoid absolutes.",
        intent: "culture_context",
        display: "culture_context",
        kind: IntentKind::ChatCasual,
    },
];

pub struct ChatLayerDecision {
    pub intent: String,
    pub confidence: f32,
    pub display: String,
    pub intent_kind: IntentKind,
}

pub async fn run_chat_layer(
    models: &ModelManager,
    text: &str,
    _labels: &RoutingLabelSet,
) -> Result<ChatLayerDecision> {
    let label_texts: Vec<&str> = CHAT_OPTIONS.iter().map(|o| o.label).collect();
    let (label, conf) = models.roberta.classify(text, &label_texts).await?;
    let option = CHAT_OPTIONS
        .iter()
        .find(|opt| opt.label == label)
        .unwrap_or(&CHAT_OPTIONS[0]);

    Ok(ChatLayerDecision {
        intent: option.intent.to_string(),
        confidence: conf,
        display: option.display.to_string(),
        intent_kind: option.kind,
    })
}
