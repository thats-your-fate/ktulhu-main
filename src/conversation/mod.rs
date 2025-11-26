use crate::model::message::Message;
use serde::{Deserialize, Serialize};

/// Role for chat messages
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum Role {
    System,
    User,
    Assistant,
}

impl Role {
    pub fn as_chatml(&self) -> &'static str {
        match self {
            Role::System => "system",
            Role::User => "user",
            Role::Assistant => "assistant",
        }
    }
}

/// Trims the conversation to fit a token limit.
/// This is safe because Minstral uses a plain concatenated ChatML format.

/// Build a full Minstral ChatML-style prompt from chat history.
///
/// Produces something like:
///
/// <|im_start|>system
/// You are a helpful assistant.
/// <|im_end|>
/// <|im_start|>user
/// Hello
/// <|im_end|>
/// <|im_start|>assistant
/// Hi there!
/// <|im_start|>user
/// What's 2+2?
/// <|im_end|>
/// <|im_start|>assistant
///
/// This is exactly what Minstral expects.
pub fn build_ministral_prompt(history: &[Message], system_prompt: Option<&str>) -> String {
    let mut out = String::new();

    // Optional system prompt
    if let Some(sys) = system_prompt {
        out.push_str("<|im_start|>system\n");
        out.push_str(sys);
        out.push_str("\n<|im_end|>\n");
    }

    for msg in history {
        let role = match msg.role.as_str() {
            "assistant" => Role::Assistant,
            "user" => Role::User,
            "system" => Role::System,
            _ => Role::User,
        };

        let text = msg.text.clone().unwrap_or_default();

        out.push_str("<|im_start|>");
        out.push_str(role.as_chatml());
        out.push_str("\n");
        out.push_str(&text);
        out.push_str("\n<|im_end|>\n");
    }

    // Model must continue as assistant
    out.push_str("<|im_start|>assistant\n");

    out
}

pub fn trim_history(mut history: Vec<Message>, max_messages: usize) -> Vec<Message> {
    if history.len() <= max_messages {
        return history;
    }

    history.reverse();
    history.truncate(max_messages);
    history.reverse();
    history
}
