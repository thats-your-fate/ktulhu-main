use crate::{attachments::message_attachment_summaries, model::message::Message};
use serde::{Deserialize, Serialize};

pub const STOP_SEQS: &[&str] = &["<|", "<|im_end|>"];

/// Role for chat messages
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum Role {
    System,
    User,
    Assistant,
}

impl Role {
    #[inline]
    pub fn as_chatml(&self) -> &'static str {
        match self {
            Role::System => "system",
            Role::User => "user",
            Role::Assistant => "assistant",
        }
    }
}

/// Build a full Mistral ChatML-style prompt from chat history.
///
/// IMPORTANT:
/// - The assistant block is intentionally LEFT OPEN.
/// - The model is EXPECTED to emit `<|im_end|>`.
/// - Inference MUST stop on `<|im_end|>`.
pub fn build_mistral_prompt(history: &[Message], system_prompt: Option<&str>) -> String {
    let mut out = String::new();

    // Exactly one system prompt (preferred)
    if let Some(sys) = system_prompt {
        out.push_str("<|im_start|>system\n");
        out.push_str(&sanitize_chatml_text(sys));
        out.push_str("\n<|im_end|>\n");
    }

    for msg in history {
        let role = match msg.role.as_str() {
            "system" => Role::System,
            "user" => Role::User,
            "assistant" => Role::Assistant,
            _ => continue, // skip non-chat entries
        };

        let Some(text) = msg.text.as_deref().filter(|t| !t.is_empty()) else {
            continue;
        };

        let mut body = sanitize_chatml_text(text);
        let attachment_notes = message_attachment_summaries(&msg.attachments);
        if !attachment_notes.is_empty() {
            body.push_str("\n\n[Attachments]\n");
            for note in attachment_notes {
                body.push_str("- ");
                body.push_str(&sanitize_chatml_text(&note));
                body.push('\n');
            }
        }

        out.push_str("<|im_start|>");
        out.push_str(role.as_chatml());
        out.push('\n');
        out.push_str(&body);
        out.push_str("\n<|im_end|>\n");
    }

    // Leave assistant block OPEN on purpose
    out.push_str("<|im_start|>assistant\n");

    out
}

/// Temporary compatibility shim for legacy call sites expecting the old typo.
pub fn build_ministral_prompt(history: &[Message], system_prompt: Option<&str>) -> String {
    build_mistral_prompt(history, system_prompt)
}

/// Trim history by message count (token-based trimming should be done upstream)
pub fn trim_history(mut history: Vec<Message>, max_messages: usize) -> Vec<Message> {
    if history.len() <= max_messages {
        return history;
    }

    history.drain(0..history.len() - max_messages);
    history
}

/// Escape any ChatML control tokens inside user/system text.
///
/// This prevents users from breaking prompt structure.
pub fn sanitize_chatml_text(text: &str) -> String {
    text.replace("<|", "<\\|").replace("|>", "|\\>")
}

/// Remove ChatML structural markers and role headers from generated text.
pub fn strip_chatml_markers(text: &str) -> String {
    const START: &str = "<|im_start|>";
    const END: &str = "<|im_end|>";
    let mut cleaned = String::with_capacity(text.len());
    let mut idx = 0;

    while idx < text.len() {
        let rest = &text[idx..];
        if rest.starts_with(START) {
            idx += START.len();
            if let Some(pos) = text[idx..].find('\n') {
                idx += pos + 1;
            } else {
                break;
            }
            continue;
        }

        if rest.starts_with(END) {
            idx += END.len();
            if let Some(ch) = text[idx..].chars().next() {
                if ch == '\n' {
                    idx += ch.len_utf8();
                }
            }
            continue;
        }

        if let Some(ch) = rest.chars().next() {
            cleaned.push(ch);
            idx += ch.len_utf8();
        } else {
            break;
        }
    }

    cleaned
}

/// Trim output if a new ChatML block was partially emitted.
pub fn trim_partial_chatml(text: &str) -> &str {
    let mut end = STOP_SEQS
        .iter()
        .filter_map(|seq| text.find(seq))
        .min()
        .unwrap_or(text.len());

    if end > 0 && text[..end].ends_with('<') {
        end -= 1;
    }

    &text[..end]
}
