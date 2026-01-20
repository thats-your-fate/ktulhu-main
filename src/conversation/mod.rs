use crate::{attachments::message_attachment_summaries, model::message::Message};
use minijinja::Environment;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;
use tracing::{error, warn};

pub const STOP_SEQS: &[&str] = &["</s>", "<|im_end|>"];
const BOS_TOKEN: &str = "<s>";
const EOS_TOKEN: &str = "</s>";
const CHAT_TEMPLATE_NAME: &str = "hf_chat_template";

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum Role {
    System,
    User,
    Assistant,
}

impl Role {
    #[inline]
    pub fn is_user(&self) -> bool {
        matches!(self, Role::User)
    }

    #[inline]
    pub fn is_assistant(&self) -> bool {
        matches!(self, Role::Assistant)
    }
}

static TEMPLATE_STATE: OnceLock<TemplateState> = OnceLock::new();

struct TemplateState {
    env: Environment<'static>,
}

impl TemplateState {
    fn render(&self, ctx: &impl Serialize) -> Result<String, minijinja::Error> {
        self.env.get_template(CHAT_TEMPLATE_NAME)?.render(ctx)
    }
}

#[derive(Serialize)]
struct TemplateMessage {
    role: String,
    context: MessageTemplateContext,
}

#[derive(Serialize)]
struct TemplateContext<'a> {
    bos_token: &'a str,
    eos_token: &'a str,
    messages: &'a [TemplateMessage],
    tools: &'a [serde_json::Value],
}

#[derive(Serialize, Default)]
struct MessageTemplateContext {
    body: Option<String>,
    attachments: Vec<String>,
}

impl MessageTemplateContext {
    fn is_empty(&self) -> bool {
        self.body.is_none() && self.attachments.is_empty()
    }
}

pub fn build_mistral_prompt(history: &[Message], system_prompt: Option<&str>) -> String {
    let state = TEMPLATE_STATE.get_or_init(load_template_state);
    render_with_template(state, history, system_prompt)
        .unwrap_or_else(|err| panic!("chat template rendering failed: {err}"))
}

pub fn trim_history(mut history: Vec<Message>, max_messages: usize) -> Vec<Message> {
    if history.len() <= max_messages {
        return history;
    }
    history.drain(0..history.len() - max_messages);
    history
}

fn sanitize_template_text(text: &str) -> String {
    text.replace("<s>", "<\\s>")
        .replace("</s>", "<\\/s>")
        .replace("[INST]", "[ INST ]")
        .replace("[/INST]", "[ /INST ]")
        .replace("[SYSTEM_PROMPT]", "[ SYSTEM_PROMPT ]")
        .replace("[/SYSTEM_PROMPT]", "[ /SYSTEM_PROMPT ]")
}

pub fn strip_chatml_markers(text: &str) -> String {
    text.replace("<s>", "")
        .replace("</s>", "")
        .replace("[SYSTEM_PROMPT]", "")
        .replace("[/SYSTEM_PROMPT]", "")
        .replace("[INST]", "")
        .replace("[/INST]", "")
        .replace("<|im_start|>", "")
        .replace("<|im_end|>", "")
}

pub fn trim_partial_chatml(text: &str) -> &str {
    let mut end = STOP_SEQS
        .iter()
        .filter_map(|seq| text.find(seq))
        .min()
        .unwrap_or(text.len());

    if end > text.len() {
        end = text.len();
    }

    if end > 0 && !text.is_char_boundary(end) {
        let original_end = end;
        while end > 0 && !text.is_char_boundary(end) {
            end -= 1;
        }
        error!(
            original_end,
            adjusted_end = end,
            "trim_partial_chatml adjusted to utf8 boundary"
        );
    }

    if end > 0 && text[..end].ends_with('<') {
        let mut adjusted = end - 1;
        while adjusted > 0 && !text.is_char_boundary(adjusted) {
            adjusted -= 1;
        }
        if adjusted != end - 1 {
            error!(
                original_end = end,
                adjusted_end = adjusted,
                "trim_partial_chatml removed dangling utf8 byte"
            );
        }
        end = adjusted;
    }

    &text[..end]
}

fn load_template_state() -> TemplateState {
    let path = locate_chat_template().unwrap_or_else(|| {
        panic!(
            "chat_template.jinja not found; set CHAT_TEMPLATE_PATH or place one in the repo root"
        )
    });

    let raw = fs::read_to_string(&path)
        .unwrap_or_else(|err| panic!("failed to read chat template ({}): {err}", path.display()));

    let template_src = Box::leak(raw.into_boxed_str());
    let mut env = Environment::new();
    env.add_template(CHAT_TEMPLATE_NAME, template_src)
        .unwrap_or_else(|err| {
            panic!(
                "failed to compile chat template ({}): {err}",
                path.display()
            )
        });
    TemplateState { env }
}

fn locate_chat_template() -> Option<PathBuf> {
    if let Some(explicit) = env::var_os("CHAT_TEMPLATE_PATH") {
        let candidate = PathBuf::from(explicit);
        if candidate.exists() {
            return Some(candidate);
        }
        warn!(
            target = "conversation",
            path = candidate.display().to_string(),
            "CHAT_TEMPLATE_PATH does not exist"
        );
    }

    let repo_root = Path::new("chat_template.jinja");
    if repo_root.exists() {
        return Some(repo_root.to_path_buf());
    }

    let mut stack: VecDeque<PathBuf> = VecDeque::new();
    stack.push_back(PathBuf::from("models"));
    while let Some(dir) = stack.pop_front() {
        let read_dir = match fs::read_dir(&dir) {
            Ok(rd) => rd,
            Err(_) => continue,
        };
        for entry in read_dir.flatten() {
            let path = entry.path();
            if path.is_dir() {
                stack.push_back(path);
                continue;
            }
            if path
                .file_name()
                .and_then(|name| name.to_str())
                .map(|name| name.eq_ignore_ascii_case("chat_template.jinja"))
                .unwrap_or(false)
            {
                return Some(path);
            }
        }
    }

    None
}

fn render_with_template(
    state: &TemplateState,
    history: &[Message],
    system_prompt: Option<&str>,
) -> Result<String, minijinja::Error> {
    let messages = template_messages(history, system_prompt);
    let tools: [serde_json::Value; 0] = [];
    let ctx = TemplateContext {
        bos_token: BOS_TOKEN,
        eos_token: EOS_TOKEN,
        messages: &messages,
        tools: &tools,
    };
    state.render(&ctx)
}

fn template_messages(history: &[Message], system_prompt: Option<&str>) -> Vec<TemplateMessage> {
    let mut messages = Vec::new();

    if let Some(sys) = system_prompt.filter(|s| !s.trim().is_empty()) {
        let context = MessageTemplateContext {
            body: Some(sanitize_template_text(sys.trim())),
            attachments: Vec::new(),
        };
        messages.push(TemplateMessage {
            role: "system".into(),
            context,
        });
    }

    for msg in history {
        let role = match msg.role.as_str() {
            "user" => "user",
            "assistant" => "assistant",
            "system" => "system",
            _ => continue,
        };

        if role == "system" {
            let context = assemble_message_context(msg);
            if !context.is_empty() {
                messages.push(TemplateMessage {
                    role: role.into(),
                    context,
                });
            }
            continue;
        }

        let context = assemble_message_context(msg);
        if context.is_empty() {
            continue;
        }
        push_with_alternation(&mut messages, role, context);
    }

    messages
}

fn push_with_alternation(
    messages: &mut Vec<TemplateMessage>,
    role: &str,
    context: MessageTemplateContext,
) {
    if let Some(last) = messages.last() {
        if last.role == role {
            let filler_role = if role == "user" { "assistant" } else { "user" };
            messages.push(TemplateMessage {
                role: filler_role.into(),
                context: MessageTemplateContext::default(),
            });
        }
    }

    messages.push(TemplateMessage {
        role: role.into(),
        context,
    });
}

fn assemble_message_context(msg: &Message) -> MessageTemplateContext {
    let body = msg
        .text
        .as_deref()
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(sanitize_template_text);

    let attachments = message_attachment_summaries(&msg.attachments)
        .into_iter()
        .map(|a| sanitize_template_text(&a))
        .collect();

    MessageTemplateContext { body, attachments }
}
