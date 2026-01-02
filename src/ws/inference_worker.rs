use axum::extract::ws::Message as WsMessage;
use serde_json::json;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use tokio::sync::mpsc;
use tracing::{debug, info};
use uuid::Uuid;

use crate::conversation::{strip_chatml_markers, trim_partial_chatml};
use crate::db::DBLayer;
use crate::inference::{byte_decoder::tidy_decoded_text, InferenceService};
use crate::model::message::Message;

use super::handler::touch_chat;

pub struct InferenceJob {
    pub prompt: String,
    pub chat_id: String,
    pub session_id: String,
    pub sender: mpsc::Sender<WsMessage>,
    pub infer: Arc<InferenceService>,
    pub db: Arc<DBLayer>,
    pub cancel: Arc<AtomicBool>,
}

#[derive(Clone)]
pub struct InferenceWorker {
    tx: mpsc::Sender<InferenceJob>,
}

#[derive(serde::Deserialize, Debug)]
struct SummaryJson {
    summary: String,
}

impl InferenceWorker {
    pub fn new(queue_size: usize) -> Self {
        let (tx, rx) = mpsc::channel(queue_size);
        tokio::spawn(worker_loop(rx));
        Self { tx }
    }

    pub fn try_enqueue(&self, job: InferenceJob) -> bool {
        match self.tx.try_send(job) {
            Ok(()) => true,
            Err(mpsc::error::TrySendError::Full(_)) => false,
            Err(mpsc::error::TrySendError::Closed(_)) => false,
        }
    }

    pub async fn enqueue(
        &self,
        job: InferenceJob,
    ) -> Result<(), mpsc::error::SendError<InferenceJob>> {
        self.tx.send(job).await
    }
}

fn blackhole_ws_sender() -> mpsc::Sender<WsMessage> {
    let (tx, mut rx) = mpsc::channel(8);

    tokio::spawn(async move {
        while rx.recv().await.is_some() {
            // intentionally discard
        }
    });

    tx
}

pub fn warmup_job(infer: Arc<InferenceService>, db: Arc<DBLayer>) -> InferenceJob {
    InferenceJob {
        prompt: "Hello".to_string(), // minimal token path
        chat_id: "__warmup__".into(),
        session_id: "__warmup__".into(),
        sender: blackhole_ws_sender(),
        infer,
        db,
        cancel: Arc::new(AtomicBool::new(false)),
    }
}

fn should_generate_summary(history: &[Message]) -> bool {
    // 1) If summary already exists, do NOT generate again
    if history.iter().any(|m| m.role == "summary") {
        return false;
    }

    // 2) Generate summary as soon as the first user message exists
    let user_count = history.iter().filter(|m| m.role == "user").count();

    user_count == 1
}

async fn worker_loop(mut rx: mpsc::Receiver<InferenceJob>) {
    while let Some(job) = rx.recv().await {
        process_job(job).await;
    }
}
async fn process_job(job: InferenceJob) {
    if job.cancel.load(Ordering::SeqCst) {
        return;
    }

    if job.sender.is_closed() {
        return;
    }

    let is_warmup = job.chat_id == "__warmup__";

    info!(
        chat_id = job.chat_id.as_str(),
        session_id = job.session_id.as_str(),
        stop_sequences = "<|im_end|>",
        "starting mistral stream"
    );

    let mut stream = job
        .infer
        .generate_stream(job.prompt.clone(), job.cancel.clone());

    let mut assistant_reply = String::new();

    while let Some(token) = stream.recv().await {
        if is_warmup {
            break; // first token is enough
        }

        if token.contains("<|im_end|>") {
            break;
        }

        assistant_reply.push_str(token.as_str());

        let msg = serde_json::json!({
            "type": "assistant",
            "token": token
        });

        if job.cancel.load(Ordering::SeqCst) {
            break;
        }

        if job.sender.is_closed() {
            break;
        }

        if job
            .sender
            .send(WsMessage::Text(msg.to_string().into()))
            .await
            .is_err()
        {
            break;
        }
    }

    let final_response = trim_partial_chatml(&strip_chatml_markers(&assistant_reply)).to_string();
    let final_response = tidy_decoded_text(&final_response);

    let assistant_msg = Message {
        id: Uuid::new_v4().to_string(),
        chat_id: job.chat_id.clone(),
        session_id: Some(job.session_id.clone()),
        user_id: None,
        device_hash: None,
        role: "assistant".into(),
        text: Some(final_response.clone()),
        language: None,
        attachments: Vec::new(),
        liked: false,
        ts: chrono::Utc::now().timestamp(),
    };

    if let Err(err) = job.db.save_message(&assistant_msg).await {
        eprintln!(
            "failed to save assistant message {}: {err}",
            assistant_msg.id
        );
    }

    let _ = touch_chat(&job.db, &assistant_msg.chat_id, None).await;

    // -----------------------
    // LOAD UPDATED HISTORY
    // -----------------------
    let history = job
        .db
        .list_messages_for_chat(&job.chat_id)
        .await
        .unwrap_or_default();

    // -----------------------
    // SUMMARY TRIGGER (correct!)
    // -----------------------
    if should_generate_summary(&history) {
        debug!("summary triggered for chat {}", job.chat_id);

        tokio::spawn({
            let db = job.db.clone();
            let infer = job.infer.clone();
            let chat_id = job.chat_id.clone();
            let ws_tx = job.sender.clone();

            async move {
                if let Err(e) = generate_summary_message(db, infer, chat_id, ws_tx).await {
                    eprintln!("summary generation failed: {e}");
                }
            }
        });
    }

    let done_msg = serde_json::json!({
        "type": "assistant",
        "done": true
    });

    if job
        .sender
        .send(WsMessage::Text(done_msg.to_string().into()))
        .await
        .is_err()
    {
        return;
    }
}

pub async fn generate_summary_message(
    db: Arc<DBLayer>,
    infer: Arc<InferenceService>,
    chat_id: String,
    ws_tx: mpsc::Sender<WsMessage>,
) -> anyhow::Result<()> {
    let history = db
        .list_messages_for_chat(&chat_id)
        .await
        .unwrap_or_default();

    // Prevent duplicates
    if history.iter().any(|m| m.role == "summary") {
        return Ok(());
    }

    let user_messages: Vec<&Message> = history.iter().filter(|m| m.role == "user").collect();

    // Extract ONLY first few user messages
    let text = user_messages
        .iter()
        .take(3)
        .filter_map(|m| m.text.as_ref())
        .cloned()
        .collect::<Vec<_>>()
        .join("\n");

    if text.trim().is_empty() {
        return Ok(());
    }

    // NEW: Phi summarizer with strict JSON output and fallback.
    let language_hint = user_messages
        .iter()
        .filter_map(|m| m.language.as_deref())
        .find(|lang| !lang.trim().is_empty());

    let normalized_lang = language_hint.and_then(|lang| normalize_language_code(lang));
    let language_instruction = normalized_lang
        .as_deref()
        .map(|code| {
            format!(
                "- The \"summary\" value must be written in {}.",
                language_display_name(code)
            )
        })
        .unwrap_or_else(|| "- The \"summary\" value must be written in English.".to_string());

    let prompt = format!(
        r#"You output ONLY valid JSON with a single field "summary".
- Summarize the user's request in at most 3 lower-case words.
- Use short, meaningful keywords without punctuation.
- If the intent is unclear or generic, set "summary" to "general request".
{language_instruction}
Text to summarize:
{text}
JSON:"#,
        text = text,
        language_instruction = language_instruction
    );

    let summary_raw = infer.phi.generate_with_prompt(&prompt, 64).await?;
    let summary = extract_summary(&summary_raw);

    // Save summary message
    let msg = Message {
        id: Uuid::new_v4().to_string(),
        chat_id: chat_id.clone(),
        session_id: None,
        user_id: None,
        device_hash: None,
        role: "summary".into(),
        text: Some(summary.clone()),
        language: normalized_lang.clone(),
        attachments: Vec::new(),
        liked: false,
        ts: chrono::Utc::now().timestamp(),
    };

    db.save_message(&msg).await?;
    let _ = touch_chat(&db, &chat_id, None).await;

    // Send to UI
    let summary_msg = json!({
        "type": "summary",
        "chat_id": chat_id,
        "message_id": msg.id,
        "role": "summary",
        "text": summary,
        "ts": msg.ts,
        "language": normalized_lang,
    });

    let _ = ws_tx
        .send(WsMessage::Text(summary_msg.to_string().into()))
        .await;

    Ok(())
}

fn extract_summary(raw: &str) -> String {
    if let Some(summary) = parse_summary_json(raw) {
        return summary;
    }

    normalize_summary(raw)
}

fn parse_summary_json(raw: &str) -> Option<String> {
    let trimmed = raw.trim();

    if let Ok(obj) = serde_json::from_str::<SummaryJson>(trimmed) {
        return Some(normalize_summary(&obj.summary));
    }

    let start = trimmed.find('{')?;
    let end = trimmed.rfind('}')?;
    if end <= start {
        return None;
    }

    let slice = &trimmed[start..=end];
    serde_json::from_str::<SummaryJson>(slice)
        .map(|obj| normalize_summary(&obj.summary))
        .ok()
}

fn normalize_summary(text: &str) -> String {
    let cleaned = text
        .replace(['\n', '\r'], " ")
        .split_whitespace()
        .take(3)
        .map(|w| w.trim_matches(|c: char| c.is_ascii_punctuation()))
        .filter(|w| !w.is_empty())
        .collect::<Vec<_>>()
        .join(" ")
        .to_lowercase();

    if cleaned.is_empty() {
        "general request".to_string()
    } else {
        cleaned
    }
}

fn normalize_language_code(raw: &str) -> Option<String> {
    let normalized = raw
        .split(|c| c == '-' || c == '_')
        .next()
        .map(|part| part.trim().to_lowercase())
        .unwrap_or_default();

    if normalized.is_empty() {
        None
    } else {
        Some(normalized)
    }
}

fn language_display_name(code: &str) -> String {
    match code {
        "es" => "Spanish".to_string(),
        "pt" => "Portuguese".to_string(),
        "ru" => "Russian".to_string(),
        "fr" => "French".to_string(),
        "de" => "German".to_string(),
        "it" => "Italian".to_string(),
        "en" => "English".to_string(),
        _ => code.to_string(),
    }
}
