use axum::extract::ws::Message as WsMessage;
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

        let history_snapshot = history.clone();
        tokio::spawn({
            let db = job.db.clone();
            let chat_id = job.chat_id.clone();
            let ws_tx = job.sender.clone();
            let history_data = history_snapshot;

            async move {
                if let Err(e) = generate_summary_message(db, chat_id, ws_tx, history_data).await {
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
    chat_id: String,
    ws_tx: mpsc::Sender<WsMessage>,
    history: Vec<Message>,
) -> anyhow::Result<()> {
    if history.iter().any(|m| m.role == "summary") {
        return Ok(());
    }

    let user_messages: Vec<&Message> = history.iter().filter(|m| m.role == "user").collect();

    let text_snippets = history
        .iter()
        .filter_map(|m| m.text.as_ref())
        .take(4)
        .cloned()
        .collect::<Vec<_>>()
        .join("\n");

    if text_snippets.trim().is_empty() {
        return Ok(());
    }

    let language_hint = user_messages
        .iter()
        .filter_map(|m| m.language.as_deref())
        .find(|lang| !lang.trim().is_empty());

    let normalized_lang = language_hint.and_then(|lang| normalize_language_code(lang));
    let summary = normalize_summary(&text_snippets);

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

    let summary_msg = serde_json::json!({
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
