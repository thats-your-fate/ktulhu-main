use axum::extract::ws::Message as WsMessage;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use tokio::sync::mpsc;
use tracing::{debug, info};
use uuid::Uuid;

use crate::conversation::{
    build_mistral_prompt, strip_chatml_markers, trim_history, trim_partial_chatml,
};
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
    if history.iter().any(|m| m.role == "summary") {
        return false;
    }

    let user_count = history.iter().filter(|m| m.role == "user").count();
    let assistant_count = history.iter().filter(|m| m.role == "assistant").count();

    user_count > 0 && assistant_count >= 1
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
        meta: None,
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
        if let Err(e) = generate_summary_message(
            job.db.clone(),
            job.chat_id.clone(),
            job.sender.clone(),
            history.clone(),
            job.infer.clone(),
        )
        .await
        {
            eprintln!("summary generation failed: {e}");
        }
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
    infer: Arc<InferenceService>,
) -> anyhow::Result<()> {
    if history.iter().any(|m| m.role == "summary") {
        return Ok(());
    }

    let language_hint = history
        .iter()
        .filter_map(|m| m.language.as_deref())
        .find(|lang| !lang.trim().is_empty());

    let normalized_lang = language_hint.and_then(|lang| normalize_language_code(lang));
    let summary_prompt = build_summary_prompt(&history);
    if summary_prompt.is_empty() {
        return Ok(());
    }

    let cancel = Arc::new(AtomicBool::new(false));
    let raw = infer
        .generate_completion(summary_prompt, cancel.clone())
        .await?;
    cancel.store(true, Ordering::SeqCst);
    let trimmed = trim_partial_chatml(&raw);
    let cleaned = strip_chatml_markers(trimmed).trim().to_string();
    if cleaned.is_empty() {
        return Ok(());
    }

    let msg = Message {
        id: Uuid::new_v4().to_string(),
        chat_id: chat_id.clone(),
        session_id: None,
        user_id: None,
        device_hash: None,
        role: "summary".into(),
        text: Some(cleaned.clone()),
        language: normalized_lang.clone(),
        attachments: Vec::new(),
        liked: false,
        ts: chrono::Utc::now().timestamp(),
        meta: None,
    };

    db.save_message(&msg).await?;
    let _ = touch_chat(&db, &chat_id, None).await;

    let summary_msg = serde_json::json!({
        "type": "summary",
        "chat_id": chat_id,
        "message_id": msg.id,
        "role": "summary",
        "text": cleaned,
        "ts": msg.ts,
        "language": normalized_lang,
    });

    let _ = ws_tx
        .send(WsMessage::Text(summary_msg.to_string().into()))
        .await;

    Ok(())
}

const SUMMARY_PROMPT: &str = "Summarize user message to display in ui as chat summary with at most 20 characters.\nAvoid punctuation and keep it lowercase and plain text. If request is in other language than English, summarize in that language.\n";

fn build_summary_prompt(history: &[Message]) -> String {
    if history.is_empty() {
        return String::new();
    }

    let mut summary_history: Vec<Message> = history
        .iter()
        .filter(|m| m.role == "user")
        .cloned()
        .collect();

    if summary_history.is_empty() {
        return String::new();
    }

    summary_history = trim_history(summary_history, 6);
    build_mistral_prompt(&summary_history, Some(SUMMARY_PROMPT))
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
