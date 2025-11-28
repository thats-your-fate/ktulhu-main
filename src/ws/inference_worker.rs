use axum::extract::ws::Message as WsMessage;
use serde_json::json;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use tokio::sync::mpsc;
use uuid::Uuid;

use crate::conversation::build_summary_prompt;
use crate::db::DBLayer;
use crate::inference::InferenceService;
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

    pub async fn enqueue(
        &self,
        job: InferenceJob,
    ) -> Result<(), mpsc::error::SendError<InferenceJob>> {
        self.tx.send(job).await
    }
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

    let mut stream = job.infer.generate_stream(job.prompt, job.cancel.clone());

    let mut assistant_reply = String::new();

    while let Some(token) = stream.recv().await {
        assistant_reply.push_str(&token);

        let msg = serde_json::json!({
            "type": "assistant",
            "token": token
        });

        if job.cancel.load(Ordering::SeqCst) {
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

    let assistant_msg = Message {
        id: Uuid::new_v4().to_string(),
        chat_id: job.chat_id.clone(),
        session_id: Some(job.session_id.clone()),
        user_id: None,
        device_hash: None,
        role: "assistant".into(),
        text: Some(assistant_reply.clone()),
        ts: chrono::Utc::now().timestamp(),
    };

    if let Err(err) = job.db.save_message(&assistant_msg).await {
        eprintln!(
            "failed to save assistant message {}: {err}",
            assistant_msg.id
        );
    }
    let _ = touch_chat(&job.db, &assistant_msg.chat_id, None).await;

    // Check if chat already has a summary
    let has_summary = {
        let history = job
            .db
            .list_messages_for_chat(&job.chat_id)
            .await
            .unwrap_or_default();
        history.iter().any(|m| m.role == "summary")
    };

    // Queue summary generation once per chat
    if !has_summary {
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

    let _ = job
        .sender
        .send(WsMessage::Text(done_msg.to_string().into()))
        .await;
}

/// Generate and persist a chat summary, then push to client.
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

    // Skip if a summary already exists to avoid duplicates.
    if history.iter().any(|m| m.role == "summary") {
        return Ok(());
    }

    // Require at least one message to summarize.
    if history.is_empty() {
        return Ok(());
    }

    let prompt = build_summary_prompt(&history);

    // If we somehow have no user text, skip generating a summary.
    if prompt.trim().is_empty() {
        return Ok(());
    }

    let raw = infer.generate(&prompt, 128).await?;

    // Attempt direct JSON parsing
    let parsed: SummaryJson = match serde_json::from_str(&raw) {
        Ok(v) => v,
        Err(_) => {
            // Try to extract the first JSON object inside the raw text
            if let Some(start) = raw.find('{') {
                if let Some(end) = raw.rfind('}') {
                    let candidate = &raw[start..=end];

                    match serde_json::from_str(candidate) {
                        Ok(v) => v,
                        Err(_) => {
                            eprintln!("SUMMARY PARSE WARNING: fallback to raw text");
                            SummaryJson {
                                summary: raw.trim().to_string(),
                            }
                        }
                    }
                } else {
                    SummaryJson {
                        summary: raw.trim().to_string(),
                    }
                }
            } else {
                SummaryJson {
                    summary: raw.trim().to_string(),
                }
            }
        }
    };

    let summary = parsed.summary.trim().to_string();

    let msg = Message {
        id: Uuid::new_v4().to_string(),
        chat_id: chat_id.clone(),
        session_id: None,
        user_id: None,
        device_hash: None,
        role: "summary".into(),
        text: Some(summary.clone()),
        ts: chrono::Utc::now().timestamp(),
    };

    db.save_message(&msg).await?;
    let _ = touch_chat(&db, &chat_id, None).await;

    let summary_msg = json!({
        "type": "summary",
        "chat_id": chat_id,
        "message_id": msg.id,
        "role": "summary",
        "text": summary,
        "ts": msg.ts,
    });

    let _ = ws_tx
        .send(WsMessage::Text(summary_msg.to_string().into()))
        .await;

    Ok(())
}
