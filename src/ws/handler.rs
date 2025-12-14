use axum::extract::ws::{Message as WsMessage, WebSocket};
use axum::extract::State;
use axum::{response::IntoResponse, routing::get, Router};

use futures_util::{SinkExt, StreamExt};
use serde::Deserialize;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use tokio::sync::{mpsc, Mutex};
use tokio::time::{timeout, Duration};

use crate::conversation::{build_ministral_prompt, trim_history};
use crate::db::DBLayer;
use crate::inference::InferenceService;
use crate::manager::ModelManager;
use crate::model::chat::Chat;
use crate::model::message::{Message, MessageAttachment};
use crate::storage::StorageService;
use crate::ws::inference_worker::{InferenceJob, InferenceWorker};
use anyhow::anyhow;
use tracing::{debug, info};
use uuid::Uuid;
// ------------------------------------------------------------
// TYPES
// ------------------------------------------------------------
#[derive(Clone)]
pub struct AppState {
    pub db: Arc<DBLayer>,
    pub models: Arc<ModelManager>,
    pub infer: Arc<InferenceService>,
    pub worker: InferenceWorker,
    pub jwt_secret: String,
    pub google_client_id: String,
    pub apple_client_id: String,
    pub storage: Arc<StorageService>,
}

#[derive(Deserialize)]
pub struct PromptMsg {
    pub msg_type: MsgType,
    pub request_id: String,
    pub chat_id: String,
    pub session_id: String,
    pub device_hash: String,
    pub text: String,
    #[serde(default)]
    pub attachments: Vec<IncomingAttachment>,
}

#[derive(Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MsgType {
    Prompt,
    Register,
    Cancel,
}

#[derive(Deserialize, Debug, Clone)]
pub struct IncomingAttachment {
    pub id: String,
    pub filename: String,
    #[serde(default)]
    pub mime_type: Option<String>,
    pub path: String,
    pub size: usize,
    #[serde(default)]
    pub description: Option<String>,
}

#[derive(Debug, Default)]
struct WsSession {
    device_hash: Option<String>,
    session_id: Option<String>,
    chat_id: Option<String>,
    cancel: Arc<AtomicBool>,
}

// ------------------------------------------------------------
// ROUTER
// ------------------------------------------------------------
pub fn ws_router() -> Router<AppState> {
    Router::new().route("/ws", get(ws_handler))
}

async fn ws_handler(
    ws: axum::extract::WebSocketUpgrade,
    State(state): State<AppState>,
) -> impl IntoResponse {
    ws.on_upgrade(move |socket| handle_socket(socket, state))
}

// ------------------------------------------------------------
// WEBSOCKET HANDLER (SPLIT SOCKET)
// ------------------------------------------------------------
async fn handle_socket(socket: WebSocket, state: AppState) {
    let (mut ws_sender, mut receiver) = socket.split();

    let session = Arc::new(Mutex::new(WsSession::default()));
    let (tx, mut rx) = mpsc::channel::<WsMessage>(128);

    // Dedicated writer task keeps websocket flushing smoothly.
    let writer = tokio::spawn(async move {
        while let Some(msg) = rx.recv().await {
            if ws_sender.send(msg).await.is_err() {
                break;
            }
            // Avoid getting stuck forever on flush.
            if timeout(Duration::from_secs(10), ws_sender.flush())
                .await
                .is_err()
            {
                break;
            }
        }
    });

    while let Some(Ok(msg)) = receiver.next().await {
        match msg {
            WsMessage::Text(raw) => {
                let parsed: PromptMsg = match serde_json::from_str(raw.as_str()) {
                    Ok(v) => v,
                    Err(_) => {
                        send_json(&tx, json_error("Invalid JSON")).await;
                        continue;
                    }
                };

                match parsed.msg_type {
                    MsgType::Register => {
                        handle_register(parsed, &session, &tx).await;
                    }

                    MsgType::Prompt => {
                        // Reset cancel
                        {
                            let s = session.lock().await;
                            s.cancel.store(false, Ordering::SeqCst);
                        }

                        // -----------------------------------------------------
                        // 1) CLASSIFICATION — this is the only added section
                        // -----------------------------------------------------
                        let scope = crate::classifier::scope::classify_scope(
                            &state.models,
                            parsed.text.as_str(),
                        )
                        .await
                        .unwrap_or_else(|_| "unknown".into());

                        let routing_result = match crate::classifier::routing::route_intent(
                            &state.models,
                            parsed.text.as_str(),
                        )
                        .await
                        {
                            Ok(v) => v,
                            Err(err) => {
                                eprintln!("intent routing failed: {err}");
                                crate::classifier::routing::IntentRoutingResult::default()
                            }
                        };

                        if routing_result.multi_intent {
                            send_json(
                                &tx,
                                serde_json::json!({
                                    "type": "system",
                                    "event": "multi_intent_blocked",
                                    "message": "Please ask one question at a time so I can answer accurately."
                                }),
                            )
                            .await;
                            continue;
                        }

                        for note in &routing_result.notes {
                            info!(
                                chat_id = parsed.chat_id.as_str(),
                                request_id = parsed.request_id.as_str(),
                                note = note.as_str(),
                                "intent routing note"
                            );
                        }

                        if routing_result.clarification_needed {
                            send_json(
                                &tx,
                                serde_json::json!({
                                    "type": "system",
                                    "event": "intent_clarification_needed",
                                    "message": "Just to confirm—are you looking for advice, a short task response, or general thoughts?"
                                }),
                            )
                            .await;
                        }

                        let intent = routing_result.intent.clone();
                        let intent_confidence = routing_result.confidence;

                        info!(
                            chat_id = parsed.chat_id.as_str(),
                            request_id = parsed.request_id.as_str(),
                            intent = intent.as_str(),
                            intent_confidence,
                            "intent decision triggered"
                        );

                        // Send classifier debug meta
                        send_json(
                            &tx,
                            serde_json::json!({
                                "type": "classifier_debug",
                                "scope": scope,
                                "intent": intent,
                                "intent_confidence": intent_confidence
                            }),
                        )
                        .await;

                        // -----------------------------------------------------
                        // 2) HANDLE NON-MODEL CASES (stop here)

                        // -----------------------------------------------------

                        // -----------------------------------------------------
                        // 3) NORMAL CHAT FLOW (no_rag) — original logic resumes
                        // -----------------------------------------------------

                        // Ensure chat exists (create if missing)
                        let chat_id = match ensure_chat_for_device(
                            &state.db,
                            parsed.chat_id.clone(),
                            parsed.device_hash.clone(),
                        )
                        .await
                        {
                            Ok(cid) => cid,
                            Err(e) => {
                                eprintln!("failed to ensure chat: {e}");
                                send_json(&tx, json_error("chat_init_failed")).await;
                                continue;
                            }
                        };

                        // Inform client if a new chat id was created
                        if chat_id != parsed.chat_id {
                            send_json(
                                &tx,
                                serde_json::json!({
                                    "type": "system",
                                    "event": "chat_created",
                                    "chat_id": chat_id
                                }),
                            )
                            .await;
                        }

                        // Persist attachments (if any)
                        let attachments = match persist_attachments(&parsed.attachments) {
                            Ok(files) => files,
                            Err(e) => {
                                eprintln!("failed to store attachments: {e}");
                                send_json(&tx, json_error("attachment_error")).await;
                                continue;
                            }
                        };

                        let mut user_text = parsed.text.clone();

                        if !parsed.attachments.is_empty() {
                            if let Some(summary) =
                                compose_attachment_descriptions(&parsed.attachments)
                            {
                                user_text.push_str("\n\n[Vision attachments]\n");
                                user_text.push_str(&summary);
                                debug!("attachment descriptions provided: {}", summary);
                                send_json(
                                    &tx,
                                    serde_json::json!({
                                        "type": "vision_summary",
                                        "chat_id": chat_id,
                                        "summary": summary
                                    }),
                                )
                                .await;
                            }
                        }

                        // Load chat history
                        let mut history = state
                            .db
                            .list_messages_for_chat(&chat_id)
                            .await
                            .unwrap_or_default();

                        // Prepare user message
                        let msg_id = if parsed.request_id.is_empty() {
                            Uuid::new_v4().to_string()
                        } else {
                            parsed.request_id.clone()
                        };

                        let user_msg = Message {
                            id: msg_id,
                            chat_id: chat_id.clone(),
                            session_id: Some(parsed.session_id.clone()),
                            user_id: None,
                            device_hash: Some(parsed.device_hash.clone()),
                            role: "user".into(),
                            text: Some(user_text.clone()),
                            attachments: attachments.clone(),
                            ts: chrono::Utc::now().timestamp(),
                        };

                        history.push(user_msg.clone());

                        // Trim long histories
                        history = trim_history(history, 24);

                        // Build Mistral prompt
                        let system_prompt = crate::prompts::prompt_for_intent(&intent).to_string();

                        let prompt_for_model =
                            build_ministral_prompt(&history, Some(&system_prompt));

                        // Save user message
                        if let Err(err) = state.db.save_message(&user_msg).await {
                            eprintln!("failed to save user message {}: {err}", user_msg.id);
                        }
                        let _ =
                            touch_chat(&state.db, &chat_id, Some(parsed.device_hash.clone())).await;

                        // Share cancel flag
                        let cancel_flag = {
                            let s = session.lock().await;
                            s.cancel.clone()
                        };

                        // Queue inference job — ORIGINAL logic
                        let job = InferenceJob {
                            prompt: prompt_for_model,
                            chat_id: chat_id.clone(),
                            session_id: parsed.session_id.clone(),
                            sender: tx.clone(),
                            infer: state.infer.clone(),
                            db: state.db.clone(),
                            cancel: cancel_flag,
                        };

                        if let Err(err) = state.worker.enqueue(job).await {
                            eprintln!("failed to enqueue inference job: {err}");
                            send_json(&tx, json_error("inference_unavailable")).await;
                        }
                    }

                    MsgType::Cancel => {
                        // Actually set cancel flag!
                        {
                            let s = session.lock().await;
                            s.cancel.store(true, Ordering::SeqCst);
                        }
                        send_json(&tx, json_system("cancel_ack")).await;
                    }
                }
            }
            WsMessage::Ping(payload) => {
                let _ = tx.send(WsMessage::Pong(payload)).await;
            }
            WsMessage::Close(_) => break,
            _ => {}
        };
    }

    // Socket closed → set cancel flag
    {
        let s = session.lock().await;
        s.cancel.store(true, Ordering::SeqCst);
    }

    // Drop sender to stop writer task
    drop(tx);
    let _ = writer.await;
}

// ------------------------------------------------------------
// REGISTER HANDLER
// ------------------------------------------------------------
async fn handle_register(
    msg: PromptMsg,
    session: &Arc<Mutex<WsSession>>,
    sender: &mpsc::Sender<WsMessage>,
) {
    let mut s = session.lock().await;

    s.device_hash = Some(msg.device_hash);
    s.session_id = Some(msg.session_id);
    s.chat_id = Some(msg.chat_id);

    send_json(
        sender,
        serde_json::json!({
            "type": "system",
            "event": "registered",
            "session_id": s.session_id,
            "chat_id": s.chat_id,
            "device_hash": s.device_hash,
        }),
    )
    .await;
}

fn persist_attachments(incoming: &[IncomingAttachment]) -> Result<Vec<MessageAttachment>, String> {
    let mut files = Vec::new();

    for item in incoming {
        if item.id.trim().is_empty()
            || item.filename.trim().is_empty()
            || item.path.trim().is_empty()
        {
            return Err("attachment reference is missing id/filename/path".to_string());
        }

        files.push(MessageAttachment {
            id: item.id.clone(),
            filename: item.filename.clone(),
            mime_type: item.mime_type.clone(),
            path: item.path.clone(),
            size: item.size,
        });
    }

    Ok(files)
}

fn compose_attachment_descriptions(attachments: &[IncomingAttachment]) -> Option<String> {
    let mut lines = Vec::new();

    for item in attachments {
        if let Some(desc) = item.description.as_deref() {
            if !desc.trim().is_empty() {
                lines.push(format!("{}: {}", item.filename, desc.trim()));
                continue;
            }
        }
        if let Some(mime) = &item.mime_type {
            lines.push(format!("{} ({} bytes, {})", item.filename, item.size, mime));
        } else {
            lines.push(format!("{} ({} bytes)", item.filename, item.size));
        }
    }

    if lines.is_empty() {
        None
    } else {
        Some(lines.join("\n"))
    }
}

// ------------------------------------------------------------
// SEND JSON WRAPPER
// ------------------------------------------------------------
async fn send_json(sender: &mpsc::Sender<WsMessage>, value: serde_json::Value) {
    let msg = WsMessage::Text(value.to_string().into());
    let _ = sender.send(msg).await;
}

fn json_error(msg: &str) -> serde_json::Value {
    serde_json::json!({
        "type": "error",
        "message": msg
    })
}

fn json_system(event: &str) -> serde_json::Value {
    serde_json::json!({
        "type": "system",
        "event": event
    })
}

// ------------------------------------------------------------
// STREAMING INFERENCE HELPERS
// ------------------------------------------------------------
pub(crate) async fn ensure_chat_for_device(
    db: &DBLayer,
    chat_id: String,
    device_hash: String,
) -> anyhow::Result<String> {
    if chat_id.is_empty() {
        return Err(anyhow!("chat_id is required"));
    }

    if let Some(_) = db.load_chat(&chat_id).await? {
        return Ok(chat_id);
    }

    let chat = Chat {
        id: chat_id.clone(),
        title: None,
        user_id: None,
        device_hash: Some(device_hash),
        updated_ts: chrono::Utc::now().timestamp(),
        meta: None,
    };
    db.save_chat(&chat).await?;
    Ok(chat_id)
}

pub(crate) async fn touch_chat(
    db: &DBLayer,
    chat_id: &str,
    device_hash: Option<String>,
) -> anyhow::Result<bool> {
    // ---------------------------------------------------------
    // 1. Load chat or initialize new
    // ---------------------------------------------------------
    let mut chat = db.load_chat(chat_id).await?.unwrap_or(Chat {
        id: chat_id.to_string(),
        title: None,
        user_id: None,
        device_hash: device_hash.clone(),
        updated_ts: chrono::Utc::now().timestamp(),
        meta: Some(serde_json::json!({})),
    });

    // Ensure meta exists
    if chat.meta.is_none() {
        chat.meta = Some(serde_json::json!({}));
    }

    // ---------------------------------------------------------
    // 2. Detect summary presence ONLY if not already stored
    // ---------------------------------------------------------
    let meta = chat.meta.as_mut().unwrap();

    let has_summary_cached = meta
        .get("has_summary")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);

    let has_summary = if has_summary_cached {
        true
    } else {
        // Scan messages only once
        let detected = db
            .list_messages_for_chat(chat_id)
            .await
            .map(|msgs| msgs.iter().any(|m| m.role == "summary"))
            .unwrap_or(false);

        // Store result so we never need to scan again
        if detected {
            meta["has_summary"] = serde_json::json!(true);
        }

        detected
    };

    // ---------------------------------------------------------
    // 3. Update device hash ONLY IF chat has none yet
    // ---------------------------------------------------------
    if chat.device_hash.is_none() {
        if let Some(hash) = device_hash {
            chat.device_hash = Some(hash);
        }
    }

    // ---------------------------------------------------------
    // 4. Update timestamp + save chat
    // ---------------------------------------------------------
    chat.updated_ts = chrono::Utc::now().timestamp();
    db.save_chat(&chat).await?;

    Ok(has_summary)
}
