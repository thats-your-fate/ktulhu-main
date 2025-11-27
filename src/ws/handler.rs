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
use crate::model::chat::Chat;
use crate::model::message::Message;
use crate::ws::inference_worker::{generate_summary_message, InferenceJob, InferenceWorker};
use anyhow::anyhow;
use uuid::Uuid;
// ------------------------------------------------------------
// TYPES
// ------------------------------------------------------------
#[derive(Clone)]
pub struct AppState {
    pub db: Arc<DBLayer>,
    pub infer: Arc<InferenceService>,
    pub worker: InferenceWorker,
}

#[derive(Deserialize)]
pub struct PromptMsg {
    pub msg_type: MsgType,
    pub request_id: String,
    pub chat_id: String,
    pub session_id: String,
    pub device_hash: String,
    pub text: String,
}

#[derive(Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MsgType {
    Prompt,
    Register,
    Cancel,
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

                        // Ensure chat exists (create if missing / empty id)
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

                        // Load chat history from DB
                        let mut history = state
                            .db
                            .list_messages_for_chat(&chat_id)
                            .await
                            .unwrap_or_default();

                        // Append new user message to history; tolerate missing request_id
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
                            text: Some(parsed.text.clone()),
                            ts: chrono::Utc::now().timestamp(),
                        };

                        history.push(user_msg.clone());

                        // (Optional) Trim long histories
                        history = trim_history(history, 24);

                        // Build full Minstral prompt
                        let prompt_for_model =
                            build_ministral_prompt(&history, Some("You are a helpful assistant."));

                        // Save the new user message
                        if let Err(err) = state.db.save_message(&user_msg).await {
                            eprintln!("failed to save user message {}: {err}", user_msg.id);
                        }
                        let has_summary =
                            touch_chat(&state.db, &chat_id, Some(parsed.device_hash.clone()))
                                .await
                                .unwrap_or(true);

                        // Kick off summary generation on first user request if missing
                        if !has_summary {
                            tokio::spawn({
                                let db = state.db.clone();
                                let infer = state.infer.clone();
                                let chat_id = chat_id.clone();
                                let ws_tx = tx.clone();
                                async move {
                                    if let Err(e) =
                                        generate_summary_message(db, infer, chat_id, ws_tx).await
                                    {
                                        eprintln!("summary generation failed: {e}");
                                    }
                                }
                            });
                        }

                        // Share cancel flag
                        let cancel_flag = {
                            let s = session.lock().await;
                            s.cancel.clone()
                        };

                        // Queue inference for sequential worker execution
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
    // Check if any summary already exists
    let has_summary = db
        .list_messages_for_chat(chat_id)
        .await
        .map(|msgs| msgs.iter().any(|m| m.role == "summary"))
        .unwrap_or(false);

    let mut chat = db.load_chat(chat_id).await?.unwrap_or(Chat {
        id: chat_id.to_string(),
        title: None,
        user_id: None,
        device_hash: device_hash.clone(),
        updated_ts: chrono::Utc::now().timestamp(),
        meta: None,
    });

    if let Some(hash) = device_hash {
        chat.device_hash = Some(hash);
    }
    chat.updated_ts = chrono::Utc::now().timestamp();
    db.save_chat(&chat).await?;
    Ok(has_summary)
}
