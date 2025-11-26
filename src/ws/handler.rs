use axum::extract::ws::{Message as WsMessage, WebSocket};
use axum::extract::State;
use axum::{response::IntoResponse, routing::get, Router};

use futures_util::{SinkExt, StreamExt};
use serde::Deserialize;
use std::sync::Arc;
use tokio::sync::{mpsc, Mutex};
use tokio::time::{timeout, Duration};

use crate::conversation::{build_ministral_prompt, trim_history};
use crate::db::DBLayer;
use crate::inference::InferenceService;
use crate::model::chat::Chat;
use crate::model::message::Message;
use std::sync::atomic::{AtomicBool, Ordering};
use uuid::Uuid;
// ------------------------------------------------------------
// TYPES
// ------------------------------------------------------------
#[derive(Clone)]
pub struct AppState {
    pub db: Arc<DBLayer>,
    pub infer: Arc<InferenceService>,
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

                        // Append new user message to history
                        let user_msg = Message {
                            id: parsed.request_id.clone(),
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
                        let _ =
                            touch_chat(&state.db, &chat_id, Some(parsed.device_hash.clone())).await;

                        // Share cancel flag
                        let cancel_flag = {
                            let s = session.lock().await;
                            s.cancel.clone()
                        };

                        // Spawn inference on formatted prompt
                        spawn_inference_task(
                            prompt_for_model,
                            chat_id.clone(),
                            parsed.session_id.clone(),
                            tx.clone(),
                            state.infer.clone(),
                            state.db.clone(),
                            cancel_flag,
                        );
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
// STREAMING INFERENCE TASK
// ------------------------------------------------------------
fn spawn_inference_task(
    prompt: String,
    chat_id: String,
    session_id: String,
    sender: mpsc::Sender<WsMessage>,
    infer: Arc<InferenceService>,
    db: Arc<DBLayer>,
    cancel: Arc<AtomicBool>,
) {
    tokio::spawn(async move {
        let mut stream = infer.generate_stream(prompt, cancel.clone());

        let mut assistant_reply = String::new();

        while let Some(token) = stream.recv().await {
            assistant_reply.push_str(&token);

            let msg = serde_json::json!({
                "type": "assistant",
                "token": token
            });

            if cancel.load(Ordering::SeqCst) {
                break;
            }

            if sender
                .send(WsMessage::Text(msg.to_string().into()))
                .await
                .is_err()
            {
                break;
            }
        }

        // Save assistant message
        let assistant_msg = Message {
            id: uuid::Uuid::new_v4().to_string(),
            chat_id,
            session_id: Some(session_id),
            user_id: None,
            device_hash: None,
            role: "assistant".into(),
            text: Some(assistant_reply.clone()),
            ts: chrono::Utc::now().timestamp(),
        };

        if let Err(err) = db.save_message(&assistant_msg).await {
            eprintln!(
                "failed to save assistant message {}: {err}",
                assistant_msg.id
            );
        }
        let _ = touch_chat(&db, &assistant_msg.chat_id, None).await;

        // Send done
        let done_msg = serde_json::json!({
            "type": "assistant",
            "done": true
        });

        let _ = sender
            .send(WsMessage::Text(done_msg.to_string().into()))
            .await;
    });
}

async fn ensure_chat_for_device(
    db: &DBLayer,
    chat_id: String,
    device_hash: String,
) -> anyhow::Result<String> {
    if let Some(_) = db.load_chat(&chat_id).await? {
        return Ok(chat_id);
    }

    let new_id = if chat_id.is_empty() {
        Uuid::new_v4().to_string()
    } else {
        chat_id
    };

    let chat = Chat {
        id: new_id.clone(),
        title: None,
        user_id: None,
        device_hash: Some(device_hash),
        updated_ts: chrono::Utc::now().timestamp(),
        meta: None,
    };
    db.save_chat(&chat).await?;
    Ok(new_id)
}

async fn touch_chat(
    db: &DBLayer,
    chat_id: &str,
    device_hash: Option<String>,
) -> anyhow::Result<()> {
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
    Ok(())
}
