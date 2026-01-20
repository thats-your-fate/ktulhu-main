use axum::extract::ws::{Message as WsMessage, WebSocket};
use axum::extract::State;
use axum::{response::IntoResponse, routing::get, Router};

use futures_util::{SinkExt, StreamExt};
use serde::Deserialize;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use tokio::sync::{mpsc, Mutex};
use tokio::time::{timeout, Duration};

use crate::attachments::{attachment_summaries, IncomingAttachment};
use crate::conversation::{build_mistral_prompt, trim_history};
use crate::db::DBLayer;
use crate::inference::InferenceService;
use crate::internal_api::handlers::ensure_chat_for_device;
use crate::manager::ModelManager;
use crate::model::chat::Chat;
use crate::model::message::{Message, MessageAttachment};
use crate::payment::PaymentService;
use crate::prompts;
use crate::ws::inference_worker::{InferenceJob, InferenceWorker};
use anyhow::{anyhow, Error};
use tracing::{debug, info, warn};
use uuid::Uuid;

const CLASSIFIER_TIMEOUT: Duration = Duration::from_secs(15);
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
    pub payment: Option<PaymentService>,
}

#[derive(Deserialize, Debug)]
pub struct PromptMsg {
    pub msg_type: MsgType,
    pub request_id: String,
    pub chat_id: String,
    pub session_id: String,
    pub device_hash: String,
    pub text: String,
    #[serde(default)]
    pub language: Option<String>,
    #[serde(default)]
    pub attachments: Vec<IncomingAttachment>,
}

#[derive(Deserialize, Debug)]
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
    let (tx, mut rx) = mpsc::channel::<WsMessage>(32);

    // Dedicated writer task keeps websocket flushing smoothly.
    let writer = tokio::spawn(async move {
        while let Some(msg) = rx.recv().await {
            match timeout(Duration::from_secs(30), ws_sender.send(msg)).await {
                Ok(Ok(_)) => {}
                Ok(Err(_)) => break,
                Err(_) => continue,
            }
        }
    });

    'socket_loop: while let Some(Ok(msg)) = receiver.next().await {
        match msg {
            WsMessage::Text(raw) => {
                let parsed: PromptMsg = match serde_json::from_str(raw.as_str()) {
                    Ok(v) => v,
                    Err(_) => {
                        if let Err(err) = send_json(&tx, json_error("Invalid JSON")).await {
                            eprintln!("failed to send ws message: {err}");
                            break 'socket_loop;
                        }
                        continue;
                    }
                };

                tokio::task::yield_now().await;

                if !matches!(parsed.msg_type, MsgType::Register) {
                    info!(
                        chat_id = parsed.chat_id.as_str(),
                        session_id = parsed.session_id.as_str(),
                        request_id = parsed.request_id.as_str(),
                        msg_type = ?parsed.msg_type,
                        device_hash = parsed.device_hash.as_str(),
                        language = parsed.language.as_deref().unwrap_or(""),
                        attachments = parsed.attachments.len(),
                        text = parsed.text.as_str(),
                        "incoming ws message"
                    );
                }

                match parsed.msg_type {
                    MsgType::Register => {
                        if let Err(err) = handle_register(parsed, &session, &tx).await {
                            eprintln!("failed to send ws message: {err}");
                            break 'socket_loop;
                        }
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
                        let attachment_notes = attachment_summaries(&parsed.attachments);
                        let classification_text = if attachment_notes.is_empty() {
                            parsed.text.clone()
                        } else {
                            let mut augmented = parsed.text.clone();
                            augmented.push_str("\n\n[Attachments]\n");
                            for note in &attachment_notes {
                                augmented.push_str("- ");
                                augmented.push_str(note);
                                augmented.push('\n');
                            }
                            augmented
                        };
                        let attachment_summary_combined = if attachment_notes.is_empty() {
                            None
                        } else {
                            let combined = attachment_notes.join("\n");
                            info!(
                                chat_id = parsed.chat_id.as_str(),
                                request_id = parsed.request_id.as_str(),
                                attachments = attachment_notes.len(),
                                summary = combined.as_str(),
                                "attachment summary generated"
                            );
                            Some(combined)
                        };

                        let stored_attachments: Vec<MessageAttachment> = parsed
                            .attachments
                            .iter()
                            .map(|att| MessageAttachment {
                                id: att.id.clone(),
                                filename: att.filename.clone(),
                                mime_type: att.mime_type.clone(),
                                preview_base64: att.preview_base64.clone(),
                                path: att.path.clone(),
                                size: None,
                                description: att.description.clone(),
                                ocr_text: att.ocr_text.clone(),
                                labels: att.labels.clone().unwrap_or_default(),
                            })
                            .collect();

                        let routing_result = classify_with_timeout(
                            state.models.clone(),
                            classification_text.clone(),
                            parsed.language.clone(),
                        )
                        .await;
                        let prompt_plan = prompts::build_prompt_plan(&routing_result);
                        let rendered_system_prompt =
                            prompts::render_prompt(&prompt_plan, parsed.language.as_deref());

                        let routing_language = routing_result.language.clone();

                        let decision_chain = if routing_result.notes.is_empty() {
                            "n/a".to_string()
                        } else {
                            routing_result.notes.join(" → ")
                        };
                        info!(
                            chat_id = parsed.chat_id.as_str(),
                            request_id = parsed.request_id.as_str(),
                            speech_act = routing_result.speech_act.label.as_str(),
                            domain = routing_result.domain.label.as_str(),
                            expectation = routing_result.expectation.label.as_str(),
                            routing_language = routing_result.language.as_str(),
                            prompt_key = routing_result.prompt_key.as_str(),
                            routing_path = ?routing_result.routing_path,
                            intent_kind = ?routing_result.final_intent_kind,
                            chain = decision_chain.as_str(),
                            "intent decision summary"
                        );

                        let classifier_meta = build_classifier_metadata(&routing_result);

                        // Send classifier debug meta
                        let classifier_payload = serde_json::json!({
                            "type": "classifier_debug",
                            "intent_result": routing_result.clone(),
                        });
                        if let Err(err) = send_json(&tx, classifier_payload).await {
                            eprintln!("failed to send ws message: {err}");
                            break 'socket_loop;
                        }

                        // Ensure chat exists (create if missing)
                        let chat_id = match ensure_chat_for_device(
                            &state.db,
                            parsed.chat_id.as_str(),
                            parsed.device_hash.as_str(),
                        )
                        .await
                        {
                            Ok(cid) => cid,
                            Err(e) => {
                                eprintln!("failed to ensure chat: {e}");
                                if let Err(err) =
                                    send_json(&tx, json_error("chat_init_failed")).await
                                {
                                    eprintln!("failed to send ws message: {err}");
                                    break 'socket_loop;
                                }
                                continue;
                            }
                        };

                        // Inform client if a new chat id was created
                        if chat_id != parsed.chat_id {
                            if let Err(err) = send_json(
                                &tx,
                                serde_json::json!({
                                    "type": "system",
                                    "event": "chat_created",
                                    "chat_id": chat_id
                                }),
                            )
                            .await
                            {
                                eprintln!("failed to send ws message: {err}");
                                break 'socket_loop;
                            }
                        }

                        let user_text = parsed.text.clone();

                        if let Some(combined) = attachment_summary_combined.clone() {
                            debug!("attachment descriptions provided: {}", combined);
                            if let Err(err) = send_json(
                                &tx,
                                serde_json::json!({
                                    "type": "vision_summary",
                                    "chat_id": chat_id,
                                    "summary": combined
                                }),
                            )
                            .await
                            {
                                eprintln!("failed to send ws message: {err}");
                                break 'socket_loop;
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
                            language: Some(routing_language.clone()),
                            attachments: stored_attachments.clone(),
                            liked: false,
                            ts: chrono::Utc::now().timestamp(),
                            meta: Some(classifier_meta),
                        };

                        if matches!(history.last().map(|m| m.role.as_str()), Some("user")) {
                            if let Some(removed) = history.pop() {
                                if let Err(err) =
                                    state.db.delete_message(&chat_id, &removed.id).await
                                {
                                    warn!(
                                        chat_id = chat_id.as_str(),
                                        message_id = removed.id.as_str(),
                                        "failed to delete duplicate user message: {err}"
                                    );
                                }
                            }
                        }

                        history.push(user_msg.clone());

                        // Trim long histories
                        history = trim_history(history, 24);

                        // Build chat prompt
                        let base_prompt =
                            build_mistral_prompt(&history, Some(&rendered_system_prompt));
                        info!(
                            chat_id = parsed.chat_id.as_str(),
                            session_id = parsed.session_id.as_str(),
                            prompt = rendered_system_prompt.as_str(),
                            "rendered system prompt"
                        );

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

                        let prompt_for_model = base_prompt;

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

                        if !state.worker.try_enqueue(job) {
                            eprintln!("inference worker busy, rejecting request");
                            let _ = send_json(&tx, json_error("server_busy")).await;
                            continue;
                        }
                    }

                    MsgType::Cancel => {
                        // Actually set cancel flag!
                        {
                            let s = session.lock().await;
                            s.cancel.store(true, Ordering::SeqCst);
                        }
                        if let Err(err) = send_json(&tx, json_system("cancel_ack")).await {
                            eprintln!("failed to send ws message: {err}");
                            break 'socket_loop;
                        }
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
) -> anyhow::Result<()> {
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
    .await?;

    Ok(())
}

async fn classify_with_timeout(
    models: Arc<ModelManager>,
    text: String,
    language: Option<String>,
) -> crate::classifier::routing::IntentRoutingResult {
    use crate::classifier::routing::IntentRoutingResult;
    let handle = tokio::task::spawn_blocking(move || {
        let routing =
            crate::classifier::routing::route_intent(&models, text.as_str(), language.as_deref())?;
        Ok::<IntentRoutingResult, Error>(routing)
    });

    match tokio::time::timeout(CLASSIFIER_TIMEOUT, handle).await {
        Ok(Ok(Ok(result))) => result,
        Ok(Ok(Err(err))) => {
            eprintln!("intent routing failed: {err}");
            IntentRoutingResult::default()
        }
        Ok(Err(join_err)) => {
            eprintln!("classifier task panicked: {join_err}");
            IntentRoutingResult::default()
        }
        Err(_) => {
            eprintln!(
                "intent routing timed out after {:?}, using default profile",
                CLASSIFIER_TIMEOUT
            );
            IntentRoutingResult::default()
        }
    }
}

fn build_classifier_metadata(
    result: &crate::classifier::routing::IntentRoutingResult,
) -> serde_json::Value {
    serde_json::json!({
        "classifier": {
            "speech_act": &result.speech_act,
            "domain": &result.domain,
            "expectation": &result.expectation,
            "phatic": result.phatic.as_ref(),
            "support": result.support.as_ref(),
        },
        "intent": {
            "language": result.language.as_str(),
            "prompt_key": result.prompt_key.as_str(),
            "routing_path": result.routing_path,
            "final_intent_kind": result.final_intent_kind,
            "support_intent": result.support_intent,
        }
    })
}

// ------------------------------------------------------------
// SEND JSON WRAPPER
// ------------------------------------------------------------
async fn send_json(
    sender: &mpsc::Sender<WsMessage>,
    value: serde_json::Value,
) -> anyhow::Result<()> {
    let msg = WsMessage::Text(value.to_string().into());

    match timeout(Duration::from_secs(30), sender.send(msg)).await {
        Ok(Ok(_)) => Ok(()),
        Ok(Err(_)) => Err(anyhow!("ws channel closed")),
        Err(_) => Ok(()),
    }
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
