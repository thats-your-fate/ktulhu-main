use crate::{model::chat::Chat, model::message::Message, ws::AppState};

use axum::{extract::Path, extract::State, Json};
use serde_json::json;
use std::cmp::Reverse;
use uuid::Uuid;

#[derive(Debug, serde::Serialize)]
pub struct MessagesResponse {
    pub chat_id: String,
    pub messages: Vec<Message>,
}

pub async fn get_thread(
    Path(chat_id): Path<String>,
    State(state): State<AppState>,
) -> Json<serde_json::Value> {
    match state.db.list_messages_for_chat(&chat_id).await {
        Ok(mut msgs) => {
            msgs.sort_by_key(|m| m.ts);
            Json(json!({
                "chat_id": chat_id,
                "messages": msgs,
                "source": "db"
            }))
        }
        Err(e) => Json(json!({
            "chat_id": chat_id,
            "messages": [],
            "error": e.to_string()
        })),
    }
}

pub async fn delete_thread(
    Path(chat_id): Path<String>,
    State(state): State<AppState>,
) -> Json<serde_json::Value> {
    match state.db.delete_thread(&chat_id).await {
        Ok(()) => Json(json!({
            "chat_id": chat_id,
            "deleted": true,
            "source": ["memory", "db"]
        })),
        Err(e) => Json(json!({
            "chat_id": chat_id,
            "deleted": false,
            "error": e.to_string()
        })),
    }
}

pub async fn list_chats_by_device(
    Path(device_hash): Path<String>,
    State(state): State<AppState>,
) -> Json<serde_json::Value> {
    match state.db.list_chats_for_device(&device_hash).await {
        Ok(mut chats) => {
            chats.sort_by_key(|c| Reverse(c.updated_ts));
            let chats = chats
                .into_iter()
                .map(|c| {
                    json!({
                        "chat_id": c.id,
                        "title": c.title,
                        "user_id": c.user_id,
                        "device_hash": c.device_hash,
                        "updated_ts": c.updated_ts,
                        "meta": c.meta
                    })
                })
                .collect::<Vec<_>>();

            Json(json!({ "device_hash": device_hash, "chats": chats }))
        }
        Err(e) => Json(json!({
            "device_hash": device_hash,
            "chats": [],
            "error": e.to_string()
        })),
    }
}

pub async fn list_messages_by_device(
    Path(device_hash): Path<String>,
    State(state): State<AppState>,
) -> Json<serde_json::Value> {
    match state.db.list_chats_for_device(&device_hash).await {
        Ok(chats) => {
            let mut messages = Vec::new();
            for chat in chats {
                if let Ok(mut msgs) = state.db.list_messages_for_chat(&chat.id).await {
                    messages.append(&mut msgs);
                }
            }
            messages.sort_by_key(|m| m.ts);
            Json(json!({
                "device_hash": device_hash,
                "messages": messages,
            }))
        }
        Err(e) => Json(json!({
            "device_hash": device_hash,
            "messages": [],
            "error": e.to_string()
        })),
    }
}

pub async fn list_messages_for_chat(
    State(state): State<AppState>,
    Path(chat_id): Path<String>,
) -> Json<MessagesResponse> {
    let mut msgs = state
        .db
        .list_messages_for_chat(&chat_id)
        .await
        .unwrap_or_default();

    msgs.sort_by_key(|m| m.ts);

    Json(MessagesResponse {
        chat_id,
        messages: msgs,
    })
}

/// Ensure a chat exists for the given id/device; create one if missing.
pub async fn ensure_chat_for_device(
    db: &crate::db::DBLayer,
    chat_id: &str,
    device_hash: &str,
) -> anyhow::Result<String> {
    if let Some(_) = db.load_chat(chat_id).await? {
        return Ok(chat_id.to_string());
    }

    let new_id = if chat_id.is_empty() {
        Uuid::new_v4().to_string()
    } else {
        chat_id.to_string()
    };

    let chat = Chat {
        id: new_id.clone(),
        title: None,
        user_id: None,
        device_hash: Some(device_hash.to_string()),
        updated_ts: chrono::Utc::now().timestamp(),
        meta: None,
    };
    db.save_chat(&chat).await?;
    Ok(new_id)
}

pub async fn list_chats_by_user(
    State(state): State<AppState>,
    Path(user_id): Path<String>,
) -> Result<Json<serde_json::Value>, (axum::http::StatusCode, String)> {
    // Collect all chats (explicit + devices)
    let chats = state
        .db
        .list_chats_for_user(&user_id)
        .await
        .map_err(|e| (axum::http::StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    Ok(Json(serde_json::json!({
        "user_id": user_id,
        "count": chats.len(),
        "chats": chats
    })))
}
