use crate::{
    model::{
        chat::Chat,
        message::Message,
        user::UserRole,
    },
    ws::AppState,
};

use axum::{
    extract::{Path, Query, State},
    http::StatusCode,
    response::Html,
    Json,
};
use chrono::Utc;
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::cmp::Reverse;
use uuid::Uuid;

#[derive(Debug, serde::Serialize)]
pub struct MessagesResponse {
    pub chat_id: String,
    pub messages: Vec<Message>,
}

#[derive(Debug, Serialize)]
pub struct AdminOverview {
    pub total_users: usize,
    pub total_devices: usize,
    pub total_chats: usize,
    pub total_messages: usize,
    pub liked_messages: usize,
    pub recent_chats: Vec<AdminChatSummary>,
}

#[derive(Debug, Serialize)]
pub struct AdminChatSummary {
    pub chat_id: String,
    pub summary: Option<String>,
    pub device_hash: Option<String>,
    pub message_count: usize,
    pub liked_count: usize,
    pub updated_ts: i64,
}

#[derive(Debug, Deserialize)]
pub struct SummaryUpdatePayload {
    pub summary: String,
    #[serde(default)]
    pub language: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct MessageLikePayload {
    pub liked: bool,
}

#[derive(Debug, Deserialize)]
pub struct UpdateUserRolePayload {
    pub role: UserRole,
}

#[derive(Debug, Deserialize)]
pub struct LatestMessagesQuery {
    #[serde(default = "default_latest_limit")]
    pub limit: usize,
}

fn default_latest_limit() -> usize {
    25
}

pub async fn update_summary(
    Path(chat_id): Path<String>,
    State(state): State<AppState>,
    Json(payload): Json<SummaryUpdatePayload>,
) -> Json<serde_json::Value> {
    if payload.summary.trim().is_empty() {
        return Json(json!({
            "chat_id": chat_id,
            "updated": false,
            "error": "summary cannot be empty"
        }));
    }

    if let Err(err) = state.db.remove_messages_by_role(&chat_id, "summary").await {
        return Json(json!({
            "chat_id": chat_id,
            "updated": false,
            "error": err.to_string()
        }));
    }

    let msg = Message {
        id: Uuid::new_v4().to_string(),
        chat_id: chat_id.clone(),
        session_id: None,
        user_id: None,
        device_hash: None,
        role: "summary".into(),
        text: Some(payload.summary.trim().to_string()),
        language: payload.language.clone(),
        attachments: Vec::new(),
        liked: false,
        ts: Utc::now().timestamp(),
    };

    match state.db.save_message(&msg).await {
        Ok(()) => Json(json!({
            "chat_id": chat_id,
            "summary_id": msg.id,
            "updated": true
        })),
        Err(err) => Json(json!({
            "chat_id": chat_id,
            "updated": false,
            "error": err.to_string()
        })),
    }
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

pub async fn delete_message(
    Path((chat_id, message_id)): Path<(String, String)>,
    State(state): State<AppState>,
) -> Json<serde_json::Value> {
    match state.db.delete_message(&chat_id, &message_id).await {
        Ok(true) => Json(json!({
            "chat_id": chat_id,
            "message_id": message_id,
            "deleted": true
        })),
        Ok(false) => Json(json!({
            "chat_id": chat_id,
            "message_id": message_id,
            "deleted": false,
            "error": "message_not_found"
        })),
        Err(e) => Json(json!({
            "chat_id": chat_id,
            "message_id": message_id,
            "deleted": false,
            "error": e.to_string()
        })),
    }
}

pub async fn set_message_liked(
    Path((chat_id, message_id)): Path<(String, String)>,
    State(state): State<AppState>,
    Json(payload): Json<MessageLikePayload>,
) -> Json<serde_json::Value> {
    match state
        .db
        .set_message_liked(&chat_id, &message_id, payload.liked)
        .await
    {
        Ok(true) => Json(json!({
            "chat_id": chat_id,
            "message_id": message_id,
            "liked": payload.liked,
            "updated": true
        })),
        Ok(false) => Json(json!({
            "chat_id": chat_id,
            "message_id": message_id,
            "liked": payload.liked,
            "updated": false,
            "error": "message_not_found"
        })),
        Err(e) => Json(json!({
            "chat_id": chat_id,
            "message_id": message_id,
            "liked": payload.liked,
            "updated": false,
            "error": e.to_string()
        })),
    }
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

pub async fn admin_latest_messages(
    State(state): State<AppState>,
    Query(query): Query<LatestMessagesQuery>,
) -> Json<serde_json::Value> {
    let limit = query.limit.clamp(1, 200);
    match state.db.list_recent_messages(limit).await {
        Ok(messages) => Json(json!({
            "limit": limit,
            "count": messages.len(),
            "messages": messages
        })),
        Err(err) => Json(json!({
            "limit": limit,
            "count": 0,
            "messages": [],
            "error": err.to_string()
        })),
    }
}

pub async fn admin_page() -> Html<&'static str> {
    Html(include_str!("admin.html"))
}

pub async fn admin_users_page() -> Html<&'static str> {
    Html(include_str!("users.html"))
}

pub async fn admin_list_users(State(state): State<AppState>) -> Json<serde_json::Value> {
    let mut users = state.db.list_users().await.unwrap_or_default();
    users.sort_by_key(|u| Reverse(u.created_ts));

    let rows: Vec<serde_json::Value> = users
        .into_iter()
        .map(|user| {
            json!({
                "id": user.id,
                "name": user.name,
                "email": user.email,
                "role": user.role,
                "generation_count": user.generation_count,
                "generation_limit": user.generation_limit(),
                "generations_remaining": user.generations_remaining(),
                "created_ts": user.created_ts,
                "can_generate": user.can_generate_now()
            })
        })
        .collect();

    Json(json!({
        "count": rows.len(),
        "users": rows
    }))
}

pub async fn admin_update_user_role(
    Path(user_id): Path<String>,
    State(state): State<AppState>,
    Json(payload): Json<UpdateUserRolePayload>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    let mut user = state
        .db
        .load_user(&user_id)
        .await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?
        .ok_or((StatusCode::NOT_FOUND, "user_not_found".to_string()))?;

    user.role = payload.role;
    state
        .db
        .save_user(&user)
        .await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    Ok(Json(json!({
        "user_id": user.id,
        "role": user.role,
        "can_generate": user.can_generate_now(),
        "generation_count": user.generation_count,
        "generation_limit": user.generation_limit(),
        "generations_remaining": user.generations_remaining()
    })))
}

pub async fn admin_overview(State(state): State<AppState>) -> Json<AdminOverview> {
    let users = state.db.list_users().await.unwrap_or_default();
    let devices = state.db.list_all_devices().await.unwrap_or_default();
    let chats = state.db.list_chats().await.unwrap_or_default();

    let mut total_messages = 0usize;
    let mut liked_messages = 0usize;
    let mut chat_rows = Vec::new();

    for chat in chats.iter() {
        let messages = state
            .db
            .list_messages_for_chat(&chat.id)
            .await
            .unwrap_or_default();
        let liked_count = messages.iter().filter(|m| m.liked).count();
        let message_count = messages.len();

        total_messages += message_count;
        liked_messages += liked_count;

        let summary_text = messages
            .iter()
            .rev()
            .find(|m| m.role == "summary")
            .and_then(|m| m.text.clone());

        chat_rows.push(AdminChatSummary {
            chat_id: chat.id.clone(),
            summary: summary_text,
            device_hash: chat.device_hash.clone(),
            message_count,
            liked_count,
            updated_ts: chat.updated_ts,
        });
    }

    chat_rows.sort_by_key(|c| Reverse(c.updated_ts));
    chat_rows.truncate(25);

    Json(AdminOverview {
        total_users: users.len(),
        total_devices: devices.len(),
        total_chats: chats.len(),
        total_messages,
        liked_messages,
        recent_chats: chat_rows,
    })
}
