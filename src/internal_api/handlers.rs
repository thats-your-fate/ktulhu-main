use crate::{
    internal_api::types::{InternalGenerateRequest, InternalGenerateResponse},
    model::chat::Chat,
    ws::handler::AppState,
};
use axum::extract::{Json, Path, State};
use serde_json::json;
use uuid::Uuid;

pub async fn internal_generate(
    State(state): State<AppState>,
    Json(req): Json<InternalGenerateRequest>,
) -> Json<InternalGenerateResponse> {
    let max = if req.max_tokens == 0 {
        128
    } else {
        req.max_tokens
    };

    let out = state
        .infer
        .generate(&req.prompt, max)
        .await
        .unwrap_or_else(|_| "Internal error".into());

    Json(InternalGenerateResponse { output: out })
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
            chats.sort_by_key(|c| c.updated_ts);
            Json(json!({ "device_hash": device_hash, "chats": chats }))
        }
        Err(e) => Json(json!({
            "device_hash": device_hash,
            "chats": [],
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
