use axum::{
    extract::{Path, State},
    Json,
};

use crate::{
    api::types::MessagesResponse,
    api::types::{GenerateRequest, GenerateResponse},
    ws::AppState,
};

pub async fn generate_handler(
    State(state): State<AppState>,
    Json(req): Json<GenerateRequest>,
) -> Json<GenerateResponse> {
    let max = if req.max_tokens == 0 {
        128
    } else {
        req.max_tokens
    };

    let out = state
        .infer
        .generate(&req.prompt, max)
        .await
        .unwrap_or_else(|_| "Error generating text".into());

    Json(GenerateResponse { output: out })
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
