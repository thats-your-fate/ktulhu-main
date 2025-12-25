use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};

use axum::{
    extract::State,
    headers::{authorization::Bearer, Authorization},
    http::StatusCode,
    Json, TypedHeader,
};
use chrono::Utc;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::{
    auth::jwt::decode_jwt,
    conversation::{build_mistral_prompt, strip_chatml_markers, trim_partial_chatml},
    model::{
        message::Message,
        user::{User, UserRole},
    },
    prompts,
    ws::AppState,
};

#[derive(Debug, Deserialize)]
pub struct GenerateRequest {
    pub prompt: String,
    #[serde(default)]
    pub intent: Option<String>,
    #[serde(default)]
    pub language: Option<String>,
    #[serde(default)]
    pub system_prompt: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct GenerateResponse {
    pub request_id: String,
    pub user_id: String,
    pub role: UserRole,
    pub system_prompt: String,
    pub output: String,
}

#[derive(Debug, Serialize)]
pub struct ProfileResponse {
    pub user_id: String,
    pub email: Option<String>,
    pub created_ts: i64,
    pub role: UserRole,
    pub can_generate: bool,
}

pub async fn generate(
    State(state): State<AppState>,
    TypedHeader(auth): TypedHeader<Authorization<Bearer>>,
    Json(payload): Json<GenerateRequest>,
) -> Result<Json<GenerateResponse>, (StatusCode, String)> {
    if payload.prompt.trim().is_empty() {
        return Err((StatusCode::BAD_REQUEST, "prompt_required".into()));
    }

    let user = authenticate_user(&state, auth.token()).await?;
    if !user.role.can_access_generation() {
        return Err((StatusCode::FORBIDDEN, "paid_plan_required".into()));
    }

    let request_id = Uuid::new_v4().to_string();
    let intent = payload
        .intent
        .clone()
        .unwrap_or_else(|| prompts::default_intent().to_string());

    let system_prompt = payload
        .system_prompt
        .clone()
        .unwrap_or_else(|| prompts::prompt_for_intent(&intent, payload.language.as_deref()));

    let mut history = Vec::with_capacity(1);
    history.push(Message {
        id: Uuid::new_v4().to_string(),
        chat_id: format!("api:{}", user.id),
        session_id: None,
        user_id: Some(user.id.clone()),
        device_hash: None,
        role: "user".into(),
        text: Some(payload.prompt.clone()),
        language: payload.language.clone(),
        attachments: Vec::new(),
        liked: false,
        ts: Utc::now().timestamp(),
    });

    let chatml_prompt = build_mistral_prompt(&history, Some(&system_prompt));
    let cancel = Arc::new(AtomicBool::new(false));
    let raw = state
        .infer
        .mistral
        .generate_completion(chatml_prompt, cancel.clone())
        .await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    cancel.store(true, Ordering::SeqCst);

    let trimmed = trim_partial_chatml(&raw);
    let cleaned = strip_chatml_markers(trimmed).trim().to_string();

    Ok(Json(GenerateResponse {
        request_id,
        user_id: user.id,
        role: user.role,
        system_prompt,
        output: cleaned,
    }))
}

pub async fn profile(
    State(state): State<AppState>,
    TypedHeader(auth): TypedHeader<Authorization<Bearer>>,
) -> Result<Json<ProfileResponse>, (StatusCode, String)> {
    let user = authenticate_user(&state, auth.token()).await?;

    Ok(Json(ProfileResponse {
        user_id: user.id.clone(),
        email: user.email.clone(),
        created_ts: user.created_ts,
        role: user.role.clone(),
        can_generate: user.role.can_access_generation(),
    }))
}

async fn authenticate_user(state: &AppState, token: &str) -> Result<User, (StatusCode, String)> {
    let user_id = decode_jwt(token, &state.jwt_secret)
        .map_err(|_| (StatusCode::UNAUTHORIZED, "invalid_token".into()))?;
    let user = state
        .db
        .load_user(&user_id)
        .await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?
        .ok_or((StatusCode::UNAUTHORIZED, "user_not_found".into()))?;
    Ok(user)
}
