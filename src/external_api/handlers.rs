use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};

use axum::{extract::State, http::StatusCode, Json};
use axum_extra::typed_header::{TypedHeader, TypedHeaderRejection};
use chrono::Utc;
use headers::{authorization::Bearer, Authorization};
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
    pub generation_count: u64,
    pub generation_limit: Option<u64>,
    pub generations_remaining: Option<u64>,
}

#[derive(Debug, Serialize)]
pub struct ProfileResponse {
    pub user_id: String,
    pub email: Option<String>,
    pub created_ts: i64,
    pub role: UserRole,
    pub can_generate: bool,
    pub generation_count: u64,
    pub generation_limit: Option<u64>,
    pub generations_remaining: Option<u64>,
}

#[derive(Debug, Deserialize)]
pub struct ApiCredentialsRequest {
    pub api_key: String,
    pub api_secret: String,
}

#[derive(Debug, Serialize)]
pub struct ApiCredentialsResponse {
    pub stored: bool,
}

#[derive(Debug, Serialize)]
pub struct ApiCredentialsGenerateResponse {
    pub api_key: String,
    pub api_secret: String,
}

#[derive(Debug, Deserialize)]
pub struct ApiCredentialsValidateRequest {
    pub api_key: String,
    pub api_secret: String,
}

#[derive(Debug, Serialize)]
pub struct ApiCredentialsValidateResponse {
    pub valid: bool,
}

#[derive(Debug, Serialize)]
pub struct GenerationUsageResponse {
    pub user_id: String,
    pub generation_count: u64,
    pub generation_limit: Option<u64>,
    pub generations_remaining: Option<u64>,
}

pub async fn generate(
    State(state): State<AppState>,
    auth_header: Result<TypedHeader<Authorization<Bearer>>, TypedHeaderRejection>,
    Json(payload): Json<GenerateRequest>,
) -> Result<Json<GenerateResponse>, (StatusCode, String)> {
    if payload.prompt.trim().is_empty() {
        return Err((StatusCode::BAD_REQUEST, "prompt_required".into()));
    }

    let auth = auth_header.map_err(|_| (StatusCode::UNAUTHORIZED, "login_required".into()))?;

    let mut user = authenticate_user(&state, auth.token()).await?;
    if !user.role.can_access_generation() {
        return Err((StatusCode::FORBIDDEN, "paid_plan_required".into()));
    }

    if !user.can_generate_now() {
        return Err((StatusCode::FORBIDDEN, "free_quota_exceeded".into()));
    }

    let request_id = Uuid::new_v4().to_string();

    let system_prompt = payload.system_prompt.clone();

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
        meta: None,
    });

    let chatml_prompt = build_mistral_prompt(&history, system_prompt.as_deref());
    let cancel = Arc::new(AtomicBool::new(false));
    let raw = state
        .infer
        .generate_completion(chatml_prompt, cancel.clone())
        .await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    cancel.store(true, Ordering::SeqCst);

    let trimmed = trim_partial_chatml(&raw);
    let cleaned = strip_chatml_markers(trimmed).trim().to_string();

    user.generation_count = user.generation_count.saturating_add(1);
    state
        .db
        .save_user(&user)
        .await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    let user_id = user.id.clone();
    Ok(Json(GenerateResponse {
        request_id,
        user_id,
        role: user.role.clone(),
        system_prompt: system_prompt.unwrap_or_default(),
        output: cleaned,
        generation_count: user.generation_count,
        generation_limit: user.generation_limit(),
        generations_remaining: user.generations_remaining(),
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
        can_generate: user.can_generate_now(),
        generation_count: user.generation_count,
        generation_limit: user.generation_limit(),
        generations_remaining: user.generations_remaining(),
    }))
}

pub async fn generation_usage(
    State(state): State<AppState>,
    TypedHeader(auth): TypedHeader<Authorization<Bearer>>,
) -> Result<Json<GenerationUsageResponse>, (StatusCode, String)> {
    let user = authenticate_user(&state, auth.token()).await?;
    Ok(Json(GenerationUsageResponse {
        user_id: user.id.clone(),
        generation_count: user.generation_count,
        generation_limit: user.generation_limit(),
        generations_remaining: user.generations_remaining(),
    }))
}

pub async fn generate_api_credentials(
    State(state): State<AppState>,
    TypedHeader(auth): TypedHeader<Authorization<Bearer>>,
) -> Result<Json<ApiCredentialsGenerateResponse>, (StatusCode, String)> {
    let mut user = authenticate_user(&state, auth.token()).await?;

    let api_key = format!("key_{}", Uuid::new_v4());
    let api_secret = format!("sec_{}", Uuid::new_v4());

    user.api_key = Some(api_key.clone());
    user.api_secret = Some(api_secret.clone());

    state
        .db
        .save_user(&user)
        .await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    Ok(Json(ApiCredentialsGenerateResponse {
        api_key,
        api_secret,
    }))
}

pub async fn store_api_credentials(
    State(state): State<AppState>,
    TypedHeader(auth): TypedHeader<Authorization<Bearer>>,
    Json(payload): Json<ApiCredentialsRequest>,
) -> Result<Json<ApiCredentialsResponse>, (StatusCode, String)> {
    let api_key = payload.api_key.trim();
    let api_secret = payload.api_secret.trim();

    if api_key.is_empty() || api_secret.is_empty() {
        return Err((StatusCode::BAD_REQUEST, "credentials_required".into()));
    }

    let mut user = authenticate_user(&state, auth.token()).await?;
    user.api_key = Some(api_key.to_owned());
    user.api_secret = Some(api_secret.to_owned());

    state
        .db
        .save_user(&user)
        .await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    Ok(Json(ApiCredentialsResponse { stored: true }))
}

pub async fn validate_api_credentials(
    State(state): State<AppState>,
    TypedHeader(auth): TypedHeader<Authorization<Bearer>>,
    Json(payload): Json<ApiCredentialsValidateRequest>,
) -> Result<Json<ApiCredentialsValidateResponse>, (StatusCode, String)> {
    let api_key = payload.api_key.trim();
    let api_secret = payload.api_secret.trim();

    if api_key.is_empty() || api_secret.is_empty() {
        return Err((StatusCode::BAD_REQUEST, "credentials_required".into()));
    }

    let user = authenticate_user(&state, auth.token()).await?;
    let valid = user
        .api_key
        .as_deref()
        .map(|stored| stored == api_key)
        .unwrap_or(false)
        && user
            .api_secret
            .as_deref()
            .map(|stored| stored == api_secret)
            .unwrap_or(false);

    Ok(Json(ApiCredentialsValidateResponse { valid }))
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
