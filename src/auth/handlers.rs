use axum::{Json, extract::State, http::StatusCode};
use serde_json::Value;

use crate::{
    auth::{types::*, google::verify_google_id_token, jwt::create_jwt},
    model::user::{User, UserRole},
    ws::AppState,
};
use std::sync::Arc;

pub async fn google_login(
    State(state): State<AppState>,
    Json(req): Json<GoogleAuthRequest>,
) -> Result<Json<AuthResponse>, StatusCode> {
    let claims = verify_google_id_token(&req.id_token, &state.google_client_id)
        .await
        .map_err(|_| StatusCode::UNAUTHORIZED)?;

    let google_id = claims["sub"].as_str().unwrap().to_string();
    let name = claims["name"].as_str().map(|s| s.to_string());
    let email = claims["email"].as_str().map(|s| s.to_string());

    // -------------------------
    // 1. Load or create user
    // -------------------------
    let existing = state.db.load_user(&google_id).await.unwrap_or(None);

    let mut user = existing.unwrap_or(User {
        id: google_id.clone(),
        name: None,
        external_id: None,
        created_ts: chrono::Utc::now().timestamp(),
        meta: None,
        email: None,
        password_hash: None,
        role: UserRole::Free,
    });

    if let Some(n) = name { user.name = Some(n); }
    if let Some(e) = email { user.external_id = Some(e); }

    state.db.save_user(&user)
        .await
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

    // -------------------------
    // 2. DEVICE REGISTRATION
    // -------------------------
    let device_hash = req.device_hash;

    // Check if device already exists
    if let Some(mut device) = state.db.find_device_by_hash(&device_hash).await.unwrap_or(None) {
        // Update last seen
        device.last_seen_ts = chrono::Utc::now().timestamp();
        state.db.save_user_device(&device).await.unwrap();
    } else {
        // Create new device record
        let new_device = UserDevice {
            id: uuid::Uuid::new_v4().to_string(),
            user_id: user.id.clone(),
            device_hash,
            created_ts: chrono::Utc::now().timestamp(),
            last_seen_ts: chrono::Utc::now().timestamp(),
            meta: None,
        };

        state.db.save_user_device(&new_device)
            .await
            .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    }

    // -------------------------
    // 3. JWT includes user + device
    // -------------------------
    let jwt = create_jwt(&user.id, &state.jwt_secret)
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

    // -------------------------
    // 4. Final response
    // -------------------------
    Ok(Json(AuthResponse {
        user_id: user.id,
        jwt,
    }))
}
