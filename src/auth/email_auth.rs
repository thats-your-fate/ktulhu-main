use axum::{extract::State, Json};
use serde_json::json;
use uuid::Uuid;

use crate::auth::types::*;
use crate::auth::utils::*;
use crate::{
    model::user::{User, UserRole},
    ws::AppState,
};

pub async fn email_register_handler(
    State(state): State<AppState>,
    Json(req): Json<EmailRegisterRequest>,
) -> Result<Json<EmailAuthResponse>, (axum::http::StatusCode, String)> {
    let email = req.email.trim().to_lowercase();

    // Check existing user
    let users = state
        .db
        .list_users()
        .await
        .map_err(|e| (axum::http::StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    if users.iter().any(|u| u.email.as_deref() == Some(&email)) {
        return Err((
            axum::http::StatusCode::BAD_REQUEST,
            "Email already registered".into(),
        ));
    }

    // Hash password
    let hash = hash_password(&req.password)
        .map_err(|e| (axum::http::StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    // Create user
    let user = User {
        id: Uuid::new_v4().to_string(),
        name: None,
        email: Some(email.clone()),
        external_id: Some(format!("email:{email}")),
        created_ts: chrono::Utc::now().timestamp(),
        meta: Some(json!({
            "email": email,
            "auth_methods": ["email"]
        })),
        password_hash: Some(hash),
        api_key: None,
        api_secret: None,
        generation_count: 0,
        role: UserRole::Free,
        stripe_customer_id: None,
        stripe_subscription_id: None,
    };

    state
        .db
        .save_user(&user)
        .await
        .map_err(|e| (axum::http::StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    // Add device if needed
    if let Some(device_hash) = req.device_hash {
        let _ = state.db.add_device_for_user(&user.id, &device_hash).await;
    }

    // Issue JWT
    let jwt = create_app_jwt(&state, &user.id);

    Ok(Json(EmailAuthResponse {
        jwt,
        user_id: user.id,
        email,
    }))
}

pub async fn email_login_handler(
    State(state): State<AppState>,
    Json(req): Json<EmailLoginRequest>,
) -> Result<Json<EmailAuthResponse>, (axum::http::StatusCode, String)> {
    let email = req.email.trim().to_lowercase();

    // Load users
    let users = state
        .db
        .list_users()
        .await
        .map_err(|e| (axum::http::StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    let user = users
        .into_iter()
        .find(|u| u.email.as_deref() == Some(&email))
        .ok_or((
            axum::http::StatusCode::UNAUTHORIZED,
            "Invalid credentials".into(),
        ))?;

    // Extract password hash
    let hash = user
        .password_hash
        .clone()
        .or_else(|| {
            user.meta
                .as_ref()
                .and_then(|m| m.get("password_hash"))
                .and_then(|v| v.as_str().map(|s| s.to_string()))
        })
        .ok_or((
            axum::http::StatusCode::UNAUTHORIZED,
            "Account has no password".into(),
        ))?;

    // Verify
    let valid = verify_password(&hash, &req.password)
        .map_err(|e| (axum::http::StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    if !valid {
        return Err((
            axum::http::StatusCode::UNAUTHORIZED,
            "Invalid credentials".into(),
        ));
    }

    // Device registration
    if let Some(device_hash) = req.device_hash {
        let _ = state.db.add_device_for_user(&user.id, &device_hash).await;
    }

    // JWT
    let jwt = create_app_jwt(&state, &user.id);

    Ok(Json(EmailAuthResponse {
        jwt,
        user_id: user.id,
        email,
    }))
}
