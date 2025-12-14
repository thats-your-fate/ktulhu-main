use crate::auth::jwt::make_jwt;
use crate::db::DBLayer;
use crate::model::user::User;
use crate::ws::AppState;
use axum::{extract::State, Json};
use serde::{Deserialize, Serialize};
use serde_json::json;
use uuid::Uuid;

#[derive(Deserialize)]
pub struct FacebookLoginRequest {
    pub access_token: String,
}

#[derive(Serialize)]
pub struct FacebookLoginResponse {
    pub user_id: String,
    pub email: Option<String>,
    pub jwt: String,
}

pub async fn facebook_login_handler(
    State(state): State<AppState>,
    Json(payload): Json<FacebookLoginRequest>,
) -> Result<Json<FacebookLoginResponse>, axum::http::StatusCode> {
    // 1) Call Facebook Graph API
    let url = format!(
        "https://graph.facebook.com/me?fields=id,email&access_token={}",
        payload.access_token
    );

    let fb_res = reqwest::get(&url)
        .await
        .map_err(|_| axum::http::StatusCode::BAD_REQUEST)?;

    if !fb_res.status().is_success() {
        return Err(axum::http::StatusCode::UNAUTHORIZED);
    }

    let data: serde_json::Value = fb_res
        .json()
        .await
        .map_err(|_| axum::http::StatusCode::BAD_REQUEST)?;

    let fb_id = data["id"].as_str().unwrap_or("").to_string();
    let email = data["email"].as_str().map(|s| s.to_string());

    if fb_id.is_empty() {
        return Err(axum::http::StatusCode::BAD_REQUEST);
    }

    // 2) Upsert user (MERGE ACCOUNTS)
    let user = upsert_facebook_user(&state.db, &fb_id, email.as_deref())
        .await
        .map_err(|_| axum::http::StatusCode::INTERNAL_SERVER_ERROR)?;

    // 3) Create JWT for INTERNAL user_id
    let jwt = make_jwt(&user.id).map_err(|_| axum::http::StatusCode::INTERNAL_SERVER_ERROR)?;

    Ok(Json(FacebookLoginResponse {
        user_id: user.id,
        email,
        jwt,
    }))
}

pub async fn upsert_facebook_user(
    db: &DBLayer,
    fb_id: &str,
    email: Option<&str>,
) -> anyhow::Result<User> {
    let provider_id = format!("facebook:{fb_id}");

    let all_users = db.list_users().await?;

    // 1) If a user already contains this auth method → return it
    if let Some(u) = all_users.iter().find(|u| {
        if let Some(meta) = &u.meta {
            if let Some(methods) = meta.get("auth_methods").and_then(|v| v.as_array()) {
                return methods.iter().any(|m| m.as_str() == Some(&provider_id));
            }
        }
        false
    }) {
        return Ok(u.clone());
    }

    // 2) Merge by email
    if let Some(email) = email {
        if let Some(existing) = all_users.iter().find(|u| {
            u.meta
                .as_ref()
                .and_then(|m| m.get("email"))
                .and_then(|v| v.as_str())
                == Some(email)
        }) {
            let mut updated = existing.clone();
            let mut meta = updated.meta.clone().unwrap_or(json!({}));

            let mut methods = meta
                .get("auth_methods")
                .and_then(|v| v.as_array())
                .cloned()
                .unwrap_or_default();

            if !methods.iter().any(|m| m.as_str() == Some(&provider_id)) {
                methods.push(json!(provider_id.clone()));
            }

            meta["auth_methods"] = json!(methods);
            meta["email"] = json!(email);
            updated.meta = Some(meta);
            if updated.external_id.is_none() {
                updated.external_id = Some(provider_id.clone());
            }
            updated.email = Some(email.to_string());

            db.save_user(&updated).await?;
            return Ok(updated);
        }
    }

    // 3) Create new user
    let now = chrono::Utc::now().timestamp();
    let email_owned = email.map(|e| e.to_string());
    let user = User {
        id: Uuid::new_v4().to_string(),
        name: None,
        email: email_owned.clone(),
        external_id: Some(provider_id.clone()),
        created_ts: now,
        meta: Some(json!({
            "email": email_owned,
            "auth_methods": [provider_id],
        })),
        password_hash: None,
    };

    db.save_user(&user).await?;
    Ok(user)
}
