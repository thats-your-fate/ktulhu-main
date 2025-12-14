use axum::{extract::State, Json};
use jsonwebtoken::{
    decode, decode_header, Algorithm, DecodingKey, EncodingKey, Header, Validation,
};
use serde::{Deserialize, Serialize};
use serde_json::json;

use uuid::Uuid;

use super::google_keys::GoogleJwkCache;
use crate::{db::DBLayer, model::user::User, ws::AppState};

#[derive(Deserialize)]
pub struct GoogleAuthRequest {
    pub id_token: String,
    pub device_hash: String,
}

#[derive(Serialize)]
pub struct AuthResponse {
    pub jwt: String,
    pub user_id: String,
    pub email: Option<String>,
}

#[derive(Debug, Deserialize)]
struct GoogleClaims {
    pub iss: String,
    pub sub: String,
    pub aud: String,
    pub email: Option<String>,
    pub email_verified: Option<bool>,
    pub exp: usize,
}

#[derive(Debug, Serialize, Deserialize)]
struct AppClaims {
    pub sub: String,
    pub exp: usize,
}

pub async fn google_login_handler(
    State(state): State<AppState>,
    Json(payload): Json<GoogleAuthRequest>,
) -> Result<Json<AuthResponse>, (axum::http::StatusCode, String)> {
    if state.google_client_id.is_empty() {
        return Err((
            axum::http::StatusCode::BAD_REQUEST,
            "Google login disabled".into(),
        ));
    }

    // --- decode JWT header ---
    let header = decode_header(&payload.id_token).map_err(|e| {
        (
            axum::http::StatusCode::UNAUTHORIZED,
            format!("bad header: {e}"),
        )
    })?;
    let kid = header
        .kid
        .ok_or((axum::http::StatusCode::UNAUTHORIZED, "no kid".into()))?;

    // --- fetch Google public key ---
    let jwk_cache = GoogleJwkCache::instance();
    let jwk = jwk_cache.get_key(&kid).await.map_err(|e| {
        (
            axum::http::StatusCode::INTERNAL_SERVER_ERROR,
            format!("{e}"),
        )
    })?;

    let decoding_key = DecodingKey::from_rsa_components(&jwk.n, &jwk.e).map_err(|e| {
        (
            axum::http::StatusCode::INTERNAL_SERVER_ERROR,
            format!("invalid key: {e}"),
        )
    })?;

    let mut validation = Validation::new(Algorithm::RS256);
    validation.set_audience(&[&state.google_client_id]);
    validation.set_issuer(&["https://accounts.google.com", "accounts.google.com"]);

    // --- verify token ---
    let data =
        decode::<GoogleClaims>(&payload.id_token, &decoding_key, &validation).map_err(|e| {
            (
                axum::http::StatusCode::UNAUTHORIZED,
                format!("verify failed: {e}"),
            )
        })?;

    let claims = data.claims;

    // --- UPSERT user by google:sub ---
    let user = upsert_google_user(&state.db, &claims)
        .await
        .map_err(|e| (axum::http::StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    // --- REGISTER DEVICE FOR THIS USER ---
    if !payload.device_hash.is_empty() {
        let _ = state
            .db
            .add_device_for_user(&user.id, &payload.device_hash)
            .await;
    }

    // --- Issue our own JWT ---
    let exp = chrono::Utc::now().timestamp() as usize + 60 * 60 * 24 * 7;
    let my_claims = AppClaims {
        sub: user.id.clone(),
        exp,
    };

    let jwt = jsonwebtoken::encode(
        &Header::default(),
        &my_claims,
        &EncodingKey::from_secret(state.jwt_secret.as_bytes()),
    )
    .unwrap();

    Ok(Json(AuthResponse {
        jwt,
        user_id: user.id,
        email: claims.email,
    }))
}

async fn upsert_google_user(db: &DBLayer, claims: &GoogleClaims) -> anyhow::Result<User> {
    let provider_id = format!("google:{}", claims.sub);
    let email_ref = claims.email.as_deref();

    let all_users = db.list_users().await?;

    if let Some(user) = all_users.iter().find(|u| {
        if let Some(meta) = &u.meta {
            if let Some(methods) = meta.get("auth_methods").and_then(|v| v.as_array()) {
                return methods.iter().any(|m| m.as_str() == Some(&provider_id));
            }
        }
        false
    }) {
        return Ok(user.clone());
    }

    if let Some(email) = email_ref {
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

    let now = chrono::Utc::now().timestamp();
    let email_owned = claims.email.clone();
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
