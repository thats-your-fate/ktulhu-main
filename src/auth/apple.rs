use std::time::{SystemTime, UNIX_EPOCH};

use axum::{extract::State, Json};
use jsonwebtoken::{decode_header, Algorithm, DecodingKey, EncodingKey, Header, Validation};
use serde::{Deserialize, Serialize};
use serde_json::json;
use uuid::Uuid;

use crate::{
    db::DBLayer,
    model::user::{User, UserRole},
    ws::AppState,
};

#[derive(Deserialize)]
pub struct AppleAuthRequest {
    pub id_token: String,
}

#[derive(Serialize)]
pub struct AuthResponse {
    pub jwt: String,
    pub user_id: String,
    pub email: Option<String>,
}

// Apple ID token claims (simplified)
#[derive(Debug, Deserialize)]
struct AppleIdClaims {
    iss: String,
    sub: String,
    aud: String,
    exp: usize,
    email: Option<String>,
    email_verified: Option<String>,
}

// Your own JWT claims for clients
#[derive(Debug, Serialize, Deserialize)]
struct JwtClaims {
    sub: String,
    exp: usize,
}

#[derive(Debug, Deserialize)]
struct JwkSet {
    keys: Vec<Jwk>,
}

#[derive(Debug, Deserialize)]
struct Jwk {
    kid: String,
    kty: String,
    n: String,
    e: String,
    alg: String,
}

pub async fn apple_login_handler(
    State(state): State<AppState>,
    Json(payload): Json<AppleAuthRequest>,
) -> Result<Json<AuthResponse>, (axum::http::StatusCode, String)> {
    if state.apple_client_id.is_empty() {
        return Err((
            axum::http::StatusCode::BAD_REQUEST,
            "Apple login disabled".into(),
        ));
    }

    // 1) Decode header to find kid
    let header = decode_header(&payload.id_token).map_err(|e| {
        (
            axum::http::StatusCode::UNAUTHORIZED,
            format!("Invalid token header: {e}"),
        )
    })?;

    let kid = header.kid.ok_or_else(|| {
        (
            axum::http::StatusCode::UNAUTHORIZED,
            "Missing kid in token header".to_string(),
        )
    })?;

    if header.alg != Algorithm::RS256 {
        return Err((
            axum::http::StatusCode::UNAUTHORIZED,
            "Unsupported alg".into(),
        ));
    }

    // 2) Fetch Apple's JWKS
    let jwks: JwkSet = reqwest::get("https://appleid.apple.com/auth/keys")
        .await
        .map_err(|e| {
            (
                axum::http::StatusCode::BAD_GATEWAY,
                format!("JWKS fetch error: {e}"),
            )
        })?
        .json()
        .await
        .map_err(|e| {
            (
                axum::http::StatusCode::BAD_GATEWAY,
                format!("JWKS parse error: {e}"),
            )
        })?;

    let jwk = jwks
        .keys
        .into_iter()
        .find(|k| k.kid == kid)
        .ok_or_else(|| {
            (
                axum::http::StatusCode::UNAUTHORIZED,
                "No matching JWK".into(),
            )
        })?;

    // 3) Build RSA decoding key
    let decoding_key = DecodingKey::from_rsa_components(&jwk.n, &jwk.e).map_err(|e| {
        (
            axum::http::StatusCode::INTERNAL_SERVER_ERROR,
            format!("invalid key: {e}"),
        )
    })?;

    // 4) Validate token
    let mut validation = Validation::new(Algorithm::RS256);
    validation.set_issuer(&["https://appleid.apple.com"]);
    validation.set_audience(&[&state.apple_client_id]);

    let token_data =
        jsonwebtoken::decode::<AppleIdClaims>(&payload.id_token, &decoding_key, &validation)
            .map_err(|e| {
                (
                    axum::http::StatusCode::UNAUTHORIZED,
                    format!("Token verify failed: {e}"),
                )
            })?;

    let claims = token_data.claims;

    // 5) Load or create user
    let user = upsert_apple_user(&state.db, &claims).await.map_err(|e| {
        (
            axum::http::StatusCode::INTERNAL_SERVER_ERROR,
            format!("DB error: {e}"),
        )
    })?;

    // 6) Issue your own JWT
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs() as usize;

    let exp = now + 60 * 60 * 24 * 7; // 7 days
    let app_claims = JwtClaims {
        sub: user.id.clone(),
        exp,
    };

    let jwt = jsonwebtoken::encode(
        &Header::default(),
        &app_claims,
        &EncodingKey::from_secret(state.jwt_secret.as_bytes()),
    )
    .map_err(|e| {
        (
            axum::http::StatusCode::INTERNAL_SERVER_ERROR,
            format!("JWT sign error: {e}"),
        )
    })?;

    Ok(Json(AuthResponse {
        jwt,
        user_id: user.id,
        email: user.external_id.or(user
            .meta
            .as_ref()
            .and_then(|m| m["email"].as_str().map(|s| s.to_string()))),
    }))
}

pub async fn upsert_apple_user(db: &DBLayer, claims: &AppleIdClaims) -> anyhow::Result<User> {
    let provider_id = format!("apple:{}", claims.sub);
    let email_ref = claims.email.as_deref();

    let users = db.list_users().await?;

    if let Some(user) = users.iter().find(|u| {
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
        if let Some(existing) = users.iter().find(|u| {
            u.meta
                .as_ref()
                .and_then(|m| m.get("email"))
                .and_then(|v| v.as_str())
                == Some(email)
        }) {
            let mut merged = existing.clone();
            let mut meta = merged.meta.clone().unwrap_or(json!({}));
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
            merged.meta = Some(meta);
            if merged.external_id.is_none() {
                merged.external_id = Some(provider_id.clone());
            }
            merged.email = Some(email.to_string());

            db.save_user(&merged).await?;
            return Ok(merged);
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
        api_key: None,
        api_secret: None,
        generation_count: 0,
        role: UserRole::Free,
    };

    db.save_user(&user).await?;
    Ok(user)
}
