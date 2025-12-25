use axum::{extract::State, http::StatusCode, Json};
use chrono::Utc;
use jsonwebtoken::{decode, decode_header, Algorithm, Validation};
use serde::{Deserialize, Serialize};

use crate::{
    model::user::{User, UserRole},
    ws::AppState,
};
use super::google_keys::*;

#[derive(Debug, Deserialize, Serialize)]
pub struct GoogleClaims {
    pub iss: String,
    pub sub: String,
    pub email: Option<String>,
    pub email_verified: Option<bool>,
    pub name: Option<String>,
    pub picture: Option<String>,
    pub aud: String,
    pub exp: usize,
}

#[derive(Deserialize)]
pub struct GoogleAuthRequest {
    pub id_token: String,
}

#[derive(serde::Serialize)]
pub struct GoogleAuthResponse {
    pub jwt: String,
    pub user: User,
}

pub async fn verify_google_token(
    state: &AppState,
    id_token: &str,
) -> anyhow::Result<GoogleClaims> {
    let header = decode_header(id_token)?;
    let kid = header.kid.ok_or_else(|| anyhow::anyhow!("KID missing"))?;

    let jwks = fetch_google_jwks().await?;
    let jwk = find_key(&jwks, &kid).ok_or_else(|| anyhow::anyhow!("Unknown JWK kid"))?;

    let key = to_decoding_key(jwk)?;

    let mut validation = Validation::new(Algorithm::RS256);
    validation.set_audience(&[state.google_client_id.clone()]);
    validation.set_issuer(&["https://accounts.google.com", "accounts.google.com"]);

    let data = decode::<GoogleClaims>(id_token, &key, &validation)?;
    Ok(data.claims)
}

pub async fn google_auth_handler(
    State(state): State<AppState>,
    Json(req): Json<GoogleAuthRequest>,
) -> Result<Json<GoogleAuthResponse>, (StatusCode, String)> {
    let claims = verify_google_token(&state, &req.id_token)
        .await
        .map_err(|err| (StatusCode::UNAUTHORIZED, err.to_string()))?;

    let user_id = claims.sub.clone();

    // Check if user exists already
    let user = if let Some(u) = state
        .db
        .load_user(&user_id)
        .await
        .map_err(internal_error)?
    {
        u
    } else {
        let new_user = User {
            id: user_id.clone(),
            name: claims.name.clone(),
            email: claims.email.clone(),
            external_id: Some(user_id.clone()),
            created_ts: Utc::now().timestamp(),
            meta: None,
            password_hash: None,
            role: UserRole::Free,
        };
        state
            .db
            .save_user(&new_user)
            .await
            .map_err(internal_error)?;
        new_user
    };

    // Create server JWT (signed with backend JWT_SECRET)
    let jwt = jsonwebtoken::encode(
        &jsonwebtoken::Header::default(),
        &claims,
        &jsonwebtoken::EncodingKey::from_secret(state.jwt_secret.as_bytes()),
    )
    .map_err(internal_error)?;

    Ok(Json(GoogleAuthResponse { jwt, user }))
}

fn internal_error(err: impl std::fmt::Display) -> (StatusCode, String) {
    (StatusCode::INTERNAL_SERVER_ERROR, err.to_string())
}
