use async_trait::async_trait;

use axum::{
    extract::{FromRequestParts, Extension},
    http::{request::Parts, StatusCode},
};
use axum_extra::headers::{
    Authorization,
    authorization::Bearer,
};
use axum_extra::TypedHeader;

use serde::{Deserialize, Serialize};
use jsonwebtoken::{decode, DecodingKey, Validation};

#[derive(Clone)]
pub struct JwtState {
    pub secret: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Claims {
    pub sub: String,
    pub exp: usize,
}

pub struct AuthenticatedUser(pub Claims);

#[async_trait]
impl<S> FromRequestParts<S> for AuthenticatedUser
where
    S: Send + Sync,
{
    type Rejection = (StatusCode, &'static str);

    async fn from_request_parts(
        parts: &mut Parts,
        state: &S,
    ) -> Result<Self, Self::Rejection> {

        // -----------------------------------------
        // 1. Load JwtState from extensions
        // -----------------------------------------
        let Extension(jwt): Extension<JwtState> =
            Extension::from_request_parts(parts, state)
                .await
                .map_err(|_| (StatusCode::UNAUTHORIZED, "Missing JWT state"))?;

        // -----------------------------------------
        // 2. Extract Authorization header
        //    (AUTHORS OF AXUM-EXTRA 0.9.6 REMOVED FromRequest FOR TYPEDHEADER)
        //    → MUST USE Authorization<Bearer> DIRECTLY
        // -----------------------------------------
        let TypedHeader(Authorization(bearer)) =
            TypedHeader::<Authorization<Bearer>>::from_request_parts(parts, state)
                .await
                .map_err(|_| (StatusCode::UNAUTHORIZED, "Missing Authorization header"))?;

        // -----------------------------------------
        // 3. Decode token
        // -----------------------------------------
        let data = decode::<Claims>(
            bearer.token(),
            &DecodingKey::from_secret(jwt.secret.as_bytes()),
            &Validation::default(),
        )
        .map_err(|_| (StatusCode::UNAUTHORIZED, "Invalid or expired token"))?;

        Ok(AuthenticatedUser(data.claims))
    }
}
