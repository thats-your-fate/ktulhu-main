use anyhow::Result;
use jsonwebtoken::{decode, encode, DecodingKey, EncodingKey, Header, Validation};
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize)]
struct Claims {
    sub: String,
    exp: usize,
}

pub fn make_jwt(user_id: &str) -> Result<String> {
    let exp = chrono::Utc::now().timestamp() as usize + 60 * 60 * 24 * 30; // 30 days

    let claims = Claims {
        sub: user_id.to_string(),
        exp,
    };

    let key = std::env::var("JWT_SECRET")?;
    let token = encode(
        &Header::default(),
        &claims,
        &EncodingKey::from_secret(key.as_bytes()),
    )?;
    Ok(token)
}

pub fn decode_jwt(token: &str, secret: &str) -> Result<String> {
    let data = decode::<Claims>(
        token,
        &DecodingKey::from_secret(secret.as_bytes()),
        &Validation::default(),
    )?;
    Ok(data.claims.sub)
}
