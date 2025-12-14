use anyhow::Result;
use jsonwebtoken::{encode, EncodingKey, Header};
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
