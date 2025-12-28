use serde::{Deserialize, Serialize};

#[derive(Deserialize)]
pub struct EmailRegisterRequest {
    pub email: String,
    pub password: String,
    pub device_hash: Option<String>,
}

#[derive(Deserialize)]
pub struct EmailLoginRequest {
    pub email: String,
    pub password: String,
    pub device_hash: Option<String>,
}

#[derive(Serialize)]
pub struct EmailAuthResponse {
    pub jwt: String,
    pub user_id: String,
    pub email: String,
}
