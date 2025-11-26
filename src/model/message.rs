use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub id: String,
    pub chat_id: String,
    pub session_id: Option<String>,
    pub user_id: Option<String>,
    pub device_hash: Option<String>,
    pub role: String,
    pub text: Option<String>,
    pub ts: i64,
}
