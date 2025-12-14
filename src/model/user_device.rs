use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserDevice {
    pub id: String,          // UUID
    pub user_id: String,     // FK → User.id
    pub device_hash: String, // generated on frontend
    pub created_ts: i64,
    pub last_seen_ts: i64,
    pub meta: Option<serde_json::Value>, // optional device info (browser, OS, model)
}
