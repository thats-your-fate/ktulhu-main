use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Chat {
    pub id: String,
    pub title: Option<String>,
    pub user_id: Option<String>,
    pub device_hash: Option<String>,
    pub updated_ts: i64,
    pub meta: Option<serde_json::Value>,
}
