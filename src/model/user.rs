use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct User {
    pub id: String,
    pub name: Option<String>,
    pub external_id: Option<String>,
    pub created_ts: i64,
    pub meta: Option<serde_json::Value>,
    pub email: Option<String>,
    pub password_hash: Option<String>,
}
