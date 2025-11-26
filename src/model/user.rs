use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct User {
    /// Unique user identifier
    pub id: String,

    /// Username or display name
    pub name: Option<String>,

    /// Optional hashed device id if you want anonymous device users
    pub device_hash: Option<String>,

    /// Email, phone, or external id (optional)
    pub external_id: Option<String>,

    /// Timestamp of creation
    pub created_ts: i64,

    /// Free-form JSON metadata (roles, settings, preferences)
    pub meta: Option<serde_json::Value>,
}
