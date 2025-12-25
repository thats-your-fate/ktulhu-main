use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum UserRole {
    Free,
    Paid,
    Admin,
}

impl Default for UserRole {
    fn default() -> Self {
        UserRole::Free
    }
}

impl UserRole {
    pub fn can_access_generation(&self) -> bool {
        matches!(self, UserRole::Paid | UserRole::Admin)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct User {
    pub id: String,
    pub name: Option<String>,
    pub external_id: Option<String>,
    pub created_ts: i64,
    pub meta: Option<serde_json::Value>,
    pub email: Option<String>,
    pub password_hash: Option<String>,
    #[serde(default)]
    pub role: UserRole,
}
