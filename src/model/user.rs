use serde::{Deserialize, Serialize};

pub const FREE_GENERATION_LIMIT: u64 = 10;

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
        matches!(self, UserRole::Free | UserRole::Paid | UserRole::Admin)
    }

    pub fn generation_limit(&self) -> Option<u64> {
        match self {
            UserRole::Free => Some(FREE_GENERATION_LIMIT),
            UserRole::Paid | UserRole::Admin => None,
        }
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
    pub api_key: Option<String>,
    #[serde(default)]
    pub api_secret: Option<String>,
    #[serde(default)]
    pub generation_count: u64,
    #[serde(default)]
    pub role: UserRole,
    #[serde(default)]
    pub stripe_customer_id: Option<String>,
    #[serde(default)]
    pub stripe_subscription_id: Option<String>,
}

impl User {
    pub fn generation_limit(&self) -> Option<u64> {
        self.role.generation_limit()
    }

    pub fn generations_remaining(&self) -> Option<u64> {
        self.generation_limit()
            .map(|limit| limit.saturating_sub(self.generation_count.min(limit)))
    }

    pub fn can_generate_now(&self) -> bool {
        if !self.role.can_access_generation() {
            return false;
        }
        match self.generation_limit() {
            Some(limit) => self.generation_count < limit,
            None => true,
        }
    }
}
