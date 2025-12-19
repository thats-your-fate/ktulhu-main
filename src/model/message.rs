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
    #[serde(default)]
    pub language: Option<String>,
    #[serde(default)]
    pub attachments: Vec<MessageAttachment>,
    pub ts: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MessageAttachment {
    pub id: String,
    pub filename: String,
    #[serde(default)]
    pub mime_type: Option<String>,
    pub path: String,
    pub size: usize,
}
