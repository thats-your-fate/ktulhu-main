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
    #[serde(default)]
    pub liked: bool,
    pub ts: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MessageAttachment {
    pub id: String,
    pub filename: String,
    #[serde(default)]
    pub mime_type: Option<String>,
    #[serde(default)]
    pub preview_base64: Option<String>,
    #[serde(default)]
    pub path: Option<String>,
    #[serde(default)]
    pub size: Option<usize>,
    #[serde(default)]
    pub description: Option<String>,
    #[serde(default)]
    pub ocr_text: Option<String>,
    #[serde(default)]
    pub labels: Vec<String>,
}
