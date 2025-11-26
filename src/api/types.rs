use crate::model::message::Message;
use serde::{Deserialize, Serialize};

#[derive(Debug, Deserialize)]
pub struct GenerateRequest {
    pub prompt: String,
    #[serde(default)]
    pub max_tokens: usize,
}

#[derive(Debug, Serialize)]
pub struct GenerateResponse {
    pub output: String,
}

#[derive(Debug, Serialize)]
pub struct MessagesResponse {
    pub chat_id: String,
    pub messages: Vec<Message>,
}
