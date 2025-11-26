use serde::{Deserialize, Serialize};

#[derive(Debug, Deserialize)]
pub struct InternalGenerateRequest {
    pub prompt: String,
    #[serde(default)]
    pub max_tokens: usize,
}

#[derive(Debug, Serialize)]
pub struct InternalGenerateResponse {
    pub output: String,
}
