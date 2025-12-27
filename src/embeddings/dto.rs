use serde::{Deserialize, Serialize};

#[derive(Deserialize)]
pub struct EmbedRequest {
    pub texts: Vec<String>,
}

#[derive(Serialize)]
pub struct EmbedResponse {
    pub model: String,
    pub dims: usize,
    pub embeddings: Vec<Vec<f32>>,
}
