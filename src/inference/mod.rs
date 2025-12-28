pub mod byte_decoder;
pub mod mistral;
pub mod phi;
pub mod roberta;

use std::sync::Arc;

use mistral::MistralService;
use phi::PhiService;

pub struct InferenceService {
    pub mistral: Arc<MistralService>,
    pub phi: Arc<PhiService>,
}

impl InferenceService {
    pub fn new(
        mistral: Arc<MistralService>,
        phi: Arc<PhiService>,
    ) -> Self {
        Self {
            mistral,
            phi,
        }
    }

    pub fn generate_stream(
        &self,
        prompt: String,
        cancel: std::sync::Arc<std::sync::atomic::AtomicBool>,
    ) -> tokio::sync::mpsc::Receiver<String> {
        self.mistral.generate_stream(prompt, cancel)
    }

    pub async fn generate_completion(
        &self,
        prompt: String,
        cancel: std::sync::Arc<std::sync::atomic::AtomicBool>,
    ) -> anyhow::Result<String> {
        self.mistral.generate_completion(prompt, cancel).await
    }

}
