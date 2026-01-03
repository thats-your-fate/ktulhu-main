pub mod byte_decoder;
pub mod llama_cpp_service;
pub mod roberta_classifier;
pub mod roberta_phatic_gate;

use std::sync::Arc;

use llama_cpp_service::LlamaCppService;

pub struct InferenceService {
    engine: Arc<LlamaCppService>,
}

impl InferenceService {
    pub fn new(engine: Arc<LlamaCppService>) -> Self {
        Self { engine }
    }

    pub fn generate_stream(
        &self,
        prompt: String,
        cancel: Arc<std::sync::atomic::AtomicBool>,
    ) -> tokio::sync::mpsc::Receiver<String> {
        self.engine.generate_stream(prompt, cancel)
    }

    pub async fn generate_completion(
        &self,
        prompt: String,
        cancel: Arc<std::sync::atomic::AtomicBool>,
    ) -> anyhow::Result<String> {
        self.engine.generate_completion(prompt, cancel).await
    }
}
