pub mod byte_decoder;
pub mod ministral8_b;
pub mod ministral_8b_resoning;
pub mod phi;
pub mod roberta;

use std::sync::Arc;

use ministral8_b::Ministral8BService;
use ministral_8b_resoning::Ministral8BResoningService;
use phi::PhiService;

pub struct InferenceService {
    #[allow(dead_code)]
    pub ministral8_b: Arc<Ministral8BService>,
    pub mistral_reasoning: Arc<Ministral8BResoningService>,
    pub phi: Arc<PhiService>,
}

impl InferenceService {
    pub fn new(
        mistral_reasoning: Arc<Ministral8BResoningService>,
        ministral8_b: Arc<Ministral8BService>,
        phi: Arc<PhiService>,
    ) -> Self {
        Self {
            mistral_reasoning,
            ministral8_b,
            phi,
        }
    }

    pub fn generate_stream(
        &self,
        prompt: String,
        cancel: std::sync::Arc<std::sync::atomic::AtomicBool>,
    ) -> tokio::sync::mpsc::Receiver<String> {
        self.mistral_reasoning.generate_stream(prompt, cancel)
    }

    pub async fn generate_completion(
        &self,
        prompt: String,
        cancel: std::sync::Arc<std::sync::atomic::AtomicBool>,
    ) -> anyhow::Result<String> {
        self.mistral_reasoning.generate_completion(prompt, cancel).await
    }
}
