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
    pub mistral_reasoning: Option<Arc<Ministral8BResoningService>>,
    pub ministral8_b: Option<Arc<Ministral8BService>>,
    pub phi: Arc<PhiService>,
}

impl InferenceService {
    pub fn new(
        mistral_reasoning: Option<Arc<Ministral8BResoningService>>,
        ministral8_b: Option<Arc<Ministral8BService>>,
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
        if let Some(service) = &self.mistral_reasoning {
            service.generate_stream(prompt, cancel)
        } else if let Some(service) = &self.ministral8_b {
            service.generate_stream(prompt, cancel)
        } else {
            panic!("No Mistral service loaded");
        }
    }

    pub async fn generate_completion(
        &self,
        prompt: String,
        cancel: std::sync::Arc<std::sync::atomic::AtomicBool>,
    ) -> anyhow::Result<String> {
        if let Some(service) = &self.mistral_reasoning {
            service.generate_completion(prompt, cancel).await
        } else if let Some(service) = &self.ministral8_b {
            service.generate_completion(prompt, cancel).await
        } else {
            anyhow::bail!("No Mistral service loaded");
        }
    }
}
