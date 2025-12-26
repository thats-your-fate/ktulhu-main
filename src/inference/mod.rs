pub mod byte_decoder;
pub mod mistral;
pub mod phi;
pub mod roberta;

use std::sync::Arc;

use mistral::MistralService;
use phi::PhiService;
use roberta::RobertaService;

pub struct InferenceService {
    pub mistral: Arc<MistralService>,
    pub phi: Arc<PhiService>,
    pub roberta: Arc<RobertaService>,
}

impl InferenceService {
    pub fn new(
        mistral: Arc<MistralService>,
        phi: Arc<PhiService>,
        roberta: Arc<RobertaService>,
    ) -> Self {
        Self {
            mistral,
            phi,
            roberta,
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

    // --------------------------
    // Internal summarization (Phi-3)
    // --------------------------
    pub async fn summarize_with_phi(&self, text: &str) -> anyhow::Result<String> {
        self.phi.summarize(text, 160).await
    }

    // --------------------------
    // Classification (RoBERTa)
    // --------------------------
    pub async fn classify_with_roberta(&self, text: &str) -> anyhow::Result<String> {
        let (label, _) = self
            .roberta
            .classify(text, &["positive", "neutral", "negative"])
            .await?;
        Ok(label)
    }
}
