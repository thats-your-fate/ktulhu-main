use anyhow::{anyhow, Result};
use candle_core::{DType, Device, IndexOp, Tensor};
use candle_nn::VarBuilder;
use tokenizers::Tokenizer;
use crate::inference::byte_decoder::tidy_decoded_text;

// ✔️ Candle 0.9.1 Phi-3 model
use candle_transformers::models::phi3::{Config as Phi3Config, Model as Phi3Model};

use std::{fs, path::PathBuf, sync::Arc};
use tokio::sync::Mutex;

// ---------------------------------------------------------
// Generic config wrapper: we forward raw JSON → Phi3Config
// ---------------------------------------------------------
#[derive(Debug, Clone, serde::Deserialize)]
pub struct PhiConfig {
    #[serde(flatten)]
    pub inner: serde_json::Value,
}

// ---------------------------------------------------------
// Actual model wrapper
// ---------------------------------------------------------
pub struct PhiModel {
    inner: Phi3Model,
}

impl PhiModel {
    pub fn new(cfg: &PhiConfig, vb: VarBuilder) -> candle_core::Result<Self> {
        let cfg: Phi3Config = serde_json::from_value(cfg.inner.clone())
            .map_err(|e| candle_core::Error::Msg(format!("Phi3 config parse error: {e}")))?;

        let model = Phi3Model::new(&cfg, vb)?;
        Ok(Self { inner: model })
    }

    pub fn forward(&mut self, x: &Tensor, pos: usize) -> candle_core::Result<Tensor> {
        // pos = seqlen_offset (same as Mistral)
        self.inner.forward(x, pos)
    }

    pub fn clear_kv_cache(&mut self) {
        self.inner.clear_kv_cache();
    }
}

// ---------------------------------------------------------
// PUBLIC SERVICE: INTERNAL-ONLY WORKER MODEL (GPU1)
// ---------------------------------------------------------
pub struct PhiService {
    pub model: Arc<Mutex<PhiModel>>,
    pub tokenizer: Arc<Tokenizer>,
    pub device: Device,
}

impl PhiService {
    /// Constructor used by your ModelManager:
    /// `PhiService::new_with(PathBuf::from("phi3mini"), 1)`
    pub async fn new_with(snapshot_dir: PathBuf, device_id: usize) -> Result<Self> {
        let device = Device::new_cuda(device_id)?;
        println!("🐦 Phi-3 → Using CUDA:{device_id} ({device:?})");
        println!("📁 Phi snapshot: {}", snapshot_dir.display());

        // 1) Tokenizer
        let tokenizer_path = snapshot_dir.join("tokenizer.json");
        let tokenizer = Arc::new(
            Tokenizer::from_file(&tokenizer_path)
                .map_err(|e| anyhow!("Phi tokenizer error: {e}"))?,
        );

        // 2) Config
        let config_path = snapshot_dir.join("config.json");
        let cfg: PhiConfig = serde_json::from_slice(&fs::read(&config_path)?)?;

        // 3) Shards from index
        let index_path = snapshot_dir.join("model.safetensors.index.json");
        let index_json: serde_json::Value = serde_json::from_slice(&fs::read(&index_path)?)?;

        let shards = index_json["weight_map"]
            .as_object()
            .ok_or_else(|| anyhow!("Phi index.json: weight_map is not an object"))?
            .values()
            .map(|v| {
                let file = v
                    .as_str()
                    .ok_or_else(|| anyhow!("invalid shard entry in Phi index.json"))?;
                Ok(snapshot_dir.join(file))
            })
            .collect::<Result<Vec<_>>>()?;

        println!("📦 Phi-3 shards: {}", shards.len());

        // 4) mmap weights
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&shards, DType::F16, &device)? };

        // 5) Init model
        let model = Arc::new(Mutex::new(PhiModel::new(&cfg, vb)?));

        println!("🚀 Phi-3 Mini loaded on CUDA:{device_id} (internal worker)");

        Ok(Self {
            model,
            tokenizer,
            device,
        })
    }

    // -----------------------------------------------------------------
    // LOW-LEVEL: core generation (greedy, internal use only)
    // -----------------------------------------------------------------
    pub async fn generate_greedy(&self, prompt: &str, max_new_tokens: usize) -> Result<String> {
        // Clear KV cache for each job
        {
            let mut m = self.model.lock().await;
            m.clear_kv_cache();
        }

        // Encode
        let enc = self
            .tokenizer
            .encode(prompt, true)
            .map_err(|e| anyhow!("Phi encode error: {e}"))?;

        let mut tokens = enc.get_ids().to_vec();
        let prompt_len = tokens.len();

        // EOS handling: adapt to Phi tokenizer (tweak if needed)
        let eos = self
            .tokenizer
            .token_to_id("<eos>")
            .or_else(|| self.tokenizer.token_to_id("</s>"))
            .unwrap_or(u32::MAX);

        let mut pos = 0usize;

        for _ in 0..max_new_tokens {
            // first step: whole prompt; later: last token only
            let ctx: &[u32] = if pos == 0 {
                &tokens
            } else {
                std::slice::from_ref(tokens.last().unwrap())
            };

            let input = Tensor::new(ctx, &self.device)?.unsqueeze(0)?;

            // forward
            let logits = {
                let mut m = self.model.lock().await;
                let out = m.forward(&input, pos)?;
                let seq_len = out.dim(1)?;
                out.i((0, seq_len - 1))?.to_dtype(DType::F32)?
            };

            pos += ctx.len();

            // greedy argmax
            let next = argmax_u32(&logits)?.ok_or_else(|| anyhow!("Phi: empty logits"))?;
            tokens.push(next);

            if next == eos {
                break;
            }
        }

        let gen_slice = &tokens[prompt_len..];
        if gen_slice.is_empty() {
            return Ok(String::new());
        }

        let text = self
            .tokenizer
            .decode(gen_slice, false)
            .map_err(|e| anyhow!("Phi decode error: {e}"))?;

        Ok(tidy_decoded_text(&text))
    }

    // -----------------------------------------------------------------
    // HIGH-LEVEL HELPERS FOR INTERNAL TASKS
    // -----------------------------------------------------------------

    /// Short, neutral summary. Perfect for internal news / doc summaries.
    #[allow(dead_code)]
    pub async fn summarize(&self, text: &str, max_tokens: usize) -> Result<String> {
        let prompt = format!(
            "Summarize the following text clearly and concisely in a few sentences:\n\n{}",
            text
        );
        self.generate_greedy(&prompt, max_tokens).await
    }

    /// Run a custom prompt verbatim (no additional template).
    pub async fn generate_with_prompt(&self, prompt: &str, max_tokens: usize) -> Result<String> {
        self.generate_greedy(prompt, max_tokens).await
    }

    /// Extract up to N keywords, returns them as Vec<String>.
    #[allow(dead_code)]
    pub async fn extract_keywords(&self, text: &str, max_keywords: usize) -> Result<Vec<String>> {
        let prompt = format!(
            "Extract up to {} short keywords from the following text, separated by commas:\n\n{}",
            max_keywords, text
        );

        let raw = self.generate_greedy(&prompt, 128).await?;
        let keywords = raw
            .split(',')
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect::<Vec<_>>();

        Ok(keywords)
    }

    /// Classify into one of several labels (e.g. politics, tech, sports).
    #[allow(dead_code)]
    pub async fn classify_into(&self, text: &str, labels: &[&str]) -> Result<String> {
        let label_list = labels.join(", ");
        let prompt = format!(
            "Classify the following text into exactly ONE of the following labels: {labels}.\n\
             Reply with ONLY the label:\n\n{text}",
            labels = label_list,
            text = text
        );

        let raw = self.generate_greedy(&prompt, 32).await?;
        // very light post-processing: take first matching label
        for &label in labels {
            if raw.to_lowercase().contains(&label.to_lowercase()) {
                return Ok(label.to_string());
            }
        }
        Ok(raw.trim().to_string())
    }
}

// ---------------------------------------------------------
// Helpers
// ---------------------------------------------------------

/// Argmax over last dimension, returning token id as u32.
/// Assumes logits is 1D (vocab) or squeezable to 1D.
fn argmax_u32(logits: &Tensor) -> Result<Option<u32>> {
    let v = logits.flatten_all()?.to_vec1::<f32>()?;
    if v.is_empty() {
        return Ok(None);
    }

    let mut max_idx = 0usize;
    let mut max_val = v[0];
    for (i, &val) in v.iter().enumerate().skip(1) {
        if val > max_val {
            max_val = val;
            max_idx = i;
        }
    }

    Ok(Some(max_idx as u32))
}
