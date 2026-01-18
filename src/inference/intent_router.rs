use anyhow::{anyhow, Context, Result};
use candle::{DType, Device, Module, Tensor};
use candle_nn::{Linear, VarBuilder};
use candle_transformers::models::xlm_roberta::{Config, XLMRobertaModel};
use std::fs;
use std::path::{Path, PathBuf};
use tokenizers::Tokenizer;
use tracing::warn;

pub struct IntentLogits {
    pub phatic: Option<Vec<f32>>,
    pub speech_act: Vec<f32>,
    pub domain: Vec<f32>,
    pub expectation: Vec<f32>,
    pub support: Option<Vec<f32>>,
}

pub struct RobertaIntentRouter {
    model: RouterModel,
    tokenizer: Tokenizer,
    device: Device,
    max_len: usize,
    include_phatic: bool,
}

impl RobertaIntentRouter {
    pub fn load(snapshot: PathBuf, device_id: usize, with_phatic: bool) -> Result<Self> {
        let tokenizer_path = snapshot.join("tokenizer.json");
        if !tokenizer_path.exists() {
            return Err(anyhow!(
                "tokenizer.json not found under {}",
                snapshot.display()
            ));
        }

        let config = load_config(&snapshot)?;
        let mut tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| anyhow!("Tokenizer load failed ({}): {e}", tokenizer_path.display()))?;
        tokenizer.with_padding(None);
        let _ = tokenizer.with_truncation(None);

        let weights_path = find_model_weights(&snapshot)
            .ok_or_else(|| anyhow!("no model weights found under {}", snapshot.display()))?;

        let max_len = std::env::var("INTENT_ROUTER_SEQ_LEN")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .map(|len| len.min(config.max_position_embeddings))
            .unwrap_or(config.max_position_embeddings);

        let device = build_device(device_id)?;
        let dtype = DType::F16;
        let vb = build_var_builder(&weights_path, dtype, &device)?;
        let model = RouterModel::load(&config, vb, with_phatic)?;
        let include_phatic = with_phatic && model.has_phatic();

        Ok(Self {
            model,
            tokenizer,
            device,
            max_len,
            include_phatic,
        })
    }

    pub fn classify(&self, text: &str) -> Result<IntentLogits> {
        let (ids, non_padding_len) = tokenize_ids(&self.tokenizer, text, self.max_len)?;
        let seq_len = ids.len();
        let attention_mask: Vec<u32> = (0..seq_len)
            .map(|idx| if idx < non_padding_len { 1 } else { 0 })
            .collect();
        let token_type_ids = vec![0u32; seq_len];

        let ids_tensor = tensor_from_slice(&ids, seq_len, &self.device)?;
        let mask_tensor = tensor_from_slice(&attention_mask, seq_len, &self.device)?;
        let tt_tensor = tensor_from_slice(&token_type_ids, seq_len, &self.device)?;

        let outputs = self
            .model
            .forward(&ids_tensor, &mask_tensor, &tt_tensor)
            .context("intent router forward pass failed")?;

        let speech_act = tensor_to_vec(outputs.speech_act)?;
        let domain = tensor_to_vec(outputs.domain)?;
        let expectation = tensor_to_vec(outputs.expectation)?;
        let support = outputs.support.map(tensor_to_vec).transpose()?;
        let phatic_logits = if self.include_phatic {
            outputs.phatic.map(tensor_to_vec).transpose()?
        } else {
            None
        };

        Ok(IntentLogits {
            phatic: phatic_logits,
            speech_act,
            domain,
            expectation,
            support,
        })
    }
}

struct RouterModel {
    roberta: XLMRobertaModel,
    phatic: Option<RouterHead>,
    speech: RouterHead,
    domain: RouterHead,
    expectation: RouterHead,
    support: Option<RouterHead>,
    weight_dtype: DType,
}

struct RouterHead {
    dense: Linear,
    classifier: Linear,
}

struct RouterOutputs {
    speech_act: Tensor,
    domain: Tensor,
    expectation: Tensor,
    phatic: Option<Tensor>,
    support: Option<Tensor>,
}

impl RouterModel {
    fn load(cfg: &Config, vb: VarBuilder, include_phatic: bool) -> Result<Self> {
        let roberta = XLMRobertaModel::new(cfg, vb.pp("roberta"))?;
        let phatic = if include_phatic {
            RouterHead::maybe_new(&vb.pp("phatic_head"))?
        } else {
            None
        };
        let speech = RouterHead::new(&vb.pp("speech_head"))?;
        let domain = RouterHead::new(&vb.pp("domain_head"))?;
        let expectation = RouterHead::new(&vb.pp("expectation_head"))?;
        let support = RouterHead::maybe_new(&vb.pp("support_head"))?;

        Ok(Self {
            roberta,
            phatic,
            speech,
            domain,
            expectation,
            support,
            weight_dtype: vb.dtype(),
        })
    }

    fn has_phatic(&self) -> bool {
        self.phatic.is_some()
    }

    fn forward(
        &self,
        input_ids: &Tensor,
        attention_mask: &Tensor,
        token_type_ids: &Tensor,
    ) -> candle::Result<RouterOutputs> {
        let hidden =
            self.roberta
                .forward(input_ids, attention_mask, token_type_ids, None, None, None)?;
        let mut features = pool_features(&hidden, attention_mask)?;
        if features.dtype() != self.weight_dtype {
            features = features.to_dtype(self.weight_dtype)?;
        }
        let speech_act = self.speech.forward(&features)?;
        let domain = self.domain.forward(&features)?;
        let expectation = self.expectation.forward(&features)?;
        let phatic = match &self.phatic {
            Some(head) => Some(head.forward(&features)?),
            None => None,
        };
        let support = match &self.support {
            Some(head) => Some(head.forward(&features)?),
            None => None,
        };

        Ok(RouterOutputs {
            speech_act,
            domain,
            expectation,
            phatic,
            support,
        })
    }
}

impl RouterHead {
    fn new(vb: &VarBuilder) -> Result<Self> {
        let dense = load_linear(&vb.pp("0"))?;
        let classifier = load_linear(&vb.pp("2"))?;
        Ok(Self { dense, classifier })
    }

    fn maybe_new(vb: &VarBuilder) -> Result<Option<Self>> {
        if vb.pp("0").contains_tensor("weight") && vb.pp("2").contains_tensor("weight") {
            Ok(Some(Self::new(vb)?))
        } else {
            Ok(None)
        }
    }

    fn forward(&self, features: &Tensor) -> candle::Result<Tensor> {
        let hidden = self.dense.forward(features)?;
        self.classifier.forward(&hidden.tanh()?)
    }
}

pub fn logits_argmax(logits: &[f32]) -> Result<(usize, f32)> {
    if logits.is_empty() {
        return Err(anyhow!("empty logits tensor"));
    }
    let (idx, value) = logits
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        .ok_or_else(|| anyhow!("empty logits tensor"))?;
    Ok((idx, *value))
}

fn tokenize_ids(tokenizer: &Tokenizer, text: &str, max_len: usize) -> Result<(Vec<u32>, usize)> {
    let enc = tokenizer
        .encode(text, true)
        .map_err(|e| anyhow!("Tokenizer encode error: {e}"))?;
    let mut ids = enc.get_ids().to_vec();
    if ids.is_empty() {
        ids.push(0);
    }
    let mut non_padding_len = ids.len();
    if ids.len() > max_len {
        ids.truncate(max_len);
        non_padding_len = max_len;
    }
    let pad_id = pad_token_id(tokenizer);
    if ids.len() < max_len {
        ids.resize(max_len, pad_id);
    }
    Ok((ids, non_padding_len))
}

fn load_config(snapshot: &Path) -> Result<Config> {
    let path = snapshot.join("config.json");
    if !path.exists() {
        return Err(anyhow!(
            "config.json not found under {}",
            snapshot.display()
        ));
    }
    let raw = fs::read(&path).with_context(|| format!("failed to read {}", path.display()))?;
    Ok(serde_json::from_slice(&raw)
        .with_context(|| format!("failed to parse {}", path.display()))?)
}

fn find_model_weights(snapshot: &Path) -> Option<PathBuf> {
    let candidates = [
        "model.safetensors",
        "pytorch_model.bin",
        "model.bin",
        "adapter_model.bin",
    ];
    for candidate in candidates {
        let path = snapshot.join(candidate);
        if path.exists() {
            return Some(path);
        }
    }
    None
}

fn build_var_builder(path: &Path, dtype: DType, device: &Device) -> Result<VarBuilder<'static>> {
    let ext = path
        .extension()
        .and_then(|s| s.to_str())
        .unwrap_or_default()
        .to_ascii_lowercase();
    if ext == "safetensors" {
        let files = vec![path.to_path_buf()];
        unsafe {
            VarBuilder::from_mmaped_safetensors(&files, dtype, device)
                .map_err(|e| anyhow!("failed to load {}: {e}", path.display()))
        }
    } else {
        VarBuilder::from_pth(path, dtype, device)
            .map_err(|e| anyhow!("failed to load {}: {e}", path.display()))
    }
}

fn tensor_from_slice(data: &[u32], seq_len: usize, device: &Device) -> candle::Result<Tensor> {
    Tensor::new(data, device)?.reshape((1, seq_len))
}

fn tensor_to_vec(tensor: Tensor) -> Result<Vec<f32>> {
    let logits = tensor
        .to_dtype(DType::F32)?
        .to_vec2::<f32>()
        .map_err(|e| anyhow!("failed to decode logits: {e}"))?;
    logits
        .into_iter()
        .next()
        .ok_or_else(|| anyhow!("logits tensor missing batch dimension"))
}

fn load_linear(vb: &VarBuilder) -> Result<Linear> {
    let weight = vb.get_unchecked("weight")?;
    let bias = vb.get_unchecked("bias")?;
    Ok(Linear::new(weight, Some(bias)))
}

fn pool_features(hidden: &Tensor, attention_mask: &Tensor) -> candle::Result<Tensor> {
    let hidden = hidden.to_dtype(DType::F32)?;
    let cls = hidden.narrow(1, 0, 1)?.squeeze(1)?;
    let mask = attention_mask.to_dtype(DType::F32)?;
    let mask_expanded = mask.unsqueeze(2)?;
    let masked_hidden = hidden.broadcast_mul(&mask_expanded)?;
    let sum_hidden = masked_hidden.sum(1)?;
    let denom = mask.sum(1)?.clamp(1f32, f32::MAX)?;
    let mean_hidden = sum_hidden.broadcast_div(&denom.unsqueeze(1)?)?;
    Tensor::cat(&[cls, mean_hidden], 1)
}

fn build_device(device_id: usize) -> Result<Device> {
    match std::env::var("INTENT_ROUTER_DEVICE")
        .ok()
        .filter(|s| !s.trim().is_empty())
    {
        Some(pref) => parse_device_preference(pref, device_id),
        None => try_cuda_device(device_id),
    }
}

fn parse_device_preference(value: String, default_gpu: usize) -> Result<Device> {
    let trimmed = value.trim();
    let lower = trimmed.to_ascii_lowercase();
    if lower == "cpu" {
        Ok(Device::Cpu)
    } else if lower.starts_with("cuda") || lower.starts_with("gpu") {
        let ordinal = trimmed
            .split(':')
            .nth(1)
            .and_then(|part| part.parse::<usize>().ok())
            .unwrap_or(default_gpu);
        Device::new_cuda(ordinal).map_err(|err| {
            anyhow!(
                "requested CUDA device {} but initialization failed: {err}",
                ordinal
            )
        })
    } else {
        warn!(
            "unrecognized INTENT_ROUTER_DEVICE value '{}', defaulting to auto",
            trimmed
        );
        try_cuda_device(default_gpu)
    }
}

fn try_cuda_device(device_id: usize) -> Result<Device> {
    Device::new_cuda(device_id).map_err(|err| {
        anyhow!(
            "failed to initialize CUDA device {} ({err}). Build with the `intent-router-cuda` \
             feature and ensure CUDA libraries are available.",
            device_id
        )
    })
}

fn pad_token_id(tokenizer: &Tokenizer) -> u32 {
    tokenizer
        .get_padding()
        .map(|params| params.pad_id)
        .or_else(|| tokenizer.token_to_id("<pad>"))
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn run_router_classification() {
        let snapshot = PathBuf::from("models/robertaTunedHeads");
        if !snapshot.join("model.safetensors").exists() {
            eprintln!(
                "intent router snapshot missing under {}, skipping test",
                snapshot.display()
            );
            return;
        }
        let router =
            RobertaIntentRouter::load(snapshot, 0, true).expect("failed to load router model");
        let result = router
            .classify("This is a quick smoke test")
            .expect("router inference failed");
        assert_eq!(result.speech_act.len(), 5);
    }
}
