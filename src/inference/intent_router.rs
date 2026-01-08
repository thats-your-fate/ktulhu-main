use anyhow::{anyhow, Result};
use candle_core::{DType, Device, IndexOp, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::bert::{BertModel, Config as BertConfig};
use std::{fs, path::PathBuf};
use tokenizers::Tokenizer;
use tracing::warn;

const PHATIC_CLASSES: usize = 2;
const SPEECH_ACT_CLASSES: usize = 5;
const DOMAIN_CLASSES: usize = 7;
const EXPECTATION_CLASSES: usize = 5;
const SUPPORT_CLASSES: usize = 2;

pub struct IntentLogits {
    pub phatic: Option<Tensor>,
    pub speech_act: Tensor,
    pub domain: Tensor,
    pub expectation: Tensor,
    pub support: Option<Tensor>,
}

struct Head {
    dense_w: Tensor,
    dense_b: Tensor,
    out_w: Tensor,
    out_b: Tensor,
}

impl Head {
    fn load(vb: &VarBuilder, prefix: &str, hidden: usize, out: usize) -> Result<Self> {
        let dense_w = vb.pp(prefix).pp("0").get((hidden, hidden), "weight")?;
        let dense_b = vb.pp(prefix).pp("0").get(hidden, "bias")?;
        let out_w = vb.pp(prefix).pp("2").get((out, hidden), "weight")?;
        let out_b = vb.pp(prefix).pp("2").get(out, "bias")?;
        Ok(Self {
            dense_w,
            dense_b,
            out_w,
            out_b,
        })
    }

    fn forward(&self, cls: &Tensor) -> Result<Tensor> {
        let x = cls
            .unsqueeze(0)?
            .matmul(&self.dense_w.t()?)?
            .broadcast_add(&self.dense_b)?
            .tanh()?;

        let logits = x
            .matmul(&self.out_w.t()?)?
            .broadcast_add(&self.out_b)?
            .squeeze(0)?;
        Ok(logits)
    }
}

fn try_load_head(vb: &VarBuilder, prefix: &str, hidden: usize, out: usize) -> Result<Option<Head>> {
    match Head::load(vb, prefix, hidden, out) {
        Ok(head) => Ok(Some(head)),
        Err(err) => {
            warn!("head {prefix} missing or invalid: {err}. continuing without it");
            Ok(None)
        }
    }
}

pub struct RobertaIntentRouter {
    model: BertModel,
    tokenizer: Tokenizer,
    device: Device,
    max_len: usize,
    phatic_head: Option<Head>,
    speech_head: Head,
    domain_head: Head,
    expectation_head: Head,
    support_head: Option<Head>,
}

impl RobertaIntentRouter {
    pub fn load(snapshot: PathBuf, device_id: usize, with_phatic: bool) -> Result<Self> {
        let device = Device::new_cuda(device_id)?;

        let tokenizer_path = snapshot.join("tokenizer.json");
        let mut tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| anyhow!("Tokenizer load failed ({}): {e}", tokenizer_path.display()))?;
        tokenizer.with_padding(None);
        let _ = tokenizer.with_truncation(None);

        let mut config: BertConfig =
            serde_json::from_slice(&fs::read(snapshot.join("config.json"))?)?;
        config.type_vocab_size = 1;
        let max_len = 256usize;

        let weights = snapshot.join("model.safetensors");
        if !weights.exists() {
            return Err(anyhow!("model.safetensors not found in {:?}", snapshot));
        }

        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[weights], DType::F32, &device)? };
        let model = BertModel::load(vb.pp("roberta"), &config)?;
        let hidden = config.hidden_size;

        let feature_dim = hidden * 2;

        let phatic_head = if with_phatic {
            Some(Head::load(&vb, "phatic_head", feature_dim, PHATIC_CLASSES)?)
        } else {
            None
        };
        let speech_head = Head::load(&vb, "speech_head", feature_dim, SPEECH_ACT_CLASSES)?;
        let domain_head = Head::load(&vb, "domain_head", feature_dim, DOMAIN_CLASSES)?;
        let expectation_head =
            Head::load(&vb, "expectation_head", feature_dim, EXPECTATION_CLASSES)?;
        let support_head = try_load_head(&vb, "support_head", feature_dim, SUPPORT_CLASSES)?;
        Ok(Self {
            model,
            tokenizer,
            device,
            max_len,
            phatic_head,
            speech_head,
            domain_head,
            expectation_head,
            support_head,
        })
    }

    pub fn classify(&self, text: &str) -> Result<IntentLogits> {
        let enc = self
            .tokenizer
            .encode(text, true)
            .map_err(|e| anyhow!("Tokenizer encode error: {e}"))?;

        let mut ids = enc.get_ids().to_vec();
        if ids.is_empty() {
            ids.push(0);
        }
        if ids.len() > self.max_len {
            ids.truncate(self.max_len);
        }
        let seq_len = ids.len();

        let input = Tensor::new(ids.as_slice(), &self.device)?.unsqueeze(0)?;
        let mask = Tensor::ones(&[1, seq_len], DType::I64, &self.device)?;
        let token_type_ids = Tensor::zeros(input.dims(), DType::I64, &self.device)?;

        // [1, seq, hidden]
        let hidden = self.model.forward(&input, &token_type_ids, Some(&mask))?;

        // CLS token: [hidden]
        let cls = hidden.i((0, 0))?;

        // Mask-aware mean pooling
        // hidden: [1, seq, hidden] → [seq, hidden]
        let token_embeddings = hidden.i(0)?;

        // mask: [1, seq] → [seq, 1] → float
        let mask_f = mask.i(0)?.unsqueeze(1)?.to_dtype(DType::F32)?;

        // Zero-out padding tokens
        let masked = token_embeddings.broadcast_mul(&mask_f)?;

        // Sum over seq
        let sum = masked.sum(0)?;

        // Count real tokens
        let count = mask_f.sum(0)?;

        // Mean: [hidden]
        let mean = sum.broadcast_div(&count)?;

        // Concatenate → [hidden * 2] (e.g. 2048)
        let features = Tensor::cat(&[cls, mean], 0)?;

        // Heads
        let phatic = if let Some(h) = &self.phatic_head {
            Some(h.forward(&features)?)
        } else {
            None
        };
        let speech_act = self.speech_head.forward(&features)?;
        let domain = self.domain_head.forward(&features)?;
        let expectation = self.expectation_head.forward(&features)?;
        let support = if let Some(h) = &self.support_head {
            Some(h.forward(&features)?)
        } else {
            None
        };

        Ok(IntentLogits {
            phatic,
            speech_act,
            domain,
            expectation,
            support,
        })
    }
}

pub fn logits_argmax(logits: &Tensor) -> Result<(usize, f32)> {
    let values = logits.to_vec1::<f32>()?;
    let (idx, value) = values
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        .ok_or_else(|| anyhow!("empty logits tensor"))?;
    Ok((idx, *value))
}
