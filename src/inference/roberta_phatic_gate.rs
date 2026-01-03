use anyhow::{anyhow, Result};
use candle_core::{DType, Device, IndexOp, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::bert::{BertModel, Config as BertConfig};
use std::{fs, path::PathBuf};
use tokenizers::Tokenizer;

pub struct RobertaPhaticGate {
    model: BertModel,
    head_dense_w: Tensor,
    head_dense_b: Tensor,
    head_out_w: Tensor,
    head_out_b: Tensor,
    tokenizer: Tokenizer,
    device: Device,
    max_len: usize,
}

impl RobertaPhaticGate {
    pub fn load(snapshot: PathBuf, device_id: usize) -> Result<Self> {
        let device = Device::new_cuda(device_id)?;

        let tokenizer_path = snapshot.join("tokenizer.json");
        let mut tokenizer = Tokenizer::from_file(&tokenizer_path).map_err(|e| {
            anyhow!(
                "Tokenizer load failed ({}): {e}",
                tokenizer_path.display()
            )
        })?;
        tokenizer
            .with_truncation(None)
            .map_err(|e| anyhow!("Tokenizer truncation config failed: {e}"))?;
        tokenizer.with_padding(None);
        let enc = tokenizer
            .encode("hi there", true)
            .map_err(|e| anyhow!("Tokenizer debug encode failed: {e}"))?;
        println!("TOKENS: {:?}", enc.get_tokens());
        println!("IDS: {:?}", enc.get_ids());

        let mut config: BertConfig =
            serde_json::from_slice(&fs::read(snapshot.join("config.json"))?)?;
        config.type_vocab_size = 1;

        let max_len = 256usize;

        let weights = snapshot.join("model.safetensors");
        if !weights.exists() {
            return Err(anyhow!("model.safetensors not found in {:?}", snapshot));
        }

        let vb =
            unsafe { VarBuilder::from_mmaped_safetensors(&[weights], DType::F32, &device)? };
        let model = BertModel::load(vb.pp("roberta"), &config)?;

        let head_dense_w = vb
            .pp("classifier.dense")
            .get((config.hidden_size, config.hidden_size), "weight")?;
        let head_dense_b = vb
            .pp("classifier.dense")
            .get(config.hidden_size, "bias")?;
        let head_out_w = vb
            .pp("classifier.out_proj")
            .get((2, config.hidden_size), "weight")?;
        let head_out_b = vb
            .pp("classifier.out_proj")
            .get(2, "bias")?;

        Ok(Self {
            model,
            head_dense_w,
            head_dense_b,
            head_out_w,
            head_out_b,
            tokenizer,
            device,
            max_len,
        })
    }

    pub fn classify(&self, text: &str) -> Result<(usize, f32)> {
        let enc = self
            .tokenizer
            .encode(text, true)
            .map_err(|e| anyhow!("Tokenizer encode error: {e}"))?;
        let mut ids = enc.get_ids().to_vec();
        if ids.len() > self.max_len {
            ids.truncate(self.max_len);
        }
        let seq_len = ids.len();

        let input = Tensor::new(ids.as_slice(), &self.device)?.unsqueeze(0)?;
        let mask = Tensor::ones(&[1, seq_len], DType::I64, &self.device)?;
        let hidden = self.forward_hidden(&input, &mask)?;
        let cls = hidden.i((0, 0))?;

        let x = cls
            .unsqueeze(0)?
            .matmul(&self.head_dense_w.t()?)?
            .broadcast_add(&self.head_dense_b)?
            .tanh()?;

        let logits = x
            .matmul(&self.head_out_w.t()?)?
            .broadcast_add(&self.head_out_b)?
            .squeeze(0)?;
        let last_dim = logits.dims().len().saturating_sub(1);
        let probs = candle_nn::ops::softmax(&logits, last_dim)?;

        let values = probs.to_vec1::<f32>()?;
        let (idx, conf) = values
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .ok_or_else(|| anyhow!("empty logits"))?;

        Ok((idx, *conf))
    }

    fn forward_hidden(&self, input: &Tensor, mask: &Tensor) -> Result<Tensor> {
        let token_type_ids = Tensor::zeros(input.dims(), DType::I64, &self.device)?;
        Ok(self.model.forward(input, &token_type_ids, Some(mask))?)
    }
}
