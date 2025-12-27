use anyhow::{anyhow, Result};
use candle_core::{DType, Device, IndexOp, Tensor};
use candle_nn::VarBuilder;

use std::{fs, path::PathBuf, sync::Arc};
use tokio::sync::Mutex;

use candle_transformers::models::bert::{BertModel, Config as BertConfig};
use tokenizers::pre_tokenizers::byte_level::ByteLevel;
use tokenizers::{
    models::bpe::BPE,
    normalizers::{unicode::NFC, NormalizerWrapper},
    pre_tokenizers::PreTokenizerWrapper,
    Tokenizer,
};
pub struct RobertaService {
    pub model: Arc<Mutex<BertModel>>,
    pub tokenizer: Arc<Tokenizer>,
    pub device: Device,
    pub vocab_size: u32,
    pub max_len: usize,
}

impl RobertaService {
    pub async fn new_with(snapshot_dir: PathBuf, device_id: usize) -> Result<Self> {
        let device = Device::new_cuda(device_id)?;
        println!("🟦 RoBERTa → CUDA:{device_id} ({device:?})");
        println!("📁 RoBERTa snapshot: {}", snapshot_dir.display());

        let vocab = snapshot_dir.join("vocab.json");
        let merges = snapshot_dir.join("merges.txt");

        // ---- BPE builder ----
        let bpe = BPE::from_file(
            vocab
                .to_str()
                .ok_or_else(|| anyhow!("Invalid vocab path"))?,
            merges
                .to_str()
                .ok_or_else(|| anyhow!("Invalid merges path"))?,
        )
        .unk_token("<unk>".to_string())
        .build()
        .map_err(|e| anyhow!("BPE tokenizer build error: {e}"))?;

        // ---- Build tokenizer ----
        let mut tokenizer = Tokenizer::new(bpe);

        // Normalizer: must be wrapped in Option<NormalizerWrapper>
        tokenizer.with_normalizer(Some(NormalizerWrapper::NFC(NFC)));

        // Pre-tokenizer: must be wrapped in Option<PreTokenizerWrapper>
        // tokenizer.with_pre_tokenizer(Some(PreTokenizerWrapper::Whitespace(Whitespace::default())));
        tokenizer.with_pre_tokenizer(Some(PreTokenizerWrapper::ByteLevel(ByteLevel::default())));
        let tokenizer = Arc::new(tokenizer);

        // Config
        let mut config: BertConfig =
            serde_json::from_slice(&fs::read(snapshot_dir.join("config.json"))?)?;
        config.type_vocab_size = 1;
        config.layer_norm_eps = 1e-5;
        let vocab_size = config.vocab_size as u32;
        let max_len = (config.max_position_embeddings as usize)
            .saturating_sub(2)
            .max(16);

        // Single safetensors file — no index.json for this model
        let safetensor_file = snapshot_dir.join("model.safetensors");

        if !safetensor_file.exists() {
            return Err(anyhow!("model.safetensors not found in {:?}", snapshot_dir));
        }

        let shards = vec![safetensor_file];
        println!("📦 RoBERTa shards = 1");

        println!("📦 RoBERTa shards = {}", shards.len());

        // Load weights (mmaped)
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&shards, DType::F32, &device)? };

        // Initialize model
        let model = Arc::new(Mutex::new(BertModel::load(vb, &config)?));

        println!("🚀 Loaded RoBERTa on CUDA:{device_id}");

        Ok(Self {
            model,
            tokenizer,
            device,
            vocab_size,
            max_len,
        })
    }

pub async fn embed(&self, text: &str) -> Result<Vec<f32>> {
    // 1) Tokenize
    let enc = self
        .tokenizer
        .encode(text, true)
        .map_err(|e| anyhow!("RoBERTa tokenizer encode error: {e}"))?;

    let ids_u32 = enc.get_ids().to_vec();
    if let Some(&max_id) = ids_u32.iter().max() {
        if max_id >= self.vocab_size {
            return Err(anyhow!(
                "token id overflow: saw max_id={} but vocab_size={}",
                max_id,
                self.vocab_size
            ));
        }
    }

    let mut ids = ids_u32;
    if ids.len() > self.max_len {
        ids.truncate(self.max_len);
    }
    let seq_len = ids.len();

    // 2) Build tensors
    let input = Tensor::new(ids.as_slice(), &self.device)?
        .unsqueeze(0)?; // [1, seq]

    let mask = Tensor::ones(&[1, seq_len], DType::I64, &self.device)?;
    let token_type_ids = Tensor::zeros(input.dims(), DType::I64, &self.device)?;

    // 3) Forward
    let hidden = {
        let m = self.model.lock().await;
        m.forward(&input, &token_type_ids, Some(&mask))?
    };
    // hidden: [1, seq, 768]

    // 4) Mean pooling (exclude padding)
    let hidden = hidden.squeeze(0)?; // [seq, 768]
    let mask_f = mask.squeeze(0)?.to_dtype(DType::F32)?; // [seq]

    let hidden = hidden.to_vec2::<f32>()?;
    let mask_vec = mask_f.to_vec1::<f32>()?;

    let dims = hidden.first().map(|row| row.len()).unwrap_or(0);
    let mut sum = vec![0f32; dims];
    let mut count = 0f32;

    for (row, &w) in hidden.iter().zip(mask_vec.iter()) {
        count += w;
        for (acc, &val) in sum.iter_mut().zip(row.iter()) {
            *acc += val * w;
        }
    }

    if count <= 1e-6 {
        count = 1e-6;
    }
    for val in sum.iter_mut() {
        *val /= count;
    }
    let mut emb = sum;

    // 5) L2 normalize
    l2_normalize(&mut emb);

    Ok(emb)
}


    /// Simple text classification (cosine similarity to label embeddings)
    pub async fn classify(&self, text: &str, labels: &[&str]) -> Result<(String, f32)> {
        let text_emb = self.embed(text).await?;

        let mut best: Option<String> = None;
        let mut best_score = -999999.0;

        for &lbl in labels {
            let lbl_emb = self.embed(lbl).await?;
            let score = cosine(&text_emb, &lbl_emb);

            if score > best_score {
                best_score = score;
                best = Some(lbl.to_string());
            }
        }

        Ok((best.unwrap_or_else(|| "unknown".into()), best_score))
    }
}

/// cosine similarity helper
fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let mut dot = 0.0;
    let mut na = 0.0;
    let mut nb = 0.0;

    for (&x, &y) in a.iter().zip(b.iter()) {
        dot += x * y;
        na += x * x;
        nb += y * y;
    }

    dot / (na.sqrt() * nb.sqrt()).max(1e-9)
}


fn l2_normalize(v: &mut [f32]) {
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
    for x in v.iter_mut() {
        *x /= norm;
    }
}
