use anyhow::{anyhow, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::bert::{BertModel, Config as BertConfig};
use std::{fs, path::PathBuf, sync::Mutex};
use tokenizers::Tokenizer;

pub struct ClassifierOutput {
    pub speech_act: Tensor,
    pub domain: Tensor,
    pub expectation: Tensor,
}

pub struct RobertaClassifier {
    model: Mutex<BertModel>,
    tokenizer: Tokenizer,
    device: Device,
    vocab_size: u32,
    max_len: usize,
    speech_embeds: Vec<Vec<f32>>,
    domain_embeds: Vec<Vec<f32>>,
    expectation_embeds: Vec<Vec<f32>>,
}

const SPEECH_LABELS: &[&str] = &["social", "sharing", "asking", "directing", "collaborative"];
const DOMAIN_LABELS: &[&str] = &["technical", "personal", "social", "legal", "other"];
const EXPECTATION_LABELS: &[&str] = &["none", "info", "advice", "action"];

impl RobertaClassifier {
    pub fn load(snapshot_dir: PathBuf, device_id: usize) -> Result<Self> {
        let device = Device::new_cuda(device_id)?;
        println!("🟦 RoBERTa classifier → CUDA:{device_id}");

        let tokenizer_path = snapshot_dir.join("tokenizer.json");
        let tokenizer = Tokenizer::from_file(&tokenizer_path).map_err(|e| {
            anyhow!(
                "Tokenizer load failed ({}): {e}",
                tokenizer_path.display()
            )
        })?;

        let mut config: BertConfig =
            serde_json::from_slice(&fs::read(snapshot_dir.join("config.json"))?)?;
        config.type_vocab_size = 1;
        config.layer_norm_eps = 1e-5;
        let vocab_size = config.vocab_size as u32;
        let max_len = (config.max_position_embeddings as usize)
            .saturating_sub(2)
            .max(16);

        let weights = snapshot_dir.join("model.safetensors");
        if !weights.exists() {
            return Err(anyhow!("model.safetensors not found in {:?}", snapshot_dir));
        }

        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[weights], DType::F32, &device)? };
        let model = BertModel::load(vb.pp("roberta"), &config)?;

        let classifier = Self {
            model: Mutex::new(model),
            tokenizer,
            device,
            vocab_size,
            max_len,
            speech_embeds: Vec::new(),
            domain_embeds: Vec::new(),
            expectation_embeds: Vec::new(),
        };

        let speech_embeds = classifier.build_label_embeds(SPEECH_LABELS)?;
        let domain_embeds = classifier.build_label_embeds(DOMAIN_LABELS)?;
        let expectation_embeds = classifier.build_label_embeds(EXPECTATION_LABELS)?;

        Ok(Self {
            speech_embeds,
            domain_embeds,
            expectation_embeds,
            ..classifier
        })
    }

    fn build_label_embeds(&self, labels: &[&str]) -> Result<Vec<Vec<f32>>> {
        labels
            .iter()
            .map(|lbl| self.embed(lbl))
            .collect()
    }

    fn embed(&self, text: &str) -> Result<Vec<f32>> {
        let enc = self
            .tokenizer
            .encode(text, true)
            .map_err(|e| anyhow!("Tokenizer encode error: {e}"))?;

        let ids_u32 = enc.get_ids().to_vec();
        if let Some(&max_id) = ids_u32.iter().max() {
            if max_id >= self.vocab_size {
                return Err(anyhow!(
                    "token id overflow: saw max_id={} but vocab_size={}",
                    max_id, self.vocab_size
                ));
            }
        }

        let mut ids = ids_u32;
        if ids.len() > self.max_len {
            ids.truncate(self.max_len);
        }
        let seq_len = ids.len();

        let input = Tensor::new(ids.as_slice(), &self.device)?.unsqueeze(0)?;
        let mask = Tensor::ones(&[1, seq_len], DType::I64, &self.device)?;
        let token_type_ids = Tensor::zeros(input.dims(), DType::I64, &self.device)?;

        let hidden = {
            let model = self.model.lock().unwrap();
            model.forward(&input, &token_type_ids, Some(&mask))?
        };

        let hidden = hidden.squeeze(0)?;
        let mask_f = mask.squeeze(0)?.to_dtype(DType::F32)?;

        let hidden_vec = hidden.to_vec2::<f32>()?;
        let mask_vec = mask_f.to_vec1::<f32>()?;

        let dims = hidden_vec.first().map(|row| row.len()).unwrap_or(0);
        let mut sum = vec![0f32; dims];
        let mut count = 0f32;

        for (row, &w) in hidden_vec.iter().zip(mask_vec.iter()) {
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

        l2_normalize(&mut sum);
        Ok(sum)
    }

    fn score_against(&self, text_emb: &[f32], label_embs: &[Vec<f32>]) -> Vec<f32> {
        label_embs
            .iter()
            .map(|lbl| cosine(text_emb, lbl))
            .collect::<Vec<_>>()
    }

    pub fn classify(&self, text: &str) -> Result<ClassifierOutput> {
        let emb = self.embed(text)?;
        let speech_scores = self.score_against(&emb, &self.speech_embeds);
        let domain_scores = self.score_against(&emb, &self.domain_embeds);
        let expectation_scores = self.score_against(&emb, &self.expectation_embeds);

        Ok(ClassifierOutput {
            speech_act: Tensor::from_slice(&speech_scores, speech_scores.len(), &Device::Cpu)?,
            domain: Tensor::from_slice(&domain_scores, domain_scores.len(), &Device::Cpu)?,
            expectation: Tensor::from_slice(&expectation_scores, expectation_scores.len(), &Device::Cpu)?,
        })
    }
}

pub fn logits_argmax(logits: &Tensor) -> Result<(usize, f32)> {
    let values = logits.to_vec1::<f32>()?;
    let (idx, value) = values
        .iter()
        .enumerate()
        .max_by(|a, b| a
            .1
            .partial_cmp(b.1)
            .unwrap_or(std::cmp::Ordering::Equal))
        .ok_or_else(|| anyhow!("empty logits tensor"))?;
    Ok((idx, *value))
}

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
