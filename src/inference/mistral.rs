use anyhow::{anyhow, Result};
use candle_core::{DType, Device, IndexOp, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::generation::LogitsProcessor;
use candle_transformers::models::mistral::{Config as MistralConfig, Model as Mistral};
use tokenizers::Tokenizer;

use std::sync::atomic::{AtomicBool, Ordering};
use std::{fs, path::PathBuf, sync::Arc};
use tokio::sync::{mpsc, Mutex};
pub struct InferenceService {
    pub model: Arc<Mutex<Mistral>>,
    pub tokenizer: Arc<Tokenizer>,
    pub device: Device,
}

impl InferenceService {
    /// Initialize tokenizer + config + model + mmap weights
    pub async fn new() -> Result<Self> {
        let device = Device::new_cuda(0)?;
        println!("🔌 CUDA device: {:?}", device);

        let dir = find_snapshot_dir()?;
        println!("📁 Using local model snapshot: {}", dir.display());

        let tokenizer_path = dir.join("tokenizer.json");
        let config_path = dir.join("config.json");
        let index_path = dir.join("model.safetensors.index.json");

        // ---------------- Tokenizer ----------------
        let tokenizer = Arc::new(
            Tokenizer::from_file(&tokenizer_path)
                .map_err(|e| anyhow!("Tokenizer load error: {}", e))?,
        );

        // ---------------- Config ----------------
        let cfg: MistralConfig = serde_json::from_slice(&fs::read(&config_path)?)?;

        // ---------------- Shards ----------------
        let index_json: serde_json::Value = serde_json::from_slice(&fs::read(&index_path)?)?;

        let mut shards = vec![];
        for v in index_json["weight_map"].as_object().unwrap().values() {
            shards.push(dir.join(v.as_str().unwrap()));
        }
        println!("📦 Found {} shards", shards.len());

        // ---------------- Load weights (mmap) ----------------
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&shards, DType::F16, &device)? };

        // ---------------- Init Model ----------------
        let model = Arc::new(Mutex::new(Mistral::new(&cfg, vb)?));

        println!("🚀 Mistral model loaded with KV cache enabled!");

        Ok(Self {
            model,
            tokenizer,
            device,
        })
    }

    /// Full non-streaming generation (returns final string)
    pub async fn generate(&self, prompt: &str, max: usize) -> Result<String> {
        // 🔥 Reset KV cache for a fresh generation
        {
            let mut m = self.model.lock().await;
            m.clear_kv_cache();
        }

        let enc = self
            .tokenizer
            .encode(prompt, true)
            .map_err(|e| anyhow!("encode error: {}", e))?;

        let mut tokens = enc.get_ids().to_vec();
        let eos = self.tokenizer.token_to_id("</s>").unwrap_or(u32::MAX);

        let mut lp = LogitsProcessor::new(seed(), Some(0.7), None);
        let mut pos = 0usize;

        for _ in 0..max {
            let ctx = if pos == 0 {
                tokens.clone()
            } else {
                vec![*tokens.last().unwrap()]
            };

            let input = Tensor::new(&ctx[..], &self.device)?.unsqueeze(0)?;

            let mut m = self.model.lock().await;
            let out = m.forward(&input, pos)?;
            drop(m);

            let seq_len = out.dim(1)?;
            let logits = out.i((0, seq_len - 1))?.to_dtype(DType::F32)?;

            let next = lp.sample(&logits)?;
            tokens.push(next);

            pos += ctx.len();

            if next == eos {
                break;
            }
        }

        let text = self
            .tokenizer
            .decode(&tokens, false)
            .map_err(|e| anyhow!("decode error: {}", e))?;

        Ok(text)
    }

    pub fn generate_stream(
        &self,
        prompt: String,
        cancel: Arc<AtomicBool>,
    ) -> mpsc::Receiver<String> {
        let model = self.model.clone();
        let tokenizer = self.tokenizer.clone();
        let device = self.device.clone();

        let (tx, rx) = mpsc::channel(64);
        tokio::spawn(async move {
            if let Err(e) =
                run_generation(model, tokenizer, device, prompt, tx, cancel.clone()).await
            {
                eprintln!("❌ streaming error: {e}");
            }
        });

        rx
    }
}

/// STREAMING GENERATION LOOP — CLEAN, CHATGPT-STYLE STREAMING
pub async fn run_generation(
    model: Arc<Mutex<Mistral>>,
    tokenizer: Arc<Tokenizer>,
    device: Device,
    user_prompt: String,
    tx: mpsc::Sender<String>,
    cancel: Arc<AtomicBool>,
) -> Result<()> {
    /* -------------------------------------------------------
     * 1) CLEAR KV CACHE (correct)
     * ----------------------------------------------------- */
    {
        let mut m = model.lock().await;
        m.clear_kv_cache();
    }

    /* -------------------------------------------------------
     * 2) ENCODE PROMPT
     * ----------------------------------------------------- */
    use tokenizers::EncodeInput;

    let enc = tokenizer
        .encode(EncodeInput::Single(user_prompt.clone().into()), true)
        .map_err(|e| anyhow!("encode error: {e}"))?;

    let mut tokens: Vec<u32> = enc.get_ids().to_vec();

    // Determine EOS
    let eos = tokenizer
        .token_to_id("<eos>")
        .or_else(|| tokenizer.token_to_id("</s>"))
        .unwrap_or(u32::MAX);

    /* -------------------------------------------------------
     * 3) CORRECT POS LOGIC
     *    pos = number of tokens already processed into KV
     * ----------------------------------------------------- */
    let mut pos = 0usize;

    let mut lp = LogitsProcessor::new(seed(), Some(0.7), None);

    const MAX_NEW_TOKENS: usize = 4096;

    /* -------------------------------------------------------
     * 4) GENERATION LOOP
     * ----------------------------------------------------- */
    for _ in 0..MAX_NEW_TOKENS {
        if cancel.load(Ordering::SeqCst) || tx.is_closed() {
            return Ok(());
        }

        // First step: feed whole prompt
        // Later steps: feed *only* the last generated token
        let ctx: &[u32] = if pos == 0 {
            &tokens
        } else {
            let last = tokens.last().unwrap();
            std::slice::from_ref(last)
        };

        let input = Tensor::new(ctx, &device)?.unsqueeze(0)?;

        // Forward pass
        let logits = {
            let mut m = model.lock().await;
            let out = m.forward(&input, pos)?;
            let seq_len = out.dim(1)?;

            out.i((0, seq_len - 1))?.to_dtype(DType::F32)?
        };

        // Update KV position
        pos += ctx.len();

        // Sample next token
        let next_id = lp.sample(&logits)?;
        tokens.push(next_id);

        // Stop at EOS
        if next_id == eos {
            break;
        }

        /* ---------------------------------------------------
         * 5) DECODE SINGLE TOKEN
         * ------------------------------------------------- */
        let mut piece = match tokenizer.decode(&[next_id], false) {
            Ok(s) => s,
            Err(_) => continue,
        };

        // Replace SentencePiece underline "▁" with space
        if piece.contains('\u{2581}') {
            piece = piece.replace('\u{2581}', " ");
        }

        // Skip empty or totally invisible control sequences
        if piece.is_empty() || piece == "\u{200b}" {
            continue;
        }

        // Deliver token immediately
        if cancel.load(Ordering::SeqCst) || tx.is_closed() {
            return Ok(());
        }

        let _ = tx.send(piece).await;

        // Yield to keep IO tasks (ws flush) moving even if sampling is busy.
        tokio::task::yield_now().await;
    }

    Ok(())
}

/// Locate HuggingFace snapshot folder
fn find_snapshot_dir() -> Result<PathBuf> {
    let base = PathBuf::from("/srv/mistral/ktulhuTest/ministral_8b");

    if base.exists() && base.is_dir() {
        Ok(base)
    } else {
        Err(anyhow!("Snapshot dir does not exist: {}", base.display()))
    }
}

/// Random seed
fn seed() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs()
}
