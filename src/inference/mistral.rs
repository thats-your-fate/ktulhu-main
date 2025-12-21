use anyhow::{anyhow, Result};
use candle_core::{DType, Device, IndexOp, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::generation::LogitsProcessor;
use candle_transformers::models::mistral::{Config as MistralConfig, Model as Mistral};
use tokenizers::Tokenizer;
use tracing::info;

use std::sync::atomic::{AtomicBool, Ordering};
use std::{fs, path::PathBuf, sync::Arc};
use tokio::sync::{mpsc, Mutex};

use crate::conversation::{trim_partial_chatml, STOP_SEQS};

// ---------------------------------------------------------
// PUBLIC SERVICE
// ---------------------------------------------------------
pub struct MistralService {
    pub model: Arc<Mutex<Mistral>>,
    pub tokenizer: Arc<Tokenizer>,
    pub device: Device,
}

impl MistralService {
    // -----------------------------------------------------
    // Constructor with explicit model directory + GPU ID
    // -----------------------------------------------------
    pub async fn new_with(snapshot_dir: PathBuf, device_id: usize) -> Result<Self> {
        let device = Device::new_cuda(device_id)?;
        println!("🔌 Mistral → Using CUDA:{device_id} ({device:?})");
        println!("📁 Snapshot: {}", snapshot_dir.display());

        // ---- Load tokenizer ----
        let tokenizer_path = snapshot_dir.join("tokenizer.json");
        let tokenizer = Arc::new(
            Tokenizer::from_file(&tokenizer_path).map_err(|e| anyhow!("Tokenizer error: {e}"))?,
        );

        // Sanity check emoji round-trip to catch decoding regressions.
        {
            let emoji_encoding = tokenizer
                .encode("😊", false)
                .map_err(|e| anyhow!("Tokenizer emoji encode error: {e}"))?;
            let ids = emoji_encoding.get_ids().to_vec();
            let decoded = tokenizer
                .decode(ids.as_slice(), false)
                .map_err(|e| anyhow!("Tokenizer emoji decode error: {e}"))?;
            assert!(
                !decoded.contains('\u{FFFD}'),
                "Tokenizer still emits replacement character!"
            );
            info!("Tokenizer emoji test OK: {}", decoded);
        }

        // ---- Load config ----
        let config_path = snapshot_dir.join("config.json");
        let cfg: MistralConfig = serde_json::from_slice(&fs::read(&config_path)?)?;

        // ---- Load safetensor shards ----
        let index_path = snapshot_dir.join("model.safetensors.index.json");
        let index_json: serde_json::Value = serde_json::from_slice(&fs::read(&index_path)?)?;

        let shards = index_json["weight_map"]
            .as_object()
            .unwrap()
            .values()
            .map(|v| snapshot_dir.join(v.as_str().unwrap()))
            .collect::<Vec<_>>();

        println!("📦 Found {} Mistral shards", shards.len());

        // ---- mmap the model weights ----
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&shards, DType::F16, &device)? };

        let model = Arc::new(Mutex::new(Mistral::new(&cfg, vb)?));

        println!("🚀 Mistral loaded on CUDA:{device_id} with KV cache enabled");

        Ok(Self {
            model,
            tokenizer,
            device,
        })
    }

    // -----------------------------------------------------
    // Streaming generation API
    // -----------------------------------------------------
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
                run_mistral_stream(model, tokenizer, device, prompt, tx, cancel.clone()).await
            {
                eprintln!("❌ streaming error: {e}");
            }
        });

        rx
    }

    pub async fn generate_completion(
        &self,
        prompt: String,
        cancel: Arc<AtomicBool>,
    ) -> Result<String> {
        run_mistral_completion(
            self.model.clone(),
            self.tokenizer.clone(),
            self.device.clone(),
            prompt,
            cancel,
        )
        .await
    }
}

// ---------------------------------------------------------
// MISTRAL STREAMING LOOP
// ---------------------------------------------------------
pub async fn run_mistral_stream(
    model: Arc<Mutex<Mistral>>,
    tokenizer: Arc<Tokenizer>,
    device: Device,
    user_prompt: String,
    tx: mpsc::Sender<String>,
    cancel: Arc<AtomicBool>,
) -> Result<()> {
    {
        let mut m = model.lock().await;
        m.clear_kv_cache();
    }

    use tokenizers::EncodeInput;
    let enc = tokenizer
        .encode(EncodeInput::Single(user_prompt.into()), true)
        .map_err(|e| anyhow!("Tokenizer encode error: {e}"))?;
    let mut tokens = enc.get_ids().to_vec();
    let eos = tokenizer
        .token_to_id("<eos>")
        .or_else(|| tokenizer.token_to_id("</s>"))
        .unwrap_or(u32::MAX);

    let mut pos = 0usize;
    let mut lp = LogitsProcessor::new(seed(), Some(0.7), None);
    let mut generated_tokens: Vec<u32> = Vec::new();
    let mut emitted_len = 0usize;
    let mut boundary_fixer = BoundaryFixer::default();

    const MAX_NEW_TOKENS: usize = 4096;

    for _ in 0..MAX_NEW_TOKENS {
        if cancel.load(Ordering::SeqCst) || tx.is_closed() {
            return Ok(());
        }

        let ctx: &[u32] = if pos == 0 {
            &tokens
        } else {
            std::slice::from_ref(tokens.last().unwrap())
        };

        let input = Tensor::new(ctx, &device)?.unsqueeze(0)?;

        let logits = {
            let mut m = model.lock().await;
            let out = m.forward(&input, pos)?;
            let seq_len = out.dim(1)?;
            out.i((0, seq_len - 1))?.to_dtype(DType::F32)?
        };

        pos += ctx.len();

        let next_id = lp.sample(&logits)?;
        tokens.push(next_id);

        if next_id == eos {
            break;
        }

        generated_tokens.push(next_id);

        let decoded = match tokenizer.decode(&generated_tokens, false) {
            Ok(s) => s,
            Err(_) => continue,
        };

        let mut normalized = decoded.replace('\u{2581}', " ");
        normalized = normalized.replace('\u{200b}', "");

        if normalized.contains('\u{FFFD}') {
            continue;
        }

        let pending_suffix = longest_stop_prefix_suffix(&normalized);
        let safe_end = normalized.len().saturating_sub(pending_suffix);
        let candidate = &normalized[..safe_end];
        let cleaned = trim_partial_chatml(candidate).to_string();
        let stop_hit = cleaned.len() < candidate.len();

        if cleaned.len() < emitted_len {
            emitted_len = cleaned.len();
        }

        if cleaned.len() > emitted_len {
            let delta = &cleaned[emitted_len..];
            if !delta.is_empty() {
                if cancel.load(Ordering::SeqCst) || tx.is_closed() {
                    return Ok(());
                }

                let formatted = boundary_fixer.apply(delta);
                if !formatted.is_empty() {
                    tx.send(formatted)
                        .await
                        .map_err(|e| anyhow!("stream send error: {e}"))?;
                }
            }
            emitted_len = cleaned.len();
        }

        if stop_hit {
            break;
        }

        tokio::task::yield_now().await;
    }

    Ok(())
}

async fn run_mistral_completion(
    model: Arc<Mutex<Mistral>>,
    tokenizer: Arc<Tokenizer>,
    device: Device,
    user_prompt: String,
    cancel: Arc<AtomicBool>,
) -> Result<String> {
    {
        let mut m = model.lock().await;
        m.clear_kv_cache();
    }

    use tokenizers::EncodeInput;
    let enc = tokenizer
        .encode(EncodeInput::Single(user_prompt.into()), true)
        .map_err(|e| anyhow!("Tokenizer encode error: {e}"))?;
    let mut tokens = enc.get_ids().to_vec();

    let eos = tokenizer
        .token_to_id("<eos>")
        .or_else(|| tokenizer.token_to_id("</s>"))
        .unwrap_or(u32::MAX);

    let mut pos = 0usize;
    let mut lp = LogitsProcessor::new(seed(), Some(0.7), None);
    let mut output = String::new();

    const MAX_NEW_TOKENS: usize = 4096;

    for _ in 0..MAX_NEW_TOKENS {
        if cancel.load(Ordering::SeqCst) {
            return Err(anyhow!("cancelled"));
        }

        let ctx: &[u32] = if pos == 0 {
            &tokens
        } else {
            std::slice::from_ref(tokens.last().unwrap())
        };

        let input = Tensor::new(ctx, &device)?.unsqueeze(0)?;

        let logits = {
            let mut m = model.lock().await;
            let out = m.forward(&input, pos)?;
            let seq_len = out.dim(1)?;
            out.i((0, seq_len - 1))?.to_dtype(DType::F32)?
        };

        pos += ctx.len();

        let next_id = lp.sample(&logits)?;
        tokens.push(next_id);

        if next_id == eos {
            break;
        }

        let mut piece = match tokenizer.decode(&[next_id], false) {
            Ok(s) => s,
            Err(_) => continue,
        };

        if piece.contains('\u{2581}') {
            piece = piece.replace('\u{2581}', " ");
        }
        if piece.is_empty() || piece == "\u{200b}" {
            continue;
        }

        output.push_str(&piece);

        if STOP_SEQS.iter().any(|seq| output.contains(seq)) {
            break;
        }

        tokio::task::yield_now().await;
    }

    let trimmed = trim_partial_chatml(&output).to_string();
    let mut fixer = BoundaryFixer::default();
    Ok(fixer.apply(&trimmed))
}

// ---------------------------------------------------------
// Helpers
// ---------------------------------------------------------
fn seed() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs()
}

fn longest_stop_prefix_suffix(text: &str) -> usize {
    STOP_SEQS
        .iter()
        .map(|seq| {
            let max_len = seq.len().min(text.len());
            (1..=max_len)
                .rev()
                .find(|len| text.ends_with(&seq[..*len]))
                .unwrap_or(0)
        })
        .max()
        .unwrap_or(0)
}

#[derive(Default)]
struct BoundaryFixer {
    last_char: Option<char>,
}

impl BoundaryFixer {
    fn apply(&mut self, chunk: &str) -> String {
        let mut out = String::with_capacity(chunk.len());

        for ch in chunk.chars() {
            if let Some(prev) = self.last_char {
                let prev_is_alpha = prev.is_alphabetic();
                let prev_is_digit = prev.is_ascii_digit();
                let ch_is_alpha = ch.is_alphabetic();
                let ch_is_digit = ch.is_ascii_digit();

                if prev_is_alpha && ch_is_digit && !prev.is_whitespace() && !ch.is_whitespace() {
                    out.push(' ');
                } else if prev_is_digit
                    && ch_is_alpha
                    && !prev.is_whitespace()
                    && !ch.is_whitespace()
                {
                    out.push(' ');
                }
            }

            out.push(ch);
            self.last_char = Some(ch);
        }

        out
    }
}
