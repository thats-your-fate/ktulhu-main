use anyhow::{Result, anyhow};
use candle_core::{Device, Tensor, DType, IndexOp};
use candle_nn::VarBuilder;
use candle_transformers::generation::LogitsProcessor;
use candle_transformers::models::mistral::{Config as MistralConfig, Model as Mistral};
use tokenizers::Tokenizer;

use std::io::{self, Write};
use std::path::{Path, PathBuf};

const MODEL_DIR: &str = "models--mistralai--Mistral-7B-Instruct-v0.2";

/// Find the HuggingFace snapshot directory for the Mistral model.
fn find_snapshot_dir() -> Result<PathBuf> {
    let home = std::env::var("HOME")?;
    let base = Path::new(&home)
        .join(".cache/huggingface/hub")
        .join(MODEL_DIR)
        .join("snapshots");

    for entry in std::fs::read_dir(&base)? {
        let path = entry?.path();
        if path.is_dir() {
            return Ok(path);
        }
    }
    Err(anyhow!("No snapshot directory found in {}", base.display()))
}

#[tokio::main]
async fn main() -> Result<()> {
    // -------------------------
    // CUDA device
    // -------------------------
    let device = Device::new_cuda(0)?;
    println!("🔌 Using CUDA {:?}", device);

    // -------------------------
    // Model directory lookup
    // -------------------------
    let dir = find_snapshot_dir()?;
    println!("📁 Using local model at: {}", dir.display());

    let tokenizer_path = dir.join("tokenizer.json");
    let config_path    = dir.join("config.json");
    let index_path     = dir.join("model.safetensors.index.json");

    // -------------------------
    // Tokenizer
    // -------------------------
    let tokenizer = Tokenizer::from_file(&tokenizer_path)
        .map_err(|e| anyhow!("Tokenizer load error: {}", e))?;

    // HF mistral tokenizer uses </s> as EOS
    let eos_id = tokenizer.token_to_id("</s>").unwrap_or(u32::MAX);

    // -------------------------
    // Config
    // -------------------------
    let cfg: MistralConfig =
        serde_json::from_slice(&std::fs::read(&config_path)?)?;

    // -------------------------
    // Collect shard files
    // -------------------------
    let index_json: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&index_path)?)?;

    let mut shards = vec![];
    for v in index_json["weight_map"].as_object().unwrap().values() {
        shards.push(dir.join(v.as_str().unwrap()));
    }

    println!("📦 Found {} weight shards", shards.len());

    // -------------------------
    // Load weights
    // -------------------------
    let vb = unsafe {
        VarBuilder::from_mmaped_safetensors(&shards, DType::F16, &device)?
    };

    let mut model = Mistral::new(&cfg, vb)?;
    println!("🚀 Mistral loaded!");

    // -------------------------
    // Prompt
    // -------------------------
    print!("> ");
    io::stdout().flush()?;
    let mut prompt = String::new();
    io::stdin().read_line(&mut prompt)?;
    let prompt = prompt.trim().to_string();

    // Encode using HF tokenizer
let enc = tokenizer.encode(prompt.as_str(), true)
    .map_err(|e| anyhow!("Tokenizer encode error: {}", e))?;

    let mut tokens = enc.get_ids().to_vec();

    print!("{prompt}");
    io::stdout().flush()?;

    // -------------------------
    // Sampler
    // -------------------------
    let mut lp = LogitsProcessor::new(seed(), Some(0.7), None);
    let mut pos = 0usize;

    // -------------------------
    // Inference loop
    // -------------------------
for _ in 0..256 {
    // Feed only the last token after the first step
    let context = if pos == 0 {
        tokens.clone()
    } else {
        vec![*tokens.last().unwrap()]
    };

    let input = Tensor::new(&context[..], &device)?.unsqueeze(0)?;
    let out = model.forward(&input, pos)?;

    // out shape: [1, seq, vocab]
    let seq_len = out.dim(1)?;
    let logits = out.i((0, seq_len - 1))?;      // last token
    let logits = logits.to_dtype(DType::F32)?;  // f32 for sampling

    let next = lp.sample(&logits)?;
    tokens.push(next);

let decoded = tokenizer
    .decode(&[next], false)
    .map_err(|e| anyhow!("decode error: {}", e))?;

print!("{decoded}");
io::stdout().flush()?;


    pos += context.len();

    if next == eos_id {
        break;
    }
}


    println!("\n✔ Done");
    Ok(())
}

/// Random seed helper
fn seed() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs()
}
