use anyhow::{anyhow, Result};
use std::{path::PathBuf, sync::Arc};

use crate::inference::{
    llama_cpp_service::LlamaCppService,
    roberta_classifier::RobertaClassifier,
    roberta_phatic_gate::RobertaPhaticGate,
};

pub struct ModelManager {
    pub mistral_llama: Arc<LlamaCppService>,
    pub roberta: Arc<RobertaClassifier>,
    pub phatic_gate: Arc<RobertaPhaticGate>,
}

impl ModelManager {
    pub async fn new() -> Result<Self> {
        let roberta_dir = PathBuf::from("/home/yaro/projects/ktulhu-main/models/roberta1");

        let env_llama_cli_bin = std::env::var("LLAMA_CLI_BIN").ok().filter(|s| !s.trim().is_empty());
        let env_llama_cli_model =
            std::env::var("LLAMA_CLI_MODEL").ok().filter(|s| !s.trim().is_empty());

        let default_llama_bin = PathBuf::from("llama.cpp/build/bin/llama-cli");
        let llama_cli_bin_path = env_llama_cli_bin
            .as_deref()
            .map(PathBuf::from)
            .or_else(|| {
                if default_llama_bin.exists() {
                    Some(default_llama_bin.clone())
                } else {
                    None
                }
            });
        if env_llama_cli_bin.is_none() {
            match &llama_cli_bin_path {
                Some(path) => println!(
                    "ℹ️  LLAMA_CLI_BIN not set – defaulting to {}",
                    path.display()
                ),
                None => println!(
                    "ℹ️  llama-cli binary not found (looked for {})",
                    default_llama_bin.display()
                ),
            }
        }

        let llama_cli_model_candidates = [
            PathBuf::from(
                "models/Ministral3-14B-Resoning-gguf/Ministral-3-14B-Instruct-2512-Q8_0.gguf",
            ),
            PathBuf::from(
                "models/Ministral3-14B-Resoning-gguf/Ministral-3-14B-Reasoning-2512-Q8_0.gguf",
            ),
        ];
        let llama_cli_model_path = env_llama_cli_model
            .as_deref()
            .map(PathBuf::from)
            .or_else(|| {
                llama_cli_model_candidates
                    .iter()
                    .find(|p| p.exists())
                    .cloned()
            });
        if env_llama_cli_model.is_none() {
            match &llama_cli_model_path {
                Some(path) => println!(
                    "ℹ️  LLAMA_CLI_MODEL not set – defaulting to {}",
                    path.display()
                ),
                None => {
                    let default_dir = PathBuf::from("models/Ministral3-14B-Resoning-gguf");
                    println!(
                        "ℹ️  No GGUF detected in {}",
                        default_dir.display()
                    );
                }
            }
        }

        let llama_ctx_size = std::env::var("LLAMA_CLI_CTX")
            .ok()
            .and_then(|v| v.parse::<u32>().ok())
            .unwrap_or(4096);
        let llama_max_tokens = std::env::var("LLAMA_CLI_MAX_TOKENS")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(512);
        let llama_temp = std::env::var("LLAMA_CLI_TEMP")
            .ok()
            .and_then(|v| v.parse::<f32>().ok())
            .unwrap_or(0.8);
        let llama_top_p = std::env::var("LLAMA_CLI_TOP_P")
            .ok()
            .and_then(|v| v.parse::<f32>().ok())
            .unwrap_or(0.9);
        let llama_top_k = std::env::var("LLAMA_CLI_TOP_K")
            .ok()
            .and_then(|v| v.parse::<i32>().ok())
            .unwrap_or(40);
        let llama_gpu_layers = std::env::var("LLAMA_CLI_NGL")
            .ok()
            .and_then(|v| v.parse::<i32>().ok());
        let llama_threads = std::env::var("LLAMA_CLI_THREADS")
            .ok()
            .and_then(|v| v.parse::<i32>().ok());

        let mistral_llama = match (llama_cli_bin_path, llama_cli_model_path) {
            (Some(_bin), Some(model)) => Arc::new(LlamaCppService::new(
                model,
                llama_ctx_size,
                llama_max_tokens,
                llama_temp,
                llama_top_p,
                llama_top_k,
                llama_gpu_layers,
                llama_threads,
            )?),
            _ => {
                return Err(anyhow!(
                    "LLAMA_CLI_MODEL not configured and default GGUF not found"
                ))
            }
        };

        let classifier_dir = roberta_dir.clone();
        let roberta = tokio::task::spawn_blocking(move || {
            RobertaClassifier::load(classifier_dir, 0)
        })
        .await??;
        let roberta = Arc::new(roberta);

        let env_phatic_dir =
            std::env::var("PHATIC_MODEL_DIR").ok().filter(|s| !s.trim().is_empty());
        let phatic_dir = env_phatic_dir
            .as_deref()
            .map(PathBuf::from)
            .or_else(|| {
                let candidate = PathBuf::from("/home/yaro/projects/ktulhu-main/models/roberta1/out");
                if candidate.exists() {
                    Some(candidate)
                } else {
                    None
                }
            })
            .ok_or_else(|| {
                anyhow!(
                    "PHATIC_MODEL_DIR not configured and models/roberta/out was not found"
                )
            })?;
        match env_phatic_dir {
            Some(_) => println!("ℹ️  PHATIC_MODEL_DIR -> {}", phatic_dir.display()),
            None => println!(
                "ℹ️  PHATIC_MODEL_DIR not set – defaulting to {}",
                phatic_dir.display()
            ),
        }

        if !phatic_dir.join("tokenizer.json").exists() {
            return Err(anyhow!(
                "phatic gate tokenizer.json not found under {}",
                phatic_dir.display()
            ));
        }
        if !phatic_dir.join("config.json").exists() {
            return Err(anyhow!(
                "phatic gate config.json not found under {}",
                phatic_dir.display()
            ));
        }
        if !phatic_dir.join("model.safetensors").exists() {
            return Err(anyhow!(
                "phatic gate model.safetensors not found under {}",
                phatic_dir.display()
            ));
        }

        let phatic_gate = tokio::task::spawn_blocking(move || {
            RobertaPhaticGate::load(phatic_dir, 0)
        })
        .await??;
        let phatic_gate = Arc::new(phatic_gate);

        Ok(Self {
            mistral_llama,
            roberta,
            phatic_gate,
        })
    }
}
