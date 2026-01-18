use anyhow::{anyhow, Result};
use std::{path::PathBuf, sync::Arc};

use crate::inference::{intent_router::RobertaIntentRouter, llama_cpp_service::LlamaCppService};

pub struct ModelManager {
    pub mistral_llama: Arc<LlamaCppService>,
    pub intent_router: Arc<RobertaIntentRouter>,
}

impl ModelManager {
    pub async fn new() -> Result<Self> {
        let default_intent_router_dir =
            PathBuf::from("/home/yaro/projects/ktulhu-main/models/robertaTunedHeads");

        let env_llama_cli_bin = std::env::var("LLAMA_CLI_BIN")
            .ok()
            .filter(|s| !s.trim().is_empty());
        let env_llama_cli_model = std::env::var("LLAMA_CLI_MODEL")
            .ok()
            .filter(|s| !s.trim().is_empty());

        let default_llama_bin = PathBuf::from("llama.cpp/build/bin/llama-cli");
        let llama_cli_bin_path = env_llama_cli_bin.as_deref().map(PathBuf::from).or_else(|| {
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
        let llama_cli_model_path =
            env_llama_cli_model
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
                    println!("ℹ️  No GGUF detected in {}", default_dir.display());
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

        let env_intent_router_dir = std::env::var("INTENT_ROUTER_DIR")
            .ok()
            .filter(|s| !s.trim().is_empty());
        let legacy_phatic_dir = if env_intent_router_dir.is_none() {
            std::env::var("PHATIC_MODEL_DIR")
                .ok()
                .filter(|s| !s.trim().is_empty())
        } else {
            None
        };
        let (intent_router_dir, log_msg) = if let Some(dir) = env_intent_router_dir {
            (PathBuf::from(&dir), format!("INTENT_ROUTER_DIR -> {}", dir))
        } else if let Some(dir) = legacy_phatic_dir {
            (
                PathBuf::from(&dir),
                format!("PHATIC_MODEL_DIR (legacy) -> {}", dir),
            )
        } else {
            (
                default_intent_router_dir.clone(),
                format!(
                    "INTENT_ROUTER_DIR not set – defaulting to {}",
                    default_intent_router_dir.display()
                ),
            )
        };
        println!("ℹ️  {log_msg}");

        let intent_router_dir = if intent_router_dir.exists() {
            intent_router_dir
        } else {
            let fallback = intent_router_dir.join("out");
            if fallback.exists() {
                println!(
                    "ℹ️  intent router directory missing, falling back to {}",
                    fallback.display()
                );
                fallback
            } else {
                intent_router_dir
            }
        };

        if !intent_router_dir.join("tokenizer.json").exists() {
            return Err(anyhow!(
                "tokenizer.json not found under {}",
                intent_router_dir.display()
            ));
        }
        if !intent_router_dir.join("config.json").exists() {
            return Err(anyhow!(
                "config.json not found under {}",
                intent_router_dir.display()
            ));
        }
        let has_weights = ["model.safetensors", "pytorch_model.bin", "model.bin"]
            .iter()
            .any(|name| intent_router_dir.join(name).exists());
        if !has_weights {
            return Err(anyhow!(
                "no model weights found under {} (expected model.safetensors or pytorch_model.bin)",
                intent_router_dir.display()
            ));
        }

        let use_phatic_head = std::env::var("INTENT_ROUTER_PHATIC")
            .ok()
            .map(|v| v != "0")
            .unwrap_or(true);

        let router_dir_clone = intent_router_dir.clone();
        let intent_router = tokio::task::spawn_blocking(move || {
            RobertaIntentRouter::load(router_dir_clone, 0, use_phatic_head)
        })
        .await??;
        let intent_router = Arc::new(intent_router);

        Ok(Self {
            mistral_llama,
            intent_router,
        })
    }
}
