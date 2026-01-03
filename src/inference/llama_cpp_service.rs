use anyhow::{anyhow, bail, Result};
use rand::{thread_rng, Rng};
use std::ffi::CString;
use std::os::raw::c_char;
use std::os::unix::ffi::OsStrExt;
use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use tokio::sync::mpsc;

#[allow(
    non_camel_case_types,
    non_snake_case,
    non_upper_case_globals,
    dead_code,
    clippy::all
)]
mod ffi {
    include!(concat!(env!("OUT_DIR"), "/llama_bindings.rs"));
}

static BACKEND_ONCE: OnceLock<()> = OnceLock::new();
static BACKEND_USERS: AtomicUsize = AtomicUsize::new(0);

fn init_backend() {
    BACKEND_ONCE.get_or_init(|| unsafe {
        ffi::llama_backend_init();
    });
    BACKEND_USERS.fetch_add(1, Ordering::SeqCst);
}

fn shutdown_backend() {
    if BACKEND_USERS.fetch_sub(1, Ordering::SeqCst) == 1 {
        unsafe {
            ffi::llama_backend_free();
        }
    }
}

pub struct LlamaCppService {
    inner: Arc<Mutex<LlamaInner>>,
}

struct LlamaInner {
    model: *mut ffi::llama_model,
    ctx: *mut ffi::llama_context,
    sampler: *mut ffi::llama_sampler,
    vocab: *const ffi::llama_vocab,
    eos_token: ffi::llama_token,
    n_batch: i32,
    max_tokens: usize,
    n_past: i32,
}

unsafe impl Send for LlamaInner {}
unsafe impl Sync for LlamaInner {}

impl Drop for LlamaInner {
    fn drop(&mut self) {
        unsafe {
            if !self.sampler.is_null() {
                ffi::llama_sampler_free(self.sampler);
            }
            if !self.ctx.is_null() {
                ffi::llama_free(self.ctx);
            }
            if !self.model.is_null() {
                ffi::llama_model_free(self.model);
            }
        }
        shutdown_backend();
    }
}

impl LlamaCppService {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        model_path: impl AsRef<Path>,
        ctx_length: u32,
        max_tokens: usize,
        temperature: f32,
        top_p: f32,
        top_k: i32,
        gpu_layers: Option<i32>,
        threads: Option<i32>,
    ) -> Result<Self> {
        init_backend();
        println!("⚡️ llama.cpp params: ctx={ctx_length} max_tokens={max_tokens} temp={temperature} top_p={top_p} top_k={top_k} gpu_layers={:?} threads={:?}", gpu_layers, threads);
        let path = model_path.as_ref();
        if !path.exists() {
            bail!("GGUF model not found at {}", path.display());
        }

        let path_cstr = CString::new(path.as_os_str().as_bytes())
            .map_err(|_| anyhow!("model path contains interior null byte"))?;

        let mut model_params = unsafe { ffi::llama_model_default_params() };
        model_params.n_gpu_layers = gpu_layers.unwrap_or(-1);
        model_params.use_mmap = true;

        let model =
            unsafe { ffi::llama_model_load_from_file(path_cstr.as_ptr(), model_params) };
        if model.is_null() {
            shutdown_backend();
            bail!("failed to load model from {}", path.display());
        }

        let vocab = unsafe { ffi::llama_model_get_vocab(model) };
        if vocab.is_null() {
            unsafe {
                ffi::llama_model_free(model);
            }
            shutdown_backend();
            bail!("model vocabulary unavailable");
        }

        let mut ctx_params = unsafe { ffi::llama_context_default_params() };
        ctx_params.n_ctx = ctx_length;
        ctx_params.n_batch = 512;
        ctx_params.n_ubatch = 512;
        let threads = threads.unwrap_or_else(|| num_cpus::get_physical() as i32);
        ctx_params.n_threads = threads;
        ctx_params.n_threads_batch = threads;
        ctx_params.offload_kqv = true;

        let ctx = unsafe { ffi::llama_init_from_model(model, ctx_params) };
        if ctx.is_null() {
            unsafe {
                ffi::llama_model_free(model);
            }
            shutdown_backend();
            bail!("failed to create llama context");
        }

        let mut sampler_params = unsafe { ffi::llama_sampler_chain_default_params() };
        sampler_params.no_perf = true;

        let sampler = unsafe { ffi::llama_sampler_chain_init(sampler_params) };
        if sampler.is_null() {
            unsafe {
                ffi::llama_free(ctx);
                ffi::llama_model_free(model);
            }
            shutdown_backend();
            bail!("failed to create sampler chain");
        }

        unsafe {
            if top_k > 0 {
                let topk = ffi::llama_sampler_init_top_k(top_k);
                ffi::llama_sampler_chain_add(sampler, topk);
            }
            if top_p < 0.9999 {
                let topp = ffi::llama_sampler_init_top_p(top_p, 1);
                ffi::llama_sampler_chain_add(sampler, topp);
            }
            if (temperature - 1.0).abs() > f32::EPSILON {
                let temp = ffi::llama_sampler_init_temp(temperature);
                ffi::llama_sampler_chain_add(sampler, temp);
            }
            let seed = thread_rng().gen();
            let dist = ffi::llama_sampler_init_dist(seed);
            ffi::llama_sampler_chain_add(sampler, dist);
        }

        let eos_token = unsafe { ffi::llama_vocab_eos(vocab) };

        let inner = LlamaInner {
            model,
            ctx,
            sampler,
            vocab,
            eos_token,
            n_batch: 512,
            max_tokens,
            n_past: 0,
        };

        Ok(Self {
            inner: Arc::new(Mutex::new(inner)),
        })
    }

    pub fn generate_stream(
        &self,
        prompt: String,
        cancel: Arc<AtomicBool>,
    ) -> mpsc::Receiver<String> {
        let (tx, rx) = mpsc::channel(128);
        let inner = self.inner.clone();
        tokio::task::spawn_blocking(move || {
            if let Err(err) =
                LlamaInner::run_with_guard(inner, &prompt, cancel, tx.clone())
            {
                let _ = tx.blocking_send(format!("llama.cpp error: {err}"));
            }
        });
        rx
    }

    pub async fn generate_completion(
        &self,
        prompt: String,
        cancel: Arc<AtomicBool>,
    ) -> Result<String> {
        let mut rx = self.generate_stream(prompt, cancel);
        let mut out = String::new();
        while let Some(chunk) = rx.recv().await {
            out.push_str(&chunk);
        }
        Ok(out)
    }
}

impl LlamaInner {
    fn run_with_guard(
        inner: Arc<Mutex<LlamaInner>>,
        prompt: &str,
        cancel: Arc<AtomicBool>,
        tx: mpsc::Sender<String>,
    ) -> Result<()> {
        let mut guard = inner
            .lock()
            .map_err(|_| anyhow!("failed to lock llama context"))?;
        guard.run(prompt, cancel, tx)
    }

    fn run(
        &mut self,
        prompt: &str,
        cancel: Arc<AtomicBool>,
        tx: mpsc::Sender<String>,
    ) -> Result<()> {
        unsafe {
            let mem = ffi::llama_get_memory(self.ctx);
            ffi::llama_memory_clear(mem, true);
            ffi::llama_sampler_reset(self.sampler);
        }
        self.n_past = 0;

        let prompt_tokens = self.tokenize(prompt)?;
        self.decode_sequence(&prompt_tokens)?;
        let mut pending = Vec::new();

        for _ in 0..self.max_tokens {
            if cancel.load(Ordering::SeqCst) {
                break;
            }
            let token = unsafe { ffi::llama_sampler_sample(self.sampler, self.ctx, -1) };
            if token == self.eos_token || token == ffi::LLAMA_TOKEN_NULL {
                break;
            }
            unsafe {
                ffi::llama_sampler_accept(self.sampler, token);
            }
            let piece = self.render_token_bytes(token)?;
            if piece.is_empty() {
                self.flush_pending(&mut pending, &tx)?;
                continue;
            }
            pending.extend_from_slice(&piece);
            self.flush_pending(&mut pending, &tx)?;
            self.decode_sequence(std::slice::from_ref(&token))?;
        }
        // Flush any remaining buffered bytes.
        self.flush_pending(&mut pending, &tx)?;
        Ok(())
    }

    fn tokenize(&self, text: &str) -> Result<Vec<ffi::llama_token>> {
        let mut buf =
            vec![0 as ffi::llama_token; text.len().max(32)];
        let bytes = text.as_bytes();
        let text_ptr = bytes.as_ptr() as *const c_char;
        loop {
            let res = unsafe {
                ffi::llama_tokenize(
                    self.vocab,
                    text_ptr,
                    bytes.len() as i32,
                    buf.as_mut_ptr(),
                    buf.len() as i32,
                    false,
                    true,
                )
            };
            if res >= 0 {
                buf.truncate(res as usize);
                return Ok(buf);
            }
            let needed = (-res) as usize + 8;
            buf.resize(needed, 0);
        }
    }

    fn decode_sequence(&mut self, tokens: &[ffi::llama_token]) -> Result<()> {
        if tokens.is_empty() {
            return Ok(());
        }
        let mut processed = 0usize;
        while processed < tokens.len() {
            let take = (tokens.len() - processed).min(self.n_batch as usize);
            let chunk = &tokens[processed..processed + take];
            let mut batch = unsafe { ffi::llama_batch_init(self.n_batch, 0, 1) };
            unsafe {
                let token_slice =
                    std::slice::from_raw_parts_mut(batch.token, chunk.len());
                token_slice.copy_from_slice(chunk);

                let pos_slice =
                    std::slice::from_raw_parts_mut(batch.pos, chunk.len());
                for (i, slot) in pos_slice.iter_mut().enumerate() {
                    *slot = (self.n_past + i as i32) as ffi::llama_pos;
                }

                let n_seq_slice =
                    std::slice::from_raw_parts_mut(batch.n_seq_id, chunk.len());
                let seq_heads =
                    std::slice::from_raw_parts_mut(batch.seq_id, chunk.len());
                let logits_slice =
                    std::slice::from_raw_parts_mut(batch.logits, chunk.len());

                for i in 0..chunk.len() {
                    n_seq_slice[i] = 1;
                    let seq_slot =
                        std::slice::from_raw_parts_mut(seq_heads[i], 1);
                    seq_slot[0] = 0;
                    logits_slice[i] = if i == chunk.len() - 1 { 1 } else { 0 };
                }
            }
            batch.n_tokens = chunk.len() as i32;
            let err = unsafe { ffi::llama_decode(self.ctx, batch) };
            unsafe { ffi::llama_batch_free(batch) };
            if err != 0 {
                bail!("llama_decode failed with code {}", err);
            }
            processed += chunk.len();
            self.n_past += chunk.len() as i32;
        }
        Ok(())
    }

    fn render_token_bytes(&self, token: ffi::llama_token) -> Result<Vec<u8>> {
        let mut buf = vec![0u8; 64];
        loop {
            let res = unsafe {
                ffi::llama_token_to_piece(
                    self.vocab,
                    token,
                    buf.as_mut_ptr() as *mut c_char,
                    buf.len() as i32,
                    0,
                    false,
                )
            };
            if res >= 0 {
                let slice = &buf[..res as usize];
                return Ok(slice.to_vec());
            }
            let needed = (-res) as usize + 8;
            buf.resize(needed, 0);
        }
    }

    fn flush_pending(
        &self,
        pending: &mut Vec<u8>,
        tx: &mpsc::Sender<String>,
    ) -> Result<()> {
        loop {
            if pending.is_empty() {
                return Ok(());
            }
            match std::str::from_utf8(pending) {
                Ok(valid) => {
                    if !valid.is_empty() {
                        if tx.blocking_send(valid.to_string()).is_err() {
                            return Ok(());
                        }
                    }
                    pending.clear();
                    return Ok(());
                }
                Err(err) => {
                    let valid_up_to = err.valid_up_to();
                    if valid_up_to > 0 {
                        let chunk = unsafe {
                            std::str::from_utf8_unchecked(&pending[..valid_up_to])
                        };
                        if !chunk.is_empty() && tx.blocking_send(chunk.to_string()).is_err() {
                            return Ok(());
                        }
                        pending.drain(..valid_up_to);
                        continue;
                    }
                    if let Some(error_len) = err.error_len() {
                        // Drop invalid sequence and emit replacement char.
                        pending.drain(..error_len);
                        if tx.blocking_send("�".to_string()).is_err() {
                            return Ok(());
                        }
                    } else {
                        // Need more bytes for a valid UTF-8 sequence.
                        return Ok(());
                    }
                }
            }
        }
    }
}
