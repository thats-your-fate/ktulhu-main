# llama.cpp Backend Environment

The server can swap the internal Candle backend for the local `llama.cpp` bindings when the following environment variables are set.

## Required

- `LLAMA_CLI_BIN`
  - Absolute or relative path to `llama-cli`. We only use this to locate the compiled `libllama.so`.
  - Defaults to `llama.cpp/build/bin/llama-cli` if present (so the sibling shared libraries are found).
- `LLAMA_CLI_MODEL`
  - Path to a GGUF checkpoint.
  - Defaults to the first GGUF found under `models/Ministral3-14B-Resoning-gguf/` (prefers the Instruct file).

## Optional Tuning

- `LLAMA_CLI_CTX` (default: `4096`)
- `LLAMA_CLI_MAX_TOKENS` (default: `512`)
- `LLAMA_CLI_TEMP` (default: `0.8`)
- `LLAMA_CLI_TOP_P` (default: `0.9`)
- `LLAMA_CLI_TOP_K` (default: `40`)
- `LLAMA_CLI_NGL` (GPU offload layers, default: auto)
- `LLAMA_CLI_THREADS` (default: auto)
- `LLAMA_CPP_LIBDIR` (optional) override search path for `libllama.so` if it lives outside `llama.cpp/build/bin`

When both required paths resolve, `ModelManager` logs `⚙️ Using llama.cpp backend for Mistral GGUF` and all Mistral traffic streams through the in-process `libllama` bindings instead of Candle.

## Classifier Paths

- `INTENT_ROUTER_DIR`
  - Optional override for the multi-head RoBERTa intent router checkpoint (falls back to the legacy `PHATIC_MODEL_DIR` env var or `models/roberta1`).
- `INTENT_ROUTER_PHATIC`
  - Set to `0` when the checkpoint does not contain the optional phatic head.
