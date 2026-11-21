# llm-bench-cli

A lightweight, high-performance C++20 command-line runner and inference benchmarking tool built directly on top of [`llama.cpp`](https://github.com/ggml-org/llama.cpp).

## Project Goal

`llm-bench-cli` is designed to measure real-world systems performance and hardware telemetry for local Large Language Model (LLM) inference without Python runtime overhead. 

Key engineering objectives:
* **Deterministic Resource Management:** Uses modern C++20 RAII (`std::unique_ptr` with custom deleters) to wrap `libllama` C pointers, guaranteeing leak-free model and KV-cache lifecycles.
* **Granular Latency Profiling:** Separates **Time-To-First-Token (TTFT / Prefill)** from **Autoregressive Decode Speed** using microsecond-precision monotonic clocks (`std::chrono::steady_clock`).
* **OS-Level Memory Telemetry:** Queries kernel APIs (`getrusage` on Linux/macOS, `GetProcessMemoryInfo` on Windows) to track Peak Resident Set Size (RSS).
* **CI/CD Ready:** Exports structured JSON reports for automated regression tracking across model quantizations, thread counts, and hardware backends.

---

## Prerequisites

* **C++20 Compiler:** GCC 11+, Clang 14+, or MSVC 2022+
* **CMake:** Version 3.20 or higher
* **Git:** Required by CMake `FetchContent` to pull `llama.cpp` automatically
* **Model Weights:** A quantized model in `.gguf` format (e.g., Llama-3.2, Qwen2.5, or Mistral from [Hugging Face](https://huggingface.co))

---

## How to Build

The project uses CMake `FetchContent` to download and compile `llama.cpp` automatically.

### Standard CPU Build
```bash
git clone https://github.com/avatar-2k/llm-benchmark-cli.git
cd llm-benchmark-cli

cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

### Hardware-Accelerated Builds (Optional)
Pass standard `ggml` backend flags during the CMake configuration step to enable GPU offloading:

* **NVIDIA CUDA:**
  ```bash
  cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON
  cmake --build build -j
  ```
* **Apple Silicon (Metal):**
  ```bash
  cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_METAL=ON
  cmake --build build -j
  ```
* **Vulkan (AMD / Intel / Cross-Platform):**
  ```bash
  cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_VULKAN=ON
  cmake --build build -j
  ```

---

## How to Run

### 1. Download a GGUF Model
Create a `models/` directory and download any `.gguf` model file:
```bash
mkdir -p models
# Example: Place your downloaded .gguf file inside models/
```

### 2. Basic Benchmark Run
Run inference with token streaming enabled and print the telemetry summary:
```bash
./build/llm-bench -m models/your-model-q4_k_m.gguf
```

### 3. Custom Configuration & JSON Export
Benchmark with 8 CPU threads, 32 offloaded GPU layers, and export metrics to JSON while suppressing console text streaming:
```bash
./build/llm-bench \
  -m models/your-model-q4_k_m.gguf \
  -p "Explain the difference between latency and throughput in distributed systems." \
  -n 256 \
  -t 8 \
  -ngl 32 \
  --quiet \
  --json benchmark_results.json
```

### CLI Options

| Flag | Description | Default |
| :--- | :--- | :--- |
| `-m <path>` | Path to the `.gguf` model file *(Required)* | — |
| `-p <prompt>` | Input prompt string | Built-in systems prompt |
| `-n <tokens>` | Maximum number of tokens to generate | `128` |
| `-t <threads>` | Number of CPU threads for prompt & token generation | `4` |
| `-ngl <layers>` | Number of transformer layers to offload to GPU | `0` |
| `--json <file>` | Write structured benchmark metrics to a JSON file | Disabled |
| `--quiet` | Suppress streamed token output during generation | Disabled |

---

## Sample Output

### Terminal Report
```text
[INFO] Loading model: models/Llama-3.2-3B-Instruct-Q4_K_M.gguf...
[INFO] Running inference benchmark...

================ BENCHMARK REPORT ================
Model Load Time   : 342.18 ms
TTFT (Prefill)    : 48.65 ms (18 tokens)
Prompt Speed      : 369.99 tok/s
Generation Speed  : 64.12 tok/s (256 tokens)
Peak Memory (RSS) : 2318.40 MB
==================================================
```

### JSON Output (`--json`)
```json
{
  "model": "models/Llama-3.2-3B-Instruct-Q4_K_M.gguf",
  "threads": 8,
  "gpu_layers": 32,
  "model_load_ms": 342.18,
  "ttft_ms": 48.65,
  "prompt_tokens": 18,
  "prompt_tps": 369.99,
  "generated_tokens": 256,
  "decode_tps": 64.12,
  "peak_rss_mb": 2318.40
}
```

---

## Project Structure

```text
llm-bench-cli/
├── CMakeLists.txt        # CMake build config with FetchContent for llama.cpp
├── README.md             # Project documentation
└── src/
    ├── main.cpp          # CLI argument parsing, reporting, and JSON export
    ├── engine.hpp        # RAII wrappers for llama_model, context, and sampler
    ├── engine.cpp        # Tokenization, prefill batching, and decode loop
    └── telemetry.hpp     # High-resolution timers and cross-platform RSS memory tracking
```
