#pragma once
#include "llama.h"
#include "telemetry.hpp"
#include <memory>
#include <string>

struct EngineConfig {
    std::string model_path;
    int n_ctx = 2048;
    int n_threads = 4;
    int n_gpu_layers = 0; // Set > 0 to offload to CUDA/Metal/Vulkan
    int max_tokens = 128;
};

class InferenceEngine {
public:
    explicit InferenceEngine(const EngineConfig& config);
    ~InferenceEngine();

    BenchmarkMetrics run_benchmark(const std::string& prompt, bool stream_output = true);

private:
    EngineConfig config_;
    double load_time_ms_ = 0.0;

    // Custom deleters for deterministic RAII memory management
    struct ModelDeleter { void operator()(llama_model* m) { llama_model_free(m); } };
    struct ContextDeleter { void operator()(llama_context* c) { llama_free(c); } };
    struct SamplerDeleter { void operator()(llama_sampler* s) { llama_sampler_free(s); } };

    std::unique_ptr<llama_model, ModelDeleter> model_;
    std::unique_ptr<llama_context, ContextDeleter> ctx_;
    std::unique_ptr<llama_sampler, SamplerDeleter> sampler_;
};
