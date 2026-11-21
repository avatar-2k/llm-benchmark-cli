#include "engine.hpp"

#include <iostream>
#include <stdexcept>
#include <vector>

InferenceEngine::InferenceEngine(const EngineConfig& config) : config_(config) {
    llama_backend_init();

    ScopedTimer load_timer;

    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = config_.n_gpu_layers;

    model_.reset(llama_model_load_from_file(config_.model_path.c_str(), model_params));
    if (!model_) {
        throw std::runtime_error("unable to load model: " + config_.model_path);
    }

    llama_context_params context_params = llama_context_default_params();
    context_params.n_ctx = config_.n_ctx;
    context_params.n_batch = config_.n_ctx;
    context_params.n_threads = config_.n_threads;
    context_params.n_threads_batch = config_.n_threads;
    context_params.no_perf = false;

    ctx_.reset(llama_init_from_model(model_.get(), context_params));
    if (!ctx_) {
        throw std::runtime_error("unable to create inference context");
    }

    llama_sampler_chain_params sampler_params = llama_sampler_chain_default_params();
    sampler_params.no_perf = false;
    sampler_.reset(llama_sampler_chain_init(sampler_params));
    llama_sampler_chain_add(sampler_.get(), llama_sampler_init_greedy());

    load_time_ms_ = load_timer.elapsed_ms();
}

InferenceEngine::~InferenceEngine() {
    llama_backend_free();
}

BenchmarkMetrics InferenceEngine::run_benchmark(const std::string& prompt, bool stream_output) {
    BenchmarkMetrics metrics;
    metrics.model_load_ms = load_time_ms_;

    const llama_vocab* vocab = llama_model_get_vocab(model_.get());
    const int prompt_count = -llama_tokenize(
        vocab, prompt.c_str(), prompt.size(), nullptr, 0, true, true);
    if (prompt_count <= 0) {
        throw std::runtime_error("failed to tokenize prompt");
    }

    std::vector<llama_token> prompt_tokens(prompt_count);
    if (llama_tokenize(
            vocab, prompt.c_str(), prompt.size(),
            prompt_tokens.data(), prompt_tokens.size(), true, true) < 0) {
        throw std::runtime_error("failed to tokenize prompt");
    }

    metrics.prompt_tokens = prompt_count;
    ScopedTimer prompt_timer;

    llama_batch prompt_batch = llama_batch_get_one(prompt_tokens.data(), prompt_count);
    if (llama_decode(ctx_.get(), prompt_batch) != 0) {
        throw std::runtime_error("failed to evaluate prompt");
    }

    metrics.ttft_ms = prompt_timer.elapsed_ms();
    metrics.prompt_tps = metrics.ttft_ms > 0.0
        ? (prompt_count * 1000.0 / metrics.ttft_ms)
        : 0.0;

    llama_token eos = llama_vocab_eos(vocab);
    llama_token token = llama_sampler_sample(sampler_.get(), ctx_.get(), -1);
    if (llama_vocab_is_eog(vocab, token)) {
        token = eos;
    }

    ScopedTimer decode_timer;
    for (int i = 0; i < config_.max_tokens; ++i) {
        if (llama_vocab_is_eog(vocab, token)) {
            break;
        }

        if (stream_output) {
            char piece[128];
            const int piece_size = llama_token_to_piece(vocab, token, piece, sizeof(piece), 0, true);
            if (piece_size < 0) {
                throw std::runtime_error("failed to convert generated token");
            }
            std::cout.write(piece, piece_size);
            std::cout.flush();
        }

        ++metrics.generated_tokens;
        llama_sampler_accept(sampler_.get(), token);

        llama_batch next_batch = llama_batch_get_one(&token, 1);
        if (llama_decode(ctx_.get(), next_batch) != 0) {
            throw std::runtime_error("failed during token generation");
        }

        token = llama_sampler_sample(sampler_.get(), ctx_.get(), -1);
    }

    metrics.decode_tps = metrics.generated_tokens > 0 && decode_timer.elapsed_ms() > 0.0
        ? (metrics.generated_tokens * 1000.0 / decode_timer.elapsed_ms())
        : 0.0;
    metrics.peak_rss_mb = get_peak_rss_mb();
    return metrics;
}
