#include "engine.hpp"
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>

void print_usage() {
    std::cout << "Usage: llm-bench -m <model.gguf> [options]\n"
              << "Options:\n"
              << "  -m <path>     Path to GGUF model file (required)\n"
              << "  -p <prompt>   Input prompt string\n"
              << "  -n <tokens>   Max tokens to generate (default: 128)\n"
              << "  -t <threads>  CPU threads to use (default: 4)\n"
              << "  -ngl <layers> GPU layers to offload (default: 0)\n"
              << "  --json <file> Export benchmark metrics to JSON\n"
              << "  --quiet       Suppress token streaming output\n";
}

void export_json(const std::string& path, const EngineConfig& cfg, const BenchmarkMetrics& m) {
    std::ofstream out(path);
    out << "{\n"
        << "  \"model\": \"" << cfg.model_path << "\",\n"
        << "  \"threads\": " << cfg.n_threads << ",\n"
        << "  \"gpu_layers\": " << cfg.n_gpu_layers << ",\n"
        << "  \"model_load_ms\": " << m.model_load_ms << ",\n"
        << "  \"ttft_ms\": " << m.ttft_ms << ",\n"
        << "  \"prompt_tokens\": " << m.prompt_tokens << ",\n"
        << "  \"prompt_tps\": " << m.prompt_tps << ",\n"
        << "  \"generated_tokens\": " << m.generated_tokens << ",\n"
        << "  \"decode_tps\": " << m.decode_tps << ",\n"
        << "  \"peak_rss_mb\": " << m.peak_rss_mb << "\n"
        << "}\n";
}

int main(int argc, char** argv) {
    EngineConfig config;
    std::string prompt = "Explain the difference between latency and throughput in distributed systems.";
    std::string json_out;
    bool stream = true;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-m" && i + 1 < argc) config.model_path = argv[++i];
        else if (arg == "-p" && i + 1 < argc) prompt = argv[++i];
        else if (arg == "-n" && i + 1 < argc) config.max_tokens = std::stoi(argv[++i]);
        else if (arg == "-t" && i + 1 < argc) config.n_threads = std::stoi(argv[++i]);
        else if (arg == "-ngl" && i + 1 < argc) config.n_gpu_layers = std::stoi(argv[++i]);
        else if (arg == "--json" && i + 1 < argc) json_out = argv[++i];
        else if (arg == "--quiet") stream = false;
        else { print_usage(); return 1; }
    }

    if (config.model_path.empty()) {
        print_usage();
        return 1;
    }

    try {
        std::cout << "[INFO] Loading model: " << config.model_path << "...\n";
        InferenceEngine engine(config);

        std::cout << "[INFO] Running inference benchmark...\n\n";
        BenchmarkMetrics metrics = engine.run_benchmark(prompt, stream);

        std::cout << std::fixed << std::setprecision(2);
        std::cout << "\n================ BENCHMARK REPORT ================\n"
                  << "Model Load Time   : " << metrics.model_load_ms << " ms\n"
                  << "TTFT (Prefill)    : " << metrics.ttft_ms << " ms (" << metrics.prompt_tokens << " tokens)\n"
                  << "Prompt Speed      : " << metrics.prompt_tps << " tok/s\n"
                  << "Generation Speed  : " << metrics.decode_tps << " tok/s (" << metrics.generated_tokens << " tokens)\n"
                  << "Peak Memory (RSS) : " << metrics.peak_rss_mb << " MB\n"
                  << "==================================================\n";

        if (!json_out.empty()) {
            export_json(json_out, config, metrics);
            std::cout << "[INFO] Metrics saved to " << json_out << "\n";
        }
    } catch (const std::exception& ex) {
        std::cerr << "[ERROR] " << ex.what() << "\n";
        return 1;
    }

    return 0;
}
