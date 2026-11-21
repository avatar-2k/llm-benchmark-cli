#pragma once
#include <chrono>
#include <cstdint>

#if defined(__linux__) || defined(__APPLE__)
#include <sys/resource.h>
#elif defined(_WIN32)
#include <windows.h>
#include <psapi.h>
#endif

struct BenchmarkMetrics {
    double model_load_ms = 0.0;
    double ttft_ms = 0.0;              // Time to First Token (Prompt Eval)
    double prompt_tps = 0.0;           // Prompt tokens / sec
    double decode_tps = 0.0;           // Generated tokens / sec
    int prompt_tokens = 0;
    int generated_tokens = 0;
    double peak_rss_mb = 0.0;
};

class ScopedTimer {
public:
    ScopedTimer() : start_(std::chrono::steady_clock::now()) {}
    double elapsed_ms() const {
        auto now = std::chrono::steady_clock::now();
        return std::chrono::duration<double, std::milli>(now - start_).count();
    }
private:
    std::chrono::steady_clock::time_point start_;
};

inline double get_peak_rss_mb() {
#if defined(__linux__)
    struct rusage usage;
    if (getrusage(RUSAGE_SELF, &usage) == 0) {
        return static_cast<double>(usage.ru_maxrss) / 1024.0; // Linux reports KB
    }
#elif defined(__APPLE__)
    struct rusage usage;
    if (getrusage(RUSAGE_SELF, &usage) == 0) {
        return static_cast<double>(usage.ru_maxrss) / (1024.0 * 1024.0); // macOS reports Bytes
    }
#elif defined(_WIN32)
    PROCESS_MEMORY_COUNTERS pmc;
    if (GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc))) {
        return static_cast<double>(pmc.PeakWorkingSetSize) / (1024.0 * 1024.0);
    }
#endif
    return 0.0;
}
