#include "littlebit.h"
#include "internal.h"

#include "ggml-backend.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <mutex>
#include <thread>
#include <vector>

#ifndef LITTLEBIT_VERSION
#define LITTLEBIT_VERSION "0.0.0"
#endif

namespace {

thread_local std::string g_error;
thread_local std::string g_log_error; // first error line logged by llama.cpp or ggml on this thread

std::mutex       g_log_mutex;
littlebit_log_cb g_log_cb = [](int level, const char * text, void *) {
    if (level >= 3) {
        fputs(text, stderr);
    }
};
void * g_log_user = nullptr;

} // namespace

namespace littlebit_internal {

void log_forward(ggml_log_level level, const char * text, void *) {
    thread_local ggml_log_level last = GGML_LOG_LEVEL_INFO;
    if (level != GGML_LOG_LEVEL_CONT) {
        last = level;
    }
    if (last == GGML_LOG_LEVEL_ERROR && g_log_error.empty()) {
        g_log_error = text;
        while (!g_log_error.empty() && g_log_error.back() == '\n') {
            g_log_error.pop_back();
        }
    }
    std::lock_guard<std::mutex> lock(g_log_mutex);
    if (g_log_cb) {
        g_log_cb((int) last, text, g_log_user);
    }
}

void init() {
    static std::once_flag once;
    std::call_once(once, [] {
        ggml_log_set(log_forward, nullptr);
        ggml_backend_load_all();
    });
}

void set_error(const std::string & what) { g_error = what; }
std::string & log_error() { return g_log_error; }

double now_ms() {
    using namespace std::chrono;
    return duration<double, std::milli>(steady_clock::now().time_since_epoch()).count();
}

} // namespace littlebit_internal

extern "C" {

const char * littlebit_version(void)    { return LITTLEBIT_VERSION; }
const char * littlebit_last_error(void) { return g_error.c_str(); }

void littlebit_set_log(littlebit_log_cb cb, void * user) {
    std::lock_guard<std::mutex> lock(g_log_mutex);
    g_log_cb   = cb;
    g_log_user = user;
}

// one value per CPU, or nothing if the file is missing for some CPU
static std::vector<long> read_cpus(int32_t n, const char * file) {
    std::vector<long> res;
    for (int32_t i = 0; i < n; ++i) {
        std::ifstream f("/sys/devices/system/cpu/cpu" + std::to_string(i) + "/" + file);
        long v = 0;
        if (!(f >> v) || v <= 0) {
            return {};
        }
        res.push_back(v);
    }
    return res;
}

// Heterogeneous CPUs (phones, watches): the cores within 80% of the fastest. The kernel's capacity is preferred to the
// clock because small and big cores can run at similar clocks.
// Homogeneous CPUs: leave one core free from 4 cores up, two from 8 up - ggml's workers spin.
int32_t littlebit_default_threads(void) {
    const int32_t n = std::max(1, (int32_t) std::thread::hardware_concurrency());
    for (const char * file : { "cpu_capacity", "cpufreq/cpuinfo_max_freq" }) {
        const std::vector<long> speed = read_cpus(n, file);
        if (speed.empty()) {
            continue;
        }
        const long top = *std::max_element(speed.begin(), speed.end());
        const int32_t big = (int32_t) std::count_if(speed.begin(), speed.end(), [top](long v) { return v * 10 >= top * 8; });
        if (big < n) {
            return big;
        }
    }
    return n >= 8 ? n - 2 : n >= 4 ? n - 1 : n;
}

} // extern "C"
