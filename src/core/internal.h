// what the engines share; not installed
#pragma once

#include "ggml.h"

#include <exception>
#include <stdexcept>
#include <string>

namespace littlebit_internal {

void init();                              // backends and logging, once
void log_forward(ggml_log_level level, const char * text, void * user);
void set_error(const std::string & what); // littlebit_last_error() of this thread
std::string & log_error();                // first error line logged by llama.cpp or ggml on this thread
double now_ms();

[[noreturn]] inline void fail(const std::string & what) {
    throw std::runtime_error(what);
}

// body of a C API function: exceptions become littlebit_last_error()
template <typename R, typename F> R guarded(R on_error, F && f) {
    try {
        set_error("");
        log_error().clear();
        return f();
    } catch (const std::exception & e) {
        set_error(e.what());
    } catch (...) {
        set_error("unknown error");
    }
    return on_error;
}

} // namespace littlebit_internal
