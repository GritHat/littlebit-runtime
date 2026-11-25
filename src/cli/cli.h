// shared by the parts of the command line tool
#pragma once

#include "littlebit.h"

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

struct options {
    std::string cmd, model, system, tokens, out, kv;
    std::optional<std::string> prompt;
    std::optional<int32_t> max_tokens, n_ctx, n_batch, n_threads, top_k, n_gpu_layers;
    std::optional<float> temperature, top_p;
    std::optional<uint32_t> seed;
    bool verbose = false, mmap = true, repack = true;
};

[[noreturn]] void die(const std::string & msg);
[[noreturn]] void die_api(const char * what);
std::vector<int32_t> parse_tokens(const std::string & s);
void write_floats(const std::string & path, const std::vector<float> & x);
bool print_text(const char * text, size_t len, void *); // a littlebit_text_cb
void set_log(const options & o);

int cmd_llm(const options & o);                        // run, chat, bench, info, logits
int cmd_serve(const std::vector<std::string> & args);
int cmd_ncm(const options & o);
