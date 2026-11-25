// littlebit: command line front end of the LittleBit runtime.
#include "littlebit.h"

#include <cerrno>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <string>
#include <vector>

#if !defined(_WIN32)
#include <unistd.h>
#endif

namespace {

const char * USAGE =
    "usage: littlebit <command> MODEL.gguf [options]\n"
    "\n"
    "commands:\n"
    "  run     generate from a prompt            (-p TEXT, or stdin)\n"
    "  chat    interactive chat                  (--system TEXT)\n"
    "  bench   measure prompt and decode speed\n"
    "  info    show model information\n"
    "  logits  dump logits of given token ids    (--tokens 1,2,3 --out FILE)\n"
    "  serve   OpenAI-compatible HTTP server     (extra options go to llama-server)\n"
    "  version\n"
    "\n"
    "options:\n"
    "  -p, --prompt TEXT     prompt\n"
    "  -n, --max-tokens N    tokens to generate, -1 for no limit   (256)\n"
    "  -c, --ctx N           context size                          (2048)\n"
    "  -b, --batch N         prompt batch size                     (256)\n"
    "  -t, --threads N       CPU threads                           (automatic)\n"
    "      --temp X          temperature, 0 for greedy             (0.7)\n"
    "      --top-k N / --top-p X / --seed N                        (40 / 0.95 / random)\n"
    "      --kv TYPE         KV cache type: f16, q8_0, f32         (f16)\n"
    "  -ngl N                layers to offload to a GPU            (0)\n"
    "      --no-mmap         read the model into memory\n"
    "      --no-repack       keep weights memory-mapped (less memory, slower on ARM)\n"
    "  -v, --verbose         show runtime log\n";

struct options {
    std::string cmd, model, prompt, system, tokens, out;
    bool has_prompt = false, verbose = false, has_threads = false, has_ctx = false;
    littlebit_model_params   mp = littlebit_default_model_params();
    littlebit_session_params sp = littlebit_default_session_params();
    littlebit_gen_params     gp = littlebit_default_gen_params();
};

[[noreturn]] void die(const std::string & msg) {
    fprintf(stderr, "littlebit: %s\n", msg.c_str());
    exit(1);
}

[[noreturn]] void die_api(const char * what) {
    die(std::string(what) + ": " + littlebit_last_error());
}

long long to_int(const std::string & s, const char * opt, long long lo, long long hi) {
    errno = 0;
    char * end = nullptr;
    const long long v = strtoll(s.c_str(), &end, 10);
    if (s.empty() || *end || errno || v < lo || v > hi) {
        die(std::string("invalid value '") + s + "' for " + opt);
    }
    return v;
}

float to_float(const std::string & s, const char * opt) {
    char * end = nullptr;
    const float v = strtof(s.c_str(), &end);
    if (s.empty() || *end || !(v >= 0.0f) || v > 1e6f) {
        die(std::string("invalid value '") + s + "' for " + opt);
    }
    return v;
}

options parse(const std::vector<std::string> & args) {
    options o;
    o.cmd = args[0];
    for (size_t i = 1; i < args.size(); ++i) {
        const std::string & a = args[i];
        auto val = [&]() -> const std::string & {
            if (i + 1 >= args.size()) {
                die("missing value for " + a);
            }
            return args[++i];
        };
        if      (a == "-p" || a == "--prompt")     { o.prompt = val(); o.has_prompt = true; }
        else if (a == "-n" || a == "--max-tokens") { o.gp.max_tokens = (int32_t) to_int(val(), "-n", -1, INT_MAX); }
        else if (a == "-c" || a == "--ctx")        { o.sp.n_ctx = (int32_t) to_int(val(), "-c", 1, INT_MAX); o.has_ctx = true; }
        else if (a == "-b" || a == "--batch")      { o.sp.n_batch = (int32_t) to_int(val(), "-b", 1, INT_MAX); }
        else if (a == "-t" || a == "--threads")    { o.sp.n_threads = (int32_t) to_int(val(), "-t", 1, 512); o.has_threads = true; }
        else if (a == "--temp")                    { o.gp.temperature = to_float(val(), "--temp"); }
        else if (a == "--top-k")                   { o.gp.top_k = (int32_t) to_int(val(), "--top-k", 0, INT_MAX); }
        else if (a == "--top-p")                   { o.gp.top_p = to_float(val(), "--top-p"); }
        else if (a == "--seed")                    { o.gp.seed = (uint32_t) to_int(val(), "--seed", 0, UINT_MAX); }
        else if (a == "--system")                  { o.system = val(); }
        else if (a == "--tokens")                  { o.tokens = val(); }
        else if (a == "--out")                     { o.out = val(); }
        else if (a == "-ngl")                      { o.mp.n_gpu_layers = (int32_t) to_int(val(), "-ngl", -1, INT_MAX); }
        else if (a == "--no-mmap")                 { o.mp.use_mmap = false; }
        else if (a == "--no-repack")               { o.mp.repack = false; }
        else if (a == "-v" || a == "--verbose")    { o.verbose = true; }
        else if (a == "--kv") {
            const std::string & v = val();
            o.sp.kv_type = v == "f16" ? LITTLEBIT_KV_F16 : v == "q8_0" ? LITTLEBIT_KV_Q8_0 : v == "f32" ? LITTLEBIT_KV_F32
                         : (die("invalid value '" + v + "' for --kv"), LITTLEBIT_KV_F16);
        }
        else if (!a.empty() && a[0] == '-')        { die("unknown option " + a); }
        else if (o.model.empty())                  { o.model = a; }
        else                                       { die("unexpected argument '" + a + "'"); }
    }
    if (o.model.empty()) {
        die("no model given\n\n" + std::string(USAGE));
    }
    return o;
}

struct runtime {
    littlebit_model   * model   = nullptr;
    littlebit_session * session = nullptr;
    ~runtime() {
        littlebit_session_free(session);
        littlebit_model_free(model);
    }
};

void open(runtime & rt, const options & o, bool with_session) {
    if (o.verbose) {
        littlebit_set_log([](int, const char * t, void *) { fputs(t, stderr); }, nullptr);
    } else {
        littlebit_set_log([](int level, const char * t, void *) { if (level == 4) { fputs(t, stderr); } }, nullptr);
    }
    if (!(rt.model = littlebit_model_load(o.model.c_str(), &o.mp))) {
        die_api("cannot load model");
    }
    if (with_session && !(rt.session = littlebit_session_new(rt.model, &o.sp))) {
        die_api("cannot create session");
    }
}

bool print_text(const char * text, size_t len, void *) {
    fwrite(text, 1, len, stdout);
    fflush(stdout);
    return true;
}

void print_stats(const littlebit_session * s) {
    const littlebit_stats st = littlebit_session_stats(s);
    fprintf(stderr, "[prompt %d tokens, %.2f tok/s | generated %d tokens, %.2f tok/s]\n",
            st.n_prompt, st.prompt_ms > 0 ? 1e3 * st.n_prompt / st.prompt_ms : 0.0,
            st.n_generated, st.generate_ms > 0 ? 1e3 * st.n_generated / st.generate_ms : 0.0);
}

int cmd_run(options & o) {
    if (!o.has_prompt) {
        o.prompt.assign(std::istreambuf_iterator<char>(std::cin), std::istreambuf_iterator<char>());
    }
    runtime rt;
    open(rt, o, true);
    const int rc = littlebit_generate(rt.session, o.prompt.c_str(), &o.gp, print_text, nullptr);
    if (rc < 0) {
        die_api("generation failed");
    }
    printf("\n");
    print_stats(rt.session);
    return 0;
}

int cmd_chat(options & o) {
    runtime rt;
    open(rt, o, true);
    littlebit_model_info info;
    if (littlebit_model_get_info(rt.model, &info) != 0) {
        die_api("cannot read model info");
    }
    if (!info.has_chat_template) {
        die("this model has no chat template; use 'littlebit run'");
    }
    fprintf(stderr, "%s - /reset clears the conversation, /quit or Ctrl-D leaves\n", info.description);
    std::string line;
    while (fprintf(stderr, "> "), std::getline(std::cin, line)) {
        if (line == "/quit") {
            break;
        }
        if (line == "/reset") {
            littlebit_session_reset(rt.session);
            continue;
        }
        if (line.empty()) {
            continue;
        }
        const int rc = littlebit_chat(rt.session, o.system.c_str(), line.c_str(), &o.gp, print_text, nullptr);
        printf("\n");
        if (rc < 0) {
            fprintf(stderr, "littlebit: %s\n", littlebit_last_error());
        } else {
            print_stats(rt.session);
            if (rc == LITTLEBIT_STOP_CONTEXT) {
                fprintf(stderr, "littlebit: context full, use /reset\n");
            }
        }
    }
    return 0;
}

int cmd_info(options & o) {
    runtime rt;
    open(rt, o, false);
    littlebit_model_info info;
    if (littlebit_model_get_info(rt.model, &info) != 0) {
        die_api("cannot read model info");
    }
    printf("description      %s\n", info.description);
    printf("architecture     %s\n", info.arch);
    printf("littlebit format %s\n", info.format_version ? std::to_string(info.format_version).c_str() : "none (ordinary GGUF)");
    printf("tensor data      %.1f MiB\n", info.size_bytes / 1048576.0);
    printf("vocabulary       %d\n", info.n_vocab);
    printf("training context %d\n", info.n_ctx_train);
    printf("chat template    %s\n", info.has_chat_template ? "yes" : "no");
    return 0;
}

int cmd_bench(options & o) {
    const int n_prompt = 128, n_gen = 64;
    o.sp.n_ctx = std::max(o.sp.n_ctx, n_prompt + n_gen + 8);
    if (!o.has_threads) {
        o.sp.n_threads = littlebit_default_threads();
    }
    runtime rt;
    open(rt, o, true);
    littlebit_model_info info;
    if (littlebit_model_get_info(rt.model, &info) != 0) {
        die_api("cannot read model info");
    }
    std::string prompt;
    while (true) {
        const int32_t n = littlebit_count_tokens(rt.model, (prompt + " the").c_str());
        if (n < 0) {
            die_api("benchmark failed");
        }
        if (n > n_prompt) {
            break;
        }
        prompt += " the";
    }
    o.gp.temperature = 0;
    o.gp.max_tokens  = n_gen;
    for (int pass = 0; pass < 2; ++pass) { // the first pass pages the model in
        littlebit_session_reset(rt.session);
        if (littlebit_generate(rt.session, prompt.c_str(), &o.gp, nullptr, nullptr) < 0) {
            die_api("benchmark failed");
        }
    }
    printf("%s, %d threads\n", info.description, o.sp.n_threads);
    print_stats(rt.session);
    return 0;
}

int cmd_logits(options & o) {
    if (o.tokens.empty() || o.out.empty()) {
        die("logits needs --tokens and --out");
    }
    std::vector<int32_t> toks;
    size_t pos = 0;
    while (pos <= o.tokens.size()) {
        const size_t next = std::min(o.tokens.find(',', pos), o.tokens.size());
        toks.push_back((int32_t) to_int(o.tokens.substr(pos, next - pos), "--tokens", 0, INT_MAX));
        pos = next + 1;
    }
    if (!o.has_ctx) {
        o.sp.n_ctx = std::max<int32_t>(o.sp.n_ctx, (int32_t) toks.size());
    }
    runtime rt;
    open(rt, o, true);
    littlebit_model_info info;
    if (littlebit_model_get_info(rt.model, &info) != 0) {
        die_api("cannot read model info");
    }
    std::vector<float> logits(toks.size() * (size_t) info.n_vocab);
    if (littlebit_eval(rt.session, toks.data(), (int32_t) toks.size(), logits.data(), logits.size()) != 0) {
        die_api("evaluation failed");
    }
    FILE * f = fopen(o.out.c_str(), "wb");
    if (!f) {
        die("cannot open " + o.out);
    }
    const bool ok = fwrite(logits.data(), sizeof(float), logits.size(), f) == logits.size();
    if (fclose(f) != 0 || !ok) {
        die("cannot write " + o.out);
    }
    return 0;
}

// llama-server from the same build already understands LittleBit models
int cmd_serve(const std::vector<std::string> & args) {
#if defined(_WIN32)
    (void) args;
    die("run llama-server.exe -m MODEL.gguf instead");
#else
    if (args.size() < 2 || args[1].empty() || args[1][0] == '-') {
        die("usage: littlebit serve MODEL.gguf [llama-server options]");
    }
    std::vector<std::string> cmd = { "llama-server", "-m", args[1] };
    bool has_threads = false;
    for (size_t i = 2; i < args.size(); ++i) {
        has_threads |= args[i] == "-t" || args[i] == "--threads";
        cmd.push_back(args[i]);
    }
    if (!has_threads) {
        cmd.insert(cmd.end(), { "-t", std::to_string(littlebit_default_threads()) });
    }
    std::vector<char *> argv;
    for (auto & c : cmd) {
        argv.push_back(&c[0]);
    }
    argv.push_back(nullptr);
    char self[PATH_MAX];
    const ssize_t n = readlink("/proc/self/exe", self, sizeof(self) - 1);
    if (n > 0) {
        self[n] = '\0';
        std::string path(self);
        path = path.substr(0, path.rfind('/')) + "/llama-server";
        if (access(path.c_str(), X_OK) == 0) {
            execv(path.c_str(), argv.data());
        }
    }
    execvp("llama-server", argv.data());
    die("llama-server not found next to littlebit or in PATH (build with -DLITTLEBIT_SERVER=ON)");
#endif
}

} // namespace

int main(int argc, char ** argv) {
    const std::vector<std::string> args(argv + 1, argv + argc);
    if (args.empty() || args[0] == "-h" || args[0] == "--help" || args[0] == "help") {
        fputs(USAGE, args.empty() ? stderr : stdout);
        return args.empty() ? 1 : 0;
    }
    if (args[0] == "version" || args[0] == "--version") {
        printf("littlebit %s\n", littlebit_version());
        return 0;
    }
    if (args[0] == "serve") {
        return cmd_serve(args);
    }
    if (args[0] != "run" && args[0] != "chat" && args[0] != "info" && args[0] != "bench" && args[0] != "logits") {
        die("unknown command '" + args[0] + "'\n\n" + USAGE);
    }
    options o = parse(args);
    if (o.cmd == "run")  { return cmd_run(o); }
    if (o.cmd == "chat") { return cmd_chat(o); }
    if (o.cmd == "info") { return cmd_info(o); }
    if (o.cmd == "bench") { return cmd_bench(o); }
    return cmd_logits(o);
}
