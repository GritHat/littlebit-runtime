// littlebit: command line front end of the LittleBit runtime.
#include "cli.h"

#include <cerrno>
#include <climits>
#include <cstdio>
#include <cstdlib>

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
    "  ncm     Next-Concept model: solve sudokus (-p PUZZLE, or one per line on stdin) or sample text\n"
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
        if      (a == "-p" || a == "--prompt")     { o.prompt = val(); }
        else if (a == "-n" || a == "--max-tokens") { o.max_tokens = (int32_t) to_int(val(), "-n", -1, INT_MAX); }
        else if (a == "-c" || a == "--ctx")        { o.n_ctx = (int32_t) to_int(val(), "-c", 1, INT_MAX); }
        else if (a == "-b" || a == "--batch")      { o.n_batch = (int32_t) to_int(val(), "-b", 1, INT_MAX); }
        else if (a == "-t" || a == "--threads")    { o.n_threads = (int32_t) to_int(val(), "-t", 1, 512); }
        else if (a == "--temp")                    { o.temperature = to_float(val(), "--temp"); }
        else if (a == "--top-k")                   { o.top_k = (int32_t) to_int(val(), "--top-k", 0, INT_MAX); }
        else if (a == "--top-p")                   { o.top_p = to_float(val(), "--top-p"); }
        else if (a == "--seed")                    { o.seed = (uint32_t) to_int(val(), "--seed", 0, UINT_MAX); }
        else if (a == "--system")                  { o.system = val(); }
        else if (a == "--tokens")                  { o.tokens = val(); }
        else if (a == "--out")                     { o.out = val(); }
        else if (a == "-ngl")                      { o.n_gpu_layers = (int32_t) to_int(val(), "-ngl", -1, INT_MAX); }
        else if (a == "--no-mmap")                 { o.mmap = false; }
        else if (a == "--no-repack")               { o.repack = false; }
        else if (a == "-v" || a == "--verbose")    { o.verbose = true; }
        else if (a == "--kv") {
            o.kv = val();
            if (o.kv != "f16" && o.kv != "q8_0" && o.kv != "f32") {
                die("invalid value '" + o.kv + "' for --kv");
            }
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

} // namespace

void die(const std::string & msg) {
    fprintf(stderr, "littlebit: %s\n", msg.c_str());
    exit(1);
}

void die_api(const char * what) {
    die(std::string(what) + ": " + littlebit_last_error());
}

std::vector<int32_t> parse_tokens(const std::string & s) {
    std::vector<int32_t> toks;
    size_t pos = 0;
    while (pos <= s.size()) {
        const size_t next = std::min(s.find(',', pos), s.size());
        toks.push_back((int32_t) to_int(s.substr(pos, next - pos), "--tokens", 0, INT_MAX));
        pos = next + 1;
    }
    return toks;
}

void write_floats(const std::string & path, const std::vector<float> & x) {
    FILE * f = fopen(path.c_str(), "wb");
    if (!f) {
        die("cannot open " + path);
    }
    const bool ok = fwrite(x.data(), sizeof(float), x.size(), f) == x.size();
    if (fclose(f) != 0 || !ok) {
        die("cannot write " + path);
    }
}

bool print_text(const char * text, size_t len, void *) {
    fwrite(text, 1, len, stdout);
    fflush(stdout);
    return true;
}

void set_log(const options & o) {
    if (o.verbose) {
        littlebit_set_log([](int, const char * t, void *) { fputs(t, stderr); }, nullptr);
    } else {
        littlebit_set_log([](int level, const char * t, void *) { if (level == 4) { fputs(t, stderr); } }, nullptr);
    }
}

int main(int argc, char ** argv) {
    const std::vector<std::string> args(argv + 1, argv + argc);
    if (args.empty() || args[0] == "-h" || args[0] == "--help" || args[0] == "help") {
        fputs(USAGE, args.empty() ? stderr : stdout);
        return args.empty() ? 1 : 0;
    }
    const std::string & cmd = args[0];
    if (cmd == "version" || cmd == "--version") {
        printf("littlebit %s\n", littlebit_version());
        return 0;
    }
#ifdef LITTLEBIT_WITH_LLM
    if (cmd == "serve") {
        return cmd_serve(args);
    }
    if (cmd == "run" || cmd == "chat" || cmd == "info" || cmd == "bench" || cmd == "logits") {
        return cmd_llm(parse(args));
    }
#endif
#ifdef LITTLEBIT_WITH_NCM
    if (cmd == "ncm") {
        return cmd_ncm(parse(args));
    }
#endif
    die("unknown command '" + cmd + "'\n\n" + USAGE);
}
