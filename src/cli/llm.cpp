// run, chat, bench, info, logits, serve
#include "cli.h"
#include "littlebit_llm.h"

#include <algorithm>
#include <climits>
#include <cstdio>
#include <iostream>
#include <iterator>

#if !defined(_WIN32)
#include <unistd.h>
#endif

namespace {

struct llm_options : options {
    littlebit_model_params   mp = littlebit_default_model_params();
    littlebit_session_params sp = littlebit_default_session_params();
    littlebit_gen_params     gp = littlebit_default_gen_params();

    explicit llm_options(const options & o) : options(o) {
        mp.n_gpu_layers = n_gpu_layers.value_or(mp.n_gpu_layers);
        mp.use_mmap     = mmap;
        mp.repack       = repack;
        sp.n_ctx        = n_ctx.value_or(sp.n_ctx);
        sp.n_batch      = n_batch.value_or(sp.n_batch);
        sp.n_threads    = n_threads.value_or(sp.n_threads);
        sp.kv_type      = kv == "f32" ? LITTLEBIT_KV_F32 : kv == "q8_0" ? LITTLEBIT_KV_Q8_0 : sp.kv_type;
        gp.max_tokens   = max_tokens.value_or(gp.max_tokens);
        gp.temperature  = temperature.value_or(gp.temperature);
        gp.top_k        = top_k.value_or(gp.top_k);
        gp.top_p        = top_p.value_or(gp.top_p);
        gp.seed         = seed.value_or(gp.seed);
    }
};

struct runtime {
    littlebit_model   * model   = nullptr;
    littlebit_session * session = nullptr;
    ~runtime() {
        littlebit_session_free(session);
        littlebit_model_free(model);
    }
};

void open(runtime & rt, const llm_options & o, bool with_session) {
    set_log(o);
    if (!(rt.model = littlebit_model_load(o.model.c_str(), &o.mp))) {
        die_api("cannot load model");
    }
    if (with_session && !(rt.session = littlebit_session_new(rt.model, &o.sp))) {
        die_api("cannot create session");
    }
}

void print_stats(const littlebit_session * s) {
    const littlebit_stats st = littlebit_session_stats(s);
    fprintf(stderr, "[prompt %d tokens, %.2f tok/s | generated %d tokens, %.2f tok/s]\n",
            st.n_prompt, st.prompt_ms > 0 ? 1e3 * st.n_prompt / st.prompt_ms : 0.0,
            st.n_generated, st.generate_ms > 0 ? 1e3 * st.n_generated / st.generate_ms : 0.0);
}

int cmd_run(llm_options & o) {
    if (!o.prompt) {
        o.prompt.emplace(std::istreambuf_iterator<char>(std::cin), std::istreambuf_iterator<char>());
    }
    runtime rt;
    open(rt, o, true);
    const int rc = littlebit_generate(rt.session, o.prompt->c_str(), &o.gp, print_text, nullptr);
    if (rc < 0) {
        die_api("generation failed");
    }
    printf("\n");
    print_stats(rt.session);
    return 0;
}

int cmd_chat(llm_options & o) {
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

int cmd_info(llm_options & o) {
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

int cmd_bench(llm_options & o) {
    const int n_prompt = 128, n_gen = 64;
    o.sp.n_ctx = std::max(o.sp.n_ctx, n_prompt + n_gen + 8);
    if (!o.n_threads) {
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

int cmd_logits(llm_options & o) {
    if (o.tokens.empty() || o.out.empty()) {
        die("logits needs --tokens and --out");
    }
    const std::vector<int32_t> toks = parse_tokens(o.tokens);
    if (!o.n_ctx) {
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
    write_floats(o.out, logits);
    return 0;
}

} // namespace

int cmd_llm(const options & opts) {
    llm_options o(opts);
    if (o.cmd == "run")   { return cmd_run(o); }
    if (o.cmd == "chat")  { return cmd_chat(o); }
    if (o.cmd == "info")  { return cmd_info(o); }
    if (o.cmd == "bench") { return cmd_bench(o); }
    return cmd_logits(o);
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
