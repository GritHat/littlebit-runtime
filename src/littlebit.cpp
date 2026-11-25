#include "littlebit.h"

#include "ggml-backend.h"
#include "llama.h"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#ifndef LITTLEBIT_VERSION
#define LITTLEBIT_VERSION "0.0.0"
#endif

namespace {

thread_local std::string g_error;
thread_local std::string g_llama_error; // first error line logged by libllama on this thread

std::mutex       g_log_mutex;
littlebit_log_cb g_log_cb = [](int level, const char * text, void *) {
    if (level >= 3) {
        fputs(text, stderr);
    }
};
void * g_log_user = nullptr;

void log_forward(ggml_log_level level, const char * text, void *) {
    thread_local ggml_log_level last = GGML_LOG_LEVEL_INFO;
    if (level != GGML_LOG_LEVEL_CONT) {
        last = level;
    }
    if (last == GGML_LOG_LEVEL_ERROR && g_llama_error.empty()) {
        g_llama_error = text;
        while (!g_llama_error.empty() && g_llama_error.back() == '\n') {
            g_llama_error.pop_back();
        }
    }
    std::lock_guard<std::mutex> lock(g_log_mutex);
    if (g_log_cb) {
        g_log_cb((int) last, text, g_log_user);
    }
}

void init_once() {
    static std::once_flag once;
    std::call_once(once, [] {
        llama_log_set(log_forward, nullptr);
        ggml_backend_load_all();
        llama_backend_init();
    });
}

[[noreturn]] void fail(const std::string & what) {
    throw std::runtime_error(what);
}

template <typename R, typename F> R guarded(R on_error, F && f) {
    try {
        g_error.clear();
        g_llama_error.clear();
        return f();
    } catch (const std::exception & e) {
        g_error = e.what();
    } catch (...) {
        g_error = "unknown error";
    }
    return on_error;
}

// length of the longest prefix of s that does not end inside a UTF-8 sequence
size_t utf8_complete(const std::string & s) {
    const size_t n = s.size();
    for (size_t back = 1; back <= 3 && back <= n; ++back) {
        const unsigned char c = (unsigned char) s[n - back];
        if ((c & 0xC0) == 0x80) {
            continue;
        }
        const size_t need = c >= 0xF0 ? 4 : c >= 0xE0 ? 3 : c >= 0xC0 ? 2 : 1;
        return need > back ? n - back : n;
    }
    return n;
}

std::string meta_str(const llama_model * m, const char * key) {
    char buf[128];
    const int32_t n = llama_model_meta_val_str(m, key, buf, sizeof(buf));
    return n < 0 ? std::string() : std::string(buf);
}

struct context_deleter { void operator()(llama_context * p) const { llama_free(p); } };
struct model_deleter   { void operator()(llama_model   * p) const { llama_model_free(p); } };
struct sampler_deleter { void operator()(llama_sampler * p) const { llama_sampler_free(p); } };
struct batch_holder {
    llama_batch b;
    explicit batch_holder(int32_t n) : b(llama_batch_init(n, 0, 1)) {}
    ~batch_holder() { llama_batch_free(b); }
    batch_holder(const batch_holder &) = delete;
    batch_holder & operator=(const batch_holder &) = delete;
};

std::vector<llama_token> tokenize(const llama_vocab * vocab, const std::string & text, bool add_special) {
    if (llama_vocab_type(vocab) == LLAMA_VOCAB_TYPE_NONE) {
        fail("the model has no tokenizer");
    }
    const int32_t n = -llama_tokenize(vocab, text.data(), (int32_t) text.size(), nullptr, 0, add_special, true);
    if (n < 0) {
        fail("tokenization failed");
    }
    std::vector<llama_token> toks((size_t) n);
    if (n > 0 && llama_tokenize(vocab, text.data(), (int32_t) text.size(), toks.data(), n, add_special, true) != n) {
        fail("tokenization failed");
    }
    return toks;
}

double now_ms() {
    using namespace std::chrono;
    return duration<double, std::milli>(steady_clock::now().time_since_epoch()).count();
}

} // namespace

struct littlebit_model {
    std::unique_ptr<llama_model, model_deleter> model;
    const llama_vocab * vocab = nullptr;
};

struct littlebit_session {
    littlebit_model * model = nullptr;
    std::unique_ptr<llama_context, context_deleter> ctx;
    int32_t n_ctx   = 0;
    int32_t n_batch = 0;
    int32_t n_past  = 0;
    llama_token pending = LLAMA_TOKEN_NULL; // sampled but not yet decoded
    std::vector<llama_token> kv;            // tokens held by the KV cache, n_past of them
    std::vector<std::pair<std::string, std::string>> history;
    littlebit_stats stats = {};

    void reset() {
        llama_memory_clear(llama_get_memory(ctx.get()), true);
        n_past  = 0;
        pending = LLAMA_TOKEN_NULL;
        kv.clear();
        history.clear();
        stats = {};
    }

    // keep the first n tokens of the KV cache; falls back to an empty cache where that is not possible
    void truncate(int32_t n) {
        if (n == n_past) {
            return;
        }
        llama_memory_t mem = llama_get_memory(ctx.get());
        if (!llama_memory_seq_rm(mem, 0, n, -1)) {
            llama_memory_clear(mem, true);
            n = 0;
        }
        n_past = n;
        kv.resize((size_t) n);
    }

    // logits: nullptr to keep only the last token's logits
    void decode(const std::vector<llama_token> & toks, float * logits) {
        const int32_t n_vocab = llama_vocab_n_tokens(model->vocab);
        batch_holder batch(n_batch);
        for (size_t i = 0; i < toks.size();) {
            const int32_t n = (int32_t) std::min<size_t>(n_batch, toks.size() - i);
            for (int32_t j = 0; j < n; ++j) {
                batch.b.token[j]     = toks[i + j];
                batch.b.pos[j]       = n_past + j;
                batch.b.n_seq_id[j]  = 1;
                batch.b.seq_id[j][0] = 0;
                batch.b.logits[j]    = logits != nullptr || i + j + 1 == toks.size();
            }
            batch.b.n_tokens = n;
            const int32_t rc = llama_decode(ctx.get(), batch.b);
            if (rc != 0) {
                fail("decode failed (" + std::to_string(rc) + ")");
            }
            if (logits) {
                for (int32_t j = 0; j < n; ++j) {
                    memcpy(logits + (i + j) * n_vocab, llama_get_logits_ith(ctx.get(), j), n_vocab * sizeof(float));
                }
            }
            kv.insert(kv.end(), toks.begin() + (std::ptrdiff_t) i, toks.begin() + (std::ptrdiff_t) i + n);
            n_past += n;
            i += (size_t) n;
        }
    }

    int generate(const std::vector<llama_token> & toks, const littlebit_gen_params & gp, littlebit_text_cb cb, void * user,
                 std::string * reply) {
        if (toks.empty()) {
            fail("empty prompt");
        }
        if (n_past + (int64_t) toks.size() >= n_ctx) {
            fail("prompt does not fit the context (" + std::to_string(n_past + toks.size()) + " tokens, n_ctx " +
                 std::to_string(n_ctx) + ")");
        }
        pending = LLAMA_TOKEN_NULL;
        stats   = {};

        std::unique_ptr<llama_sampler, sampler_deleter> smpl(llama_sampler_chain_init(llama_sampler_chain_default_params()));
        if (gp.temperature <= 0.0f) {
            llama_sampler_chain_add(smpl.get(), llama_sampler_init_greedy());
        } else {
            if (gp.top_k > 0)    { llama_sampler_chain_add(smpl.get(), llama_sampler_init_top_k(gp.top_k)); }
            if (gp.top_p < 1.0f) { llama_sampler_chain_add(smpl.get(), llama_sampler_init_top_p(gp.top_p, 1)); }
            llama_sampler_chain_add(smpl.get(), llama_sampler_init_temp(gp.temperature));
            llama_sampler_chain_add(smpl.get(), llama_sampler_init_dist(gp.seed));
        }

        const double t0 = now_ms();
        decode(toks, nullptr);
        stats.n_prompt = (int32_t) toks.size();
        const double t1 = now_ms();
        stats.prompt_ms = t1 - t0;

        std::string buf;
        int stop = LITTLEBIT_STOP_LENGTH;
        while (gp.max_tokens < 0 || stats.n_generated < gp.max_tokens) {
            const llama_token tok = llama_sampler_sample(smpl.get(), ctx.get(), -1);
            pending = tok;
            if (llama_vocab_is_eog(model->vocab, tok)) {
                stop = LITTLEBIT_STOP_EOG;
                break;
            }
            ++stats.n_generated;

            std::string piece(64, '\0');
            int32_t n = llama_token_to_piece(model->vocab, tok, &piece[0], (int32_t) piece.size(), 0, false);
            if (n < 0) {
                piece.resize((size_t) -n);
                n = llama_token_to_piece(model->vocab, tok, &piece[0], (int32_t) piece.size(), 0, false);
            }
            buf.append(piece, 0, (size_t) std::max(n, 0));
            const size_t ready = utf8_complete(buf);
            if (ready > 0) {
                if (reply) {
                    reply->append(buf, 0, ready);
                }
                const bool go = !cb || cb(buf.data(), ready, user);
                buf.erase(0, ready);
                if (!go) {
                    stop = LITTLEBIT_STOP_USER;
                    break;
                }
            }
            if (n_past >= n_ctx) {
                stop = LITTLEBIT_STOP_CONTEXT;
                break;
            }
            if (gp.max_tokens >= 0 && stats.n_generated >= gp.max_tokens) {
                break; // the last token stays pending
            }
            decode({ tok }, nullptr);
            pending = LLAMA_TOKEN_NULL;
        }
        if (!buf.empty() && stop != LITTLEBIT_STOP_USER) {
            if (reply) {
                reply->append(buf);
            }
            if (cb) {
                cb(buf.data(), buf.size(), user);
            }
        }
        stats.generate_ms = now_ms() - t1;
        return stop;
    }

    std::string render(bool add_assistant) const {
        const char * tmpl = llama_model_chat_template(model->model.get(), nullptr);
        if (!tmpl) {
            fail("the model has no chat template");
        }
        std::vector<llama_chat_message> msgs;
        size_t total = 0;
        for (const auto & m : history) {
            msgs.push_back({ m.first.c_str(), m.second.c_str() });
            total += m.first.size() + m.second.size();
        }
        std::string out(total * 2 + 256, '\0');
        int32_t n = llama_chat_apply_template(tmpl, msgs.data(), msgs.size(), add_assistant, &out[0], (int32_t) out.size());
        if (n > (int32_t) out.size()) {
            out.resize((size_t) n);
            n = llama_chat_apply_template(tmpl, msgs.data(), msgs.size(), add_assistant, &out[0], (int32_t) out.size());
        }
        if (n < 0) {
            fail("the model's chat template is not supported");
        }
        out.resize((size_t) n);
        return out;
    }
};

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

littlebit_model_params littlebit_default_model_params(void) { return { 0, true, true, false }; }
littlebit_session_params littlebit_default_session_params(void) { return { 2048, 256, 0, LITTLEBIT_KV_F16 }; }
littlebit_gen_params littlebit_default_gen_params(void) { return { 256, 0.7f, 40, 0.95f, LLAMA_DEFAULT_SEED }; }

littlebit_model * littlebit_model_load(const char * path, const littlebit_model_params * params) {
    return guarded((littlebit_model *) nullptr, [&]() -> littlebit_model * {
        if (!path) {
            fail("path is NULL");
        }
        init_once();
        const littlebit_model_params p = params ? *params : littlebit_default_model_params();
        llama_model_params mp = llama_model_default_params();
        mp.n_gpu_layers = p.n_gpu_layers;
        ggml_backend_dev_t no_devices[] = { nullptr };
        if (p.n_gpu_layers == 0) {
            mp.devices = no_devices; // otherwise a GPU may still be used for some operations
        }
        mp.use_extra_bufts = p.repack;
        mp.load_mode    = p.use_mmap ? LLAMA_LOAD_MODE_MMAP : LLAMA_LOAD_MODE_NONE;
        std::unique_ptr<littlebit_model> m(new littlebit_model);
        m->model.reset(llama_model_load_from_file(path, mp));
        if (!m->model) {
            fail(std::string("failed to load '") + path + "'" + (g_llama_error.empty() ? "" : ": " + g_llama_error));
        }
        if (p.require_littlebit && meta_str(m->model.get(), "littlebit.version").empty()) {
            fail(std::string("'") + path + "' is not a LittleBit model");
        }
        m->vocab = llama_model_get_vocab(m->model.get());
        return m.release();
    });
}

void littlebit_model_free(littlebit_model * model) { delete model; }

int littlebit_model_get_info(const littlebit_model * model, littlebit_model_info * info) {
    return guarded(-1, [&]() -> int {
        if (!model || !info) {
            fail("NULL argument");
        }
        *info = {};
        const llama_model * m = model->model.get();
        llama_model_desc(m, info->description, sizeof(info->description));
        snprintf(info->arch, sizeof(info->arch), "%s", meta_str(m, "general.architecture").c_str());
        const std::string v = meta_str(m, "littlebit.version");
        info->format_version    = v.empty() ? 0 : (uint32_t) std::stoul(v);
        info->size_bytes        = llama_model_size(m);
        info->n_vocab           = llama_vocab_n_tokens(model->vocab);
        info->n_ctx_train       = llama_model_n_ctx_train(m);
        info->has_chat_template = llama_model_chat_template(m, nullptr) != nullptr;
        return 0;
    });
}

int32_t littlebit_count_tokens(const littlebit_model * model, const char * text) {
    return guarded(-1, [&]() -> int {
        if (!model || !text) {
            fail("NULL argument");
        }
        return (int) tokenize(model->vocab, text, false).size();
    });
}

littlebit_session * littlebit_session_new(littlebit_model * model, const littlebit_session_params * params) {
    return guarded((littlebit_session *) nullptr, [&]() -> littlebit_session * {
        if (!model) {
            fail("model is NULL");
        }
        const littlebit_session_params p = params ? *params : littlebit_default_session_params();
        if (p.n_ctx <= 0 || p.n_batch <= 0 || p.n_threads < 0) {
            fail("n_ctx and n_batch must be positive and n_threads must not be negative");
        }
        llama_context_params cp = llama_context_default_params();
        const int32_t n_train = llama_model_n_ctx_train(model->model.get());
        const int32_t n_ctx   = n_train > 0 ? std::min(p.n_ctx, n_train) : p.n_ctx;
        cp.n_ctx           = (uint32_t) n_ctx;
        cp.n_batch         = (uint32_t) std::min(p.n_batch, n_ctx);
        cp.n_ubatch        = cp.n_batch;
        cp.n_threads       = p.n_threads > 0 ? p.n_threads : littlebit_default_threads();
        cp.n_threads_batch = cp.n_threads;
        switch (p.kv_type) {
            case LITTLEBIT_KV_F16:  cp.type_k = cp.type_v = GGML_TYPE_F16;  break;
            case LITTLEBIT_KV_Q8_0: cp.type_k = cp.type_v = GGML_TYPE_Q8_0; break;
            case LITTLEBIT_KV_F32:  cp.type_k = cp.type_v = GGML_TYPE_F32;  break;
            default: fail("invalid kv_type");
        }
        std::unique_ptr<littlebit_session> s(new littlebit_session);
        s->model   = model;
        s->n_ctx   = n_ctx;
        s->n_batch = (int32_t) cp.n_batch;
        s->ctx.reset(llama_init_from_model(model->model.get(), cp));
        if (!s->ctx) {
            fail("failed to create the context" + (g_llama_error.empty() ? "" : ": " + g_llama_error));
        }
        return s.release();
    });
}

void littlebit_session_free(littlebit_session * session) { delete session; }

void littlebit_session_reset(littlebit_session * session) {
    if (session) {
        session->reset();
    }
}

int32_t littlebit_session_n_past(const littlebit_session * session) { return session ? session->n_past : 0; }

littlebit_stats littlebit_session_stats(const littlebit_session * session) {
    return session ? session->stats : littlebit_stats{};
}

int littlebit_generate(littlebit_session * session, const char * prompt, const littlebit_gen_params * params,
                       littlebit_text_cb cb, void * user) {
    return guarded(-1, [&]() -> int {
        if (!session || !prompt) {
            fail("NULL argument");
        }
        const littlebit_gen_params gp = params ? *params : littlebit_default_gen_params();
        std::vector<llama_token> toks = tokenize(session->model->vocab, prompt, session->n_past == 0);
        if (session->pending != LLAMA_TOKEN_NULL) {
            toks.insert(toks.begin(), session->pending);
        }
        return session->generate(toks, gp, cb, user, nullptr);
    });
}

int littlebit_chat(littlebit_session * session, const char * system, const char * message,
                   const littlebit_gen_params * params, littlebit_text_cb cb, void * user) {
    return guarded(-1, [&]() -> int {
        if (!session || !message) {
            fail("NULL argument");
        }
        const littlebit_gen_params gp = params ? *params : littlebit_default_gen_params();
        const auto saved = session->history;
        try {
            if (session->history.empty() && system && *system) {
                session->history.emplace_back("system", system);
            }
            session->history.emplace_back("user", message);
            std::vector<llama_token> toks = tokenize(session->model->vocab, session->render(true), true);
            if ((int64_t) toks.size() >= session->n_ctx) {
                fail("the conversation does not fit the context (" + std::to_string(toks.size()) + " tokens, n_ctx " +
                     std::to_string(session->n_ctx) + ")");
            }
            // reuse the longest common prefix, but always decode at least one token to get logits
            size_t n = 0;
            while (n < session->kv.size() && n + 1 < toks.size() && session->kv[n] == toks[n]) {
                ++n;
            }
            session->truncate((int32_t) n);
            toks.erase(toks.begin(), toks.begin() + session->n_past);
            std::string reply;
            const int rc = session->generate(toks, gp, cb, user, &reply);
            session->history.emplace_back("assistant", reply);
            return rc;
        } catch (...) {
            session->history = saved;
            throw;
        }
    });
}

int littlebit_eval(littlebit_session * session, const int32_t * tokens, int32_t n_tokens, float * logits, size_t n_logits) {
    return guarded(-1, [&]() -> int {
        if (!session || !tokens || !logits || n_tokens <= 0) {
            fail("invalid argument");
        }
        const int32_t n_vocab = llama_vocab_n_tokens(session->model->vocab);
        if (n_tokens > session->n_ctx) {
            fail("too many tokens for the context");
        }
        if (n_logits < (size_t) n_tokens * (size_t) n_vocab) {
            fail("logits buffer too small");
        }
        for (int32_t i = 0; i < n_tokens; ++i) {
            if (tokens[i] < 0 || tokens[i] >= n_vocab) {
                fail("token id out of range");
            }
        }
        session->reset();
        session->decode(std::vector<llama_token>(tokens, tokens + n_tokens), logits);
        return 0;
    });
}

} // extern "C"
