// Next-Concept Modeling (arXiv 2610.14162) on ggml. A level of codes is predicted in parallel from everything coarser, so
// llama.cpp's one-token-at-a-time context does not apply. docs/NCM.md describes the file.
#include "littlebit_ncm.h"
#include "../core/internal.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpp.h"
#include "ggml-cpu.h"
#include "gguf.h"
#include "llama-littlebit.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

using namespace littlebit_internal;

namespace {

// x/2 (1 + erf(x/sqrt 2)), to about 2e-7. ggml_gelu_erf calls erff for every element and takes a large part of the run time.
void gelu_row(float * y, const float * x, int64_t n) {
    for (int64_t i = 0; i < n; ++i) {
        const float a = std::fabs(x[i]) * 0.70710678f;
        const float t = 1.0f / (1.0f + 0.3275911f * a);
        // exp(-a^2) as 2^k 2^f; beyond a = 9.3 the factor is below the smallest normal number and does not matter
        const float z = std::max(-86.0f, -a * a) * 1.44269504f; // in this order NaN becomes -86
        const int32_t k = (int32_t) (z - 0.5f);
        const float f = (z - (float) k) * 0.69314718f;
        const float p = 1.0f + f * (1.0f + f * (0.5f + f * (0.16666667f + f * (0.041666667f + f * (0.0083333333f + f * 0.0013888889f)))));
        const int32_t bits = (k + 127) * (1 << 23);
        float e;
        memcpy(&e, &bits, sizeof(e));
        const float q = t * (0.254829592f + t * (-0.284496736f + t * (1.421413741f + t * (-1.453152027f + t * 1.061405429f)))) * p * e;
        // q = erfc(a)
        y[i] = x[i] > 0.0f ? x[i] - 0.5f * x[i] * q : 0.5f * x[i] * q;
    }
}

void gelu_op(ggml_tensor * dst, const ggml_tensor * src, int ith, int nth, void *) {
    const int64_t rows = ggml_nrows(src), r0 = rows * ith / nth, r1 = rows * (ith + 1) / nth;
    for (int64_t r = r0; r < r1; ++r) {
        gelu_row((float *) ((char *) dst->data + (size_t) r * dst->nb[1]), (const float *) ((const char *) src->data + (size_t) r * src->nb[1]),
                 src->ne[0]);
    }
}

ggml_tensor * gelu(ggml_context * ctx, ggml_tensor * x) {
    return ggml_map_custom1(ctx, x, gelu_op, GGML_N_TASKS_MAX, nullptr);
}

struct linear {
    ggml_tensor * w = nullptr; // dense weight, or
    llama_littlebit_linear lb; // factors
    ggml_tensor * b = nullptr;

    ggml_tensor * operator()(ggml_context * ctx, ggml_tensor * x) const {
        ggml_tensor * y = w ? ggml_mul_mat(ctx, w, x) : llama_littlebit_mul_mat(ctx, lb, x);
        return b ? ggml_add(ctx, y, b) : y;
    }
};

struct threadpool_deleter { void operator()(ggml_threadpool * p) const { ggml_threadpool_free(p); } };

struct norm {
    ggml_tensor * w = nullptr, * b = nullptr;

    ggml_tensor * operator()(ggml_context * ctx, ggml_tensor * x) const {
        x = ggml_mul(ctx, ggml_norm(ctx, x, 1e-6f), w);
        return b ? ggml_add(ctx, x, b) : x;
    }
};

struct block {
    norm ln1, ln2;
    linear qkv, proj, up, down;
    ggml_tensor * q_norm = nullptr, * k_norm = nullptr;
};

} // namespace

struct littlebit_ncm {
    littlebit_ncm_info info = {};
    int64_t n_embd = 0, n_head = 0, n_prefix = 0, D = 0, dec_head = 0, L = 0, n_vocab = 0;
    float rope_base = 1e4f, soft_temperature = 1, temperature = 1;
    bool query_l0 = false, tied = false, soft = false, sudoku = false;
    std::vector<int64_t> levels;
    std::vector<float> head_scale;
    std::vector<int32_t> l0_strides;
    std::vector<std::string> pieces;
    int32_t eos = -1, pad = -1;

    ggml_context_ptr ctx_w, ctx_kv;
    std::vector<ggml_backend_buffer_ptr> bufs;
    std::unique_ptr<ggml_threadpool, threadpool_deleter> threads; // outlives the backend
    ggml_backend_ptr backend;
    ggml_gallocr_ptr galloc;

    linear in, dec_out;
    ggml_tensor * level_embd = nullptr, * token_embd = nullptr;
    std::vector<block> blocks;
    block dec;
    norm ln_f, dec_ln_f;
    std::vector<linear> heads, phi, l0_up;
    std::vector<ggml_tensor *> codebook_t;
    std::vector<std::vector<float>> codebook; // [K*D] per level
    std::vector<float> token_embd_f, l0_query, l0_unigram, l0_table;
    std::vector<ggml_tensor *> kv_k, kv_v;    // prefix keys (rotated) and values

    littlebit_stats stats = {};

    void load(const char * path, const littlebit_ncm_params & p);

    // run one graph; `build` returns the output and may register inputs
    struct graph_io {
        ggml_context * ctx;
        ggml_cgraph * gf;
        std::vector<std::pair<ggml_tensor *, const void *>> inputs;
        ggml_tensor * input(ggml_type type, int64_t n0, int64_t n1, const void * data) {
            ggml_tensor * t = n1 ? ggml_new_tensor_2d(ctx, type, n0, n1) : ggml_new_tensor_1d(ctx, type, n0);
            ggml_set_input(t);
            inputs.emplace_back(t, data);
            return t;
        }
    };
    template <typename F> std::vector<float> run(F && build) {
        const size_t n_nodes = 8192;
        ggml_context_ptr ctx(ggml_init({ ggml_tensor_overhead() * n_nodes + ggml_graph_overhead_custom(n_nodes, false), nullptr, true }));
        if (!ctx) {
            fail("out of memory");
        }
        graph_io io = { ctx.get(), ggml_new_graph_custom(ctx.get(), n_nodes, false), {} };
        ggml_tensor * out = build(io);
        if (out) {
            ggml_set_output(out);
            ggml_build_forward_expand(io.gf, out);
        }
        if (!ggml_gallocr_alloc_graph(galloc.get(), io.gf)) {
            fail("out of memory");
        }
        for (const auto & it : io.inputs) {
            if (it.first->buffer) { // not used by the graph otherwise
                ggml_backend_tensor_set(it.first, it.second, 0, ggml_nbytes(it.first));
            }
        }
        if (ggml_backend_graph_compute(backend.get(), io.gf) != GGML_STATUS_SUCCESS) {
            fail("computation failed");
        }
        std::vector<float> res(out ? (size_t) ggml_nelements(out) : 0);
        if (out) {
            ggml_backend_tensor_get(out, res.data(), 0, ggml_nbytes(out));
        }
        return res;
    }

    // one transformer block; pos == nullptr: no rotary embedding
    // kv_out: store keys and values there; kv_in: attend to those as well
    ggml_tensor * apply(graph_io & io, const block & b, ggml_tensor * x, int64_t heads_n, ggml_tensor * pos,
                        int il_in, int il_out) const {
        ggml_context * c = io.ctx;
        const int64_t C = x->ne[0], T = x->ne[1], hd = C / heads_n;
        ggml_tensor * h = b.qkv(c, b.ln1(c, x));
        auto split = [&](int i) {
            return ggml_view_3d(c, h, hd, heads_n, T, hd * sizeof(float), h->nb[1], (size_t) (i * C) * sizeof(float));
        };
        ggml_tensor * q = ggml_mul(c, ggml_rms_norm(c, split(0), 1e-6f), b.q_norm);
        ggml_tensor * k = ggml_mul(c, ggml_rms_norm(c, split(1), 1e-6f), b.k_norm);
        ggml_tensor * v = split(2);
        if (pos) {
            q = ggml_rope_ext(c, q, pos, nullptr, (int) hd, GGML_ROPE_TYPE_NEOX, 0, rope_base, 1, 0, 1, 0, 0);
            k = ggml_rope_ext(c, k, pos, nullptr, (int) hd, GGML_ROPE_TYPE_NEOX, 0, rope_base, 1, 0, 1, 0, 0);
        }
        q = ggml_permute(c, q, 0, 2, 1, 3);               // [hd, T, H]
        k = ggml_permute(c, k, 0, 2, 1, 3);
        v = ggml_permute(c, v, 0, 2, 1, 3);
        if (il_out >= 0) {
            ggml_build_forward_expand(io.gf, ggml_cpy(c, k, kv_k[il_out]));
            ggml_build_forward_expand(io.gf, ggml_cpy(c, v, kv_v[il_out]));
        }
        if (il_in >= 0) {
            k = ggml_concat(c, kv_k[il_in], k, 1);
            v = ggml_concat(c, kv_v[il_in], v, 1);
        }
        ggml_tensor * a = ggml_mul_mat(c, ggml_cont(c, k), ggml_cont(c, q));        // [P + T, T, H]
        a = ggml_soft_max_ext(c, a, nullptr, 1.0f / std::sqrt((float) hd), 0.0f);
        ggml_tensor * y = ggml_mul_mat(c, ggml_cont(c, ggml_transpose(c, v)), a);   // [hd, T, H]
        y = ggml_cont_2d(c, ggml_permute(c, y, 0, 2, 1, 3), C, T);
        x = ggml_add(c, x, b.proj(c, y));
        return ggml_add(c, x, b.down(c, gelu(c, b.up(c, b.ln2(c, x)))));
    }

    void run_prefix(const std::vector<int32_t> & toks) {
        std::vector<float> z((size_t) (n_prefix * D));
        for (int64_t t = 0; t < n_prefix; ++t) {
            memcpy(&z[(size_t) (t * D)], &token_embd_f[(size_t) (toks[(size_t) t] * D)], (size_t) D * sizeof(float));
        }
        std::vector<int32_t> pos((size_t) n_prefix);
        for (size_t i = 0; i < pos.size(); ++i) { pos[i] = (int32_t) i; }
        run([&](graph_io & io) -> ggml_tensor * {
            ggml_tensor * x = in(io.ctx, io.input(GGML_TYPE_F32, D, n_prefix, z.data()));
            ggml_tensor * p = io.input(GGML_TYPE_I32, n_prefix, 0, pos.data());
            for (size_t i = 0; i < blocks.size(); ++i) {
                x = apply(io, blocks[i], x, n_head, p, -1, (int) i);
            }
            return x;
        });
    }

    // logits [l * K] of level `lv` from its input [l * D]
    std::vector<float> run_level(size_t lv, const std::vector<float> & xin) {
        const int64_t l = levels[lv];
        const size_t hi = lv - (query_l0 ? 0 : 1);
        std::vector<int32_t> pos((size_t) l);
        for (size_t i = 0; i < pos.size(); ++i) { pos[i] = (int32_t) i; }
        return run([&](graph_io & io) {
            ggml_context * c = io.ctx;
            ggml_tensor * x = in(c, io.input(GGML_TYPE_F32, D, l, xin.data()));
            x = ggml_add(c, x, ggml_view_1d(c, level_embd, n_embd, hi * level_embd->nb[1]));
            ggml_tensor * p = io.input(GGML_TYPE_I32, l, 0, pos.data());
            for (size_t i = 0; i < blocks.size(); ++i) {
                x = apply(io, blocks[i], x, n_head, p, (int) i, -1);
            }
            x = ln_f(c, x);
            if (!tied) {
                return heads[hi](c, x);
            }
            // scaled negative squared distance to the codebook
            x = ggml_mul_mat(c, codebook_t[lv], ggml_mul_mat(c, heads[hi].w, x));
            return ggml_add(c, ggml_scale(c, x, head_scale[hi]), heads[hi].b);
        });
    }

    std::vector<float> run_decoder(const std::vector<float> & z) {
        return run([&](graph_io & io) {
            ggml_tensor * x = apply(io, dec, io.input(GGML_TYPE_F32, D, L, z.data()), dec_head, nullptr, -1, -1);
            return dec_out(io.ctx, dec_ln_f(io.ctx, x));
        });
    }

    // weighted codebook entries [l * D]; logits == nullptr: hard codes
    std::vector<float> embed(size_t lv, const std::vector<int32_t> & codes, const std::vector<float> & probs) const {
        const std::vector<float> & cb = codebook[lv];
        const size_t d = (size_t) D, K = cb.size() / d, l = (size_t) levels[lv];
        std::vector<float> e(l * d, 0.0f);
        for (size_t t = 0; t < l; ++t) {
            for (size_t k = probs.empty() ? (size_t) codes[t] : 0; k < K; ++k) {
                const float w = probs.empty() ? 1.0f : probs[t * K + k];
                for (size_t j = 0; j < d; ++j) {
                    e[t * d + j] += w * cb[k * d + j];
                }
                if (probs.empty()) {
                    break;
                }
            }
        }
        return e;
    }

    // add the contribution of level lv to z [L * D]
    void accumulate(size_t lv, const std::vector<int32_t> & codes, const std::vector<float> & probs, std::vector<float> & z) {
        const size_t d = (size_t) D, n = (size_t) L, l = (size_t) levels[lv];
        if (lv == 0 && !l0_table.empty()) {
            const size_t K = codebook[0].size() / d;
            for (size_t k = 0; k < K; ++k) {
                const float w = probs.empty() ? (k == (size_t) codes[0] ? 1.0f : 0.0f) : probs[k];
                for (size_t i = 0; w != 0.0f && i < n * d; ++i) {
                    z[i] += w * l0_table[k * n * d + i];
                }
            }
            return;
        }
        std::vector<float> e = embed(lv, codes, probs);
        if (lv == 0) {
            for (size_t j = 0; j < l0_up.size(); ++j) {
                const int64_t T = (int64_t) e.size() / D, s = l0_strides[j];
                e = run([&](graph_io & io) {
                    ggml_tensor * y = ggml_mul_mat(io.ctx, l0_up[j].w, io.input(GGML_TYPE_F32, D, T, e.data()));
                    return ggml_add(io.ctx, ggml_reshape_2d(io.ctx, y, D, s * T), l0_up[j].b);
                });
            }
        } else if (l != n) { // linear interpolation, half-pixel centers
            std::vector<float> u(n * d);
            for (size_t t = 0; t < n; ++t) {
                const float s = std::max(0.0f, ((float) t + 0.5f) * ((float) l / (float) n) - 0.5f);
                const size_t i0 = std::min((size_t) s, l - 1), i1 = std::min(i0 + 1, l - 1);
                const float f = s - (float) i0;
                for (size_t j = 0; j < d; ++j) {
                    u[t * d + j] = e[i0 * d + j] * (1.0f - f) + e[i1 * d + j] * f;
                }
            }
            e.swap(u);
        }
        std::vector<float> x(n * 3 * d, 0.0f); // x[t-1], x[t], x[t+1] side by side
        for (size_t t = 0; t < n; ++t) {
            for (size_t o = 0; o < 3; ++o) {
                if (t + o >= 1 && t + o - 1 < n) {
                    memcpy(&x[(t * 3 + o) * d], &e[(t + o - 1) * d], d * sizeof(float));
                }
            }
        }
        const std::vector<float> y = run([&](graph_io & io) {
            return phi[lv](io.ctx, io.input(GGML_TYPE_F32, 3 * D, L, x.data()));
        });
        for (size_t i = 0; i < z.size(); ++i) {
            z[i] += y[i];
        }
    }

    std::vector<float> pool(const std::vector<float> & z, size_t lv) const { // adaptive average
        const size_t d = (size_t) D, n = (size_t) L, l = (size_t) levels[lv];
        std::vector<float> x(l * d, 0.0f);
        for (size_t t = 0; t < l; ++t) {
            const size_t a = t * n / l, b = ((t + 1) * n + l - 1) / l;
            for (size_t i = a; i < b; ++i) {
                for (size_t j = 0; j < d; ++j) {
                    x[t * d + j] += z[i * d + j] / (float) (b - a);
                }
            }
        }
        return x;
    }

    static int32_t sample(const float * logits, size_t K, float temp, int32_t top_k, float top_p, std::mt19937_64 & rng) {
        if (temp <= 0) {
            return (int32_t) (std::max_element(logits, logits + K) - logits);
        }
        std::vector<std::pair<float, int32_t>> c(K);
        for (size_t i = 0; i < K; ++i) { c[i] = { logits[i] / temp, (int32_t) i }; }
        size_t n = K;
        if (top_p < 1 || (top_k > 0 && (size_t) top_k < K)) {
            std::sort(c.begin(), c.end(), [](const auto & a, const auto & b) { return a.first > b.first; });
            n = top_k > 0 ? std::min(K, (size_t) top_k) : K;
        }
        float mx = c[0].first;
        for (size_t i = 0; i < n; ++i) { mx = std::max(mx, c[i].first); }
        std::vector<double> p(n);
        double sum = 0;
        for (size_t i = 0; i < n; ++i) { sum += p[i] = std::exp((double) (c[i].first - mx)); }
        if (top_p < 1) {
            double acc = 0;
            for (size_t i = 0; i < n; ++i) {
                if (acc > top_p * sum) { n = i; break; }
                acc += p[i];
            }
            sum = acc;
        }
        double r = (double) (rng() >> 11) * 0x1.0p-53 * sum;
        for (size_t i = 0; i < n; ++i) {
            if ((r -= p[i]) < 0) { return c[i].second; }
        }
        return c[n - 1].second;
    }

    std::vector<int32_t> generate(const std::vector<int32_t> & prefix, const littlebit_ncm_gen_params & gp,
                                  littlebit_ncm_trace_cb trace, void * user) {
        if ((int64_t) prefix.size() != n_prefix) {
            fail("the prefix must have " + std::to_string(n_prefix) + " tokens");
        }
        for (int32_t t : prefix) {
            if (t < 0 || t >= n_vocab) {
                fail("token out of range");
            }
        }
        const double t0 = now_ms();
        const float temp = gp.temperature < 0 ? temperature : gp.temperature;
        std::mt19937_64 rng(gp.seed == 0xFFFFFFFFu ? std::random_device{}() : gp.seed);
        run_prefix(prefix);
        const double t1 = now_ms();

        std::vector<float> z((size_t) (L * D), 0.0f), x = l0_query;
        for (size_t lv = 0; lv < levels.size(); ++lv) {
            const size_t l = (size_t) levels[lv], K = codebook[lv].size() / (size_t) D;
            std::vector<int32_t> codes(l);
            std::vector<float> probs;
            if (lv == 0 && !query_l0) {
                std::vector<float> lg(K);
                for (size_t k = 0; k < K; ++k) { lg[k] = l0_unigram[k] > 0 ? std::log(l0_unigram[k]) : -INFINITY; }
                for (int32_t & c : codes) { c = sample(lg.data(), K, 1.0f, 0, 1.0f, rng); }
            } else {
                const std::vector<float> logits = run_level(lv, x);
                if (trace) {
                    trace((int32_t) lv, logits.data(), logits.size(), user);
                }
                if (soft) {
                    probs.resize(l * K);
                    for (size_t t = 0; t < l; ++t) {
                        const float * lg = &logits[t * K];
                        const float mx = *std::max_element(lg, lg + K);
                        float sum = 0;
                        for (size_t k = 0; k < K; ++k) { sum += probs[t * K + k] = std::exp((lg[k] - mx) / soft_temperature); }
                        for (size_t k = 0; k < K; ++k) { probs[t * K + k] /= sum; }
                    }
                } else {
                    for (size_t t = 0; t < l; ++t) {
                        codes[t] = sample(&logits[t * K], K, temp, gp.top_k, gp.top_p, rng);
                    }
                }
            }
            accumulate(lv, codes, probs, z);
            if (lv + 1 < levels.size()) {
                x = pool(z, lv + 1);
            }
        }
        const std::vector<float> out = run_decoder(z);
        if (trace) {
            trace(-1, out.data(), out.size(), user);
        }
        std::vector<int32_t> toks((size_t) L);
        for (size_t t = 0; t < toks.size(); ++t) {
            const float * lg = &out[t * (size_t) n_vocab];
            toks[t] = (int32_t) (std::max_element(lg, lg + n_vocab) - lg);
        }
        stats = { (int32_t) n_prefix, (int32_t) L, t1 - t0, now_ms() - t1 };
        return toks;
    }
};

void littlebit_ncm::load(const char * path, const littlebit_ncm_params & p) {
    ggml_context * meta = nullptr;
    gguf_context_ptr g(gguf_init_from_file(path, { /*.no_alloc =*/ true, /*.ctx =*/ &meta }));
    if (!g) {
        fail(std::string("cannot read ") + path);
    }
    ctx_w.reset(meta);

    auto key = [&](const std::string & k, gguf_type type, bool required = true) -> int64_t {
        const int64_t id = gguf_find_key(g.get(), k.c_str());
        if (id < 0 && !required) {
            return -1;
        }
        if (id < 0 || gguf_get_kv_type(g.get(), id) != type) {
            fail("missing or invalid metadata " + k);
        }
        return id;
    };
    auto u32 = [&](const char * k, int64_t lo, int64_t hi) {
        const int64_t v = gguf_get_val_u32(g.get(), key(k, GGUF_TYPE_UINT32));
        if (v < lo || v > hi) {
            fail(std::string("invalid ") + k);
        }
        return v;
    };
    auto f32  = [&](const char * k) { return gguf_get_val_f32(g.get(), key(k, GGUF_TYPE_FLOAT32)); };
    auto flag = [&](const char * k) { return gguf_get_val_bool(g.get(), key(k, GGUF_TYPE_BOOL)); };
    auto arr  = [&](const char * k, gguf_type type) {
        const int64_t id = key(k, GGUF_TYPE_ARRAY);
        if (gguf_get_arr_type(g.get(), id) != type) {
            fail(std::string("invalid ") + k);
        }
        return id;
    };
    if (std::string(gguf_get_val_str(g.get(), key("general.architecture", GGUF_TYPE_STRING))) != "ncm") {
        fail("not an NCM model");
    }
    if (u32("ncm.version", 0, UINT32_MAX) != 1) {
        fail("unsupported ncm.version");
    }
    const int64_t n_layer = u32("ncm.block_count", 1, 1024);
    n_embd   = u32("ncm.embedding_length", 1, 1 << 16);
    n_head   = u32("ncm.head_count", 1, n_embd);
    n_prefix = u32("ncm.prefix_length", 1, 1 << 16);
    D        = u32("ncm.code_length", 1, 1 << 16);
    dec_head = u32("ncm.decoder.head_count", 1, D);
    n_vocab  = u32("ncm.vocab_size", 1, 1 << 24);
    if (n_embd % n_head || (n_embd / n_head) % 2 || D % dec_head) {
        fail("invalid number of heads");
    }
    rope_base        = f32("ncm.rope_base");
    soft_temperature = f32("ncm.soft_temperature");
    temperature      = f32("ncm.temperature");
    if (!(rope_base > 0) || !(soft_temperature > 0) || !(temperature >= 0)) {
        fail("invalid rope_base or temperature");
    }
    query_l0 = flag("ncm.query_level0");
    tied     = flag("ncm.tied_heads");
    soft     = flag("ncm.soft_decisions");
    sudoku   = std::string(gguf_get_val_str(g.get(), key("ncm.task", GGUF_TYPE_STRING))) == "sudoku";
    {
        const int64_t id = arr("ncm.levels", GGUF_TYPE_INT32);
        const int32_t * d = (const int32_t *) gguf_get_arr_data(g.get(), id);
        levels.assign(d, d + gguf_get_arr_n(g.get(), id));
        if (levels.size() < 2 || levels.size() > 1024) {
            fail("invalid ncm.levels");
        }
        for (size_t i = 0; i < levels.size(); ++i) {
            if (levels[i] < 1 || levels[i] > (1 << 16) || (i && levels[i] < levels[i - 1])) {
                fail("invalid ncm.levels");
            }
        }
        L = levels.back();
    }
    const size_t n_lvl = levels.size(), n_heads = n_lvl - (query_l0 ? 0 : 1);
    if (tied) {
        const int64_t id = arr("ncm.head_scale", GGUF_TYPE_FLOAT32);
        const float * d = (const float *) gguf_get_arr_data(g.get(), id);
        head_scale.assign(d, d + gguf_get_arr_n(g.get(), id));
        if (head_scale.size() != n_heads) {
            fail("invalid ncm.head_scale");
        }
    }
    if (gguf_find_key(g.get(), "ncm.l0_strides") >= 0) {
        const int64_t id = arr("ncm.l0_strides", GGUF_TYPE_INT32);
        const int32_t * d = (const int32_t *) gguf_get_arr_data(g.get(), id);
        l0_strides.assign(d, d + gguf_get_arr_n(g.get(), id));
        int64_t n = levels[0];
        for (int32_t s : l0_strides) {
            if (s < 1 || s > L || (n *= s) > L) { fail("invalid ncm.l0_strides"); }
        }
        if (n != L) {
            fail("invalid ncm.l0_strides");
        }
    }
    if (gguf_find_key(g.get(), "ncm.tokenizer.pieces_hex") >= 0) {
        const int64_t id = arr("ncm.tokenizer.pieces_hex", GGUF_TYPE_STRING);
        for (size_t i = 0; i < gguf_get_arr_n(g.get(), id); ++i) {
            const std::string hex = gguf_get_arr_str(g.get(), id, i);
            std::string s;
            for (size_t j = 0; j + 1 < hex.size(); j += 2) {
                s += (char) strtol(hex.substr(j, 2).c_str(), nullptr, 16);
            }
            pieces.push_back(s);
        }
        eos = gguf_get_val_i32(g.get(), key("ncm.tokenizer.eos", GGUF_TYPE_INT32));
        pad = gguf_get_val_i32(g.get(), key("ncm.tokenizer.pad", GGUF_TYPE_INT32));
    }
    if (sudoku && (n_prefix != 91 || L != 89 || n_vocab != 12)) {
        fail("unexpected sudoku dimensions");
    }

    // ---- tensors
    std::map<const ggml_tensor *, bool> used; // value: multiplied as a weight matrix
    auto get = [&](const std::string & name, int64_t n0, int64_t n1, bool matrix = false, bool required = true, bool table = false) -> ggml_tensor * {
        ggml_tensor * t = ggml_get_tensor(meta, name.c_str());
        if (!t && !required) {
            return nullptr;
        }
        if (!t) {
            fail("missing tensor " + name);
        }
        if (t->ne[0] != n0 || t->ne[1] != n1 || t->ne[2] != 1 || t->ne[3] != 1) {
            fail("wrong shape of tensor " + name);
        }
        // weight matrices may be in any of ggml's block formats; tables (looked up by row) stay F32 or F16
        const bool block = t->type == GGML_TYPE_Q8_0 || t->type == GGML_TYPE_Q4_0 || t->type == GGML_TYPE_Q4_1 ||
                           t->type == GGML_TYPE_Q5_0 || t->type == GGML_TYPE_Q5_1 || t->type == GGML_TYPE_Q4_K ||
                           t->type == GGML_TYPE_Q5_K || t->type == GGML_TYPE_Q6_K;
        if (t->type != GGML_TYPE_F32 && !(matrix && t->type == GGML_TYPE_F16) && !(matrix && !table && block)) {
            fail("unsupported type of tensor " + name);
        }
        used[t] = matrix;
        return t;
    };
    auto lin = [&](const std::string & n, int64_t i, int64_t o) {
        linear r;
        r.b = get(n + ".bias", o, 1, false, false);
        ggml_tensor * u = ggml_get_tensor(meta, (n + ".lb_u").c_str());
        if (!u) {
            r.w = get(n + ".weight", i, o, true);
            return r;
        }
        if (gguf_find_key(g.get(), "littlebit.version") < 0 ||
            gguf_get_val_u32(g.get(), key("littlebit.version", GGUF_TYPE_UINT32)) != 1) {
            fail("unsupported littlebit.version");
        }
        if (ggml_get_tensor(meta, (n + ".weight").c_str())) {
            fail("both dense and LittleBit weights for " + n);
        }
        llama_littlebit_linear & lb = r.lb;
        auto factor = [&](const char * s) {
            ggml_tensor * t = ggml_get_tensor(meta, (n + s).c_str());
            if (t && (t->ne[2] != 1 || t->ne[3] != 1 ||
                      (t->type != GGML_TYPE_Q1_0 && t->type != GGML_TYPE_F16 && t->type != GGML_TYPE_F32))) {
                fail("invalid tensor " + n + s);
            }
            if (t) {
                used[t] = true;
            }
            return t;
        };
        lb.u    = factor(".lb_u");
        lb.v    = factor(".lb_v");
        lb.v_r  = factor(".lb_v_r");
        lb.v2   = get(n + ".lb_v2", i, 1);
        lb.v2_r = lb.v_r ? get(n + ".lb_v2_r", i, 1) : nullptr;
        const int64_t n_br = lb.v_r ? 2 : 1;
        if (!lb.v || lb.v->ne[0] < i || lb.v->ne[0] >= i + 128 || lb.u->ne[0] % n_br || lb.u->ne[1] != o ||
            lb.u->ne[0] / n_br < lb.v->ne[1] || (lb.v_r && (lb.v_r->ne[0] != lb.v->ne[0] || lb.v_r->ne[1] != lb.v->ne[1]))) {
            fail("inconsistent LittleBit tensors for " + n);
        }
        lb.n_in = i; lb.n_out = o; lb.in_pad = lb.v->ne[0]; lb.r = lb.v->ne[1]; lb.r_pad = lb.u->ne[0] / n_br;
        info.n_littlebit++;
        return r;
    };
    auto nrm = [&](const std::string & n, int64_t d, bool bias = true) {
        norm r;
        r.w = get(n + ".weight", d, 1);
        r.b = bias ? get(n + ".bias", d, 1) : nullptr;
        return r;
    };
    auto blk = [&](const std::string & n, int64_t d, int64_t h) {
        block b;
        b.ln1 = nrm(n + "ln1", d);
        b.ln2 = nrm(n + "ln2", d);
        b.q_norm = get(n + "q_norm", d / h, 1);
        b.k_norm = get(n + "k_norm", d / h, 1);
        b.qkv  = lin(n + "qkv", d, 3 * d);
        b.proj = lin(n + "proj", d, d);
        b.up   = lin(n + "up", d, 4 * d);
        b.down = lin(n + "down", 4 * d, d);
        return b;
    };

    in = lin("in", D, n_embd);
    level_embd = get("level_embd", n_embd, (int64_t) n_heads);
    token_embd = get("token_embd", D, n_vocab, true, true, true);
    for (int64_t i = 0; i < n_layer; ++i) {
        blocks.push_back(blk("blk." + std::to_string(i) + ".", n_embd, n_head));
    }
    ln_f     = nrm("ln_f", n_embd);
    dec      = blk("dec.", D, dec_head);
    dec_ln_f = nrm("dec.ln_f", D, false);
    dec_out.w = get("dec.output", D, n_vocab, true);
    std::vector<ggml_tensor *> books;
    for (size_t i = 0; i < n_lvl; ++i) {
        const std::string name = "codebook." + std::to_string(i);
        const ggml_tensor * t = ggml_get_tensor(meta, name.c_str());
        books.push_back(get(name, D, t ? t->ne[1] : 0, true, true, true));
    }
    for (size_t h = 0; h < n_heads; ++h) {
        const size_t lv = h + n_lvl - n_heads;
        const std::string n = "head." + std::to_string(h);
        const int64_t K = books[lv]->ne[1];
        linear r;
        if (tied) {
            r.w = get(n + ".proj", n_embd, D, true);
            r.b = get(n + ".bias", K, 1);
        } else {
            r = lin(n, n_embd, K);
        }
        heads.push_back(r);
    }
    codebook_t = books;
    ggml_tensor * query = query_l0 ? get("l0_query", D, levels[0]) : nullptr;
    ggml_tensor * unigram = query_l0 ? nullptr : get("level0_unigram", books[0]->ne[1], 1);
    ggml_tensor * table = nullptr;
    if (l0_strides.empty()) {
        if (levels[0] != 1) {
            fail("level-0 table with more than one position");
        }
        table = get("l0_table", L * D, books[0]->ne[1], true, true, true);
    }
    for (size_t j = 0; j < l0_strides.size(); ++j) {
        linear r;
        r.w = get("l0_up." + std::to_string(j) + ".weight", D, l0_strides[j] * D, true);
        r.b = get("l0_up." + std::to_string(j) + ".bias", D, 1);
        l0_up.push_back(r);
    }
    for (size_t i = 0; i < n_lvl; ++i) {
        phi.push_back(i == 0 && table ? linear() : lin("phi." + std::to_string(i), 3 * D, D));
    }

    // ---- data
    backend.reset(ggml_backend_cpu_init());
    if (!backend) {
        fail("no CPU backend");
    }
    const int32_t n_threads = p.n_threads > 0 ? p.n_threads : littlebit_default_threads();
    ggml_backend_cpu_set_n_threads(backend.get(), n_threads);
    if (n_threads > 1) { // a generation is dozens of small graphs; without a pool each would start its own threads
        ggml_threadpool_params tp = ggml_threadpool_params_default(n_threads);
        threads.reset(ggml_threadpool_new(&tp));
        if (!threads) {
            fail("cannot start threads");
        }
        ggml_backend_cpu_set_threadpool(backend.get(), threads.get());
    }
    ggml_backend_buffer_type_t cpu = ggml_backend_cpu_buffer_type();
    galloc.reset(ggml_gallocr_new(cpu));

    std::vector<ggml_backend_buffer_type_t> bufts;
    ggml_backend_dev_t dev = ggml_backend_get_device(backend.get());
    if (p.repack) {
        auto extra = (ggml_backend_dev_get_extra_bufts_t) ggml_backend_reg_get_proc_address(
                ggml_backend_dev_backend_reg(dev), "ggml_backend_dev_get_extra_bufts");
        for (ggml_backend_buffer_type_t * b = extra ? extra(dev) : nullptr; b && *b; ++b) {
            bufts.push_back(*b);
        }
    }
    // host-side copies are made from these, so they have to stay in plain memory
    std::vector<const ggml_tensor *> plain(books.begin(), books.end());
    plain.insert(plain.end(), { token_embd, table, query, unigram });
    auto supports = [&](ggml_backend_buffer_type_t bt, const ggml_tensor * w) {
        // a dry run: does this buffer type implement mul_mat for such a weight?
        ggml_context_ptr c(ggml_init({ ggml_tensor_overhead() * 4, nullptr, true }));
        ggml_tensor * t = ggml_new_tensor_2d(c.get(), w->type, w->ne[0], w->ne[1]);
        ggml_tensor * op = ggml_mul_mat(c.get(), t, ggml_new_tensor_2d(c.get(), GGML_TYPE_F32, w->ne[0], 64));
        ggml_backend_buffer_ptr b(ggml_backend_buft_alloc_buffer(bt, 0)); // empty, as in llama.cpp
        if (!b) {
            return false;
        }
        t->buffer = b.get();
        const bool ok = ggml_backend_dev_supports_op(dev, op);
        t->buffer = nullptr;
        return ok;
    };
    std::map<ggml_backend_buffer_type_t, std::vector<ggml_tensor *>> groups;
    for (ggml_tensor * t = ggml_get_first_tensor(meta); t; t = ggml_get_next_tensor(meta, t)) {
        const auto it = used.find(t);
        if (it == used.end()) {
            fail(std::string("unexpected tensor ") + ggml_get_name(t));
        }
        ggml_backend_buffer_type_t bt = cpu;
        if (it->second && std::find(plain.begin(), plain.end(), t) == plain.end()) {
            for (ggml_backend_buffer_type_t b : bufts) {
                if (supports(b, t)) { bt = b; break; }
            }
        }
        groups[bt].push_back(t);
    }
    for (auto & gr : groups) {
        const size_t align = ggml_backend_buft_get_alignment(gr.first);
        size_t size = align;
        for (ggml_tensor * t : gr.second) {
            size += GGML_PAD(ggml_backend_buft_get_alloc_size(gr.first, t), align);
        }
        ggml_backend_buffer_t buf = ggml_backend_buft_alloc_buffer(gr.first, size);
        if (!buf) {
            fail("out of memory");
        }
        bufs.emplace_back(buf);
        ggml_backend_buffer_set_usage(buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        ggml_tallocr ta = ggml_tallocr_new(buf);
        for (ggml_tensor * t : gr.second) {
            if (ggml_tallocr_alloc(&ta, t) != GGML_STATUS_SUCCESS) {
                fail("out of memory");
            }
        }
    }
    std::ifstream f(path, std::ios::binary);
    f.seekg(0, std::ios::end);
    const uint64_t file_size = (uint64_t) std::max<std::streamoff>(0, f.tellg());
    std::vector<char> tmp;
    for (int64_t i = 0; i < gguf_get_n_tensors(g.get()); ++i) {
        ggml_tensor * t = ggml_get_tensor(meta, gguf_get_tensor_name(g.get(), i));
        const uint64_t off = gguf_get_data_offset(g.get()) + gguf_get_tensor_offset(g.get(), i), n = ggml_nbytes(t);
        if (off > file_size || n > file_size - off) {
            fail("the file is truncated");
        }
        tmp.resize(n);
        f.seekg((std::streamoff) off);
        if (!f.read(tmp.data(), (std::streamsize) n)) {
            fail("cannot read the file");
        }
        ggml_backend_tensor_set(t, tmp.data(), 0, n);
        info.size_bytes += n;
    }
    auto floats = [](const ggml_tensor * t) {
        std::vector<float> r;
        if (t) {
            r.resize((size_t) ggml_nelements(t));
            if (t->type == GGML_TYPE_F16) {
                ggml_fp16_to_fp32_row((const ggml_fp16_t *) t->data, r.data(), ggml_nelements(t));
            } else {
                memcpy(r.data(), t->data, r.size() * sizeof(float));
            }
        }
        return r;
    };
    for (const ggml_tensor * t : books) {
        codebook.push_back(floats(t));
    }
    token_embd_f = floats(token_embd);
    l0_query     = floats(query);
    l0_unigram   = floats(unigram);
    l0_table     = floats(table);
    if (!query_l0 && !std::any_of(l0_unigram.begin(), l0_unigram.end(), [](float x) { return x > 0 && x <= 1; })) {
        fail("invalid level0_unigram");
    }

    ctx_kv.reset(ggml_init({ ggml_tensor_overhead() * (size_t) (2 * n_layer), nullptr, true }));
    for (int64_t i = 0; i < n_layer; ++i) {
        kv_k.push_back(ggml_new_tensor_3d(ctx_kv.get(), GGML_TYPE_F32, n_embd / n_head, n_prefix, n_head));
        kv_v.push_back(ggml_new_tensor_3d(ctx_kv.get(), GGML_TYPE_F32, n_embd / n_head, n_prefix, n_head));
    }
    ggml_backend_buffer_t kb = ggml_backend_alloc_ctx_tensors_from_buft(ctx_kv.get(), cpu);
    if (!kb) {
        fail("out of memory");
    }
    bufs.emplace_back(kb);

    info.n_prefix = (int32_t) n_prefix;
    info.n_output = (int32_t) L;
    info.n_vocab  = (int32_t) n_vocab;
    info.n_levels = (int32_t) n_lvl;
    info.n_layers = (int32_t) n_layer;
    info.n_threads = n_threads;
    info.sudoku   = sudoku;
}

// ---- C API

littlebit_ncm_params littlebit_default_ncm_params(void) { return { 0, true }; }
littlebit_ncm_gen_params littlebit_default_ncm_gen_params(void) { return { -1.0f, 0, 1.0f, 0xFFFFFFFFu }; }

littlebit_ncm * littlebit_ncm_load(const char * path, const littlebit_ncm_params * params) {
    return guarded((littlebit_ncm *) nullptr, [&]() -> littlebit_ncm * {
        if (!path) {
            fail("path is NULL");
        }
        init();
        std::unique_ptr<littlebit_ncm> m(new littlebit_ncm);
        m->load(path, params ? *params : littlebit_default_ncm_params());
        return m.release();
    });
}

void littlebit_ncm_free(littlebit_ncm * m) { delete m; }

int littlebit_ncm_get_info(const littlebit_ncm * m, littlebit_ncm_info * info) {
    return guarded(-1, [&] {
        if (!m || !info) {
            fail("invalid argument");
        }
        *info = m->info;
        return 0;
    });
}

littlebit_stats littlebit_ncm_stats(const littlebit_ncm * m) { return m ? m->stats : littlebit_stats{}; }

int32_t littlebit_ncm_generate(littlebit_ncm * m, const int32_t * prefix, int32_t n_prefix, const littlebit_ncm_gen_params * params,
                               int32_t * out, int32_t n_out, littlebit_ncm_trace_cb trace, void * user) {
    return guarded((int32_t) -1, [&] {
        if (!m || !prefix || n_prefix < 0 || !out) {
            fail("invalid argument");
        }
        if (n_out < m->L) {
            fail("the output needs room for " + std::to_string(m->L) + " tokens");
        }
        const auto toks = m->generate({ prefix, prefix + n_prefix }, params ? *params : littlebit_default_ncm_gen_params(), trace, user);
        std::copy(toks.begin(), toks.end(), out);
        return (int32_t) toks.size();
    });
}

int littlebit_ncm_sudoku(littlebit_ncm * m, const char * puzzle, char * solution) {
    return guarded(-1, [&] {
        if (!m || !puzzle || !solution) {
            fail("invalid argument");
        }
        if (!m->sudoku) {
            fail("not a sudoku model");
        }
        if (strlen(puzzle) != 81 || strspn(puzzle, ".0123456789") != 81) {
            fail("a puzzle is 81 characters: digits, and 0 or . for empty cells");
        }
        std::vector<int32_t> prefix = { 11 };
        for (int i = 0; i < 81; ++i) {
            if (i && i % 9 == 0) {
                prefix.push_back(10);
            }
            prefix.push_back(puzzle[i] == '.' ? 0 : puzzle[i] - '0');
        }
        prefix.push_back(11);
        const auto toks = m->generate(prefix, littlebit_default_ncm_gen_params(), nullptr, nullptr);
        for (int i = 0; i < 81; ++i) {
            const int32_t t = toks[(size_t) (i + i / 9)];
            solution[i] = t >= 1 && t <= 9 ? (char) ('0' + t) : '?';
        }
        solution[81] = '\0';
        return 0;
    });
}

int littlebit_ncm_text(littlebit_ncm * m, const littlebit_ncm_gen_params * params, littlebit_text_cb cb, void * user) {
    return guarded(-1, [&] {
        if (!m || !cb) {
            fail("invalid argument");
        }
        if (m->pieces.empty() || m->n_prefix != 1 || m->eos < 0) {
            fail("this model has no text tokenizer");
        }
        const auto toks = m->generate({ m->eos }, params ? *params : littlebit_default_ncm_gen_params(), nullptr, nullptr);
        std::string s;
        for (int32_t t : toks) {
            if (t == m->eos) {
                break;
            }
            if (t != m->pad && (size_t) t < m->pieces.size()) {
                s += m->pieces[(size_t) t];
            }
        }
        cb(s.data(), s.size(), user);
        return 0;
    });
}
