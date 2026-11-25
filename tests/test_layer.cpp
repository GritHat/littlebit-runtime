// Layer-level check of llama_littlebit_mul_mat on the CPU backend.
//
//   test_layer model.gguf <base tensor name, e.g. blk.0.attn_q> n_tokens x_out.bin y_out.bin [repack]
//
// With "repack", the factors go through the CPU backend's repacked layout where it has one.
//
// Loads the LittleBit factors of one layer from a GGUF, feeds random activations through the exact graph
// that llama.cpp builds, and writes x and y so that tests/test_layer.py can compare against numpy.

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "gguf.h"
#include "llama-littlebit.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

// copy w into a buffer of the CPU backend's repacking type; returns w itself if that does not apply
static ggml_tensor * repack(ggml_context * ctx, ggml_tensor * w, int & n_repacked) {
    ggml_backend_dev_t dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    auto get_extra = (ggml_backend_dev_get_extra_bufts_t) ggml_backend_reg_get_proc_address(
            ggml_backend_dev_backend_reg(dev), "ggml_backend_dev_get_extra_bufts");
    if (!w || !get_extra) {
        return w;
    }
    for (ggml_backend_buffer_type_t * bufts = get_extra(dev); bufts && *bufts; ++bufts) {
        if (strcmp(ggml_backend_buft_name(*bufts), "CPU_REPACK") != 0) {
            continue;
        }
        ggml_tensor * t = ggml_new_tensor_2d(ctx, w->type, w->ne[0], w->ne[1]);
        ggml_backend_buffer_t buf = ggml_backend_buft_alloc_buffer(*bufts, ggml_backend_buft_get_alloc_size(*bufts, t));
        ggml_backend_tensor_alloc(buf, t, ggml_backend_buffer_get_base(buf));
        ggml_tensor * op = ggml_mul_mat(ctx, t, ggml_new_tensor_2d(ctx, GGML_TYPE_F32, w->ne[0], 8));
        if (t->extra && ggml_backend_dev_supports_op(dev, op)) {
            ggml_backend_tensor_set(t, w->data, 0, ggml_nbytes(w));
            ++n_repacked;
            return t;
        }
    }
    return w;
}

int main(int argc, char ** argv) {
    if (argc < 6) {
        fprintf(stderr, "usage: %s model.gguf base n_tokens x.bin y.bin\n", argv[0]);
        return 1;
    }
    const std::string base = argv[2];
    const int n_tokens = atoi(argv[3]);

    ggml_context * wctx = nullptr;
    gguf_init_params gp = { /*.no_alloc =*/ false, /*.ctx =*/ &wctx };
    gguf_context * g = gguf_init_from_file(argv[1], gp);
    if (!g) {
        fprintf(stderr, "failed to read %s\n", argv[1]);
        return 1;
    }

    auto get = [&](const std::string & n) { return ggml_get_tensor(wctx, (base + "." + n).c_str()); };

    llama_littlebit_linear lb;
    lb.v2   = get("lb_v2");
    lb.v    = get("lb_v");
    lb.v2_r = get("lb_v2_r");
    lb.v_r  = get("lb_v_r");
    lb.u    = get("lb_u");
    if (!lb.v2 || !lb.v || !lb.u) {
        fprintf(stderr, "no LittleBit tensors for %s\n", base.c_str());
        return 1;
    }
    const int64_t n_br = lb.v_r ? 2 : 1;
    lb.n_in   = lb.v2->ne[0];
    lb.n_out  = lb.u->ne[1];
    lb.in_pad = lb.v->ne[0];
    lb.r      = lb.v->ne[1];
    lb.r_pad  = lb.u->ne[0] / n_br;

    ggml_init_params cp = { /*.mem_size =*/ 16u*1024*1024, /*.mem_buffer =*/ nullptr, /*.no_alloc =*/ true };
    ggml_context * ctx = ggml_init(cp);

    int n_repacked = 0;
    if (argc > 6 && strcmp(argv[6], "repack") == 0) {
        lb.v   = repack(ctx, lb.v,   n_repacked);
        lb.v_r = repack(ctx, lb.v_r, n_repacked);
        lb.u   = repack(ctx, lb.u,   n_repacked);
    }

    ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, lb.n_in, n_tokens);
    ggml_tensor * y = llama_littlebit_mul_mat(ctx, lb, x);
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, y);

    // a separate buffer for every tensor of the graph that has no data yet
    std::vector<ggml_tensor *> todo = { x };
    for (int i = 0; i < ggml_graph_n_nodes(gf); ++i) {
        todo.push_back(ggml_graph_node(gf, i));
    }
    std::vector<std::vector<char>> data;
    for (ggml_tensor * t : todo) {
        if (!t->data) {
            data.emplace_back(ggml_nbytes(t) + 64);
            t->data = (void *) (((uintptr_t) data.back().data() + 63) & ~(uintptr_t) 63);
        }
    }

    std::mt19937 rng(42);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    float * xd = (float *) x->data;
    for (int64_t i = 0; i < ggml_nelements(x); ++i) {
        xd[i] = nd(rng);
    }

    ggml_backend_t backend = ggml_backend_cpu_init();
    ggml_backend_cpu_set_n_threads(backend, 4);
    if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "compute failed\n");
        return 1;
    }
    ggml_backend_free(backend);

    FILE * fx = fopen(argv[4], "wb"); fwrite(x->data, 1, ggml_nbytes(x), fx); fclose(fx);
    FILE * fy = fopen(argv[5], "wb"); fwrite(y->data, 1, ggml_nbytes(y), fy); fclose(fy);
    printf("%s: in=%lld out=%lld r=%lld r_pad=%lld in_pad=%lld branches=%lld repacked=%d\n", base.c_str(),
           (long long) lb.n_in, (long long) lb.n_out, (long long) lb.r, (long long) lb.r_pad,
           (long long) lb.in_pad, (long long) n_br, n_repacked);

    ggml_free(ctx);
    ggml_free(wctx);
    gguf_free(g);
    return 0;
}
