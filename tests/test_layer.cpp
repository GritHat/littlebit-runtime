// Layer-level check of llama_littlebit_mul_mat on the CPU backend.
//
//   test_layer model.gguf <base tensor name, e.g. blk.0.attn_q> n_tokens x_out.bin y_out.bin
//
// Loads the LittleBit factors of one layer from a GGUF, feeds random activations through the exact graph
// that llama.cpp builds, and writes x and y so that tests/test_layer.py can compare against numpy.

#include "ggml.h"
#include "ggml-cpu.h"
#include "gguf.h"
#include "llama-littlebit.h"

#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

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

    ggml_init_params cp = { /*.mem_size =*/ 256u*1024*1024, /*.mem_buffer =*/ nullptr, /*.no_alloc =*/ false };
    ggml_context * ctx = ggml_init(cp);

    ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, lb.n_in, n_tokens);
    std::mt19937 rng(42);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    float * xd = (float *) x->data;
    for (int64_t i = 0; i < ggml_nelements(x); ++i) {
        xd[i] = nd(rng);
    }

    ggml_tensor * y = llama_littlebit_mul_mat(ctx, lb, x);
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, y);
    ggml_graph_compute_with_ctx(ctx, gf, 4);

    FILE * fx = fopen(argv[4], "wb"); fwrite(x->data, 1, ggml_nbytes(x), fx); fclose(fx);
    FILE * fy = fopen(argv[5], "wb"); fwrite(y->data, 1, ggml_nbytes(y), fy); fclose(fy);
    printf("%s: in=%lld out=%lld r=%lld r_pad=%lld in_pad=%lld branches=%lld\n", base.c_str(),
           (long long) lb.n_in, (long long) lb.n_out, (long long) lb.r, (long long) lb.r_pad,
           (long long) lb.in_pad, (long long) n_br);

    ggml_free(ctx);
    ggml_free(wctx);
    gguf_free(g);
    return 0;
}
