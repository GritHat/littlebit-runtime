// ncm
#include "cli.h"
#include "littlebit_ncm.h"

#include <cctype>
#include <cstdio>
#include <iostream>

int cmd_ncm(const options & o) {
    set_log(o);
    littlebit_ncm_params np = littlebit_default_ncm_params();
    np.n_threads = o.n_threads.value_or(0);
    np.repack    = o.repack;
    struct holder {
        littlebit_ncm * m = nullptr;
        ~holder() { littlebit_ncm_free(m); }
    } h;
    if (!(h.m = littlebit_ncm_load(o.model.c_str(), &np))) {
        die_api("cannot load model");
    }
    littlebit_ncm_info info;
    littlebit_ncm_get_info(h.m, &info);
    littlebit_ncm_gen_params gp = littlebit_default_ncm_gen_params();
    gp.temperature = o.temperature.value_or(gp.temperature);
    gp.top_k       = o.top_k.value_or(gp.top_k);
    gp.top_p       = o.top_p.value_or(gp.top_p);
    gp.seed        = o.seed.value_or(gp.seed);
    fprintf(stderr, "[NCM, %d layers, %d LittleBit, %.1f MB, %d threads]\n", info.n_layers, info.n_littlebit,
            info.size_bytes / 1e6, info.n_threads);
    int n = 0;
    double ms = 0;
    auto count = [&] {
        const littlebit_stats st = littlebit_ncm_stats(h.m);
        ms += st.prompt_ms + st.generate_ms;
        ++n;
    };
    if (!o.tokens.empty()) { // raw tokens in and out, and all logits
        const std::vector<int32_t> toks = parse_tokens(o.tokens);
        std::vector<int32_t> out((size_t) info.n_output);
        std::vector<float> logits;
        auto trace = [](int32_t, const float * x, size_t len, void * u) {
            auto * v = (std::vector<float> *) u;
            v->insert(v->end(), x, x + len);
        };
        if (littlebit_ncm_generate(h.m, toks.data(), (int32_t) toks.size(), &gp, out.data(), info.n_output, trace, &logits) < 0) {
            die_api("generation failed");
        }
        for (size_t i = 0; i < out.size(); ++i) {
            printf("%s%d", i ? "," : "", out[i]);
        }
        printf("\n");
        if (!o.out.empty()) {
            write_floats(o.out, logits);
        }
        count();
    } else if (info.sudoku) {
        auto solve = [&](const std::string & line) {
            char sol[82];
            if (littlebit_ncm_sudoku(h.m, line.c_str(), sol) != 0) {
                die_api("cannot solve");
            }
            printf("%s\n", sol);
            fflush(stdout);
            count();
        };
        std::string line;
        if (o.prompt) {
            solve(*o.prompt);
        } else {
            while (std::getline(std::cin, line)) {
                while (!line.empty() && isspace((unsigned char) line.back())) {
                    line.pop_back();
                }
                if (!line.empty()) {
                    solve(line);
                }
            }
        }
    } else {
        if (littlebit_ncm_text(h.m, &gp, print_text, nullptr) != 0) {
            die_api("generation failed");
        }
        printf("\n");
        count();
    }
    if (n) {
        fprintf(stderr, "[%d generated, %.1f ms each]\n", n, ms / n);
    }
    return 0;
}
