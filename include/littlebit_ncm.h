// littlebit_ncm: Next-Concept Modeling (NCM) models, which generate whole levels of codes at a time.
//
// A model holds its working memory and is not thread-safe.
#ifndef LITTLEBIT_NCM_H
#define LITTLEBIT_NCM_H

#include "littlebit.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct littlebit_ncm littlebit_ncm;

typedef struct {
    int32_t n_threads; // 0: littlebit_default_threads()
    bool    repack;    // let the CPU backend copy weights into a faster layout, where it has one (default true)
} littlebit_ncm_params;

typedef struct {
    float    temperature; // < 0: the model's own (default), 0: greedy. Unused by models with soft decisions
    int32_t  top_k;       // <= 0: disabled (default)
    float    top_p;       // >= 1: disabled (default)
    uint32_t seed;        // 0xFFFFFFFF: random (default)
} littlebit_ncm_gen_params;

typedef struct {
    int32_t  n_prefix;    // input tokens, exactly this many
    int32_t  n_output;    // output tokens, always this many
    int32_t  n_vocab;
    int32_t  n_levels;
    int32_t  n_layers;
    int32_t  n_littlebit; // LittleBit linear layers
    int32_t  n_threads;
    bool     sudoku;
    uint64_t size_bytes;  // tensor data in the file
} littlebit_ncm_info;

// Logits of one level (level >= 0), then of the output tokens (level -1).
typedef void (*littlebit_ncm_trace_cb)(int32_t level, const float * logits, size_t n, void * user);

LITTLEBIT_API littlebit_ncm_params     littlebit_default_ncm_params(void);
LITTLEBIT_API littlebit_ncm_gen_params littlebit_default_ncm_gen_params(void);
LITTLEBIT_API littlebit_ncm * littlebit_ncm_load(const char * path, const littlebit_ncm_params * params);
LITTLEBIT_API void            littlebit_ncm_free(littlebit_ncm * model);
LITTLEBIT_API int             littlebit_ncm_get_info(const littlebit_ncm * model, littlebit_ncm_info * info);
LITTLEBIT_API littlebit_stats littlebit_ncm_stats(const littlebit_ncm * model);

// Returns the number of tokens written to `out`, or < 0 on error. `trace` may be NULL.
LITTLEBIT_API int32_t littlebit_ncm_generate(littlebit_ncm * model, const int32_t * prefix, int32_t n_prefix,
                                             const littlebit_ncm_gen_params * params, int32_t * out, int32_t n_out,
                                             littlebit_ncm_trace_cb trace, void * user);

// `puzzle`: 81 digits by rows, 0 or . for empty cells. `solution`: room for 82 characters; '?' where the model produced
// no digit. The result is the model's answer, which is not checked.
LITTLEBIT_API int littlebit_ncm_sudoku(littlebit_ncm * model, const char * puzzle, char * solution);

// One unconditional sample, from a model with a text tokenizer.
LITTLEBIT_API int littlebit_ncm_text(littlebit_ncm * model, const littlebit_ncm_gen_params * params, littlebit_text_cb cb, void * user);

#ifdef __cplusplus
}
#endif

#endif // LITTLEBIT_NCM_H
