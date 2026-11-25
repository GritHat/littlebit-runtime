// littlebit: C API of the LittleBit runtime.
//
// A model is loaded once and can back several independent sessions. A session holds the KV cache and is
// not thread-safe; different sessions may be used from different threads. Functions that fail return
// NULL or a negative value, and littlebit_last_error() describes the failure on the calling thread.
#ifndef LITTLEBIT_H
#define LITTLEBIT_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#if defined(_WIN32) && defined(LITTLEBIT_SHARED)
#  ifdef LITTLEBIT_BUILD
#    define LITTLEBIT_API __declspec(dllexport)
#  else
#    define LITTLEBIT_API __declspec(dllimport)
#  endif
#elif defined(__GNUC__)
#  define LITTLEBIT_API __attribute__((visibility("default")))
#else
#  define LITTLEBIT_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef struct littlebit_model   littlebit_model;
typedef struct littlebit_session littlebit_session;

typedef enum {
    LITTLEBIT_KV_F16  = 0,
    LITTLEBIT_KV_Q8_0 = 1,
    LITTLEBIT_KV_F32  = 2,
} littlebit_kv_type;

typedef struct {
    int32_t n_gpu_layers; // layers to offload (default 0: CPU only, the only backend tested so far)
    bool    use_mmap;     // map the file instead of reading it (default true)
    bool    repack;       // let the CPU backend copy weights into a faster layout, where it has one (default true)
    bool    require_littlebit; // refuse ordinary GGUF files (default false)
} littlebit_model_params;

typedef struct {
    int32_t           n_ctx;     // context size in tokens (default 2048, capped to the training context)
    int32_t           n_batch;   // prompt batch size (default 256)
    int32_t           n_threads; // 0: littlebit_default_threads()
    littlebit_kv_type kv_type;   // default LITTLEBIT_KV_F16
} littlebit_session_params;

typedef struct {
    int32_t  max_tokens;  // < 0: until end of generation or context (default 256)
    float    temperature; // <= 0: greedy (default 0.7)
    int32_t  top_k;       // <= 0: disabled (default 40)
    float    top_p;       // >= 1: disabled (default 0.95)
    uint32_t seed;        // 0xFFFFFFFF: random (default)
} littlebit_gen_params;

typedef struct {
    char     description[128];
    char     arch[64];
    uint32_t format_version; // littlebit.version, 0 for ordinary GGUF files
    uint64_t size_bytes;     // tensor data in the file
    int32_t  n_vocab;
    int32_t  n_ctx_train;
    bool     has_chat_template;
} littlebit_model_info;

typedef struct {
    int32_t n_prompt;
    int32_t n_generated;
    double  prompt_ms;
    double  generate_ms;
} littlebit_stats;

typedef enum {
    LITTLEBIT_STOP_EOG     = 0, // end-of-generation token
    LITTLEBIT_STOP_LENGTH  = 1, // max_tokens reached
    LITTLEBIT_STOP_CONTEXT = 2, // context full
    LITTLEBIT_STOP_USER    = 3, // callback returned false
} littlebit_stop;

// Receives text (not NUL-terminated) without splitting UTF-8 sequences. Return false to stop.
typedef bool (*littlebit_text_cb)(const char * text, size_t len, void * user);

// level: 1 debug, 2 info, 3 warning, 4 error
typedef void (*littlebit_log_cb)(int level, const char * text, void * user);

LITTLEBIT_API const char * littlebit_version(void);
LITTLEBIT_API const char * littlebit_last_error(void);
LITTLEBIT_API int32_t      littlebit_default_threads(void);

// Default: warnings and errors go to stderr. NULL silences the library. Process-wide.
LITTLEBIT_API void littlebit_set_log(littlebit_log_cb cb, void * user);

LITTLEBIT_API littlebit_model_params   littlebit_default_model_params(void);
LITTLEBIT_API littlebit_session_params littlebit_default_session_params(void);
LITTLEBIT_API littlebit_gen_params     littlebit_default_gen_params(void);

LITTLEBIT_API littlebit_model * littlebit_model_load(const char * path, const littlebit_model_params * params);
LITTLEBIT_API void              littlebit_model_free(littlebit_model * model);
LITTLEBIT_API int               littlebit_model_get_info(const littlebit_model * model, littlebit_model_info * info);

// Number of tokens in `text` (without BOS), or < 0 on error.
LITTLEBIT_API int32_t littlebit_count_tokens(const littlebit_model * model, const char * text);

// Sessions must be freed before their model.
LITTLEBIT_API littlebit_session * littlebit_session_new(littlebit_model * model, const littlebit_session_params * params);
LITTLEBIT_API void                littlebit_session_free(littlebit_session * session);
LITTLEBIT_API void                littlebit_session_reset(littlebit_session * session);
LITTLEBIT_API int32_t             littlebit_session_n_past(const littlebit_session * session);

// Append `prompt` to the session and generate. Returns a littlebit_stop, or < 0 on error.
LITTLEBIT_API int littlebit_generate(littlebit_session * session, const char * prompt, const littlebit_gen_params * params,
                                     littlebit_text_cb cb, void * user);

// Append one user turn using the model's chat template and generate the reply. `system` is only used on the
// first turn and may be NULL. Template support is llama_chat_apply_template() (no Jinja interpreter).
LITTLEBIT_API int littlebit_chat(littlebit_session * session, const char * system, const char * message,
                                 const littlebit_gen_params * params, littlebit_text_cb cb, void * user);

// Timings of the last littlebit_generate() / littlebit_chat() call.
LITTLEBIT_API littlebit_stats littlebit_session_stats(const littlebit_session * session);

// Evaluate raw token ids in a fresh context; writes n_tokens * n_vocab floats to `logits`. Resets the session.
LITTLEBIT_API int littlebit_eval(littlebit_session * session, const int32_t * tokens, int32_t n_tokens,
                                 float * logits, size_t n_logits);

// ---- Next-Concept Modeling (NCM) models: whole levels of codes at a time instead of one token after another.
// A model holds its working memory and is not thread-safe.

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

#endif // LITTLEBIT_H
