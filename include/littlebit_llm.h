// littlebit_llm: language models that generate one token after another (llama.cpp).
//
// A model is loaded once and can back several independent sessions. A session holds the KV cache and is not thread-safe;
// different sessions may be used from different threads.
#ifndef LITTLEBIT_LLM_H
#define LITTLEBIT_LLM_H

#include "littlebit.h"

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

typedef enum {
    LITTLEBIT_STOP_EOG     = 0, // end-of-generation token
    LITTLEBIT_STOP_LENGTH  = 1, // max_tokens reached
    LITTLEBIT_STOP_CONTEXT = 2, // context full
    LITTLEBIT_STOP_USER    = 3, // callback returned false
} littlebit_stop;

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

#ifdef __cplusplus
}
#endif

#endif // LITTLEBIT_LLM_H
