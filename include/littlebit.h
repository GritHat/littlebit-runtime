// littlebit: what the C APIs of both engines share. See littlebit_llm.h and littlebit_ncm.h.
//
// Functions that fail return NULL or a negative value, and littlebit_last_error() describes the failure on the calling
// thread.
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

typedef struct {
    int32_t n_prompt;
    int32_t n_generated;
    double  prompt_ms;
    double  generate_ms;
} littlebit_stats;

// Receives text (not NUL-terminated) without splitting UTF-8 sequences. Return false to stop.
typedef bool (*littlebit_text_cb)(const char * text, size_t len, void * user);

// level: 1 debug, 2 info, 3 warning, 4 error
typedef void (*littlebit_log_cb)(int level, const char * text, void * user);

LITTLEBIT_API const char * littlebit_version(void);
LITTLEBIT_API const char * littlebit_last_error(void);
LITTLEBIT_API int32_t      littlebit_default_threads(void);

// Default: warnings and errors go to stderr. NULL silences the library. Process-wide.
LITTLEBIT_API void littlebit_set_log(littlebit_log_cb cb, void * user);

#ifdef __cplusplus
}
#endif

#endif // LITTLEBIT_H
