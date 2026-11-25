// C API checks: test_api MODEL.gguf  (run by tests/test_runtime.py on a tiny model)
#include "littlebit.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(x) do { if (!(x)) { fprintf(stderr, "%s:%d: %s (%s)\n", __FILE__, __LINE__, #x, littlebit_last_error()); exit(1); } } while (0)

struct sink { char text[4096]; size_t len; int calls, stop_after; };

static bool collect(const char * t, size_t n, void * user) {
    struct sink * s = user;
    if (s->len + n < sizeof(s->text)) {
        memcpy(s->text + s->len, t, n);
        s->len += n;
    }
    return ++s->calls != s->stop_after;
}

int main(int argc, char ** argv) {
    CHECK(argc == 2);
    littlebit_set_log(NULL, NULL);
    CHECK(littlebit_default_threads() >= 1);

    CHECK(littlebit_model_load("/nonexistent.gguf", NULL) == NULL);
    CHECK(strlen(littlebit_last_error()) > 0);
    CHECK(littlebit_model_load(NULL, NULL) == NULL);

    littlebit_model * m = littlebit_model_load(argv[1], NULL);
    CHECK(m);
    littlebit_model_info info;
    CHECK(littlebit_model_get_info(m, &info) == 0);
    CHECK(info.format_version == 1 && info.n_vocab > 0);

    littlebit_session_params sp = littlebit_default_session_params();
    sp.n_ctx = 0;
    CHECK(littlebit_session_new(m, &sp) == NULL);
    sp.n_ctx = 64;
    sp.n_threads = 2;
    sp.n_batch = 8; // smaller than the prompt
    littlebit_session * s = littlebit_session_new(m, &sp);
    CHECK(s);

    littlebit_gen_params gp = littlebit_default_gen_params();
    gp.temperature = 0;
    gp.max_tokens = 10;
    struct sink a = {0}, b = {0}, c = {0};
    CHECK(littlebit_generate(s, "hello world", &gp, collect, &a) >= 0);
    littlebit_stats st = littlebit_session_stats(s);
    CHECK(st.n_prompt > 8 && st.n_generated <= 10);

    // greedy generation is reproducible after a reset, and equals the same text generated in two steps
    littlebit_session_reset(s);
    CHECK(littlebit_session_n_past(s) == 0);
    gp.max_tokens = 4;
    const int rc = littlebit_generate(s, "hello world", &gp, collect, &b);
    CHECK(rc >= 0);
    if (rc == LITTLEBIT_STOP_LENGTH) {
        gp.max_tokens = 6;
        CHECK(littlebit_generate(s, "", &gp, collect, &b) >= 0);
        CHECK(a.len == b.len && memcmp(a.text, b.text, a.len) == 0);
    }

    // callback can stop; context limit is reported, oversized prompts are refused
    littlebit_session_reset(s);
    gp.max_tokens = -1;
    c.stop_after = 1;
    const int user = littlebit_generate(s, "hello", &gp, collect, &c);
    CHECK(user == LITTLEBIT_STOP_USER || user == LITTLEBIT_STOP_EOG);
    littlebit_session_reset(s);
    const int full = littlebit_generate(s, "hello", &gp, NULL, NULL);
    CHECK(full == LITTLEBIT_STOP_CONTEXT || full == LITTLEBIT_STOP_EOG);
    CHECK(littlebit_session_n_past(s) <= 64);
    littlebit_session_reset(s);
    char big[512];
    memset(big, 'a', sizeof(big) - 1);
    big[sizeof(big) - 1] = 0;
    CHECK(littlebit_generate(s, big, &gp, NULL, NULL) < 0);
    CHECK(littlebit_generate(s, NULL, &gp, NULL, NULL) < 0);

    // chat: several turns, then one that does not fit leaves the session usable
    CHECK(info.has_chat_template);
    gp.max_tokens = 3;
    littlebit_session_reset(s);
    CHECK(littlebit_chat(s, NULL, "a", &gp, NULL, NULL) >= 0);
    CHECK(littlebit_chat(s, NULL, big, &gp, NULL, NULL) < 0);

    // raw evaluation
    const int32_t ids[3] = { 5, 6, 7 };
    const size_t n = 3 * (size_t) info.n_vocab;
    float * logits = malloc(n * sizeof(float));
    CHECK(littlebit_eval(s, ids, 3, logits, n) == 0);
    CHECK(littlebit_eval(s, ids, 3, logits, n - 1) < 0);
    const int32_t bad[1] = { info.n_vocab };
    CHECK(littlebit_eval(s, bad, 1, logits, n) < 0);
    free(logits);

    littlebit_session_free(s);
    littlebit_model_free(m);
    littlebit_session_free(NULL);
    littlebit_model_free(NULL);
    printf("ok\n");
    return 0;
}
