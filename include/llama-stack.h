/* Stable C ABI for the llama-cpp Lisp binding.
 * Implementation lives in native/llama-stack.c against ggml-org/llama.cpp.
 * Do not bind llama.h structs from Lisp — they churn. */
#ifndef LLAMA_STACK_H
#define LLAMA_STACK_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#ifndef LLAMA_STACK_API
#if defined(_WIN32)
#ifdef LLAMA_STACK_BUILD
#define LLAMA_STACK_API __declspec(dllexport)
#else
#define LLAMA_STACK_API __declspec(dllimport)
#endif
#else
#define LLAMA_STACK_API __attribute__((visibility("default")))
#endif
#endif

#define LLAMA_STACK_ABI_VERSION 4

typedef enum llama_stack_status {
    LLAMA_STACK_OK = 0,
    LLAMA_STACK_INVALID = 1,
    LLAMA_STACK_MODEL_LOAD = 2,
    LLAMA_STACK_RUNTIME = 3
} llama_stack_status;

typedef struct llama_stack_engine llama_stack_engine;
typedef struct llama_stack_grammar llama_stack_grammar;

typedef struct llama_stack_load_params {
    int32_t n_ctx;
    int32_t n_gpu_layers; /* -1 = all */
    int32_t n_threads;    /* 0 = default */
} llama_stack_load_params;

typedef struct llama_stack_embedding_result {
    float *values;
    int32_t n_embeddings;
    int32_t dim;
    int32_t prompt_tokens;
} llama_stack_embedding_result;

LLAMA_STACK_API int32_t llama_stack_abi_version(void);
LLAMA_STACK_API const char *llama_stack_version(void);
LLAMA_STACK_API const char *llama_stack_last_error(void);

LLAMA_STACK_API llama_stack_status llama_stack_load(const char *model_path,
                                                    const llama_stack_load_params *params,
                                                    llama_stack_engine **out);
LLAMA_STACK_API void llama_stack_free(llama_stack_engine *engine);

LLAMA_STACK_API llama_stack_status llama_stack_embed(llama_stack_engine *engine,
                                                     const char **texts,
                                                     int32_t n_texts,
                                                     llama_stack_embedding_result *out);
LLAMA_STACK_API void llama_stack_embedding_result_free(llama_stack_embedding_result *out);

typedef struct llama_stack_complete_params {
    int32_t max_tokens;
    float temperature;
    const char *grammar;      /* GBNF; NULL / empty = unconstrained */
    const char *grammar_root; /* NULL / empty = "root" */
    const llama_stack_grammar *parsed; /* ABI 4; wins over grammar string */
} llama_stack_complete_params;

/* Compile GBNF via llama_sampler_init_grammar. Engine must outlive the handle.
 * complete clones the sampler so one parse can be reused. */
LLAMA_STACK_API llama_stack_status llama_stack_grammar_parse(
    llama_stack_engine *engine,
    const char *grammar,
    const char *grammar_root,
    llama_stack_grammar **out);
LLAMA_STACK_API void llama_stack_grammar_free(llama_stack_grammar *grammar);

LLAMA_STACK_API llama_stack_status llama_stack_complete(llama_stack_engine *engine,
                                                        const char *prompt,
                                                        int32_t max_tokens,
                                                        float temperature,
                                                        char **out_text,
                                                        int32_t *prompt_tokens,
                                                        int32_t *completion_tokens);
LLAMA_STACK_API llama_stack_status llama_stack_complete_ex(llama_stack_engine *engine,
                                                           const char *prompt,
                                                           const llama_stack_complete_params *params,
                                                           char **out_text,
                                                           int32_t *prompt_tokens,
                                                           int32_t *completion_tokens);
/* on_token: each detok delta. Return 0 = continue, non-zero = stop (still OK).
 * NULL callback = same as complete_ex. */
typedef int32_t (*llama_stack_token_cb)(const char *piece, void *user);
LLAMA_STACK_API llama_stack_status llama_stack_complete_stream(
    llama_stack_engine *engine,
    const char *prompt,
    const llama_stack_complete_params *params,
    llama_stack_token_cb on_token,
    void *user,
    char **out_text,
    int32_t *prompt_tokens,
    int32_t *completion_tokens);
LLAMA_STACK_API void llama_stack_string_free(char *s);

#ifdef __cplusplus
}
#endif

#endif
