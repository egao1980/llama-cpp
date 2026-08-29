#include "llama-stack.h"

#include "llama.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define ERR_MAX 1024

static char g_last_error[ERR_MAX] = "";
static int g_backend_ready = 0;

struct llama_stack_engine {
    struct llama_model *model;
    struct llama_context *ctx;
    int32_t n_ctx;
};

static void set_error(const char *msg) {
    snprintf(g_last_error, ERR_MAX, "%s", msg ? msg : "unknown error");
}

static void set_errorf(const char *fmt, int code) {
    snprintf(g_last_error, ERR_MAX, fmt, code);
}

static void ensure_backend(void) {
    if (!g_backend_ready) {
        llama_backend_init();
        g_backend_ready = 1;
    }
}

int32_t llama_stack_abi_version(void) {
    return LLAMA_STACK_ABI_VERSION;
}

const char *llama_stack_version(void) {
    const char *v = llama_version();
    return v ? v : "unknown";
}

const char *llama_stack_last_error(void) {
    return g_last_error;
}

static struct llama_context *make_ctx(struct llama_model *model, int32_t n_ctx,
                                      int32_t n_threads, int embeddings) {
    struct llama_context_params cparams = llama_context_default_params();
    if (n_ctx > 0) {
        cparams.n_ctx = (uint32_t)n_ctx;
        cparams.n_batch = (uint32_t)n_ctx;
        /* Encoder / non-causal graphs require n_ubatch == n_batch. */
        cparams.n_ubatch = (uint32_t)n_ctx;
    }
    if (n_threads > 0) {
        cparams.n_threads = n_threads;
        cparams.n_threads_batch = n_threads;
    }
    cparams.embeddings = embeddings ? true : false;
    if (embeddings) {
        cparams.pooling_type = LLAMA_POOLING_TYPE_UNSPECIFIED;
        cparams.attention_type = LLAMA_ATTENTION_TYPE_UNSPECIFIED;
    }
    return llama_init_from_model(model, cparams);
}

llama_stack_status llama_stack_load(const char *model_path,
                                    const llama_stack_load_params *params,
                                    llama_stack_engine **out) {
    if (!model_path || !model_path[0] || !out) {
        set_error("model path and out are required");
        return LLAMA_STACK_INVALID;
    }
    *out = NULL;
    ensure_backend();
    llama_stack_load_params defaults;
    memset(&defaults, 0, sizeof(defaults));
    defaults.n_ctx = 2048;
    defaults.n_gpu_layers = -1;
    if (!params) {
        params = &defaults;
    }
    struct llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = params->n_gpu_layers ? params->n_gpu_layers : -1;
    struct llama_model *model = llama_model_load_from_file(model_path, mparams);
    if (!model) {
        set_error("llama_model_load_from_file failed");
        return LLAMA_STACK_MODEL_LOAD;
    }
    int32_t n_ctx = params->n_ctx > 0 ? params->n_ctx : 2048;
    struct llama_context *ctx = make_ctx(model, n_ctx, params->n_threads, 1);
    if (!ctx) {
        llama_model_free(model);
        set_error("llama_init_from_model failed");
        return LLAMA_STACK_MODEL_LOAD;
    }
    llama_stack_engine *engine = (llama_stack_engine *)calloc(1, sizeof(*engine));
    if (!engine) {
        llama_free(ctx);
        llama_model_free(model);
        set_error("out of memory");
        return LLAMA_STACK_RUNTIME;
    }
    engine->model = model;
    engine->ctx = ctx;
    engine->n_ctx = n_ctx;
    *out = engine;
    return LLAMA_STACK_OK;
}

void llama_stack_free(llama_stack_engine *engine) {
    if (!engine) {
        return;
    }
    if (engine->ctx) {
        llama_free(engine->ctx);
    }
    if (engine->model) {
        llama_model_free(engine->model);
    }
    free(engine);
}

static int32_t tokenize(const struct llama_vocab *vocab, const char *text,
                        llama_token **out_tokens) {
    int32_t n = llama_tokenize(vocab, text, (int32_t)strlen(text), NULL, 0, true, true);
    if (n == INT32_MIN) {
        return -1;
    }
    if (n < 0) {
        n = -n;
    }
    if (n <= 0) {
        return 0;
    }
    llama_token *tokens = (llama_token *)malloc((size_t)n * sizeof(llama_token));
    if (!tokens) {
        return -1;
    }
    int32_t got = llama_tokenize(vocab, text, (int32_t)strlen(text), tokens, n, true, true);
    if (got < 0) {
        free(tokens);
        return -1;
    }
    *out_tokens = tokens;
    return got;
}

static llama_stack_status embed_one(llama_stack_engine *engine, const char *text,
                                    float *dest, int32_t dim) {
    const struct llama_vocab *vocab = llama_model_get_vocab(engine->model);
    llama_token *tokens = NULL;
    int32_t n = tokenize(vocab, text, &tokens);
    if (n < 0) {
        set_error("tokenize failed");
        return LLAMA_STACK_RUNTIME;
    }
    if (n == 0) {
        memset(dest, 0, (size_t)dim * sizeof(float));
        free(tokens);
        return LLAMA_STACK_OK;
    }
    if (n > engine->n_ctx) {
        free(tokens);
        set_error("prompt longer than n_ctx");
        return LLAMA_STACK_INVALID;
    }
    llama_memory_clear(llama_get_memory(engine->ctx), true);
    enum llama_pooling_type pooling = llama_pooling_type(engine->ctx);
    struct llama_batch batch = llama_batch_init(n, 0, 1);
    batch.n_tokens = n;
    for (int32_t i = 0; i < n; i++) {
        batch.token[i] = tokens[i];
        batch.pos[i] = i;
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i] = (pooling == LLAMA_POOLING_TYPE_NONE) ? (i == n - 1) : 1;
    }
    int32_t rc;
    if (llama_model_has_encoder(engine->model) && !llama_model_has_decoder(engine->model)) {
        rc = llama_encode(engine->ctx, batch);
    } else {
        rc = llama_decode(engine->ctx, batch);
    }
    llama_batch_free(batch);
    free(tokens);
    if (rc != 0) {
        set_errorf("llama_decode/encode failed (%d)", rc);
        return LLAMA_STACK_RUNTIME;
    }
    float *emb = llama_get_embeddings_seq(engine->ctx, 0);
    if (!emb) {
        emb = llama_get_embeddings_ith(engine->ctx, -1);
    }
    if (!emb) {
        set_error("no embedding output (not a pooling/embedding graph?)");
        return LLAMA_STACK_RUNTIME;
    }
    memcpy(dest, emb, (size_t)dim * sizeof(float));
    /* Match llama.cpp examples/embedding default: L2 (OpenAI / LM Studio). */
    {
        double ss = 0.0;
        int32_t i;
        for (i = 0; i < dim; i++) {
            ss += (double)dest[i] * (double)dest[i];
        }
        if (ss > 0.0) {
            float inv = (float)(1.0 / sqrt(ss));
            for (i = 0; i < dim; i++) {
                dest[i] *= inv;
            }
        }
    }
    return LLAMA_STACK_OK;
}

llama_stack_status llama_stack_embed(llama_stack_engine *engine,
                                     const char **texts,
                                     int32_t n_texts,
                                     llama_stack_embedding_result *out) {
    if (!engine || !engine->ctx || !texts || n_texts <= 0 || !out) {
        set_error("embed requires engine, texts, and out");
        return LLAMA_STACK_INVALID;
    }
    memset(out, 0, sizeof(*out));
    llama_set_embeddings(engine->ctx, true);
    int32_t dim = llama_model_n_embd_out(engine->model);
    if (dim <= 0) {
        dim = llama_model_n_embd(engine->model);
    }
    if (dim <= 0) {
        set_error("model n_embd is 0");
        return LLAMA_STACK_RUNTIME;
    }
    float *values = (float *)calloc((size_t)n_texts * (size_t)dim, sizeof(float));
    if (!values) {
        set_error("out of memory");
        return LLAMA_STACK_RUNTIME;
    }
    int32_t tokens = 0;
    for (int32_t i = 0; i < n_texts; i++) {
        const char *text = texts[i] ? texts[i] : "";
        const struct llama_vocab *vocab = llama_model_get_vocab(engine->model);
        llama_token *tmp = NULL;
        int32_t n = tokenize(vocab, text, &tmp);
        if (n > 0) {
            tokens += n;
        }
        free(tmp);
        llama_stack_status st = embed_one(engine, text, values + ((size_t)i * (size_t)dim), dim);
        if (st != LLAMA_STACK_OK) {
            free(values);
            return st;
        }
    }
    out->values = values;
    out->n_embeddings = n_texts;
    out->dim = dim;
    out->prompt_tokens = tokens;
    return LLAMA_STACK_OK;
}

void llama_stack_embedding_result_free(llama_stack_embedding_result *out) {
    if (!out) {
        return;
    }
    free(out->values);
    out->values = NULL;
    out->n_embeddings = 0;
    out->dim = 0;
    out->prompt_tokens = 0;
}

static char *detok(const struct llama_vocab *vocab, const llama_token *tokens, int32_t n) {
    if (n <= 0) {
        char *empty = (char *)malloc(1);
        if (empty) {
            empty[0] = 0;
        }
        return empty;
    }
    int32_t need = llama_detokenize(vocab, tokens, n, NULL, 0, false, false);
    if (need == INT32_MIN) {
        return NULL;
    }
    if (need < 0) {
        need = -need;
    }
    char *buf = (char *)malloc((size_t)need + 1);
    if (!buf) {
        return NULL;
    }
    int32_t got = llama_detokenize(vocab, tokens, n, buf, need, false, false);
    if (got < 0) {
        free(buf);
        return NULL;
    }
    buf[got] = 0;
    return buf;
}

llama_stack_status llama_stack_complete(llama_stack_engine *engine,
                                        const char *prompt,
                                        int32_t max_tokens,
                                        float temperature,
                                        char **out_text,
                                        int32_t *prompt_tokens,
                                        int32_t *completion_tokens) {
    if (!engine || !engine->ctx || !prompt || !out_text) {
        set_error("complete requires engine, prompt, and out_text");
        return LLAMA_STACK_INVALID;
    }
    *out_text = NULL;
    if (prompt_tokens) {
        *prompt_tokens = 0;
    }
    if (completion_tokens) {
        *completion_tokens = 0;
    }
    llama_set_embeddings(engine->ctx, false);
    const struct llama_vocab *vocab = llama_model_get_vocab(engine->model);
    llama_token *prompt_toks = NULL;
    int32_t n_prompt = tokenize(vocab, prompt, &prompt_toks);
    if (n_prompt < 0) {
        set_error("tokenize failed");
        return LLAMA_STACK_RUNTIME;
    }
    if (n_prompt == 0) {
        set_error("empty prompt");
        free(prompt_toks);
        return LLAMA_STACK_INVALID;
    }
    if (n_prompt >= engine->n_ctx) {
        free(prompt_toks);
        set_error("prompt longer than n_ctx");
        return LLAMA_STACK_INVALID;
    }
    if (max_tokens <= 0) {
        max_tokens = 32;
    }
    llama_memory_clear(llama_get_memory(engine->ctx), true);
    struct llama_batch batch = llama_batch_init(engine->n_ctx, 0, 1);
    batch.n_tokens = n_prompt;
    for (int32_t i = 0; i < n_prompt; i++) {
        batch.token[i] = prompt_toks[i];
        batch.pos[i] = i;
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i] = (i == n_prompt - 1);
    }
    int32_t rc = llama_decode(engine->ctx, batch);
    if (rc != 0) {
        llama_batch_free(batch);
        free(prompt_toks);
        set_errorf("llama_decode failed (%d)", rc);
        return LLAMA_STACK_RUNTIME;
    }
    struct llama_sampler *smpl = llama_sampler_chain_init(llama_sampler_chain_default_params());
    if (temperature <= 0.0f) {
        llama_sampler_chain_add(smpl, llama_sampler_init_greedy());
    } else {
        llama_sampler_chain_add(smpl, llama_sampler_init_temp(temperature));
        llama_sampler_chain_add(smpl, llama_sampler_init_dist(LLAMA_DEFAULT_SEED));
    }
    llama_token *gen = (llama_token *)malloc((size_t)max_tokens * sizeof(llama_token));
    if (!gen) {
        llama_sampler_free(smpl);
        llama_batch_free(batch);
        free(prompt_toks);
        set_error("out of memory");
        return LLAMA_STACK_RUNTIME;
    }
    int32_t n_gen = 0;
    int32_t pos = n_prompt;
    for (int32_t i = 0; i < max_tokens; i++) {
        llama_token id = llama_sampler_sample(smpl, engine->ctx, -1);
        llama_sampler_accept(smpl, id);
        if (llama_vocab_is_eog(vocab, id)) {
            break;
        }
        gen[n_gen++] = id;
        batch.n_tokens = 1;
        batch.token[0] = id;
        batch.pos[0] = pos++;
        batch.n_seq_id[0] = 1;
        batch.seq_id[0][0] = 0;
        batch.logits[0] = 1;
        rc = llama_decode(engine->ctx, batch);
        if (rc != 0) {
            break;
        }
    }
    llama_sampler_free(smpl);
    llama_batch_free(batch);
    free(prompt_toks);
    char *text = detok(vocab, gen, n_gen);
    free(gen);
    if (!text) {
        set_error("detokenize failed");
        return LLAMA_STACK_RUNTIME;
    }
    *out_text = text;
    if (prompt_tokens) {
        *prompt_tokens = n_prompt;
    }
    if (completion_tokens) {
        *completion_tokens = n_gen;
    }
    return LLAMA_STACK_OK;
}

void llama_stack_string_free(char *s) {
    free(s);
}
