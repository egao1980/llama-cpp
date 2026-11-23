// libllamastack: the stable C ABI (include/llama-stack.h) over ggml-org/llama.cpp.
//
// C++ so that no C++ exception from llama.cpp (the grammar sampler throws
// std::runtime_error, allocations throw std::bad_alloc) unwinds into the
// caller's frames: a C caller (SBCL through CFFI) has no unwinder for it, so
// it ends in std::terminate -> abort() -> SIGABRT. Every exported function
// runs its body through guard(), which turns an exception into a status code
// and llama_stack_last_error(). llama-cpp#17.
//
// Resources are owned by RAII holders, so an exception (or an early error
// return) frees what was allocated so far.

#include "llama-stack.h"

#include "llama.h"

#include <climits>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <memory>
#include <new>
#include <stdexcept>
#include <vector>

struct llama_stack_engine {
    llama_model *model;
    llama_context *ctx;
    int32_t n_ctx;
};

struct llama_stack_grammar {
    llama_sampler *smpl;
};

namespace {

constexpr size_t ERR_MAX = 1024;

// Per thread: a status and its message are read by the thread that made the call.
thread_local char g_last_error[ERR_MAX] = "";
bool g_backend_ready = false;

void set_error(const char *msg) {
    std::snprintf(g_last_error, ERR_MAX, "%s", msg ? msg : "unknown error");
}

void set_errorf(const char *fmt, int code) {
    std::snprintf(g_last_error, ERR_MAX, fmt, code);
}

// Run BODY; an escaping C++ exception becomes LLAMA_STACK_RUNTIME.
template <typename F>
llama_stack_status guard(F &&body) noexcept {
    try {
        return body();
    } catch (const std::bad_alloc &) {
        std::snprintf(g_last_error, ERR_MAX, "out of memory");
    } catch (const std::exception &e) {
        std::snprintf(g_last_error, ERR_MAX, "C++ exception: %s", e.what());
    } catch (...) {
        std::snprintf(g_last_error, ERR_MAX, "unknown C++ exception");
    }
    return LLAMA_STACK_RUNTIME;
}

// Same for functions with no status: the error is only recorded.
template <typename F>
void guard_void(F &&body) noexcept {
    (void)guard([&]() {
        body();
        return LLAMA_STACK_OK;
    });
}

struct sampler_deleter {
    void operator()(llama_sampler *s) const { llama_sampler_free(s); }
};
using sampler_ptr = std::unique_ptr<llama_sampler, sampler_deleter>;

struct batch_holder {
    llama_batch batch;
    explicit batch_holder(int32_t n) : batch(llama_batch_init(n, 0, 1)) {}
    ~batch_holder() { llama_batch_free(batch); }
    batch_holder(const batch_holder &) = delete;
    batch_holder &operator=(const batch_holder &) = delete;
};

// A malloc'd C string handed to the caller (freed by llama_stack_string_free).
struct c_string_deleter {
    void operator()(char *s) const { std::free(s); }
};
using c_string = std::unique_ptr<char, c_string_deleter>;

void ensure_backend() {
    if (!g_backend_ready) {
        llama_backend_init();
        g_backend_ready = true;
    }
}

llama_context *make_ctx(llama_model *model, int32_t n_ctx, int32_t n_threads,
                        bool embeddings) {
    llama_context_params cparams = llama_context_default_params();
    if (n_ctx > 0) {
        cparams.n_ctx = (uint32_t)n_ctx;
        cparams.n_batch = (uint32_t)n_ctx;
        // Encoder / non-causal graphs require n_ubatch == n_batch.
        cparams.n_ubatch = (uint32_t)n_ctx;
    }
    if (n_threads > 0) {
        cparams.n_threads = n_threads;
        cparams.n_threads_batch = n_threads;
    }
    cparams.embeddings = embeddings;
    if (embeddings) {
        cparams.pooling_type = LLAMA_POOLING_TYPE_UNSPECIFIED;
        cparams.attention_type = LLAMA_ATTENTION_TYPE_UNSPECIFIED;
    }
    return llama_init_from_model(model, cparams);
}

// Tokens of TEXT, or false when tokenization fails.
bool tokenize(const llama_vocab *vocab, const char *text, std::vector<llama_token> &out) {
    const int32_t len = (int32_t)std::strlen(text);
    int32_t n = llama_tokenize(vocab, text, len, nullptr, 0, true, true);
    if (n == INT32_MIN) {
        return false;
    }
    if (n < 0) {
        n = -n;
    }
    out.assign((size_t)n, 0);
    if (n == 0) {
        return true;
    }
    const int32_t got = llama_tokenize(vocab, text, len, out.data(), n, true, true);
    if (got < 0) {
        return false;
    }
    out.resize((size_t)got);
    return true;
}

// Text of TOKENS as a malloc'd string, or null when detokenization fails.
c_string detok(const llama_vocab *vocab, const std::vector<llama_token> &tokens) {
    const int32_t n = (int32_t)tokens.size();
    if (n <= 0) {
        c_string empty((char *)std::malloc(1));
        if (!empty) {
            throw std::bad_alloc();
        }
        empty.get()[0] = 0;
        return empty;
    }
    int32_t need = llama_detokenize(vocab, tokens.data(), n, nullptr, 0, false, false);
    if (need == INT32_MIN) {
        return nullptr;
    }
    if (need < 0) {
        need = -need;
    }
    c_string buf((char *)std::malloc((size_t)need + 1));
    if (!buf) {
        throw std::bad_alloc();
    }
    const int32_t got = llama_detokenize(vocab, tokens.data(), n, buf.get(), need, false, false);
    if (got < 0) {
        return nullptr;
    }
    buf.get()[got] = 0;
    return buf;
}

llama_stack_status embed_one(llama_stack_engine *engine, const char *text, float *dest,
                             int32_t dim) {
    const llama_vocab *vocab = llama_model_get_vocab(engine->model);
    std::vector<llama_token> tokens;
    if (!tokenize(vocab, text, tokens)) {
        set_error("tokenize failed");
        return LLAMA_STACK_RUNTIME;
    }
    const int32_t n = (int32_t)tokens.size();
    if (n == 0) {
        std::memset(dest, 0, (size_t)dim * sizeof(float));
        return LLAMA_STACK_OK;
    }
    if (n > engine->n_ctx) {
        set_error("prompt longer than n_ctx");
        return LLAMA_STACK_INVALID;
    }
    llama_memory_clear(llama_get_memory(engine->ctx), true);
    const enum llama_pooling_type pooling = llama_pooling_type(engine->ctx);
    batch_holder b(n);
    b.batch.n_tokens = n;
    for (int32_t i = 0; i < n; i++) {
        b.batch.token[i] = tokens[(size_t)i];
        b.batch.pos[i] = i;
        b.batch.n_seq_id[i] = 1;
        b.batch.seq_id[i][0] = 0;
        b.batch.logits[i] = (pooling == LLAMA_POOLING_TYPE_NONE) ? (i == n - 1) : 1;
    }
    int32_t rc;
    if (llama_model_has_encoder(engine->model) && !llama_model_has_decoder(engine->model)) {
        rc = llama_encode(engine->ctx, b.batch);
    } else {
        rc = llama_decode(engine->ctx, b.batch);
    }
    if (rc != 0) {
        set_errorf("llama_decode/encode failed (%d)", rc);
        return LLAMA_STACK_RUNTIME;
    }
    const float *emb = llama_get_embeddings_seq(engine->ctx, 0);
    if (!emb) {
        emb = llama_get_embeddings_ith(engine->ctx, -1);
    }
    if (!emb) {
        set_error("no embedding output (not a pooling/embedding graph?)");
        return LLAMA_STACK_RUNTIME;
    }
    std::memcpy(dest, emb, (size_t)dim * sizeof(float));
    // Match llama.cpp examples/embedding default: L2 (OpenAI / LM Studio).
    double ss = 0.0;
    for (int32_t i = 0; i < dim; i++) {
        ss += (double)dest[i] * (double)dest[i];
    }
    if (ss > 0.0) {
        const float inv = (float)(1.0 / std::sqrt(ss));
        for (int32_t i = 0; i < dim; i++) {
            dest[i] *= inv;
        }
    }
    return LLAMA_STACK_OK;
}

llama_stack_status load_impl(const char *model_path, const llama_stack_load_params *params,
                             llama_stack_engine **out) {
    if (!model_path || !model_path[0] || !out) {
        set_error("model path and out are required");
        return LLAMA_STACK_INVALID;
    }
    *out = nullptr;
    ensure_backend();
    llama_stack_load_params defaults;
    std::memset(&defaults, 0, sizeof(defaults));
    defaults.n_ctx = 2048;
    defaults.n_gpu_layers = -1;
    if (!params) {
        params = &defaults;
    }
    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = params->n_gpu_layers ? params->n_gpu_layers : -1;
    llama_model *model = llama_model_load_from_file(model_path, mparams);
    if (!model) {
        set_error("llama_model_load_from_file failed");
        return LLAMA_STACK_MODEL_LOAD;
    }
    struct model_holder {
        llama_model *m;
        ~model_holder() {
            if (m) {
                llama_model_free(m);
            }
        }
    } model_guard{model};
    const int32_t n_ctx = params->n_ctx > 0 ? params->n_ctx : 2048;
    llama_context *ctx = make_ctx(model, n_ctx, params->n_threads, true);
    if (!ctx) {
        set_error("llama_init_from_model failed");
        return LLAMA_STACK_MODEL_LOAD;
    }
    llama_stack_engine *engine = (llama_stack_engine *)std::calloc(1, sizeof(*engine));
    if (!engine) {
        llama_free(ctx);
        set_error("out of memory");
        return LLAMA_STACK_RUNTIME;
    }
    model_guard.m = nullptr;
    engine->model = model;
    engine->ctx = ctx;
    engine->n_ctx = n_ctx;
    *out = engine;
    return LLAMA_STACK_OK;
}

llama_stack_status embed_impl(llama_stack_engine *engine, const char **texts, int32_t n_texts,
                              llama_stack_embedding_result *out) {
    if (!engine || !engine->ctx || !texts || n_texts <= 0 || !out) {
        set_error("embed requires engine, texts, and out");
        return LLAMA_STACK_INVALID;
    }
    std::memset(out, 0, sizeof(*out));
    llama_set_embeddings(engine->ctx, true);
    int32_t dim = llama_model_n_embd_out(engine->model);
    if (dim <= 0) {
        dim = llama_model_n_embd(engine->model);
    }
    if (dim <= 0) {
        set_error("model n_embd is 0");
        return LLAMA_STACK_RUNTIME;
    }
    std::unique_ptr<float, void (*)(void *)> values(
        (float *)std::calloc((size_t)n_texts * (size_t)dim, sizeof(float)), std::free);
    if (!values) {
        set_error("out of memory");
        return LLAMA_STACK_RUNTIME;
    }
    const llama_vocab *vocab = llama_model_get_vocab(engine->model);
    int32_t tokens = 0;
    std::vector<llama_token> tmp;
    for (int32_t i = 0; i < n_texts; i++) {
        const char *text = texts[i] ? texts[i] : "";
        if (tokenize(vocab, text, tmp)) {
            tokens += (int32_t)tmp.size();
        }
        const llama_stack_status st =
            embed_one(engine, text, values.get() + ((size_t)i * (size_t)dim), dim);
        if (st != LLAMA_STACK_OK) {
            return st;
        }
    }
    out->values = values.release();
    out->n_embeddings = n_texts;
    out->dim = dim;
    out->prompt_tokens = tokens;
    return LLAMA_STACK_OK;
}

// LLAMA_STACK_RUNTIME's message for a grammar dead end after GEN.
void dead_end_error(const llama_vocab *vocab, const std::vector<llama_token> &gen) {
    c_string so_far = detok(vocab, gen);
    const char *text = so_far ? so_far.get() : "";
    // At most 200 bytes of it, cut at a UTF-8 character boundary.
    size_t n = std::strlen(text);
    if (n > 200) {
        n = 200;
        while (n > 0 && (((unsigned char)text[n]) & 0xC0) == 0x80) {
            n--;
        }
    }
    std::snprintf(g_last_error, ERR_MAX,
                  "grammar dead end: the grammar allows no token after %d generated token(s)"
                  " \"%.*s%s\". llama.cpp masks end-of-generation tokens until the grammar is"
                  " complete, so a grammar that names one (<|im_end|>, </s>) cannot finish;"
                  " end the grammar where the text ends",
                  (int)gen.size(), (int)n, text, n < std::strlen(text) ? "..." : "");
}

enum class sample_result { sampled, dead_end, failed };

// The next token from the logits of the last decode. GRMR (or null) goes
// first, CHAIN (greedy, or temp + dist) picks among what it leaves. Unlike
// llama_sampler_sample this accepts nothing, and it notices a grammar dead
// end: the grammar rules out every token. llama_sampler_sample would pick one
// anyway (greedy keeps candidate 0) and accepting it fails in llama.cpp,
// with std::runtime_error, or with GGML_ABORT (abort(), which no guard
// catches) when that token is end-of-generation. llama-cpp#17.
sample_result sample_next(llama_sampler *grmr, llama_sampler *chain, llama_context *ctx,
                               int32_t n_vocab, std::vector<llama_token_data> &cur,
                               llama_token *out) {
    const float *logits = llama_get_logits_ith(ctx, -1);
    if (!logits) {
        set_error("no logits for the last token");
        return sample_result::failed;
    }
    cur.resize((size_t)n_vocab);
    for (llama_token t = 0; t < n_vocab; t++) {
        cur[(size_t)t] = llama_token_data{t, logits[t], 0.0f};
    }
    llama_token_data_array cur_p = {cur.data(), cur.size(), -1, false};
    if (grmr) {
        // The grammar marks the tokens it rules out with -inf (end-of-generation
        // tokens too, until the grammar is complete). Only the rest go on to
        // the chain, so whatever it picks is a token the grammar accepts.
        llama_sampler_apply(grmr, &cur_p);
        size_t kept = 0;
        for (size_t i = 0; i < cur_p.size; i++) {
            if (cur_p.data[i].logit > -INFINITY) {  // false for -inf and NaN
                cur_p.data[kept++] = cur_p.data[i];
            }
        }
        if (kept == 0) {
            return sample_result::dead_end;
        }
        cur_p.size = kept;
    }
    llama_sampler_apply(chain, &cur_p);
    if (cur_p.selected < 0 || cur_p.selected >= (int64_t)cur_p.size) {
        set_error("the sampler selected no token");
        return sample_result::failed;
    }
    *out = cur_p.data[cur_p.selected].id;
    return sample_result::sampled;
}

llama_stack_status complete_impl(llama_stack_engine *engine, const char *prompt,
                                 const llama_stack_complete_params *params,
                                 llama_stack_token_cb on_token, void *user, char **out_text,
                                 int32_t *prompt_tokens, int32_t *completion_tokens) {
    if (!engine || !engine->ctx || !prompt || !out_text) {
        set_error("complete requires engine, prompt, and out_text");
        return LLAMA_STACK_INVALID;
    }
    *out_text = nullptr;
    if (prompt_tokens) {
        *prompt_tokens = 0;
    }
    if (completion_tokens) {
        *completion_tokens = 0;
    }
    llama_stack_complete_params defaults;
    std::memset(&defaults, 0, sizeof(defaults));
    defaults.max_tokens = 32;
    if (!params) {
        params = &defaults;
    }
    const int32_t max_tokens = params->max_tokens > 0 ? params->max_tokens : 32;
    const float temperature = params->temperature;
    const char *grammar = (params->grammar && params->grammar[0]) ? params->grammar : nullptr;
    const char *grammar_root =
        (params->grammar_root && params->grammar_root[0]) ? params->grammar_root : "root";
    const llama_stack_grammar *parsed = params->parsed;
    llama_set_embeddings(engine->ctx, false);
    const llama_vocab *vocab = llama_model_get_vocab(engine->model);
    std::vector<llama_token> prompt_toks;
    if (!tokenize(vocab, prompt, prompt_toks)) {
        set_error("tokenize failed");
        return LLAMA_STACK_RUNTIME;
    }
    const int32_t n_prompt = (int32_t)prompt_toks.size();
    if (n_prompt == 0) {
        set_error("empty prompt");
        return LLAMA_STACK_INVALID;
    }
    if (n_prompt >= engine->n_ctx) {
        set_error("prompt longer than n_ctx");
        return LLAMA_STACK_INVALID;
    }
    // The grammar is its own sampler, applied before the chain (sample_next).
    // Parse it before decoding: a bad grammar costs no decode.
    sampler_ptr grmr;
    if (parsed && parsed->smpl) {
        grmr.reset(llama_sampler_clone(parsed->smpl));
        if (!grmr) {
            set_error("grammar clone failed");
            return LLAMA_STACK_RUNTIME;
        }
    } else if (grammar) {
        grmr.reset(llama_sampler_init_grammar(vocab, grammar, grammar_root));
        if (!grmr) {
            set_error("grammar parse failed");
            return LLAMA_STACK_INVALID;
        }
    }
    sampler_ptr smpl(llama_sampler_chain_init(llama_sampler_chain_default_params()));
    if (!smpl) {
        set_error("sampler chain init failed");
        return LLAMA_STACK_RUNTIME;
    }
    if (temperature <= 0.0f) {
        llama_sampler_chain_add(smpl.get(), llama_sampler_init_greedy());
    } else {
        llama_sampler_chain_add(smpl.get(), llama_sampler_init_temp(temperature));
        llama_sampler_chain_add(smpl.get(), llama_sampler_init_dist(LLAMA_DEFAULT_SEED));
    }
    llama_memory_clear(llama_get_memory(engine->ctx), true);
    batch_holder b(engine->n_ctx);
    b.batch.n_tokens = n_prompt;
    for (int32_t i = 0; i < n_prompt; i++) {
        b.batch.token[i] = prompt_toks[(size_t)i];
        b.batch.pos[i] = i;
        b.batch.n_seq_id[i] = 1;
        b.batch.seq_id[i][0] = 0;
        b.batch.logits[i] = (i == n_prompt - 1);
    }
    int32_t rc = llama_decode(engine->ctx, b.batch);
    if (rc != 0) {
        set_errorf("llama_decode failed (%d)", rc);
        return LLAMA_STACK_RUNTIME;
    }
    const int32_t n_vocab = llama_vocab_n_tokens(vocab);
    std::vector<llama_token_data> cur;
    std::vector<llama_token> gen;
    gen.reserve((size_t)max_tokens);
    int32_t pos = n_prompt;
    size_t emitted = 0;
    for (int32_t i = 0; i < max_tokens && pos < engine->n_ctx; i++) {
        llama_token id = LLAMA_TOKEN_NULL;
        switch (sample_next(grmr.get(), smpl.get(), engine->ctx, n_vocab, cur, &id)) {
        case sample_result::sampled:
            break;
        case sample_result::dead_end:
            dead_end_error(vocab, gen);
            return LLAMA_STACK_RUNTIME;
        case sample_result::failed:
            return LLAMA_STACK_RUNTIME;
        }
        // The grammar allows end-of-generation only once it is complete, so a
        // grammar-constrained completion stops here with a full sentence.
        // Accept once, on each sampler: the grammar advances by the piece.
        if (llama_vocab_is_eog(vocab, id)) {
            break;
        }
        if (grmr) {
            llama_sampler_accept(grmr.get(), id);
        }
        llama_sampler_accept(smpl.get(), id);
        gen.push_back(id);
        if (on_token) {
            c_string now = detok(vocab, gen);
            if (!now) {
                set_error("detokenize failed");
                return LLAMA_STACK_RUNTIME;
            }
            const size_t now_len = std::strlen(now.get());
            int32_t stop = 0;
            if (now_len > emitted) {
                stop = on_token(now.get() + emitted, user);
                emitted = now_len;
            }
            if (stop) {
                break;
            }
        }
        b.batch.n_tokens = 1;
        b.batch.token[0] = id;
        b.batch.pos[0] = pos++;
        b.batch.n_seq_id[0] = 1;
        b.batch.seq_id[0][0] = 0;
        b.batch.logits[0] = 1;
        rc = llama_decode(engine->ctx, b.batch);
        if (rc != 0) {
            break;
        }
    }
    c_string text = detok(vocab, gen);
    if (!text) {
        set_error("detokenize failed");
        return LLAMA_STACK_RUNTIME;
    }
    *out_text = text.release();
    if (prompt_tokens) {
        *prompt_tokens = n_prompt;
    }
    if (completion_tokens) {
        *completion_tokens = (int32_t)gen.size();
    }
    return LLAMA_STACK_OK;
}

llama_stack_status grammar_parse_impl(llama_stack_engine *engine, const char *grammar,
                                      const char *grammar_root, llama_stack_grammar **out) {
    if (!engine || !engine->model || !grammar || !grammar[0] || !out) {
        set_error("grammar parse requires engine, grammar, and out");
        return LLAMA_STACK_INVALID;
    }
    *out = nullptr;
    const char *root = (grammar_root && grammar_root[0]) ? grammar_root : "root";
    const llama_vocab *vocab = llama_model_get_vocab(engine->model);
    sampler_ptr smpl(llama_sampler_init_grammar(vocab, grammar, root));
    if (!smpl) {
        set_error("grammar parse failed");
        return LLAMA_STACK_INVALID;
    }
    llama_stack_grammar *g = (llama_stack_grammar *)std::calloc(1, sizeof(*g));
    if (!g) {
        set_error("out of memory");
        return LLAMA_STACK_RUNTIME;
    }
    g->smpl = smpl.release();
    *out = g;
    return LLAMA_STACK_OK;
}

}  // namespace

extern "C" {

int32_t llama_stack_abi_version(void) {
    return LLAMA_STACK_ABI_VERSION;
}

const char *llama_stack_version(void) {
    const char *v = "unknown";
    guard_void([&]() {
        const char *got = llama_version();
        if (got) {
            v = got;
        }
    });
    return v;
}

const char *llama_stack_last_error(void) {
    return g_last_error;
}

llama_stack_status llama_stack_load(const char *model_path, const llama_stack_load_params *params,
                                    llama_stack_engine **out) {
    return guard([&]() { return load_impl(model_path, params, out); });
}

void llama_stack_free(llama_stack_engine *engine) {
    if (!engine) {
        return;
    }
    guard_void([&]() {
        if (engine->ctx) {
            llama_free(engine->ctx);
            engine->ctx = nullptr;
        }
        if (engine->model) {
            llama_model_free(engine->model);
            engine->model = nullptr;
        }
    });
    std::free(engine);
}

llama_stack_status llama_stack_embed(llama_stack_engine *engine, const char **texts,
                                     int32_t n_texts, llama_stack_embedding_result *out) {
    return guard([&]() { return embed_impl(engine, texts, n_texts, out); });
}

void llama_stack_embedding_result_free(llama_stack_embedding_result *out) {
    if (!out) {
        return;
    }
    std::free(out->values);
    out->values = nullptr;
    out->n_embeddings = 0;
    out->dim = 0;
    out->prompt_tokens = 0;
}

llama_stack_status llama_stack_complete(llama_stack_engine *engine, const char *prompt,
                                        int32_t max_tokens, float temperature, char **out_text,
                                        int32_t *prompt_tokens, int32_t *completion_tokens) {
    llama_stack_complete_params params;
    std::memset(&params, 0, sizeof(params));
    params.max_tokens = max_tokens;
    params.temperature = temperature;
    return guard([&]() {
        return complete_impl(engine, prompt, &params, nullptr, nullptr, out_text, prompt_tokens,
                             completion_tokens);
    });
}

llama_stack_status llama_stack_complete_ex(llama_stack_engine *engine, const char *prompt,
                                           const llama_stack_complete_params *params,
                                           char **out_text, int32_t *prompt_tokens,
                                           int32_t *completion_tokens) {
    return guard([&]() {
        return complete_impl(engine, prompt, params, nullptr, nullptr, out_text, prompt_tokens,
                             completion_tokens);
    });
}

llama_stack_status llama_stack_complete_stream(llama_stack_engine *engine, const char *prompt,
                                               const llama_stack_complete_params *params,
                                               llama_stack_token_cb on_token, void *user,
                                               char **out_text, int32_t *prompt_tokens,
                                               int32_t *completion_tokens) {
    return guard([&]() {
        return complete_impl(engine, prompt, params, on_token, user, out_text, prompt_tokens,
                             completion_tokens);
    });
}

void llama_stack_string_free(char *s) {
    std::free(s);
}

llama_stack_status llama_stack_guard_selftest(int32_t kind) {
    struct not_a_std_exception {
        int code;
    };
    return guard([&]() {
        switch (kind) {
        case 1:
            throw std::runtime_error("guard selftest");
        case 2:
            throw std::bad_alloc();
        case 3:
            throw not_a_std_exception{3};
        default:
            set_error("");
            return LLAMA_STACK_OK;
        }
    });
}

llama_stack_status llama_stack_grammar_parse(llama_stack_engine *engine, const char *grammar,
                                             const char *grammar_root,
                                             llama_stack_grammar **out) {
    return guard([&]() { return grammar_parse_impl(engine, grammar, grammar_root, out); });
}

void llama_stack_grammar_free(llama_stack_grammar *grammar) {
    if (!grammar) {
        return;
    }
    guard_void([&]() {
        if (grammar->smpl) {
            llama_sampler_free(grammar->smpl);
            grammar->smpl = nullptr;
        }
    });
    std::free(grammar);
}

}  // extern "C"
