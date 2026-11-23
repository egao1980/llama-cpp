# llama-cpp

CFFI + native overlays for [`ggml-org/llama.cpp`](https://github.com/ggml-org/llama.cpp). Not the LLM protocol.

Lisp binds **`include/llama-stack.h`** (`libllamastack`), a small stable ABI. `llama.h` structs are not imported — they churn. Overlay also stages `libllama` + `libggml*`.

| Overlay | Backends |
|---|---|
| `linux/amd64` | **CUDA + Vulkan** (`-DGGML_CUDA=ON -DGGML_VULKAN=ON`, archs `80;86;89;90a`) |
| `linux/arm64` | CPU |
| `darwin/arm64` | Metal + Accelerate BLAS. One dylib per library (`@rpath` / `@loader_path`); no `lib*.<SONAME>.dylib` twin. |
| `windows/amd64` | **CUDA + Vulkan** |

`arrange-native-artifacts` keys **os-arch only**, so the published linux/amd64 and windows/amd64 overlays **are** the GPU builds. CPU (or cuda-only / vulkan-only) on those pairs is local: `LLAMA_CPP_FLAVOR=cpu|cuda|vulkan` → `lib/<os>-<arch>-<flavor>/`.

llama.cpp is independent of `vllm-cpp`. Windows CUDA is a first-class overlay here.

User-mode CUDA (`libcudart` / `libcublas` / `libcublasLt`, or `cudart64_12.dll` / `cublas64_12.dll` / `cublasLt64_12.dll`) is staged next to the engine with `$ORIGIN` / same-dir load. The NVIDIA **driver** (`libcuda` / `nvcuda.dll`) and the Vulkan **ICD** are never shipped — the host must provide them. No `LD_LIBRARY_PATH`.

```
scripts/build-llama.sh          # Unix; default linux/amd64 flavor=gpu
scripts/build-llama.ps1         # Windows; default flavor=gpu
# LLAMA_CPP_FLAVOR=cpu|cuda|vulkan|gpu
# LLAMA_CPP_SKIP_CMAKE=1        # restage + relink shim only
# or: LLAMA_CPP_NATIVE=/path/with/libllamastack.dylib
```

OCI: tag `v*` or `gh workflow run publish-oci.yml`. Packager takes the artifact dir; `.asd` `:cl-repo` overlays are the inventory.

SBCL masks float traps around FFI — ggml/Metal/CUDA inexact/denormals otherwise become `FLOATING-POINT-OVERFLOW`.

```lisp
(asdf:load-system "llama-cpp")
(llama-cpp:llama-available-p)
(let ((e (llama-cpp:load-engine :model-path "/models/bge.gguf" :n-ctx 512)))
  (unwind-protect
       (llama-cpp:embed e "hello")
    (llama-cpp:free-engine e)))
```

`embed` is the GGUF path that matches LM Studio (llama.cpp archs: `bert`, `qwen3`, …). `complete` is greedy / temperature sampling. Optional `:grammar` / `:grammar-root` (GBNF, ABI 2 `llama_stack_complete_ex`) goes first on the sampler chain. `parse-grammar` (ABI 4) compiles GBNF once via `llama_sampler_init_grammar`; pass the handle as `:parsed` — complete clones it. `:on-token` (ABI 3 `llama_stack_complete_stream`) is called with each detok delta; a non-NIL return stops. `n-gpu-layers` -1 = all (CUDA or Vulkan device, whichever ggml picks).

Grammar-constrained `complete` ends when the grammar is complete (llama.cpp only allows end-of-generation then), so the text is a sentence of the grammar unless `:max-tokens` cuts it short. `libllamastack` is C++ behind the C ABI: a C++ exception from llama.cpp (the grammar sampler throws `std::runtime_error`) becomes status `:runtime` and a `llama-error` that carries it (`C++ exception: …`), never an abort of the Lisp image.

Free engines with `free-engine` (grammars with `free-grammar`). A finalizer frees an engine that becomes garbage, and `free-live-handles` frees whatever is still live, grammars first; it runs from `sb-ext:*exit-hooks*` (CCL: `*lisp-cleanup-functions*`) and before an image dump, so exiting with an engine in a global ends with the exit code, not a Metal `GGML_ASSERT` in C `atexit`.

Tests with a model: `LLAMA_CPP_TEST_MODEL=/path/stories260K.gguf LLAMA_CPP_NATIVE=$PWD/lib/<os>-<arch> cl-repo test` (the grammar and exit tests run in a child SBCL; without the variable they skip).

llm-protocol lives in [`llm-backend-llama-cpp`](https://github.com/egao1980/llm-backend-llama-cpp).

## License

Lisp / this repo: MIT. Overlay `libllama` / `libggml*`: MIT ([NOTICE](NOTICE)). Staged CUDA user-mode runtime: NVIDIA CUDA EULA.
