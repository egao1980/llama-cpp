# llama-cpp

CFFI + native overlays for [`ggml-org/llama.cpp`](https://github.com/ggml-org/llama.cpp). Not the LLM protocol.

Lisp binds **`include/llama-stack.h`** (`libllamastack`), a small stable ABI. `llama.h` structs are not imported — they churn. Overlay also stages `libllama` + `libggml*`.

| Overlay | Backends |
|---|---|
| `linux/amd64` | **CUDA + Vulkan** (`-DGGML_CUDA=ON -DGGML_VULKAN=ON`, archs `80;86;89;90a`) |
| `linux/arm64` | CPU |
| `darwin/arm64` | Metal + Accelerate BLAS |
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

`embed` is the GGUF path that matches LM Studio (llama.cpp archs: `bert`, `qwen3`, …). `complete` is greedy / temperature sampling. Optional `:grammar` / `:grammar-root` (GBNF, ABI 2 `llama_stack_complete_ex`) goes first on the sampler chain. `:on-token` (ABI 3 `llama_stack_complete_stream`) is called with each detok delta; a non-NIL return stops. `n-gpu-layers` -1 = all (CUDA or Vulkan device, whichever ggml picks).

llm-protocol lives in [`llm-backend-llama-cpp`](https://github.com/egao1980/llm-backend-llama-cpp).

## License

Lisp / this repo: MIT. Overlay `libllama` / `libggml*`: MIT ([NOTICE](NOTICE)). Staged CUDA user-mode runtime: NVIDIA CUDA EULA.
