# llama-cpp

CFFI + native overlays for [`ggml-org/llama.cpp`](https://github.com/ggml-org/llama.cpp). Not the LLM protocol.

Lisp binds **`include/llama-stack.h`** (`libllamastack`), a small stable ABI. `llama.h` structs are not imported — they churn. Overlay also stages `libllama` + `libggml*`.

```
scripts/build-llama.sh          # lib/<os>-<arch>/  (Unix; Windows → build-llama.ps1)
# LLAMA_CPP_SKIP_CMAKE=1        # restage + relink shim only
# or: LLAMA_CPP_NATIVE=/path/with/libllamastack.dylib
```

OCI: tag `v*` or `gh workflow run publish-oci.yml`. Overlays are linux/amd64+arm64 (CPU), darwin/arm64 (Metal), windows/amd64 (CPU). Packager takes the artifact dir; `.asd` `:cl-repo` overlays are the inventory.

SBCL masks float traps around FFI — ggml/Metal inexact/denormals otherwise become `FLOATING-POINT-OVERFLOW`.

```lisp
(asdf:load-system "llama-cpp")
(llama-cpp:llama-available-p)
(let ((e (llama-cpp:load-engine :model-path "/models/bge.gguf" :n-ctx 512)))
  (unwind-protect
       (llama-cpp:embed e "hello")
    (llama-cpp:free-engine e)))
```

`embed` is the GGUF path that matches LM Studio (llama.cpp archs: `bert`, `qwen3`, …). `complete` is greedy / temperature sampling.

llm-protocol lives in [`llm-backend-llama-cpp`](https://github.com/egao1980/llm-backend-llama-cpp).

## License

Lisp / this repo: MIT. Overlay `libllama` / `libggml*`: MIT ([NOTICE](NOTICE)).
