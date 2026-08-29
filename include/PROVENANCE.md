# include/llama-stack.h

Our ABI (`LLAMA_STACK_ABI_VERSION`). Implemented in `native/llama-stack.c`
against [`ggml-org/llama.cpp`](https://github.com/ggml-org/llama.cpp) `llama.h`.

| Field | Value |
|-------|-------|
| Upstream | https://github.com/ggml-org/llama.cpp |
| SPDX | MIT |
| Pin | `LLAMA_CPP_REF` in `scripts/build-llama.sh` |

Do not CFFI-bind `llama.h` structs — they change. Bump `+llama-stack-abi-version+`
together with this header.
