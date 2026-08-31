(defsystem "llama-cpp"
  :version "0.1.4"
  :description "CFFI + native overlays for ggml-org/llama.cpp (linux/amd64 and windows/amd64 are CUDA+Vulkan)"
  :author "egao1980"
  :license "MIT"
  :depends-on ("cffi" "trivial-garbage")
  :serial t
  :pathname "src"
  :components ((:file "package")
               (:file "conditions")
               (:file "ffi")
               (:file "api"))
  :in-order-to ((test-op (test-op "llama-cpp/tests")))
  :properties
  (:cl-repo
   (:cffi-libraries ("libllamastack")
    :provides ("llama-cpp")
    :overlays
    ((:platform (:os "linux" :arch "amd64")
      :layers ((:role "native-library"
                :files (("lib/linux-amd64/libllamastack.so" . "libllamastack.so")
                        ("lib/linux-amd64/libllamastack.so.0" . "libllamastack.so.0")
                        ("lib/linux-amd64/libllama.so" . "libllama.so")
                        ("lib/linux-amd64/libllama.so.0" . "libllama.so.0")
                        ("lib/linux-amd64/libggml.so" . "libggml.so")
                        ("lib/linux-amd64/libggml.so.0" . "libggml.so.0")
                        ("lib/linux-amd64/libggml-base.so" . "libggml-base.so")
                        ("lib/linux-amd64/libggml-base.so.0" . "libggml-base.so.0")
                        ("lib/linux-amd64/libggml-cpu.so" . "libggml-cpu.so")
                        ("lib/linux-amd64/libggml-cpu.so.0" . "libggml-cpu.so.0")
                        ("lib/linux-amd64/libggml-cuda.so" . "libggml-cuda.so")
                        ("lib/linux-amd64/libggml-cuda.so.0" . "libggml-cuda.so.0")
                        ("lib/linux-amd64/libggml-vulkan.so" . "libggml-vulkan.so")
                        ("lib/linux-amd64/libggml-vulkan.so.0" . "libggml-vulkan.so.0")
                        ("lib/linux-amd64/libcudart.so.12" . "libcudart.so.12")
                        ("lib/linux-amd64/libcublas.so.12" . "libcublas.so.12")
                        ("lib/linux-amd64/libcublasLt.so.12" . "libcublasLt.so.12")))))
     (:platform (:os "linux" :arch "arm64")
      :layers ((:role "native-library"
                :files (("lib/linux-arm64/libllamastack.so" . "libllamastack.so")
                        ("lib/linux-arm64/libllamastack.so.0" . "libllamastack.so.0")
                        ("lib/linux-arm64/libllama.so" . "libllama.so")
                        ("lib/linux-arm64/libllama.so.0" . "libllama.so.0")
                        ("lib/linux-arm64/libggml.so" . "libggml.so")
                        ("lib/linux-arm64/libggml.so.0" . "libggml.so.0")
                        ("lib/linux-arm64/libggml-base.so" . "libggml-base.so")
                        ("lib/linux-arm64/libggml-base.so.0" . "libggml-base.so.0")
                        ("lib/linux-arm64/libggml-cpu.so" . "libggml-cpu.so")
                        ("lib/linux-arm64/libggml-cpu.so.0" . "libggml-cpu.so.0")))))
     (:platform (:os "darwin" :arch "arm64")
      :layers ((:role "native-library"
                :files (("lib/darwin-arm64/libllamastack.dylib" . "libllamastack.dylib")
                        ("lib/darwin-arm64/libllama.dylib" . "libllama.dylib")
                        ("lib/darwin-arm64/libllama.0.dylib" . "libllama.0.dylib")
                        ("lib/darwin-arm64/libggml.dylib" . "libggml.dylib")
                        ("lib/darwin-arm64/libggml.0.dylib" . "libggml.0.dylib")
                        ("lib/darwin-arm64/libggml-base.dylib" . "libggml-base.dylib")
                        ("lib/darwin-arm64/libggml-base.0.dylib" . "libggml-base.0.dylib")
                        ("lib/darwin-arm64/libggml-cpu.dylib" . "libggml-cpu.dylib")
                        ("lib/darwin-arm64/libggml-cpu.0.dylib" . "libggml-cpu.0.dylib")
                        ("lib/darwin-arm64/libggml-metal.dylib" . "libggml-metal.dylib")
                        ("lib/darwin-arm64/libggml-metal.0.dylib" . "libggml-metal.0.dylib")
                        ("lib/darwin-arm64/libggml-blas.dylib" . "libggml-blas.dylib")
                        ("lib/darwin-arm64/libggml-blas.0.dylib" . "libggml-blas.0.dylib")))))
     (:platform (:os "windows" :arch "amd64")
      :layers ((:role "native-library"
                :files (("lib/windows-amd64/llamastack.dll" . "llamastack.dll")
                        ("lib/windows-amd64/llama.dll" . "llama.dll")
                        ("lib/windows-amd64/ggml.dll" . "ggml.dll")
                        ("lib/windows-amd64/ggml-base.dll" . "ggml-base.dll")
                        ("lib/windows-amd64/ggml-cpu.dll" . "ggml-cpu.dll")
                        ("lib/windows-amd64/ggml-cuda.dll" . "ggml-cuda.dll")
                        ("lib/windows-amd64/ggml-vulkan.dll" . "ggml-vulkan.dll")
                        ("lib/windows-amd64/cudart64_12.dll" . "cudart64_12.dll")
                        ("lib/windows-amd64/cublas64_12.dll" . "cublas64_12.dll")
                        ("lib/windows-amd64/cublasLt64_12.dll" . "cublasLt64_12.dll")))))))))

(defsystem "llama-cpp/tests"
  :depends-on ("llama-cpp" "rove")
  :pathname "tests"
  :serial t
  :components ((:file "package")
               (:file "api-test"))
  :perform (test-op (o c)
             (unless (symbol-call :rove :run c)
               (error "tests failed for ~A" (component-name c)))))
