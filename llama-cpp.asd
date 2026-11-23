(defsystem "llama-cpp"
  :version "0.2.0"
  :description "CFFI + native overlays for ggml-org/llama.cpp (linux/amd64 and windows/amd64 are CUDA+Vulkan)"
  :author "egao1980"
  :license "MIT"
  :defsystem-depends-on ("asdf-native")
  :depends-on ("cffi" "trivial-garbage")
  :serial t
  :pathname "src"
  :components ((:file "package")
               (:file "conditions")
               ;; $LLAMA_CPP_NATIVE, lib/<os>-<arch>-$LLAMA_CPP_FLAVOR/, native/ (overlay),
               ;; lib/<os>-<arch>/ (checkout). Dependencies first, each by absolute path.
               ;; All optional: a checkout without natives still loads (LLAMA-AVAILABLE-P is NIL).
               ;; CUDA user-mode runtime (GPU overlays only). Never libcuda / nvcuda (driver).
               (:native-library "cudart" :versions ("12") :names (:windows ("cudart64_12.dll"))
                :env "LLAMA_CPP_NATIVE" :directories ("lib/{pair}-{env:LLAMA_CPP_FLAVOR}/") :optional t)
               (:native-library "cublasLt" :versions ("12") :names (:windows ("cublasLt64_12.dll"))
                :env "LLAMA_CPP_NATIVE" :directories ("lib/{pair}-{env:LLAMA_CPP_FLAVOR}/") :optional t)
               (:native-library "cublas" :versions ("12") :names (:windows ("cublas64_12.dll"))
                :env "LLAMA_CPP_NATIVE" :directories ("lib/{pair}-{env:LLAMA_CPP_FLAVOR}/") :optional t)
               ;; ggml core, then the backends this platform's overlay carries.
               (:native-library "ggml-base"
                :env "LLAMA_CPP_NATIVE" :directories ("lib/{pair}-{env:LLAMA_CPP_FLAVOR}/") :optional t)
               (:native-library "ggml-cpu"
                :env "LLAMA_CPP_NATIVE" :directories ("lib/{pair}-{env:LLAMA_CPP_FLAVOR}/") :optional t)
               (:native-library "ggml-cuda"
                :env "LLAMA_CPP_NATIVE" :directories ("lib/{pair}-{env:LLAMA_CPP_FLAVOR}/") :optional t)
               (:native-library "ggml-vulkan"
                :env "LLAMA_CPP_NATIVE" :directories ("lib/{pair}-{env:LLAMA_CPP_FLAVOR}/") :optional t)
               (:native-library "ggml-metal"
                :env "LLAMA_CPP_NATIVE" :directories ("lib/{pair}-{env:LLAMA_CPP_FLAVOR}/") :optional t)
               (:native-library "ggml-blas"
                :env "LLAMA_CPP_NATIVE" :directories ("lib/{pair}-{env:LLAMA_CPP_FLAVOR}/") :optional t)
               (:native-library "ggml"
                :env "LLAMA_CPP_NATIVE" :directories ("lib/{pair}-{env:LLAMA_CPP_FLAVOR}/") :optional t)
               (:native-library "llama"
                :env "LLAMA_CPP_NATIVE" :directories ("lib/{pair}-{env:LLAMA_CPP_FLAVOR}/") :optional t)
               (:native-library "llamastack"
                :env "LLAMA_CPP_NATIVE" :directories ("lib/{pair}-{env:LLAMA_CPP_FLAVOR}/")
                :system t :optional t)
               (:file "ffi")
               (:file "api"))
  :in-order-to ((test-op (test-op "llama-cpp/tests")))
  ;; Overlays: lib/<os>-<arch>/ per platform, from the :native-library components.
  :properties (:cl-repo (:provides ("llama-cpp"))))

(defsystem "llama-cpp/tests"
  :depends-on ("llama-cpp" "rove")
  :pathname "tests"
  :serial t
  :components ((:file "package")
               (:file "api-test")
               ;; llama-cpp#17: COMPLETE :grammar in a child SBCL (LLAMA_CPP_TEST_MODEL).
               (:file "grammar-test"))
  :perform (test-op (o c)
             (unless (symbol-call :rove :run c)
               (error "tests failed for ~A" (component-name c)))))
