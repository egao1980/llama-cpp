(defsystem "llama-cpp"
  :version "0.1.0"
  :description "CFFI + native overlays for ggml-org/llama.cpp (libllamastack)"
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
                :files (("lib/linux-amd64/libllamastack.so" . "libllamastack.so")))))
     (:platform (:os "linux" :arch "arm64")
      :layers ((:role "native-library"
                :files (("lib/linux-arm64/libllamastack.so" . "libllamastack.so")))))
     (:platform (:os "darwin" :arch "arm64")
      :layers ((:role "native-library"
                :files (("lib/darwin-arm64/libllamastack.dylib" . "libllamastack.dylib")))))))))

(defsystem "llama-cpp/tests"
  :depends-on ("llama-cpp" "rove")
  :pathname "tests"
  :serial t
  :components ((:file "package")
               (:file "api-test"))
  :perform (test-op (o c)
             (unless (symbol-call :rove :run c)
               (error "tests failed for ~A" (component-name c)))))
