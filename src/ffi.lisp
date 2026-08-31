(in-package #:llama-cpp)

(defconstant +llama-stack-abi-version+ 4)

(defcenum llama-stack-status
  (:ok 0)
  (:invalid 1)
  (:model-load 2)
  (:runtime 3))

(defcstruct llama-stack-load-params
  (n-ctx :int32)
  (n-gpu-layers :int32)
  (n-threads :int32))

(defcstruct llama-stack-embedding-result
  (values :pointer)
  (n-embeddings :int32)
  (dim :int32)
  (prompt-tokens :int32))

(defcstruct llama-stack-complete-params
  (max-tokens :int32)
  (temperature :float)
  (grammar :pointer)
  (grammar-root :pointer)
  (parsed :pointer))

(define-foreign-library libllamastack
  (:darwin (:or "libllamastack.dylib" "libllamastack.0.dylib"))
  (:unix (:or "libllamastack.so" "libllamastack.so.0"))
  (:windows (:or "llamastack.dll" "libllamastack.dll"))
  (t (:default "libllamastack")))

(defun %host-os ()
  #+windows "windows"
  #+darwin "darwin"
  #+linux "linux"
  #-(or windows darwin linux) "unknown")

(defun %host-arch ()
  #+(or x86-64 x64) "amd64"
  #+(or arm64 aarch64) "arm64"
  #-(or x86-64 x64 arm64 aarch64) "unknown")

(defun %flavor ()
  (let ((v (uiop:getenv "LLAMA_CPP_FLAVOR")))
    (when (and v (plusp (length v)))
      (string-downcase v))))

(defun %native-search-dirs ()
  "Overlay native/, lib/<os>-<arch>[-flavor]/. No LD_LIBRARY_PATH / DYLD_LIBRARY_PATH."
  (let ((dirs '())
        (flavor (%flavor)))
    (dolist (var '("LLAMA_CPP_NATIVE"))
      (let ((v (uiop:getenv var)))
        (when (and v (plusp (length v)))
          (push v dirs))))
    (ignore-errors
      (let* ((sys (asdf:find-system :llama-cpp nil))
             (root (when sys (asdf:system-source-directory sys))))
        (when root
          (push (namestring (merge-pathnames "native/" root)) dirs)
          (let ((base (format nil "lib/~A-~A" (%host-os) (%host-arch))))
            (when flavor
              (push (namestring (merge-pathnames (format nil "~A-~A/" base flavor) root))
                    dirs))
            (push (namestring (merge-pathnames (format nil "~A/" base) root)) dirs)))))
    (nreverse dirs)))

(defun %lib-candidates ()
  #+windows '("llamastack.dll" "libllamastack.dll")
  #+darwin '("libllamastack.dylib")
  #+(and unix (not darwin)) '("libllamastack.so")
  #-(or windows darwin unix) '("libllamastack.so"))

(defun %cuda-runtime-names (which)
  (ecase which
    (:cudart '("libcudart.so.12" "libcudart.so.13" "libcudart.so"
               "cudart64_12.dll" "cudart64_13.dll"))
    (:cublaslt '("libcublasLt.so.12" "libcublasLt.so.13" "libcublasLt.so"
                 "cublasLt64_12.dll" "cublasLt64_13.dll"))
    (:cublas '("libcublas.so.12" "libcublas.so.13" "libcublas.so"
               "cublas64_12.dll" "cublas64_13.dll"))))

(defun %preload-cuda-runtime (dir)
  "Absolute-preload CUDA user-mode deps. Never libcuda / nvcuda (driver)."
  (dolist (which '(:cudart :cublaslt :cublas))
    (let ((abs (%find-named dir (%cuda-runtime-names which))))
      (when abs
        (handler-case (load-foreign-library abs)
          (error (e)
            (warn "llama-cpp: failed to preload ~a (~a)" abs e))))))
  t)

(defun %ggml-preload-names ()
  #+darwin '("libggml-base.dylib" "libggml-cpu.dylib" "libggml-metal.dylib"
             "libggml-blas.dylib" "libggml.dylib" "libllama.dylib")
  #+(and unix (not darwin)) '("libggml-base.so" "libggml-cpu.so"
                              "libggml-cuda.so" "libggml-vulkan.so"
                              "libggml.so" "libllama.so")
  #+windows '("ggml-base.dll" "ggml-cpu.dll" "ggml-cuda.dll" "ggml-vulkan.dll"
              "ggml.dll" "llama.dll")
  #-(or windows darwin unix) '())

(defun %find-named (dir names)
  (dolist (name names)
    (let ((p (merge-pathnames name (uiop:ensure-directory-pathname dir))))
      (when (probe-file p)
        (return (namestring (truename p)))))))

(defun %preload-deps (dir)
  (%preload-cuda-runtime dir)
  (dolist (name (%ggml-preload-names))
    (let ((abs (%find-named dir (list name))))
      (when abs
        (handler-case (load-foreign-library abs)
          (error (e)
            (warn "llama-cpp: failed to preload ~a (~a)" abs e))))))
  t)

(defvar *llama-loaded* nil)

(defun load-llama ()
  "Load libllamastack (and ggml/llama deps). Idempotent."
  (unless *llama-loaded*
    (let ((preloaded nil))
      (dolist (dir (%native-search-dirs))
        (when (and dir (uiop:directory-exists-p dir))
          (pushnew dir cffi:*foreign-library-directories* :test #'equal)
          (unless preloaded
            (let ((abs (%find-named dir (%lib-candidates))))
              (when abs
                (%preload-deps dir)
                (load-foreign-library abs)
                (setf preloaded t))))))
      (unless preloaded
        (handler-case (load-foreign-library 'libllamastack)
          (error (e)
            (error 'llama-not-loaded
                   :message (princ-to-string e))))))
    (setf *llama-loaded* t)
    (let ((abi (ignore-errors (%abi-version))))
      (when (and abi (/= abi +llama-stack-abi-version+))
        (warn "libllamastack ABI ~a != pinned header ~a" abi +llama-stack-abi-version+))))
  t)

(defun llama-available-p ()
  (or *llama-loaded*
      (handler-case (progn (load-llama) t)
        (llama-not-loaded () nil)
        (error () nil))))

(defun ensure-llama ()
  (or *llama-loaded* (load-llama)))

(defcfun ("llama_stack_abi_version" %abi-version) :int32)
(defcfun ("llama_stack_version" %version) :string)
(defcfun ("llama_stack_last_error" %last-error) :string)
(defcfun ("llama_stack_load" %load) llama-stack-status
  (model-path :string)
  (params :pointer)
  (out :pointer))
(defcfun ("llama_stack_free" %free) :void
  (engine :pointer))
(defcfun ("llama_stack_embed" %embed) llama-stack-status
  (engine :pointer)
  (texts :pointer)
  (n-texts :int32)
  (out :pointer))
(defcfun ("llama_stack_embedding_result_free" %embedding-result-free) :void
  (out :pointer))
(defcfun ("llama_stack_complete" %complete) llama-stack-status
  (engine :pointer)
  (prompt :string)
  (max-tokens :int32)
  (temperature :float)
  (out-text :pointer)
  (prompt-tokens :pointer)
  (completion-tokens :pointer))
(defcfun ("llama_stack_complete_ex" %complete-ex) llama-stack-status
  (engine :pointer)
  (prompt :string)
  (params :pointer)
  (out-text :pointer)
  (prompt-tokens :pointer)
  (completion-tokens :pointer))
(defcfun ("llama_stack_complete_stream" %complete-stream) llama-stack-status
  (engine :pointer)
  (prompt :string)
  (params :pointer)
  (on-token :pointer)
  (user :pointer)
  (out-text :pointer)
  (prompt-tokens :pointer)
  (completion-tokens :pointer))
(defcfun ("llama_stack_string_free" %string-free) :void
  (s :pointer))
(defcfun ("llama_stack_grammar_parse" %grammar-parse) llama-stack-status
  (engine :pointer)
  (grammar :string)
  (grammar-root :string)
  (out :pointer))
(defcfun ("llama_stack_grammar_free" %grammar-free) :void
  (grammar :pointer))

(defvar *on-token-fn* nil)
(defvar *on-token-error* nil)

(defcallback %token-cb :int32 ((piece :pointer) (user :pointer))
  (declare (ignore user))
  (handler-case
      (let ((s (if (or (null piece) (null-pointer-p piece))
                   ""
                   (foreign-string-to-lisp piece))))
        (if (and *on-token-fn* (funcall *on-token-fn* s))
            1
            0))
    (error (e)
      (setf *on-token-error* e)
      1)))

(defun complete-ex-available-p ()
  "T when the loaded libllamastack exports llama_stack_complete_ex (ABI 2)."
  (and *llama-loaded*
       (let ((p (ignore-errors (foreign-symbol-pointer "llama_stack_complete_ex"))))
         (and p (not (null-pointer-p p))))))

(defun complete-stream-available-p ()
  "T when the loaded libllamastack exports llama_stack_complete_stream (ABI 3)."
  (and *llama-loaded*
       (let ((p (ignore-errors (foreign-symbol-pointer "llama_stack_complete_stream"))))
         (and p (not (null-pointer-p p))))))

(defun grammar-parse-available-p ()
  "T when the loaded libllamastack exports llama_stack_grammar_parse (ABI 4)."
  (and *llama-loaded*
       (let ((p (ignore-errors (foreign-symbol-pointer "llama_stack_grammar_parse"))))
         (and p (not (null-pointer-p p))))))

;;; Soft auto-load: overlay consumers get the lib; CI without natives still loads.
(eval-when (:load-toplevel :execute)
  (handler-case (load-llama)
    (error (e)
      (warn "llama-cpp: libllamastack not loaded (~a). Set LLAMA_CPP_NATIVE or build scripts/build-llama.sh."
            e))))
