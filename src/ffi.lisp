(in-package #:llama-cpp)

(defconstant +llama-stack-abi-version+ 1)

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

(defun %native-search-dirs ()
  "Overlay native/, lib/<os>-<arch>/. No LD_LIBRARY_PATH / DYLD_LIBRARY_PATH."
  (let ((dirs '()))
    (dolist (var '("LLAMA_CPP_NATIVE"))
      (let ((v (uiop:getenv var)))
        (when (and v (plusp (length v)))
          (push v dirs))))
    (ignore-errors
      (let* ((sys (asdf:find-system :llama-cpp nil))
             (root (when sys (asdf:system-source-directory sys))))
        (when root
          (push (namestring (merge-pathnames "native/" root)) dirs)
          (push (namestring (merge-pathnames
                             (format nil "lib/~A-~A/" (%host-os) (%host-arch))
                             root))
                dirs))))
    (nreverse dirs)))

(defun %lib-candidates ()
  #+windows '("llamastack.dll" "libllamastack.dll")
  #+darwin '("libllamastack.dylib")
  #+(and unix (not darwin)) '("libllamastack.so")
  #-(or windows darwin unix) '("libllamastack.so"))

(defun %ggml-preload-names ()
  #+darwin '("libggml-base.dylib" "libggml-cpu.dylib" "libggml-metal.dylib"
             "libggml-blas.dylib" "libggml.dylib" "libllama.dylib")
  #+(and unix (not darwin)) '("libggml-base.so" "libggml-cpu.so" "libggml-cuda.so"
                              "libggml.so" "libllama.so")
  #+windows '("ggml-base.dll" "ggml-cpu.dll" "ggml-blas.dll" "ggml.dll" "llama.dll")
  #-(or windows darwin unix) '())

(defun %find-named (dir names)
  (dolist (name names)
    (let ((p (merge-pathnames name (uiop:ensure-directory-pathname dir))))
      (when (probe-file p)
        (return (namestring (truename p)))))))

(defun %preload-deps (dir)
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
(defcfun ("llama_stack_string_free" %string-free) :void
  (s :pointer))

;;; Soft auto-load: overlay consumers get the lib; CI without natives still loads.
(eval-when (:load-toplevel :execute)
  (handler-case (load-llama)
    (error (e)
      (warn "llama-cpp: libllamastack not loaded (~a). Set LLAMA_CPP_NATIVE or build scripts/build-llama.sh."
            e))))
