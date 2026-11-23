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

(defvar *llama-loaded* nil)

(defun load-llama ()
  "Load libllamastack and its ggml/CUDA dependencies. Idempotent.
ASDF load already ran the :native-library components (llama-cpp.asd); this
retries the missing ones, e.g. when LLAMA_CPP_NATIVE was set after load, and
signals LLAMA-NOT-LOADED while libllamastack is still absent."
  (unless *llama-loaded*
    (dolist (c (asdf:component-children (asdf:find-system "llama-cpp")))
      (when (typep c 'asdf-native:native-library)
        (handler-bind ((asdf-native:native-library-missing #'muffle-warning))
          (asdf-native:load-native-library c))))
    (unless (asdf-native:library-loaded-path "llama-cpp" "llamastack")
      (error 'llama-not-loaded
             :message "libllamastack not found (LLAMA_CPP_NATIVE, native/, lib/<os>-<arch>/)"))
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

(defcfun ("llama_stack_guard_selftest" %guard-selftest) llama-stack-status
  (kind :int32))

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

(defun guard-selftest-available-p ()
  "T when the loaded libllamastack exports llama_stack_guard_selftest."
  (and *llama-loaded*
       (let ((p (ignore-errors (foreign-symbol-pointer "llama_stack_guard_selftest"))))
         (and p (not (null-pointer-p p))))))

;;; Soft auto-load: overlay consumers get the lib; CI without natives still loads.
(eval-when (:load-toplevel :execute)
  (handler-case (load-llama)
    (error (e)
      (warn "llama-cpp: libllamastack not loaded (~a). Set LLAMA_CPP_NATIVE or build scripts/build-llama.sh."
            e))))
