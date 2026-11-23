(in-package #:llama-cpp/tests)

;;; llama-cpp#18. Exiting with a live engine: C exit() destroys ggml-metal's
;;; device statics under the model's residency sets and ggml_metal_device_free
;;; aborts (darwin Metal), so the process ends in SIGABRT instead of its exit
;;; code. FREE-LIVE-HANDLES frees what is left from the exit hooks. Uses the
;;; child SBCL and LLAMA_CPP_TEST_MODEL of grammar-test.lisp.

(defmacro with-model-child (&body body)
  `(cond
     ((not (llama-cpp:llama-available-p)) (skip "libllamastack not present"))
     ((not (model-path)) (skip "set LLAMA_CPP_TEST_MODEL to a GGUF (see tests/grammar-test.lisp)"))
     #-sbcl (t (skip "child-image test is SBCL-only"))
     #+sbcl (t ,@body)))

(defun keep-engine-form ()
  (with-standard-io-syntax
    (let ((*print-readably* nil))
      (format nil "(defvar cl-user::*engine* (llama-cpp:load-engine :model-path ~s :n-ctx 256))"
              (namestring (model-path))))))

(defun native-abort-p (err)
  (or (search "GGML_ASSERT" err) (search "SIGABRT" err)))

#+sbcl
(defun exit-child (&rest evals)
  "→ (values code stderr) of a child that keeps an engine in a global."
  (multiple-value-bind (out err code)
      (apply #'run-child-sbcl (keep-engine-form) evals)
    (declare (ignore out))
    (values code err)))

(deftest exit-with-a-live-engine-keeps-its-exit-code
  (with-model-child
    (multiple-value-bind (code err) (exit-child "(sb-ext:exit :code 3)")
      (ok (and (eql code 3) (not (native-abort-p err)))
          (format nil "exit code ~a~@[: ~a~]" code (and (native-abort-p err) (abort-reason err)))))))

(deftest unhandled-error-with-a-live-engine-exits-1
  ;; --non-interactive: the debugger hook exits with code 1. A failing
  ;; `cl-repo test` run takes this path.
  (with-model-child
    (multiple-value-bind (code err) (exit-child "(error \"boom\")")
      (ok (and (eql code 1) (not (native-abort-p err)))
          (format nil "exit code ~a~@[: ~a~]" code (and (native-abort-p err) (abort-reason err)))))))

(deftest free-live-handles-frees-each-handle-once
  ;; Grammar and engine freed, both marked freed; a second call and the
  ;; explicit FREE-* afterwards are no-ops.
  (with-model-child
    (multiple-value-bind (result code err)
        (run-in-child
         "(let ((g (llama-cpp:parse-grammar e \"root ::= \\\"x\\\"\"))) (list (llama-cpp:free-live-handles) (llama-cpp::engine-freed-p e) (llama-cpp::grammar-freed-p g) (llama-cpp:free-live-handles) (progn (llama-cpp:free-grammar g) (llama-cpp:free-engine e) :freed)))")
      (ok (equal result '(:ok (2 t t 0 :freed)))
          (format nil "got ~s (child exit ~a~@[: ~a~])"
                  result code (and (null result) (abort-reason err)))))))
