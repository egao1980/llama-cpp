(in-package #:llama-cpp/tests)

;;; llama-cpp#17. A C++ exception from llama.cpp's grammar sampler
;;; (llama_grammar_accept_token) escapes libllamastack's C ABI: std::terminate,
;;; abort(), SIGABRT, and SBCL drops into LDB. The completion runs in a child
;;; SBCL, so an abort fails these tests instead of killing the test runner.
;;;
;;; Needs a GGUF in LLAMA_CPP_TEST_MODEL (skipped otherwise; not
;;; LLAMA_MODEL_PATH, which LOAD-ENGINE-WITHOUT-MODEL expects unset).
;;; Reproduced with ggml-org/models tinyllamas/stories260K.gguf (1 MB):
;;;   curl -L -o /tmp/stories260K.gguf \
;;;     https://huggingface.co/ggml-org/models/resolve/main/tinyllamas/stories260K.gguf
;;;   LLAMA_CPP_TEST_MODEL=/tmp/stories260K.gguf cl-repo test
;;; A checkout has no natives: point LLAMA_CPP_NATIVE at an overlay, e.g.
;;; ~/.local/share/cl-repository/systems/llama-cpp/0.2.0/native.
;;; llama.cpp's own llama-server returns "yes" for the same model, prompt and
;;; grammar at temperature 0.

(defparameter *yes-no-grammar* "root ::= (\"yes\" | \"no\")")

(defun model-path ()
  (let ((p (uiop:getenv "LLAMA_CPP_TEST_MODEL")))
    (and p (plusp (length p)) (probe-file p) p)))

(defun child-source-registry ()
  "The directory of every loaded system, so the child loads this llama-cpp."
  `(:source-registry
    ,@(remove-duplicates
       (loop for name in (asdf:already-loaded-systems)
             for dir = (ignore-errors (asdf:system-source-directory name))
             when dir
               collect (list :directory (namestring dir)))
       :test #'equal)
    :ignore-inherited-configuration))

(defun abort-reason (err)
  (or (find-if (lambda (l) (search "uncaught exception" l))
               (uiop:split-string err :separator '(#\Newline)))
      (subseq err (max 0 (- (length err) 400)))))

#+sbcl
(defun complete-in-child (grammar &key (prompt "The cat") (max-tokens 4))
  "COMPLETE with GRAMMAR on LLAMA_CPP_TEST_MODEL in a fresh SBCL.
   → (values result exit-code stderr). RESULT is (:ok text),
   (:llama-error message), or NIL when the child died before printing one."
  (let* ((load (with-standard-io-syntax
                 (let ((*print-readably* nil))
                   (format nil "(progn (asdf:initialize-source-registry '~s) (asdf:load-system \"llama-cpp\"))"
                           (child-source-registry)))))
         (run (with-standard-io-syntax
                (let ((*print-readably* nil))
                  (format nil "(let ((e (llama-cpp:load-engine :model-path ~s :n-ctx 256))) (format t \"~~&RESULT ~~s~~%\" (handler-case (list :ok (llama-cpp:complete e ~s :max-tokens ~d :grammar ~s)) (llama-cpp:llama-error (c) (list :llama-error (princ-to-string c))))) (finish-output) (llama-cpp:free-engine e))"
                          (namestring (model-path)) prompt max-tokens grammar)))))
    (multiple-value-bind (out err code)
        (uiop:run-program (list (namestring sb-ext:*runtime-pathname*)
                                "--core" (namestring sb-ext:*core-pathname*)
                                "--noinform" "--disable-ldb" "--lose-on-corruption"
                                "--end-runtime-options"
                                "--no-sysinit" "--no-userinit" "--non-interactive"
                                "--eval" "(require :asdf)"
                                "--eval" load
                                "--eval" run)
                          :input nil :output :string :error-output :string
                          :ignore-error-status t)
      (let ((line (find-if (lambda (l) (uiop:string-prefix-p "RESULT " l))
                           (uiop:split-string out :separator '(#\Newline)))))
        (values (and line
                     (with-standard-io-syntax
                       (let ((*read-eval* nil))
                         (read-from-string line t nil :start 7))))
                code
                err)))))

(defmacro with-grammar-child ((result code err) &body body)
  `(cond
     ((not (llama-cpp:llama-available-p)) (skip "libllamastack not present"))
     ((not (llama-cpp:complete-ex-available-p)) (skip "libllamastack ABI < 2 (no grammar)"))
     ((not (model-path)) (skip "set LLAMA_CPP_TEST_MODEL to a GGUF (see tests/grammar-test.lisp)"))
     #-sbcl (t (skip "child-image test is SBCL-only"))
     #+sbcl
     (t (multiple-value-bind (,result ,code ,err) (complete-in-child *yes-no-grammar*)
          ,@body))))

(deftest child-completes-without-grammar
  ;; Control: the same child, model and prompt without :grammar return text,
  ;; so a failure below is the grammar path, not the harness.
  (cond
   ((not (llama-cpp:llama-available-p)) (skip "libllamastack not present"))
   ((not (model-path)) (skip "set LLAMA_CPP_TEST_MODEL to a GGUF (see tests/grammar-test.lisp)"))
   #-sbcl (t (skip "child-image test is SBCL-only"))
   #+sbcl
   (t (multiple-value-bind (result code err) (complete-in-child nil)
        (ok (and (eq (first result) :ok) (plusp (length (second result))))
            (format nil "got ~s (child exit ~a~@[: ~a~])"
                    result code (and (null result) (abort-reason err))))))))

(deftest grammar-complete-does-not-abort-the-image
  ;; Fix 1: catch at the ABI boundary. A C++ exception becomes a status code
  ;; and a LLAMA-ERROR, never an abort.
  (with-grammar-child (result code err)
    (ok (and result (member (first result) '(:ok :llama-error)))
        (format nil "child exited ~a without a result: ~a" code (abort-reason err)))))

(deftest grammar-complete-returns-a-sentence-of-the-grammar
  ;; Fix 2: once the grammar is complete, generation stops (EOS), so the text
  ;; is a sentence of root ::= ("yes" | "no").
  (with-grammar-child (result code err)
    (ok (and (eq (first result) :ok)
             (member (second result) '("yes" "no") :test #'string=))
        (format nil "want (:ok \"yes\"|\"no\"), got ~s (child exit ~a~@[: ~a~])"
                result code (and (null result) (abort-reason err))))))
