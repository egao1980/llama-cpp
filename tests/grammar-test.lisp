(in-package #:llama-cpp/tests)

;;; llama-cpp#17. A C++ exception from llama.cpp's grammar sampler
;;; (llama_grammar_accept_token) escapes libllamastack's C ABI: std::terminate,
;;; abort(), SIGABRT, and SBCL drops into LDB; and llama_sampler_sample already
;;; accepts the token, so a second llama_sampler_accept fed the grammar every
;;; piece twice. The completion runs in a child
;;; SBCL, so an abort fails these tests instead of killing the test runner.
;;;
;;; Needs a GGUF in LLAMA_CPP_TEST_MODEL (skipped otherwise; not
;;; LLAMA_MODEL_PATH, which LOAD-ENGINE-WITHOUT-MODEL expects unset).
;;; Reproduced with ggml-org/models tinyllamas/stories260K.gguf (1 MB):
;;;   curl -L -o /tmp/stories260K.gguf \
;;;     https://huggingface.co/ggml-org/models/resolve/main/tinyllamas/stories260K.gguf
;;;   LLAMA_CPP_TEST_MODEL=/tmp/stories260K.gguf cl-repo test
;;; A checkout has no natives: point LLAMA_CPP_NATIVE at lib/<os>-<arch>/ from
;;; scripts/build-llama.sh (LLAMA_CPP_SKIP_CMAKE=1 relinks only the shim), or at
;;; an overlay, e.g. ~/.local/share/cl-repository/systems/llama-cpp/0.2.1/native.
;;; The 0.2.0 overlay aborts in every grammar test below.
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
  (or (find-if (lambda (l) (or (search "uncaught exception" l) (search "GGML_ASSERT" l)))
               (uiop:split-string err :separator '(#\Newline)))
      (subseq err (max 0 (- (length err) 400)))))

#+sbcl
(defun run-child-sbcl (&rest evals)
  "Load this llama-cpp in a fresh SBCL (--non-interactive, no LDB), then
   evaluate each of EVALS (source text) in turn. → (values stdout stderr code)."
  (let ((load (with-standard-io-syntax
                  (let ((*print-readably* nil))
                    (format nil "(progn (asdf:initialize-source-registry '~s) (asdf:load-system \"llama-cpp\"))"
                            (child-source-registry))))))
    (multiple-value-bind (out err code)
        (uiop:run-program (append
                           (list (namestring sb-ext:*runtime-pathname*)
                                 "--core" (namestring sb-ext:*core-pathname*)
                                 "--noinform" "--disable-ldb" "--lose-on-corruption"
                                 "--end-runtime-options"
                                 "--no-sysinit" "--no-userinit" "--non-interactive"
                                 "--eval" "(require :asdf)"
                                 "--eval" load)
                           (loop for e in evals append (list "--eval" e)))
                          :input nil :output :string :error-output :string
                          :ignore-error-status t)
      (values out err code))))

#+sbcl
(defun run-in-child (call)
  "Evaluate CALL (Lisp source text; E is an engine on LLAMA_CPP_TEST_MODEL)
   in a fresh SBCL. → (values result exit-code stderr). RESULT is (:ok value),
   (:llama-error message), or NIL when the child died before printing one."
  (multiple-value-bind (out err code)
      (run-child-sbcl
       (with-standard-io-syntax
           (let ((*print-readably* nil))
             (format nil "(let ((e (llama-cpp:load-engine :model-path ~s :n-ctx 256))) (let ((*print-pretty* nil)) (format t \"~~&RESULT ~~s~~%\" (handler-case (list :ok ~a) (llama-cpp:llama-error (c) (list :llama-error (princ-to-string c)))))) (finish-output) (llama-cpp:free-engine e))"
                     (namestring (model-path)) call))))
    ;; Read from the marker, not one line: a printed string may hold newlines.
    (let ((start (search (format nil "~%RESULT ") (format nil "~%~a" out))))
      (values (and start
                   (with-standard-io-syntax
                       (let ((*read-eval* nil))
                         (ignore-errors
                           (read-from-string out t nil :start (+ start 7))))))
              code
              err))))

#+sbcl
(defun complete-in-child (grammar &key (prompt "The cat") (max-tokens 4)
                                  (temperature 0.0) (mode :grammar))
  "COMPLETE with GRAMMAR on LLAMA_CPP_TEST_MODEL in a fresh SBCL (RUN-IN-CHILD).
   MODE :grammar → (:ok text). :parsed parses GRAMMAR once and completes twice
   with the handle → (:ok (text1 text2)). :stream adds :on-token →
   (:ok (text pieces-joined))."
  (run-in-child
   (with-standard-io-syntax
       (let ((*print-readably* nil))
         (ecase mode
           (:grammar
            (format nil "(llama-cpp:complete e ~s :max-tokens ~d :temperature ~f :grammar ~s)"
                    prompt max-tokens temperature grammar))
           (:parsed
            (format nil "(let ((g (llama-cpp:parse-grammar e ~s))) (prog1 (list (llama-cpp:complete e ~s :max-tokens ~d :temperature ~f :parsed g) (llama-cpp:complete e ~s :max-tokens ~d :temperature ~f :parsed g)) (llama-cpp:free-grammar g)))"
                    grammar prompt max-tokens temperature prompt max-tokens temperature))
           (:stream
            (format nil "(let ((pieces '())) (let ((text (llama-cpp:complete e ~s :max-tokens ~d :temperature ~f :grammar ~s :on-token (lambda (p) (push p pieces) nil)))) (list text (apply #'concatenate 'string (reverse pieces)))))"
                    prompt max-tokens temperature grammar)))))))

(defmacro with-grammar-child (((result code err) &rest complete-args) &body body)
  "Run COMPLETE-IN-CHILD with COMPLETE-ARGS (default: *YES-NO-GRAMMAR*) and
   bind its values, or skip when there is no library, model or SBCL."
  `(cond
    ((not (llama-cpp:llama-available-p)) (skip "libllamastack not present"))
    ((not (llama-cpp:complete-ex-available-p)) (skip "libllamastack ABI < 2 (no grammar)"))
    ((not (model-path)) (skip "set LLAMA_CPP_TEST_MODEL to a GGUF (see tests/grammar-test.lisp)"))
    #-sbcl (t (skip "child-image test is SBCL-only"))
    #+sbcl
    (t (multiple-value-bind (,result ,code ,err)
           (complete-in-child ,@(or complete-args '(*yes-no-grammar*)))
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
  (with-grammar-child ((result code err))
    (ok (and result (member (first result) '(:ok :llama-error)))
        (format nil "child exited ~a without a result: ~a" code (abort-reason err)))))

(deftest grammar-complete-returns-a-sentence-of-the-grammar
  ;; Fix 2: once the grammar is complete, generation stops (EOS), so the text
  ;; is a sentence of root ::= ("yes" | "no").
  (with-grammar-child ((result code err))
    (ok (and (eq (first result) :ok)
             (member (second result) '("yes" "no") :test #'string=))
        (format nil "want (:ok \"yes\"|\"no\"), got ~s (child exit ~a~@[: ~a~])"
                result code (and (null result) (abort-reason err))))))

(defun yes-or-no-p* (text)
  (and (stringp text) (member text '("yes" "no") :test #'string=)))

(defun describe-child (result code err)
  (format nil "got ~s (child exit ~a~@[: ~a~])"
          result code (and (null result) (abort-reason err))))

(deftest grammar-dead-end-becomes-llama-error
  ;; No token's piece can start with NUL, so the grammar sampler rules out
  ;; every candidate. llama_sampler_sample picks one anyway (greedy keeps
  ;; candidate 0) and accepting it fails: std::runtime_error here, GGML_ABORT
  ;; when candidate 0 is end-of-generation (grammar-eog-test.lisp).
  ;; libllamastack 0.2.0 aborts the image. Now the shim sees that nothing is
  ;; left before it picks and accepts: a LLAMA-ERROR :RUNTIME that says so.
  (with-grammar-child ((result code err) "root ::= \"\\x00\"")
    (ok (and (eq (first result) :llama-error)
             (search "grammar dead end" (second result))
             (search "RUNTIME" (second result)))
        (describe-child result code err))))

(deftest guard-turns-each-exception-into-runtime
  ;; The ABI guard on its own, in the library that is loaded (no model
  ;; needed). COMPLETE no longer reaches llama.cpp's grammar throws (the dead
  ;; end is seen first), so llama_stack_guard_selftest throws instead: a
  ;; std::exception, std::bad_alloc and a non-std exception each come back as
  ;; :RUNTIME with the message, never as an abort of the child.
  (cond
   ((not (llama-cpp:llama-available-p)) (skip "libllamastack not present"))
   ((not (llama-cpp::guard-selftest-available-p))
    (skip "libllamastack without llama_stack_guard_selftest"))
   #-sbcl (t (skip "child-image test is SBCL-only"))
   #+sbcl
   (t (multiple-value-bind (out err code)
          (run-child-sbcl
           "(let ((*print-pretty* nil)) (format t \"~&RESULT ~s~%\" (loop for kind in '(0 1 2 3) collect (list (llama-cpp::%guard-selftest kind) (llama-cpp::%last-error)))) (finish-output))")
        (let* ((start (search (format nil "~%RESULT ") (format nil "~%~a" out)))
               (result (and start
                            (with-standard-io-syntax
                                (let ((*read-eval* nil))
                                  (ignore-errors
                                    (read-from-string out t nil :start (+ start 7))))))))
          (ok (equal result '((:ok "")
                              (:runtime "C++ exception: guard selftest")
                              (:runtime "out of memory")
                              (:runtime "unknown C++ exception")))
              (format nil "got ~s (child exit ~a~@[: ~a~])"
                      result code (and (null result) (abort-reason err)))))))))

(deftest bad-gbnf-is-llama-error
  ;; An undefined rule: llama_sampler_init_grammar returns NULL → :INVALID.
  (with-grammar-child ((result code err) "root ::= \"a\" | missing-rule")
    (ok (and (eq (first result) :llama-error)
             (search "grammar parse failed" (second result)))
        (describe-child result code err))))

(deftest parsed-grammar-is-reusable
  ;; PARSE-GRAMMAR once, COMPLETE :parsed twice: each call clones the sampler,
  ;; so the second completion starts from the grammar's root again.
  (with-grammar-child ((result code err) *yes-no-grammar* :mode :parsed)
    (ok (and (eq (first result) :ok)
             (every #'yes-or-no-p* (second result)))
        (describe-child result code err))))

(deftest streamed-grammar-pieces-join-to-the-text
  ;; :on-token (llama_stack_complete_stream) takes the same sampler path.
  (with-grammar-child ((result code err) *yes-no-grammar* :mode :stream)
    (ok (and (eq (first result) :ok)
             (yes-or-no-p* (first (second result)))
             (equal (first (second result)) (second (second result))))
        (describe-child result code err))))

(deftest sampled-grammar-complete-is-a-sentence
  ;; Temperature > 0 (temp + dist after the grammar) also ends at the grammar.
  (with-grammar-child ((result code err) *yes-no-grammar* :temperature 0.8)
    (ok (and (eq (first result) :ok) (yes-or-no-p* (second result)))
        (describe-child result code err))))

(deftest extreme-temperatures-keep-to-the-grammar
  ;; temp divides the logits the grammar left: 1e-30 makes them +-inf, 1e30
  ;; makes them all about 0. dist still picks among them only.
  (with-grammar-child ((result code err) *yes-no-grammar* :temperature 1e-30)
    (ok (and (eq (first result) :ok) (yes-or-no-p* (second result)))
        (describe-child result code err)))
  (with-grammar-child ((result code err) *yes-no-grammar* :temperature 1e30)
    (ok (and (eq (first result) :ok) (yes-or-no-p* (second result)))
        (describe-child result code err))))

(deftest grammar-stops-at-max-tokens
  ;; An open-ended grammar is cut at :max-tokens: a prefix of a sentence,
  ;; not an error.
  (with-grammar-child ((result code err) "root ::= \"{\" [a-z]+ \"}\"" :max-tokens 3)
    (let ((text (and (eq (first result) :ok) (second result))))
      (ok (and (stringp text)
               (plusp (length text))
               (char= #\{ (char text 0))
               (every (lambda (ch) (or (char<= #\a ch #\z) (char= ch #\})))
                      (subseq text 1)))
          (describe-child result code err)))))
