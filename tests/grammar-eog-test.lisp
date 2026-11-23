(in-package #:llama-cpp/tests)

;;; llama-cpp#17, prosecution round 1. The guard in libllamastack catches C++
;;; exceptions, but llama.cpp's grammar sampler also stops the process with
;;; GGML_ABORT (abort(), not an exception) when it has to accept an
;;; end-of-generation token while the grammar is not complete
;;; (llama-grammar.cpp, llama_grammar_accept_impl). That happens whenever the
;;; grammar rules out every candidate and the forced greedy pick (candidate 0)
;;; is an EOG token, as in vocabularies whose token 0 is <|endoftext|>
;;; (SmolLM2, StarCoder, ...). A grammar that ends with the chat end-of-turn
;;; token (GBNF token syntax, llama.cpp grammars/README.md) gets there after
;;; one word: the grammar needs <|im_end|>, the sampler forbids EOG until the
;;; grammar is complete, so nothing is left.
;;;
;;; Needs LLAMA_CPP_TEST_CHAT_MODEL, a GGUF whose token 0 is end-of-generation
;;; (skipped otherwise), e.g. SmolLM2-135M-Instruct-Q4_0.gguf (92 MB):
;;;   curl -L -o /tmp/smollm2-135m.gguf \
;;;     https://huggingface.co/bartowski/SmolLM2-135M-Instruct-GGUF/resolve/main/SmolLM2-135M-Instruct-Q4_0.gguf

(defun chat-model-path ()
  (let ((p (uiop:getenv "LLAMA_CPP_TEST_CHAT_MODEL")))
    (and p (plusp (length p)) (probe-file p) p)))

(defun native-abort-reason (err)
  "The GGML_ABORT / uncaught-exception line of a dead child's stderr."
  (or (find-if (lambda (l)
                 (or (search "fatal error" l)
                     (search "GGML_ASSERT" l)
                     (search "uncaught exception" l)))
               (uiop:split-string err :separator '(#\Newline)))
      (abort-reason err)))

#+sbcl
(defun complete-on-chat-model (grammar &key (prompt "The cat") (max-tokens 8))
  "COMPLETE with GRAMMAR (greedy) on LLAMA_CPP_TEST_CHAT_MODEL in a fresh
   SBCL. → (values result exit-code stderr), as RUN-IN-CHILD."
  (multiple-value-bind (out err code)
      (run-child-sbcl
       (with-standard-io-syntax
         (let ((*print-readably* nil))
           (format nil "(let ((e (llama-cpp:load-engine :model-path ~s :n-ctx 256))) (let ((*print-pretty* nil)) (format t \"~~&RESULT ~~s~~%\" (handler-case (list :ok (llama-cpp:complete e ~s :max-tokens ~d :grammar ~s)) (llama-cpp:llama-error (c) (list :llama-error (princ-to-string c)))))) (finish-output) (llama-cpp:free-engine e))"
                   (namestring (chat-model-path)) prompt max-tokens grammar))))
    (let ((start (search (format nil "~%RESULT ") (format nil "~%~a" out))))
      (values (and start
                   (with-standard-io-syntax
                     (let ((*read-eval* nil))
                       (ignore-errors
                        (read-from-string out t nil :start (+ start 7))))))
              code
              err))))

(defmacro with-chat-model-child (((result code err) &rest complete-args) &body body)
  "Run COMPLETE-ON-CHAT-MODEL with COMPLETE-ARGS and bind its values, or skip
   when there is no library, chat model or SBCL."
  `(cond
     ((not (llama-cpp:llama-available-p)) (skip "libllamastack not present"))
     ((not (llama-cpp:complete-ex-available-p)) (skip "libllamastack ABI < 2 (no grammar)"))
     ((not (chat-model-path))
      (skip "set LLAMA_CPP_TEST_CHAT_MODEL to a GGUF whose token 0 is EOG (see tests/grammar-eog-test.lisp)"))
     #-sbcl (t (skip "child-image test is SBCL-only"))
     #+sbcl
     (t (multiple-value-bind (,result ,code ,err) (complete-on-chat-model ,@complete-args)
          ,@body))))

(defun describe-chat-child (result code err)
  (format nil "got ~s (child exit ~a~@[: ~a~])"
          result code (and (null result) (native-abort-reason err))))

(deftest grammar-ending-in-end-of-turn-does-not-abort-the-image
  ;; root ::= ("yes" | "no") <|im_end|>. Want a value or a LLAMA-ERROR, not
  ;; SIGABRT from GGML_ABORT in llama_grammar_accept_impl.
  (with-chat-model-child ((result code err) "root ::= (\"yes\" | \"no\") <|im_end|>")
    (ok (and result (member (first result) '(:ok :llama-error)))
        (describe-chat-child result code err))))

(deftest grammar-dead-end-on-a-chat-vocab-is-llama-error
  ;; GRAMMAR-EXCEPTION-BECOMES-LLAMA-ERROR (root ::= "\x00") on a vocabulary
  ;; whose token 0 is EOG: the same dead end, but the forced pick is EOG, so
  ;; GGML_ABORT instead of std::runtime_error. Want LLAMA-ERROR, as on
  ;; stories260K.
  (with-chat-model-child ((result code err) "root ::= \"\\x00\"")
    (ok (eq (first result) :llama-error)
        (describe-chat-child result code err))))
