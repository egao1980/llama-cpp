(in-package #:llama-cpp/tests)

(deftest abi-pin
  (ok (= 4 llama-cpp:+llama-stack-abi-version+)))

(deftest available-p-does-not-crash
  (ok (member (llama-cpp:llama-available-p) '(t nil))))

(deftest load-engine-without-model
  (if (llama-cpp:llama-available-p)
      (ok (signals (llama-cpp:load-engine :model-path nil)
                   'llama-cpp:llama-missing-model))
      (skip "libllamastack not present")))

(deftest embed-rejects-non-engine
  (ok (signals (llama-cpp:embed "not-an-engine" "hi")
               'llama-cpp:llama-error)))

(deftest complete-rejects-non-engine
  (ok (signals (llama-cpp:complete "not-an-engine" "hi")
               'llama-cpp:llama-error)))

(deftest complete-rejects-grammar-without-engine
  (ok (signals (llama-cpp:complete "not-an-engine" "hi"
                                  :grammar "root ::= \"x\"")
               'llama-cpp:llama-error)))

(deftest complete-rejects-on-token-without-engine
  (ok (signals (llama-cpp:complete "not-an-engine" "hi"
                                  :on-token (lambda (p) (declare (ignore p)) nil))
               'llama-cpp:llama-error)))

(deftest complete-stream-available-p-does-not-crash
  (ok (member (llama-cpp:complete-stream-available-p) '(t nil))))

(deftest grammar-parse-available-p-does-not-crash
  (ok (member (llama-cpp:grammar-parse-available-p) '(t nil))))

(deftest parse-grammar-rejects-non-engine
  (ok (signals (llama-cpp:parse-grammar "not-an-engine" "root ::= \"x\"")
               'llama-cpp:llama-error)))

(deftest parse-grammar-rejects-empty
  (ok (signals (llama-cpp:parse-grammar "not-an-engine" "")
               'llama-cpp:llama-error)))

(deftest complete-rejects-parsed-without-engine
  (ok (signals (llama-cpp:complete "not-an-engine" "hi"
                                  :parsed (make-instance 'llama-cpp:llama-grammar
                                                         :pointer (cffi:null-pointer)))
               'llama-cpp:llama-error)))
