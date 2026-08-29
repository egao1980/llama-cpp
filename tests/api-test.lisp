(in-package #:llama-cpp/tests)

(deftest abi-pin
  (ok (= 1 llama-cpp:+llama-stack-abi-version+)))

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
