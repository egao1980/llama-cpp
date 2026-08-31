(defpackage #:llama-cpp
  (:use #:cl #:cffi)
  (:local-nicknames (#:tg #:trivial-garbage))
  (:nicknames #:stack-llama)
  (:export #:+llama-stack-abi-version+
           #:llama-error
           #:llama-error-status
           #:llama-error-message
           #:llama-not-loaded
           #:llama-missing-model

           #:libllamastack
           #:load-llama
           #:llama-available-p
           #:llama-version
           #:llama-abi-version
           #:llama-last-error

           #:llama-engine
           #:llama-engine-p
           #:engine-pointer
           #:engine-model-path
           #:load-engine
           #:free-engine
           #:ensure-engine

           #:complete
           #:complete-ex-available-p
           #:embed))

(in-package #:llama-cpp)
