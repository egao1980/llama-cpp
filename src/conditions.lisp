(in-package #:llama-cpp)

(define-condition llama-error (error)
  ((status :initarg :status :reader llama-error-status :initform nil)
   (message :initarg :message :reader llama-error-message :initform nil))
  (:report (lambda (c s)
             (format s "llama-cpp error~@[ (~a)~]~@[: ~a~]"
                     (llama-error-status c)
                     (llama-error-message c)))))

(define-condition llama-not-loaded (llama-error) ()
  (:report (lambda (c s)
             (format s "libllamastack is not loaded~@[: ~a~]" (llama-error-message c)))))

(define-condition llama-missing-model (llama-error) ()
  (:report (lambda (c s)
             (format s "llama.cpp model path missing~@[: ~a~]" (llama-error-message c)))))
