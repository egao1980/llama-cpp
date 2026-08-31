(in-package #:llama-cpp)

(defun %env (name)
  (let ((v (uiop:getenv name)))
    (and v (plusp (length v)) v)))

(defun llama-version ()
  (ensure-llama)
  (%version))

(defun llama-abi-version ()
  (ensure-llama)
  (%abi-version))

(defun llama-last-error ()
  (when *llama-loaded*
    (%last-error)))

(defun %check (status op)
  (unless (eq status :ok)
    (error 'llama-error
           :status status
           :message (format nil "~a: ~a" op (or (%last-error) status)))))

(defmacro %with-foreign-floats (&body body)
  "ggml/Metal + L2 produce denormals/inexacts; SBCL traps those as conditions."
  `(#+sbcl sb-int:with-float-traps-masked
    #+sbcl (:underflow :overflow :inexact :invalid :divide-by-zero)
    #-sbcl progn
    ,@body))

(defclass llama-engine ()
  ((pointer :initarg :pointer :accessor engine-pointer)
   (model-path :initarg :model-path :accessor engine-model-path)
   (freed :initform nil :accessor engine-freed-p)))

(defun llama-engine-p (x)
  (typep x 'llama-engine))

(defun free-engine (engine)
  (when (and engine (not (engine-freed-p engine)))
    (let ((p (engine-pointer engine)))
      (when (and p (not (null-pointer-p p)))
        (%with-foreign-floats (%free p))))
    (setf (engine-pointer engine) (null-pointer)
          (engine-freed-p engine) t))
  engine)

(defun %finalize-engine (engine)
  (ignore-errors (free-engine engine)))

(defun load-engine (&key model-path (n-ctx 2048) (n-gpu-layers -1) n-threads)
  "Load a GGUF. N-CTX keeps KV/embed batch small. N-GPU-LAYERS -1 = all."
  (ensure-llama)
  (let ((path (or model-path (%env "LLAMA_MODEL_PATH") (%env "LLAMA_CPP_MODEL"))))
    (unless (and path (plusp (length path)))
      (error 'llama-missing-model
             :message "pass :model-path or set LLAMA_MODEL_PATH"))
    (with-foreign-object (params '(:struct llama-stack-load-params))
      (dotimes (i (foreign-type-size '(:struct llama-stack-load-params)))
        (setf (mem-aref params :uint8 i) 0))
      (setf (foreign-slot-value params '(:struct llama-stack-load-params) 'n-ctx)
            (or n-ctx 2048)
            (foreign-slot-value params '(:struct llama-stack-load-params) 'n-gpu-layers)
            (or n-gpu-layers -1)
            (foreign-slot-value params '(:struct llama-stack-load-params) 'n-threads)
            (or n-threads 0))
      (with-foreign-object (out :pointer)
        (setf (mem-ref out :pointer) (null-pointer))
        (%with-foreign-floats
          (%check (%load path params out) "llama_stack_load"))
        (let* ((ptr (mem-ref out :pointer))
               (engine (make-instance 'llama-engine
                                      :pointer ptr
                                      :model-path path)))
          (tg:finalize engine (lambda () (%finalize-engine engine)))
          engine)))))

(defun ensure-engine (engine)
  (cond
    ((llama-engine-p engine) engine)
    (t (error 'llama-error :message (format nil "not a llama-engine: ~s" engine)))))

(defun embed (engine texts)
  "Blocking embed. TEXTS is a string or sequence of strings.
   → (values list-of-single-float-vectors dim prompt-tokens)."
  (let* ((e (ensure-engine engine))
         (list (cond
                 ((stringp texts) (list texts))
                 ((or (listp texts) (vectorp texts))
                  (map 'list (lambda (x)
                               (if (stringp x) x (princ-to-string x)))
                       texts))
                 (t (list (princ-to-string texts)))))
         (n (length list)))
    (when (zerop n)
      (error 'llama-error :message "embed requires at least one text"))
    (with-foreign-objects ((arr :pointer n)
                           (out '(:struct llama-stack-embedding-result)))
      (dotimes (i (foreign-type-size '(:struct llama-stack-embedding-result)))
        (setf (mem-aref out :uint8 i) 0))
      (let ((owned '()))
        (unwind-protect
             (progn
               (loop for s in list for i from 0
                     for p = (foreign-string-alloc s)
                     do (push p owned)
                        (setf (mem-aref arr :pointer i) p))
               (%with-foreign-floats
                 (%check (%embed (engine-pointer e) arr n out) "llama_stack_embed")
                 (let* ((vals (foreign-slot-value out '(:struct llama-stack-embedding-result)
                                                  'values))
                        (n-emb (foreign-slot-value out '(:struct llama-stack-embedding-result)
                                                   'n-embeddings))
                        (dim (foreign-slot-value out '(:struct llama-stack-embedding-result)
                                                 'dim))
                        (tokens (foreign-slot-value out '(:struct llama-stack-embedding-result)
                                                    'prompt-tokens))
                        (vecs (loop for i from 0 below n-emb
                                    collect
                                    (let ((v (make-array dim :element-type 'single-float)))
                                      (dotimes (j dim)
                                        (setf (aref v j)
                                              (mem-aref vals :float (+ (* i dim) j))))
                                      v))))
                   (values vecs dim tokens))))
          (dolist (p owned)
            (foreign-string-free p))
          (unless (null-pointer-p
                   (foreign-slot-value out '(:struct llama-stack-embedding-result) 'values))
            (%embedding-result-free out)))))))

(defun %with-complete-params (max-tokens temperature grammar grammar-root fn)
  (with-foreign-object (params '(:struct llama-stack-complete-params))
    (dotimes (i (foreign-type-size '(:struct llama-stack-complete-params)))
      (setf (mem-aref params :uint8 i) 0))
    (setf (foreign-slot-value params '(:struct llama-stack-complete-params)
                              'max-tokens)
          (or max-tokens 32)
          (foreign-slot-value params '(:struct llama-stack-complete-params)
                              'temperature)
          (float temperature 1f0))
    (let ((g-ptr (and grammar (foreign-string-alloc grammar)))
          (r-ptr (and grammar-root (foreign-string-alloc grammar-root))))
      (unwind-protect
           (progn
             (when g-ptr
               (setf (foreign-slot-value
                      params '(:struct llama-stack-complete-params) 'grammar)
                     g-ptr))
             (when r-ptr
               (setf (foreign-slot-value
                      params '(:struct llama-stack-complete-params)
                      'grammar-root)
                     r-ptr))
             (funcall fn params))
        (when g-ptr (foreign-string-free g-ptr))
        (when r-ptr (foreign-string-free r-ptr))))))

(defun complete (engine prompt &key (max-tokens 32) (temperature 0.0)
                  grammar grammar-root on-token)
  "Blocking completion. GRAMMAR is GBNF (GRAMMAR-ROOT defaults to \"root\").
   ON-TOKEN is (lambda (piece)) per detok delta; non-NIL return stops.
   → (values text prompt-tokens completion-tokens)."
  (let ((e (ensure-engine engine))
        (g (and grammar (plusp (length grammar)) grammar))
        (root (and grammar-root (plusp (length grammar-root)) grammar-root)))
    (unless (and prompt (plusp (length prompt)))
      (error 'llama-error :message "complete requires a prompt"))
    (when (and g (not (complete-ex-available-p)))
      (error 'llama-error
             :message "grammar requires libllamastack ABI 2 (rebuild native)"))
    (when (and on-token (not (complete-stream-available-p)))
      (error 'llama-error
             :message "streaming requires libllamastack ABI 3 (rebuild native)"))
    (with-foreign-objects ((out :pointer)
                           (pt :int32)
                           (ct :int32))
      (setf (mem-ref out :pointer) (null-pointer)
            (mem-ref pt :int32) 0
            (mem-ref ct :int32) 0)
      (%with-foreign-floats
        (cond
          (on-token
           (let ((*on-token-fn* on-token)
                 (*on-token-error* nil))
             (%with-complete-params max-tokens temperature g root
               (lambda (params)
                 (%check (%complete-stream (engine-pointer e) prompt params
                                           (callback %token-cb) (null-pointer)
                                           out pt ct)
                         "llama_stack_complete_stream")))
             (when *on-token-error*
               (error *on-token-error*))))
          ((or g (complete-ex-available-p))
           (%with-complete-params max-tokens temperature g root
             (lambda (params)
               (%check (%complete-ex (engine-pointer e) prompt params out pt ct)
                       "llama_stack_complete_ex"))))
          (t
           (%check (%complete (engine-pointer e) prompt max-tokens
                              (float temperature 1f0) out pt ct)
                   "llama_stack_complete"))))
      (let ((ptr (mem-ref out :pointer)))
        (unwind-protect
             (values (if (or (null ptr) (null-pointer-p ptr))
                         ""
                         (foreign-string-to-lisp ptr))
                     (mem-ref pt :int32)
                     (mem-ref ct :int32))
          (unless (or (null ptr) (null-pointer-p ptr))
            (%string-free ptr)))))))
