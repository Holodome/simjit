;; This file is part of Simjit project <https://simjit.org>
;;
;; See LICENSE for license and copyright information
;; SPDX-License-Identifier: Zlib

#lang clotho

(require xsmith
         xsmith/app
         racr
         json
         racket/cmdline
         racket/list
         racket/match
         racket/port
         racket/string)

(define-spec-component simjit-hir)

(struct program-ir (roots) #:transparent)
(struct store-root-ir (dtype expr kind cond) #:transparent)
(struct agg-root-ir (dtype op expr cond) #:transparent)
(struct pred-agg-root-ir (op pred) #:transparent)
(struct countif-root-ir (pred) #:transparent)

(struct const-ir (dtype bits) #:transparent)
(struct load-ir (dtype kind splat?) #:transparent)
(struct gather-ir (dtype idx) #:transparent)
(struct index-ir (dtype) #:transparent)
(struct binary-ir (dtype op left right) #:transparent)
(struct unary-ir (dtype op arg) #:transparent)
(struct cmp-ir (dtype op left right unsigned?) #:transparent)
(struct pred-binary-ir (op left right) #:transparent)
(struct pred-not-ir (arg) #:transparent)
(struct select-ir (dtype cond truthy falsy) #:transparent)
(struct int-cast-ir (dtype kind arg) #:transparent)
(struct float-cast-ir (dtype arg unsigned?) #:transparent)
(struct bitcast-ir (dtype arg) #:transparent)
(struct scatter-root-ir (dtype expr idx cond) #:transparent)

(define i8-type (base-type 'i8))
(define i16-type (base-type 'i16))
(define i32-type (base-type 'i32))
(define i64-type (base-type 'i64))
(define f32-type (base-type 'f32))
(define f64-type (base-type 'f64))
(define pred-type (base-type 'i1))
(define root-type (base-type 'root))
(define program-type (base-type 'program))

(define value-dtypes '(i8 i16 i32 i64 f32 f64))
(define int-dtypes '(i8 i16 i32 i64))
(define index-dtypes '(i32 i64))
(define float-dtypes '(f32 f64))
(define pred-agg-ops '(and or xor))

(define i8-const-bits
  '("0x0" "0x1" "0x2" "0x7f" "0x80" "0xff" "0x12" "0xf0"))
(define i16-const-bits
  '("0x0" "0x1" "0x2" "0x7fff" "0x8000" "0xffff" "0x1234" "0xfff0"))
(define i32-const-bits
  '("0x0" "0x1" "0x2" "0x7fffffff" "0x80000000" "0xffffffff" "0x12345678" "0xfffffff0"))
(define i64-const-bits
  '("0x0"
    "0x1"
    "0x2"
    "0x7fffffffffffffff"
    "0x8000000000000000"
    "0xffffffffffffffff"
    "0x123456789abcdef0"
    "0xfffffffffffffff0"))
(define f32-const-bits
  '("0x0" "0x3f800000" "0xbf800000" "0x3fa00000" "0xc0200000" "0x40700000" "0x41200000" "0xbe000000"))
(define f64-const-bits
  '("0x0"
    "0x3ff0000000000000"
    "0xbff0000000000000"
    "0x3ff4000000000000"
    "0xc004000000000000"
    "0x400e000000000000"
    "0x4024000000000000"
    "0xbfc0000000000000"))
(define pred-const-bits '("0x0" "0x1"))

(define (rand-ref xs)
  (list-ref xs (random (length xs))))

(define (render-child field node)
  ($xsmith_render-node (ast-child field node)))

(define (dtype->type dtype)
  (match dtype
    ['i8 i8-type]
    ['i16 i16-type]
    ['i32 i32-type]
    ['i64 i64-type]
    ['f32 f32-type]
    ['f64 f64-type]
    ['i1 pred-type]
    [_ (error 'simjit-xsmith (format "unknown dtype ~a" dtype))]))

(define (dtype->string dtype)
  (symbol->string dtype))

(define (op->symbol op)
  (cond
    [(symbol? op) op]
    [(string? op) (string->symbol op)]
    [else op]))

(define (shift-binary-op? op)
  (member (op->symbol op) shift-binary-ops))

(define (division-binary-op? op)
  (member (op->symbol op) division-binary-ops))

(define store-kinds '(aligned unaligned))
(define compare-ops '(lt gt le ge eq ne))
(define binary-ops '(add sub mul div udiv mod umod min max umin umax and or xor andnot shll shrl shra))
(define value-binary-ops '(add sub mul min max))
(define int-binary-ops '(div udiv mod umod umin umax and or xor andnot shll shrl shra))
(define division-binary-ops '(div udiv mod umod))
(define shift-binary-ops '(shll shrl shra rotl rotr))
(define unary-ops '(not neg abs lzcnt popcnt sqrt round floor ceil trunc))
(define shared-unary-ops '(neg abs))
(define int-unary-ops '(not lzcnt popcnt))
(define float-unary-ops '(sqrt round floor ceil trunc))
(define agg-ops '(add mul min max and or xor))
(define float-cast-modes '(float-to-int int-to-float float-widen float-narrow))
(define bitcast-modes '(to-int32 to-int64))

(define (fresh-value-type)
  (fresh-type-variable i8-type i16-type i32-type i64-type f32-type f64-type))

(define (fresh-int-type)
  (fresh-type-variable i8-type i16-type i32-type i64-type))

(define (fresh-float-type)
  (fresh-type-variable f32-type f64-type))

(define (type->dtype type)
  (cond
    [(can-unify? type i8-type) 'i8]
    [(can-unify? type i16-type) 'i16]
    [(can-unify? type i32-type) 'i32]
    [(can-unify? type i64-type) 'i64]
    [(can-unify? type f32-type) 'f32]
    [(can-unify? type f64-type) 'f64]
    [(can-unify? type pred-type) 'i1]
    [else (error 'simjit-xsmith "unable to map Xsmith type to Simjit dtype")]))

(define (node-dtype node)
  (type->dtype (concretize-type (att-value '_xsmith_type-full node) #:at-node node)))

(define (node-dtype-string node)
  (dtype->string (node-dtype node)))

(define (const-bits-for dtype slot)
  (define pool
    (match dtype
      ['i8 i8-const-bits]
      ['i16 i16-const-bits]
      ['i32 i32-const-bits]
      ['i64 i64-const-bits]
      ['f32 f32-const-bits]
      ['f64 f64-const-bits]
      [_ (error 'simjit-xsmith (format "no constant pool for dtype ~a" dtype))]))
  (list-ref pool (modulo slot (length pool))))

(define (agg-expr-type op)
  (if (eq? op 'mul) i64-type (fresh-int-type)))

(define (unary-operand-type op)
  (cond
    [(member op shared-unary-ops) (fresh-value-type)]
    [(member op int-unary-ops) (fresh-int-type)]
    [(member op float-unary-ops) (fresh-float-type)]
    [else (error 'simjit-xsmith (format "unknown unary op ~a" op))]))

(define (binary-operand-type op)
  (cond
    [(member op value-binary-ops) (fresh-value-type)]
    [(member op int-binary-ops) (fresh-int-type)]
    [else (error 'simjit-xsmith (format "unknown binary op ~a" op))]))

(add-to-grammar
 simjit-hir
 [Program #f ([roots : Root * = (+ 1 (random 4))])]
 [Root #f ()
       #:prop may-be-generated #f]
 [Expr #f ()
       #:prop may-be-generated #f]
 [PredExpr #f ()
           #:prop may-be-generated #f]

 [StoreRoot Root ([expr : Expr]
                  [kind = (rand-ref store-kinds)])
            #:prop choice-weight 4]
 [CondStoreRoot Root ([expr : Expr]
                      [cond : PredExpr]
                      [kind = (rand-ref store-kinds)])
                #:prop choice-weight 4]
 [ScatterRoot Root ([expr : Expr])
              #:prop choice-weight 1]
 [CondScatterRoot Root ([expr : Expr]
                        [cond : PredExpr])
                  #:prop choice-weight 1]
 [AggRoot Root ([op = (rand-ref agg-ops)]
                [expr : Expr])
          #:prop choice-weight 4]
 [CondAggRoot Root ([op = (rand-ref agg-ops)]
                    [expr : Expr]
                    [cond : PredExpr])
              #:prop choice-weight 4]
 [PredAggRoot Root ([pred : PredExpr]
                    [op = (rand-ref pred-agg-ops)])
              #:prop choice-weight 4]
 [CountifRoot Root ([pred : PredExpr])
              #:prop choice-weight 4]

 [PredConst PredExpr ([bits = (rand-ref pred-const-bits)])
            #:prop choice-weight 8]
 [PredNot PredExpr ([arg : PredExpr])
          #:prop choice-weight 8]
 [PredBinary PredExpr ([op = (rand-ref '(and or xor))]
                       [left : PredExpr]
                       [right : PredExpr])
             #:prop choice-weight 10]
 [Cmp PredExpr ([op = (rand-ref compare-ops)]
                [unsignedflag = (zero? (random 2))]
                [left : Expr]
                [right : Expr])
      #:prop choice-weight 12]

 [Const Expr ([slot = (random 8)])
        #:prop choice-weight 12]
 [Load Expr ([dtype = (rand-ref value-dtypes)]
             [kind = (rand-ref store-kinds)])
       #:prop choice-weight 14]
 [LoadSplat Expr ([dtype = (rand-ref value-dtypes)])
            #:prop choice-weight 10]
 [Gather Expr ([dtype = (rand-ref value-dtypes)])
         #:prop choice-weight 2]
 [Index Expr ([dtype = (rand-ref index-dtypes)])
        #:prop choice-weight 6]
 [Binary Expr ([op = (rand-ref binary-ops)]
               [left : Expr]
               [right : Expr])
         #:prop choice-weight 14]
 [Unary Expr ([op = (rand-ref unary-ops)]
              [arg : Expr])
        #:prop choice-weight 10]
 [Select Expr ([cond : PredExpr]
               [truthy : Expr]
               [falsy : Expr])
         #:prop choice-weight 10]
 [IntCast Expr ([kind = (rand-ref '(trunc sext zext))]
                [arg : Expr])
          #:prop choice-weight 6]
 [FloatCast Expr ([mode = (rand-ref float-cast-modes)]
                  [arg : Expr])
            #:prop choice-weight 6]
 [Bitcast Expr ([mode = (rand-ref bitcast-modes)]
                [arg : Expr])
          #:prop choice-weight 4])

(add-property
 simjit-hir
 type-info
 [Program
  [(fresh-type-variable)
   (lambda (n t)
     (unify! t program-type)
     (hash 'roots root-type))]]
 [StoreRoot
  [(fresh-type-variable)
   (lambda (n t)
     (unify! t root-type)
     (hash 'expr (fresh-value-type)))]]
 [CondStoreRoot
  [(fresh-type-variable)
   (lambda (n t)
     (unify! t root-type)
     (hash 'expr (fresh-value-type)
           'cond pred-type))]]
 [ScatterRoot
  [(fresh-type-variable)
   (lambda (n t)
     (unify! t root-type)
     (hash 'expr (fresh-value-type)))]]
 [CondScatterRoot
  [(fresh-type-variable)
   (lambda (n t)
     (unify! t root-type)
     (hash 'expr (fresh-value-type)
           'cond pred-type))]]
 [AggRoot
  [(fresh-type-variable)
   (lambda (n t)
     (unify! t root-type)
     (hash 'expr (fresh-value-type)))]]
 [CondAggRoot
  [(fresh-type-variable)
   (lambda (n t)
     (unify! t root-type)
     (hash 'expr (fresh-value-type)
           'cond pred-type))]]
 [PredAggRoot
  [(fresh-type-variable)
   (lambda (n t)
     (unify! t root-type)
     (hash 'pred pred-type))]]
 [CountifRoot
  [(fresh-type-variable)
   (lambda (n t)
     (unify! t root-type)
     (hash 'pred pred-type))]]

 [PredConst [pred-type (lambda (n t) (hash))]]
 [PredNot [pred-type (lambda (n t) (hash 'arg pred-type))]]
 [PredBinary [pred-type (lambda (n t) (hash 'left pred-type 'right pred-type))]]
 [Cmp
  [pred-type
   (lambda (n t)
     (define operand-type (fresh-value-type))
     (hash 'left operand-type
           'right operand-type))]]

 [Const
  [(fresh-type-variable)
   (lambda (n t)
     (unify! t (fresh-value-type))
     (hash))]]
 [Load
  [(fresh-type-variable)
   (lambda (n t)
     (unify! t (fresh-value-type))
     (hash))]]
 [LoadSplat
  [(fresh-type-variable)
   (lambda (n t)
     (unify! t (fresh-value-type))
     (hash))]]
 [Gather
  [(fresh-type-variable)
   (lambda (n t)
     (unify! t (fresh-value-type))
     (hash))]]
 [Index
  [(fresh-type-variable)
   (lambda (n t)
     (unify! t (fresh-type-variable i32-type i64-type))
     (hash))]]
 [Binary
  [(fresh-type-variable)
   (lambda (n t)
     (unify! t (fresh-value-type))
     (hash 'left (fresh-value-type)
           'right (fresh-value-type)))]]
 [Unary
  [(fresh-type-variable)
   (lambda (n t)
     (unify! t (fresh-value-type))
     (hash 'arg (fresh-value-type)))]]
 [Select
  [(fresh-type-variable)
   (lambda (n t)
     (unify! t (fresh-value-type))
     (hash 'cond pred-type
           'truthy (fresh-value-type)
           'falsy (fresh-value-type)))]]
 [IntCast
  [(fresh-type-variable)
   (lambda (n t)
     (unify! t (fresh-value-type))
     (hash 'arg (fresh-value-type)))]]
 [FloatCast
  [(fresh-type-variable)
   (lambda (n t)
     (unify! t (fresh-value-type))
     (hash 'arg (fresh-value-type)))]]
 [Bitcast
  [(fresh-type-variable)
   (lambda (n t)
     (unify! t (fresh-value-type))
     (hash 'arg (fresh-value-type)))]])

(define (section tag items)
  (if (null? items)
      (format "(~a)" tag)
      (format "(~a ~a)" tag (string-join items " "))))

(define (emit-program ir)
  (define next-arg 0)
  (define next-acc 0)
  (define next-step 0)
  (define arg-items '())
  (define acc-items '())
  (define step-items '())
  (define root-items '())

  (define (alloc-arg dtype kind)
    (define idx next-arg)
    (set! next-arg (add1 next-arg))
    (set! arg-items
          (append arg-items
                  (list (format "(arg ~a ~a ~a)" idx dtype kind))))
    idx)

  (define (alloc-step text)
    (define idx next-step)
    (set! next-step (add1 next-step))
    (set! step-items
          (append step-items (list (format "(step ~a ~a)" idx text))))
    idx)

  (define (emit-ref idx)
    (format "(step ~a)" idx))

  (define (emit-expr expr)
    (match expr
      [(const-ir dtype bits)
       (alloc-step (format "const ~a \"~a\"" dtype bits))]
      [(load-ir dtype kind #f)
       (define arg-id (alloc-arg dtype "src-arr"))
       (alloc-step (format "load ~a (arg ~a) ~a" dtype arg-id kind))]
      [(load-ir dtype _ #t)
       (define arg-id (alloc-arg dtype "src-const"))
       (alloc-step (format "load-splat ~a (arg ~a)" dtype arg-id))]
      [(gather-ir dtype idx)
       (define idx-id (emit-expr idx))
       (define arg-id (alloc-arg dtype "src-gather-arr"))
       (alloc-step (format "gather ~a ~a (arg ~a)" dtype (emit-ref idx-id) arg-id))]
      [(index-ir dtype)
       (alloc-step (format "index ~a" dtype))]
      [(binary-ir dtype op left right)
       (define left-id (emit-expr left))
       (define right-id (emit-expr right))
       (alloc-step (string-append
                    (format "binary ~a ~a ~a ~a"
                            dtype
                            op
                            (emit-ref left-id)
                            (emit-ref right-id))
                    (if (shift-binary-op? op) " shift-wraparound" "")
                    (if (division-binary-op? op) " safe-division" "")))]
      [(unary-ir dtype op arg)
       (define arg-id (emit-expr arg))
       (alloc-step (format "unary ~a ~a ~a" dtype op (emit-ref arg-id)))]
      [(cmp-ir _ op left right unsigned?)
       (define left-id (emit-expr left))
       (define right-id (emit-expr right))
       (alloc-step
        (string-append
         (format "cmp i1 ~a ~a ~a" op (emit-ref left-id) (emit-ref right-id))
         (if unsigned? " t" "")))]
      [(pred-binary-ir op left right)
       (define left-id (emit-expr left))
       (define right-id (emit-expr right))
       (alloc-step (format "predicate-binary i1 ~a ~a ~a"
                           op
                           (emit-ref left-id)
                           (emit-ref right-id)))]
      [(pred-not-ir arg)
       (define arg-id (emit-expr arg))
       (alloc-step (format "predicate-not i1 ~a" (emit-ref arg-id)))]
      [(select-ir dtype cond truthy falsy)
       (define cond-id (emit-expr cond))
       (define truthy-id (emit-expr truthy))
       (define falsy-id (emit-expr falsy))
       (alloc-step (format "select ~a ~a ~a ~a"
                           dtype
                           (emit-ref cond-id)
                           (emit-ref truthy-id)
                           (emit-ref falsy-id)))]
      [(int-cast-ir dtype kind arg)
       (define arg-id (emit-expr arg))
       (alloc-step (format "int-cast ~a ~a ~a" dtype kind (emit-ref arg-id)))]
      [(float-cast-ir dtype arg unsigned?)
       (define arg-id (emit-expr arg))
       (alloc-step
        (string-append
         (format "float-cast ~a ~a" dtype (emit-ref arg-id))
         (if unsigned? " t" "")))]
      [(bitcast-ir dtype arg)
       (define arg-id (emit-expr arg))
       (alloc-step (format "bitcast ~a ~a" dtype (emit-ref arg-id)))]))

  (define (emit-root root)
    (match root
      [(store-root-ir dtype expr kind #f)
       (define expr-id (emit-expr expr))
       (define dst-id (alloc-arg dtype "dst-arr"))
       (define step-id
         (alloc-step (format "store ~a ~a (arg ~a) ~a"
                             dtype
                             (emit-ref expr-id)
                             dst-id
                             kind)))
       (set! root-items (append root-items (list (emit-ref step-id))))]
      [(store-root-ir dtype expr kind cond)
       (define expr-id (emit-expr expr))
       (define cond-id (emit-expr cond))
       (define dst-id (alloc-arg dtype "dst-arr"))
       (define step-id
         (alloc-step (format "store ~a ~a (arg ~a) ~a ~a"
                             dtype
                             (emit-ref expr-id)
                             dst-id
                             kind
                             (emit-ref cond-id))))
       (set! root-items (append root-items (list (emit-ref step-id))))]
      [(scatter-root-ir dtype expr idx #f)
       (define expr-id (emit-expr expr))
       (define idx-id (emit-expr idx))
       (define dst-id (alloc-arg dtype "dst-arr"))
       (define step-id
         (alloc-step (format "scatter ~a ~a ~a (arg ~a)"
                             dtype
                             (emit-ref expr-id)
                             (emit-ref idx-id)
                             dst-id)))
       (set! root-items (append root-items (list (emit-ref step-id))))]
      [(scatter-root-ir dtype expr idx cond)
       (define expr-id (emit-expr expr))
       (define idx-id (emit-expr idx))
       (define cond-id (emit-expr cond))
       (define dst-id (alloc-arg dtype "dst-arr"))
       (define step-id
         (alloc-step (format "scatter ~a ~a ~a (arg ~a) ~a"
                             dtype
                             (emit-ref expr-id)
                             (emit-ref idx-id)
                             dst-id
                             (emit-ref cond-id))))
       (set! root-items (append root-items (list (emit-ref step-id))))]
      [(agg-root-ir dtype op expr #f)
       (define expr-id (emit-expr expr))
       (define dst-id (alloc-arg dtype "dst-scalar"))
       (define acc-id next-acc)
       (set! next-acc (add1 next-acc))
       (define step-id
         (alloc-step (format "acc-arith-bin ~a ~a ~a (acc ~a)"
                             dtype
                             op
                             (emit-ref expr-id)
                             acc-id)))
       (set! acc-items
             (append acc-items
                     (list (format "(acc ~a ~a (arg ~a) (step ~a))"
                                   acc-id
                                   dtype
                                   dst-id
                                   step-id))))
       (set! root-items (append root-items (list (emit-ref step-id))))]
      [(agg-root-ir dtype op expr cond)
       (define expr-id (emit-expr expr))
       (define cond-id (emit-expr cond))
       (define dst-id (alloc-arg dtype "dst-scalar"))
       (define acc-id next-acc)
       (set! next-acc (add1 next-acc))
       (define step-id
         (alloc-step (format "acc-arith-bin ~a ~a ~a (acc ~a) ~a"
                             dtype
                             op
                             (emit-ref expr-id)
                             acc-id
                             (emit-ref cond-id))))
       (set! acc-items
             (append acc-items
                     (list (format "(acc ~a ~a (arg ~a) (step ~a))"
                                   acc-id
                                   dtype
                                   dst-id
                                   step-id))))
       (set! root-items (append root-items (list (emit-ref step-id))))]
      [(pred-agg-root-ir op pred)
       (define pred-id (emit-expr pred))
       (define dst-id (alloc-arg "i1" "dst-scalar"))
       (define acc-id next-acc)
       (set! next-acc (add1 next-acc))
       (define step-id
         (alloc-step (format "acc-predicate-bin i1 ~a ~a (acc ~a)"
                             op
                             (emit-ref pred-id)
                             acc-id)))
       (set! acc-items
             (append acc-items
                     (list (format "(acc ~a i1 (arg ~a) (step ~a))"
                                   acc-id
                                   dst-id
                                   step-id))))
       (set! root-items (append root-items (list (emit-ref step-id))))]
      [(countif-root-ir pred)
       (define pred-id (emit-expr pred))
       (define dst-id (alloc-arg "i64" "dst-scalar"))
       (define acc-id next-acc)
       (set! next-acc (add1 next-acc))
       (define step-id
         (alloc-step (format "countif i64 ~a (acc ~a)"
                             (emit-ref pred-id)
                             acc-id)))
       (set! acc-items
             (append acc-items
                     (list (format "(acc ~a i64 (arg ~a) (step ~a))"
                                   acc-id
                                   dst-id
                                   step-id))))
       (set! root-items (append root-items (list (emit-ref step-id))))]))

  (for ([root (program-ir-roots ir)])
    (emit-root root))

  (string-append
   "(func "
   (section "args" arg-items)
   (if (null? acc-items)
       ""
       (string-append " " (section "accs" acc-items)))
   " "
   (section "steps" step-items)
   " "
   (section "roots" root-items)
   ")"))

(define (expr-ir-dtype expr)
  (match expr
    [(const-ir dtype _) dtype]
    [(load-ir dtype _ _) dtype]
    [(gather-ir dtype _) dtype]
    [(index-ir dtype) dtype]
    [(binary-ir dtype _ _ _) dtype]
    [(unary-ir dtype _ _) dtype]
    [(select-ir dtype _ _ _) dtype]
    [(int-cast-ir dtype _ _) dtype]
    [(float-cast-ir dtype _ _) dtype]
    [(bitcast-ir dtype _) dtype]
    [_ (error 'simjit-xsmith "expected value expression IR")]))

(define (normalize-binary-dtype op preferred)
  (cond
    [(member op int-binary-ops)
     (if (member preferred int-dtypes) preferred 'i32)]
    [else
     (if (member preferred value-dtypes) preferred 'i32)]))

(define (normalize-unary-dtype op preferred)
  (cond
    [(member op shared-unary-ops)
     (if (member preferred value-dtypes) preferred 'i32)]
    [(member op int-unary-ops)
     (if (member preferred int-dtypes) preferred 'i32)]
    [(member op float-unary-ops)
     (if (member preferred float-dtypes) preferred 'f32)]
    [else (error 'simjit-xsmith (format "unknown unary op ~a" op))]))

(define (default-float-cast-dtype node)
  (match (ast-child 'mode node)
    ['float-to-int
     (if (member 'f64 (expr-possible-dtypes (ast-child 'arg node))) 'i64 'i32)]
    ['int-to-float
     (if (member 'i64 (expr-possible-dtypes (ast-child 'arg node))) 'f64 'f32)]
    ['float-widen 'f64]
    ['float-narrow 'f32]
    [_ (error 'simjit-xsmith "unknown float cast mode")]))

(define (default-bitcast-dtype node)
  (match (ast-child 'mode node)
    ['to-int32 'i32]
    ['to-float32 'f32]
    ['to-int64 'i64]
    ['to-float64 'f64]
    [_ (error 'simjit-xsmith "unknown bitcast mode")]))

(define (intersect-dtypes first-list . rest-lists)
  (foldl (lambda (next current)
           (filter (lambda (dtype) (member dtype next)) current))
         first-list
         rest-lists))

(define (dtype-choice-weight dtype)
  (match dtype
    [(or 'i8 'i16) 1]
    [_ 3]))

(define (weighted-rand-ref weighted-items)
  (define total (apply + (map cdr weighted-items)))
  (define target (random total))
  (let loop ([remaining weighted-items]
             [seen 0])
    (match remaining
      [(list) (car (car weighted-items))]
      [(cons (cons item weight) rest)
       (define next (+ seen weight))
       (if (< target next)
           item
           (loop rest next))])))

(define (rand-dtype candidates)
  (weighted-rand-ref
   (map (lambda (dtype) (cons dtype (dtype-choice-weight dtype))) candidates)))

(define (choose-dtype candidates [preferred #f])
  (cond
    [(and preferred (member preferred candidates)) preferred]
    [(null? candidates) (or preferred 'i32)]
    [else (rand-dtype candidates)]))

(define (fallback-expr-ir dtype node)
  (define slot
    (if (and (ast-node? node) (member (ast-node-type node) '(Const)))
        (ast-child 'slot node)
        0))
  (const-ir (dtype->string dtype) (const-bits-for dtype slot)))

(define (expr-possible-dtypes node)
  (match (ast-node-type node)
    ['Const value-dtypes]
    ['Load value-dtypes]
    ['LoadSplat value-dtypes]
    ['Gather value-dtypes]
    ['Index index-dtypes]
    ['Binary
     (intersect-dtypes
      (if (member (ast-child 'op node) int-binary-ops) int-dtypes value-dtypes)
      (expr-possible-dtypes (ast-child 'left node))
      (expr-possible-dtypes (ast-child 'right node)))]
    ['Unary
     (intersect-dtypes
      (cond
        [(member (ast-child 'op node) shared-unary-ops) value-dtypes]
        [(member (ast-child 'op node) int-unary-ops) int-dtypes]
        [(member (ast-child 'op node) float-unary-ops) float-dtypes]
        [else (error 'simjit-xsmith "unknown unary op")])
      (expr-possible-dtypes (ast-child 'arg node)))]
    ['Select
     (intersect-dtypes
      (expr-possible-dtypes (ast-child 'truthy node))
      (expr-possible-dtypes (ast-child 'falsy node)))]
    ['IntCast (if (eq? (ast-child 'kind node) 'trunc) '(i32) '(i64))]
    ['FloatCast
     (match (ast-child 'mode node)
       ['float-to-int int-dtypes]
       ['int-to-float float-dtypes]
       ['float-widen '(f64)]
       ['float-narrow '(f32)]
       [_ (error 'simjit-xsmith "unknown float cast mode")])]
    ['Bitcast (list (default-bitcast-dtype node))]
    [_ (error 'simjit-xsmith (format "no dtype analysis for node kind ~a" (ast-node-type node)))]))

(define (cmp-possible-dtypes node)
  (intersect-dtypes
   (if (ast-child 'unsignedflag node) int-dtypes value-dtypes)
   (expr-possible-dtypes (ast-child 'left node))
   (expr-possible-dtypes (ast-child 'right node))))

(define render-pred-node #f)
(define render-expr-node #f)
(define render-root-node #f)

(set! render-pred-node
  (lambda (node)
    (match (ast-node-type node)
      ['PredConst (const-ir "i1" (ast-child 'bits node))]
      ['PredNot (pred-not-ir (render-pred-node (ast-child 'arg node)))]
      ['PredBinary
       (pred-binary-ir (ast-child 'op node)
                       (render-pred-node (ast-child 'left node))
                       (render-pred-node (ast-child 'right node)))]
      ['Cmp
       (define dtype (choose-dtype (cmp-possible-dtypes node)))
       (cmp-ir (dtype->string dtype)
               (ast-child 'op node)
               (render-expr-node (ast-child 'left node) dtype)
               (render-expr-node (ast-child 'right node) dtype)
               (ast-child 'unsignedflag node))]
      [_ (error 'simjit-xsmith (format "unsupported predicate node ~a" (ast-node-type node)))])))

(set! render-expr-node
  (lambda (node [expected-dtype #f])
    (if (and expected-dtype
             (not (member expected-dtype (expr-possible-dtypes node))))
        (fallback-expr-ir expected-dtype node)
        (match (ast-node-type node)
          ['Const
           (define dtype (choose-dtype (expr-possible-dtypes node) expected-dtype))
           (const-ir (dtype->string dtype) (const-bits-for dtype (ast-child 'slot node)))]
          ['Load
           (define dtype (choose-dtype (expr-possible-dtypes node) expected-dtype))
           (load-ir (dtype->string dtype) (ast-child 'kind node) #f)]
          ['LoadSplat
           (define dtype (choose-dtype (expr-possible-dtypes node) expected-dtype))
           (load-ir (dtype->string dtype) 'aligned #t)]
          ['Gather
           (define dtype (choose-dtype (expr-possible-dtypes node) expected-dtype))
           (gather-ir (dtype->string dtype) (index-ir "i32"))]
          ['Index
           (define dtype (choose-dtype (expr-possible-dtypes node) expected-dtype))
           (index-ir (dtype->string dtype))]
          ['Binary
           (define dtype (choose-dtype (expr-possible-dtypes node) expected-dtype))
           (binary-ir (dtype->string dtype)
                      (ast-child 'op node)
                      (render-expr-node (ast-child 'left node) dtype)
                      (render-expr-node (ast-child 'right node) dtype))]
          ['Unary
           (define dtype (choose-dtype (expr-possible-dtypes node) expected-dtype))
           (unary-ir (dtype->string dtype)
                     (ast-child 'op node)
                     (render-expr-node (ast-child 'arg node) dtype))]
          ['Select
           (define dtype (choose-dtype (expr-possible-dtypes node) expected-dtype))
           (select-ir (dtype->string dtype)
                      (render-pred-node (ast-child 'cond node))
                      (render-expr-node (ast-child 'truthy node) dtype)
                      (render-expr-node (ast-child 'falsy node) dtype))]
          ['IntCast
           (define kind (ast-child 'kind node))
           (define dtype (if (eq? kind 'trunc) 'i32 'i64))
           (define src-dtype (if (eq? kind 'trunc) 'i64 'i32))
           (int-cast-ir (dtype->string dtype)
                        kind
                        (render-expr-node (ast-child 'arg node) src-dtype))]
          ['FloatCast
           (define mode (ast-child 'mode node))
           (define dtype (choose-dtype (expr-possible-dtypes node) expected-dtype))
           (define src-dtype
             (match mode
               ['float-to-int (if (eq? dtype 'i64) 'f64 'f32)]
               ['int-to-float (if (eq? dtype 'f64) 'i64 'i32)]
               ['float-widen 'f32]
               ['float-narrow 'f64]
               [_ (error 'simjit-xsmith "unknown float cast mode")]))
           (float-cast-ir (dtype->string dtype)
                          (render-expr-node (ast-child 'arg node) src-dtype)
                          #f)]
          ['Bitcast
           (define mode (ast-child 'mode node))
           (define-values (dtype src-dtype)
             (match mode
               ['to-int32 (values 'i32 'f32)]
               ['to-float32 (values 'f32 'i32)]
               ['to-int64 (values 'i64 'f64)]
               ['to-float64 (values 'f64 'i64)]
               [_ (error 'simjit-xsmith "unknown bitcast mode")]))
           (bitcast-ir (dtype->string dtype)
                       (render-expr-node (ast-child 'arg node) src-dtype))]
          [_ (error 'simjit-xsmith (format "unsupported expression node ~a" (ast-node-type node)))]))))

(set! render-root-node
  (lambda (node)
    (match (ast-node-type node)
      ['StoreRoot
       (define expr-ir
         (render-expr-node (ast-child 'expr node)
                           (choose-dtype (expr-possible-dtypes (ast-child 'expr node)))))
       (store-root-ir (expr-ir-dtype expr-ir) expr-ir (ast-child 'kind node) #f)]
      ['CondStoreRoot
       (define expr-ir
         (render-expr-node (ast-child 'expr node)
                           (choose-dtype (expr-possible-dtypes (ast-child 'expr node)))))
       (store-root-ir (expr-ir-dtype expr-ir)
                      expr-ir
                      (ast-child 'kind node)
                      (render-pred-node (ast-child 'cond node)))]
      ['ScatterRoot
       (define expr-ir
         (render-expr-node (ast-child 'expr node)
                           (choose-dtype (expr-possible-dtypes (ast-child 'expr node)))))
       (scatter-root-ir (expr-ir-dtype expr-ir) expr-ir (index-ir "i32") #f)]
      ['CondScatterRoot
       (define expr-ir
         (render-expr-node (ast-child 'expr node)
                           (choose-dtype (expr-possible-dtypes (ast-child 'expr node)))))
       (scatter-root-ir (expr-ir-dtype expr-ir)
                        expr-ir
                        (index-ir "i32")
                        (render-pred-node (ast-child 'cond node)))]
      ['AggRoot
       (define dtype
         (choose-dtype
          (intersect-dtypes
           (expr-possible-dtypes (ast-child 'expr node))
           (if (eq? (ast-child 'op node) 'mul) '(i64) int-dtypes))))
       (agg-root-ir (dtype->string dtype)
                    (ast-child 'op node)
                    (render-expr-node (ast-child 'expr node) dtype)
                    #f)]
      ['CondAggRoot
       (define dtype
         (choose-dtype
          (intersect-dtypes
           (expr-possible-dtypes (ast-child 'expr node))
           (if (eq? (ast-child 'op node) 'mul) '(i64) int-dtypes))))
       (agg-root-ir (dtype->string dtype)
                    (ast-child 'op node)
                    (render-expr-node (ast-child 'expr node) dtype)
                    (render-pred-node (ast-child 'cond node)))]
      ['PredAggRoot
       (pred-agg-root-ir (ast-child 'op node)
                         (render-pred-node (ast-child 'pred node)))]
      ['CountifRoot
       (countif-root-ir (render-pred-node (ast-child 'pred node)))]
      [_ (error 'simjit-xsmith (format "unsupported root node ~a" (ast-node-type node)))])))

(add-property
 simjit-hir
 render-node-info
 [Program
  (lambda (n)
    (program-ir (map render-root-node
                     (ast-children (ast-child 'roots n)))))]

 [StoreRoot
  (lambda (n)
    (store-root-ir (node-dtype-string (ast-child 'expr n))
                   (render-child 'expr n)
                   (ast-child 'kind n)
                   #f))]
 [CondStoreRoot
  (lambda (n)
    (store-root-ir (node-dtype-string (ast-child 'expr n))
                   (render-child 'expr n)
                   (ast-child 'kind n)
                   (render-child 'cond n)))]
 [ScatterRoot
  (lambda (n)
    (scatter-root-ir (node-dtype-string (ast-child 'expr n))
                     (render-child 'expr n)
                     (index-ir "i32")
                     #f))]
 [CondScatterRoot
  (lambda (n)
    (scatter-root-ir (node-dtype-string (ast-child 'expr n))
                     (render-child 'expr n)
                     (index-ir "i32")
                     (render-child 'cond n)))]
 [AggRoot
  (lambda (n)
    (agg-root-ir (node-dtype-string (ast-child 'expr n))
                 (ast-child 'op n)
                 (render-child 'expr n)
                 #f))]
 [CondAggRoot
  (lambda (n)
    (agg-root-ir (node-dtype-string (ast-child 'expr n))
                 (ast-child 'op n)
                 (render-child 'expr n)
                 (render-child 'cond n)))]
 [PredAggRoot (lambda (n) (pred-agg-root-ir (ast-child 'op n) (render-child 'pred n)))]
 [CountifRoot (lambda (n) (countif-root-ir (render-child 'pred n)))]

 [PredConst (lambda (n) (const-ir "i1" (ast-child 'bits n)))]
 [PredNot (lambda (n) (pred-not-ir (render-child 'arg n)))]
 [PredBinary
  (lambda (n) (pred-binary-ir (ast-child 'op n) (render-child 'left n) (render-child 'right n)))]
 [Cmp
  (lambda (n)
    (cmp-ir (node-dtype-string (ast-child 'left n))
            (ast-child 'op n)
            (render-child 'left n)
            (render-child 'right n)
            (ast-child 'unsignedflag n)))]

 [Const
  (lambda (n)
    (define dtype (node-dtype n))
    (const-ir (dtype->string dtype) (const-bits-for dtype (ast-child 'slot n))))]
 [Load
  (lambda (n)
    (load-ir (node-dtype-string n)
             (ast-child 'kind n)
             #f))]
 [LoadSplat (lambda (n) (load-ir (node-dtype-string n) 'aligned #t))]
 [Gather (lambda (n) (gather-ir (node-dtype-string n) (index-ir "i32")))]
 [Index (lambda (n) (index-ir (node-dtype-string n)))]
 [Binary
  (lambda (n)
    (binary-ir (node-dtype-string n)
               (ast-child 'op n)
               (render-child 'left n)
               (render-child 'right n)))]
 [Unary
  (lambda (n)
    (unary-ir (node-dtype-string n)
              (ast-child 'op n)
              (render-child 'arg n)))]
 [Select
  (lambda (n)
    (select-ir (node-dtype-string n)
               (render-child 'cond n)
               (render-child 'truthy n)
               (render-child 'falsy n)))]
 [IntCast
  (lambda (n)
    (int-cast-ir (node-dtype-string n)
                 (ast-child 'kind n)
                 (render-child 'arg n)))]
 [FloatCast
  (lambda (n)
    (float-cast-ir (node-dtype-string n)
                   (render-child 'arg n)
                   #f))]
 [Bitcast
  (lambda (n)
    (bitcast-ir (node-dtype-string n)
                (render-child 'arg n)))])

(define-xsmith-interface-functions
  [simjit-hir]
  #:program-node Program
  #:type-thunks (list (lambda () program-type)
                      (lambda () root-type)
                      (lambda () pred-type)
                      (lambda () i8-type)
                      (lambda () i16-type)
                      (lambda () i32-type)
                      (lambda () i64-type)
                      (lambda () f32-type)
                      (lambda () f64-type))
  #:comment-wrap (lambda (lines) "")
  #:format-render emit-program
  #:default-max-depth 6
  #:features ([gather #f]
              [scatter #f]
              [pack #f]
              [permute #f]
              [sum128 #f]
              [fpclass #f]))

(define (extract-serialized-program output)
  (define match-posns (regexp-match-positions #rx"\\(func" output))
  (if match-posns
      (string-trim (substring output (caar match-posns)))
      (let ([trimmed (string-trim output)])
        (if (string-prefix? trimmed "(")
            trimmed
            (error 'simjit-xsmith
                   (format "generator output does not contain a serialized '(func ...)' program; raw=~s"
                           trimmed))))))

(define (generate-raw-program #:seed seed
                              #:max-depth max-depth
                              #:gather gather?
                              #:scatter scatter?
                              #:pack pack?
                              #:permute permute?
                              #:sum128 sum128?
                              #:fpclass fpclass?)
  (extract-serialized-program
   (with-output-to-string
     (lambda ()
       (simjit-hir-generate
        #:seed seed
        #:max-depth max-depth
        #:with-gather gather?
        #:with-scatter scatter?
        #:with-pack pack?
        #:with-permute permute?
        #:with-sum128 sum128?
        #:with-fpclass fpclass?)))))

(module+ main
  (define seed #f)
  (define count 1)
  (define start-index 0)
  (define max-depth 6)
  (define profile-name "default")
  (define jsonl? #f)
  (define gather? #f)
  (define scatter? #f)
  (define pack? #f)
  (define permute? #f)
  (define sum128? #f)
  (define fpclass? #f)

  (define (parse-bool who v)
    (cond
      [(member v '("true" "1" "#t")) #t]
      [(member v '("false" "0" "#f")) #f]
      [else (error who (format "expected boolean, got ~a" v))]))

  (command-line
   #:program "simjit-xsmith"
   #:once-each
   [("--seed") s "Base seed" (set! seed (string->number s))]
   [("--count") c "Number of programs to emit" (set! count (string->number c))]
   [("--start-index") i "Starting program index" (set! start-index (string->number i))]
   [("--max-depth") d "Xsmith max depth" (set! max-depth (string->number d))]
   [("--profile-name") p "Profile label for JSONL output" (set! profile-name p)]
   [("--jsonl") "Emit one JSON record per program" (set! jsonl? #t)]
   [("--with-gather") v "Enable gather feature" (set! gather? (parse-bool '--with-gather v))]
   [("--with-scatter") v "Enable scatter feature" (set! scatter? (parse-bool '--with-scatter v))]
   [("--with-pack") v "Enable pack feature" (set! pack? (parse-bool '--with-pack v))]
   [("--with-permute") v "Enable permute feature" (set! permute? (parse-bool '--with-permute v))]
   [("--with-sum128") v "Enable sum128 feature" (set! sum128? (parse-bool '--with-sum128 v))]
   [("--with-fpclass") v "Enable fpclass feature" (set! fpclass? (parse-bool '--with-fpclass v))])

  (unless (and (integer? seed) (exact? seed))
    (error 'simjit-xsmith "--seed must be an exact integer"))
  (unless (and (integer? count) (exact? count) (>= count 1))
    (error 'simjit-xsmith "--count must be a positive integer"))
  (unless (and (integer? start-index) (exact? start-index) (>= start-index 0))
    (error 'simjit-xsmith "--start-index must be a non-negative integer"))
  (unless (and (integer? max-depth) (exact? max-depth) (>= max-depth 1))
    (error 'simjit-xsmith "--max-depth must be a positive integer"))
  (unless (or jsonl? (= count 1))
    (error 'simjit-xsmith "use --jsonl when --count is greater than 1"))

  (for ([program-index (in-range start-index (+ start-index count))])
    (define program-seed (+ seed program-index))
    (with-handlers
        ([exn:fail?
          (lambda (e)
            (eprintf "program_seed=~a program_index=~a: ~a\n"
                     program-seed
                     program-index
                     (exn-message e))
            (raise e))])
      (define serialized
        (generate-raw-program
         #:seed program-seed
         #:max-depth max-depth
         #:gather gather?
         #:scatter scatter?
         #:pack pack?
         #:permute permute?
         #:sum128 sum128?
         #:fpclass fpclass?))
      (if jsonl?
          (begin
            (write-json
             (hash 'base_seed seed
                   'program_index program-index
                   'program_seed program-seed
                   'profile profile-name
                   'serialized serialized))
            (newline))
          (displayln serialized)))))
