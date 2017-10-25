Deep Spell Optimization Module
==============================

Class Architecture
------------------

```
(deepspell)
                     /-----------------------\
                     | DSModelBase           |
                     | + graph: tf.Graph     |
                     | + finish_init_base()  |
                     \----A------------A-----/
                          '            | __init__
                          '       /---------\
                          '       | <Model> |
                          '       \----A----/
                          '            |
- - - - - - - - - - - - - ' - - - - - -|- - - - - - - - -
(deepspell_optimization)  '            |
                          '            |
        /-----------------'---------\  |
        | DSModelOptimizerMixin     |  |
        | + finish_init_base() {}   |  |
        | + finish_init_optimizer() |  |
        | + _train(...)             |  |
        \-----------------A---------/  |
                          | __init__#1 | __init__#0
                       /------------------\
                       | <ModelOptimizer> |
                       \------------------/
```
