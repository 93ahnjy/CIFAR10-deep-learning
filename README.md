# CIFAR10-deep-learning
딥러닝 과목 때 프로젝트로 만들어본 Convnet<br>
성능은 79% 나왔음.<br><br>

## 1. 네트워크 구조
| Layer | Output shape |
|---|:---|
Conv2d	| (None, 32, 32, 32)
Activation (Elu) |	(None, 32, 32, 32)
Conv2d|	(None, 32, 32, 48)
Activation (Elu)|	(None, 32, 32, 48)
Max pool	|(None, 16, 16, 48)

|---|:---|
Conv2d	|(None, 16, 16, 48)
Activation (Elu)|	(None, 16, 16, 48)
Conv2d|	(None, 16, 16, 64)
Activation (Elu)|	(None, 16, 16, 64)
Max pool	|(None, 8, 8, 64)
Drop out 0.5	|(None, 8, 8, 64)
Conv2d	|(None, 8, 8, 128)
Activation (Elu)|	(None, 8, 8, 128)
Conv2d	|(None, 8, 8, 64)
Activation (Elu)|	(None, 8, 8, 64)
Max pool	|(None, 4, 4, 64)
Drop out 0.6	|(None, 4, 4, 64)
Flatten	|(None, 1024)
Dense|	(None, 10)
Softmax|	(None, 10)
