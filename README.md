# CIFAR10-deep-learning
딥러닝 과목 때 프로젝트로 만들어본 Convnet<br>
성능은 최고 79% 나왔음.<br><br>
참고로 ipynb파일은 올리지 않았으니 이걸로 돌려봤자 안됨 ㅇㅇ.<br><br>

## 1. 네트워크 구조
| Layer | Output shape |
|---|:---|
Conv2d	| (None, 32, 32, 32)
Activation (Elu) |	(None, 32, 32, 32)
Conv2d|	(None, 32, 32, 48)
Activation (Elu)|	(None, 32, 32, 48)
Max pool	|(None, 16, 16, 48)
Conv2d	|(None, 16, 16, 48)
Activation (Elu)|	(None, 16, 16, 48)
Conv2d|	(None, 16, 16, 64)
Activation (Elu)|	(None, 16, 16, 64)
Max pool	|(None, 8, 8, 64)
Drop out (0.5)	|(None, 8, 8, 64)
Conv2d	|(None, 8, 8, 128)
Activation (Elu)|	(None, 8, 8, 128)
Conv2d	|(None, 8, 8, 64)
Activation (Elu)|	(None, 8, 8, 64)
Max pool	|(None, 4, 4, 64)
Drop out (0.6)	|(None, 4, 4, 64)
Flatten	|(None, 1024)
Dense|	(None, 10)
Softmax|	(None, 10)
<br>
<br>

## 2. 사용 환경
### * Google colab
<br>

![google colab](https://user-images.githubusercontent.com/43025974/50833588-06ffe800-1395-11e9-82b9-c211cd2a1286.png)

<br>
<br>

## 3. 사용된 기법
&nbsp;&nbsp;**1. Data augmentation** <br>
&nbsp;&nbsp;&nbsp;&nbsp;- image flip (right, left), rotation 등을 사용. 근데 나중에 들어보니 rotation을 빼고 augmentation을 사용 시 더 효과가 좋았다고 함.<br><br>

&nbsp;&nbsp;**2. ELU(Exponential linear unit) 사용**  <br>
&nbsp;&nbsp;&nbsp;&nbsp;- ELU가 cifar에서 괜찮은 성능을 나타낸다고 해서 사용. relu보다 일단 오르기는 했음. 하지만 아직도 애매함.<br><br>

&nbsp;&nbsp;**3. Dropout 사용**  <br>
&nbsp;&nbsp;&nbsp;&nbsp;- 3개 이상 부터는 효과는 좋을 텐데 5분내로 끝내기가 쉽지 않았다. 그래도 없는 것 보단 나았음.<br><br>

&nbsp;&nbsp;**4. weight regularization 사용**  <br>
&nbsp;&nbsp;&nbsp;&nbsp;- 어떤 weight의 성분이 급격히 커지면 억제하는 역할을 한다. L2 regularizer 사용. 그다지 효과가 있어보이지는 않았다.<br><br>
  ```python
  loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=Y, logits=logits)) +\
                    0.001*tf.nn.l2_loss(W_conv1) +\
                    0.001*tf.nn.l2_loss(W_conv2) +\
                    0.001*tf.nn.l2_loss(W_conv3) +\
                    0.001*tf.nn.l2_loss(W_conv4) +\
                    0.001*tf.nn.l2_loss(W_conv5) +\
                    0.001*tf.nn.l2_loss(W_conv6) +\
                    0.001*tf.nn.l2_loss(W_fc1)        
  ```

&nbsp;&nbsp;**5. batch normalization 사용 조절**  
&nbsp;&nbsp;&nbsp;&nbsp;- ELU가 batch normalization 역할을 한다고 하며, 또 dropout과 비슷한 성질이라 같이 쓸 필요가 없다고 한다. 확실히 줄여도 성능이 떨어지지 않기는 했고 훨씬 .<br><br>
