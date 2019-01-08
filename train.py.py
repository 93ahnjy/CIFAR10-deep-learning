import tensorflow as tf
import numpy as np
import math
import time
import os
import matplotlib.pyplot as plt
from CIFAR_10_augment import CIFAR_10
from google.colab import files
 

###################

 
# parameters
 

##################

tf.reset_default_graph()

epoch = 53
batch_size = 512
train_size = 30000

 

X = tf.placeholder(tf.float32, shape=[None, 32, 32, 3])
Y = tf.placeholder(tf.float32, shape=[None, 10])

keep_prob1 = tf.placeholder(tf.float32)
keep_prob2 = tf.placeholder(tf.float32)
keep_prob3 = tf.placeholder(tf.float32)
keep_prob4 = tf.placeholder(tf.float32)

dataset = CIFAR_10()

  

########

##  [data, label] = dataset.next('train', batch_size)

## -> train_data, train_label is returned by number of batch_size

##  [data, label] = dataset.next('train', batch_size)

## -> validation_data, validation_label is returned by number of batch_size

#######################################################

## make model

## simple example for CIFAR_10 clacification


 

# layer 1, 2
# W_conv1 = tf.get_variable("W_conv1", shape=[3, 3, 3, 32], initializer=tf.contrib.layers.xavier_initializer())
W_conv1 = tf.Variable(tf.random_normal(shape=[3, 3, 3, 32], stddev=2e-2))
L1 = tf.nn.conv2d(X, W_conv1, strides=[1, 1, 1, 1], padding='SAME')
L1 = tf.nn.elu(L1)

W_conv2 = tf.Variable(tf.random_normal(shape=[3, 3, 32, 48], stddev=2e-2))
L2 = tf.nn.conv2d(L1, W_conv2, strides=[1, 1, 1, 1], padding='SAME')
L2 = tf.nn.elu(L2)
L2 = tf.nn.max_pool(L2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')




# layer 3, 4
W_conv3 = tf.Variable(tf.random_normal(shape=[3, 3, 48, 48], stddev=2e-2))
L3 = tf.nn.conv2d(L2, W_conv3, strides=[1, 1, 1, 1], padding='SAME')
L3 = tf.nn.elu(L3)


W_conv4 = tf.Variable(tf.random_normal(shape=[3, 3, 48, 64], stddev=2e-2))
L4 = tf.nn.conv2d(L3, W_conv4, strides=[1, 1, 1, 1], padding='SAME')
L4 = tf.nn.elu(L4)
L4 = tf.nn.max_pool(L4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
L4 = tf.nn.dropout(L4, keep_prob2) 


  
# layer 5, 6
W_conv5 = tf.Variable(tf.random_normal(shape=[3, 3, 64, 128], stddev=2e-2))
L5 = tf.nn.conv2d(L4, W_conv5, strides=[1, 1, 1, 1], padding='SAME')  
L5 = tf.nn.elu(L5)


W_conv6 = tf.Variable(tf.random_normal(shape=[3, 3, 128, 64], stddev=2e-2))
L6 = tf.nn.conv2d(L5, W_conv6, strides=[1, 1, 1, 1], padding='SAME')  
L6 = tf.nn.elu(L6)
L6 = tf.nn.max_pool(L6, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
L6 = tf.nn.dropout(L6, keep_prob3)  



# fully connected layer
W_fc1 = tf.Variable(tf.random_normal(shape=[4 * 4 * 64, 10], stddev=2e-2))
h_conv_flat = tf.reshape(L6, [-1, 4 * 4 * 64])
logits = tf.matmul(h_conv_flat, W_fc1)


y_pred = tf.nn.softmax(logits)


 

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=Y, logits=logits)) +\
                    0.001*tf.nn.l2_loss(W_conv1) +\
                    0.001*tf.nn.l2_loss(W_conv2) +\
                    0.001*tf.nn.l2_loss(W_conv3) +\
                    0.001*tf.nn.l2_loss(W_conv4) +\
                    0.001*tf.nn.l2_loss(W_conv5) +\
                    0.001*tf.nn.l2_loss(W_conv6) +\
                    0.001*tf.nn.l2_loss(W_fc1)
            
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
learning_rate = 0.0015


with tf.control_dependencies(update_ops):
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

 

 

#######################################################

 
train_acc_list = []
valid_acc_list = []
valid_acc_list_nochange = []
epoch_list = []
max_valid_acc = 0;
tmp_lr = 0;

correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))



total_batch = train_size // batch_size
total_time = time.time() 




sess = tf.Session()
sess.run(tf.global_variables_initializer())


saver = tf.train.Saver()
if not os.path.isdir('./model'):
  os.mkdir('./model')
model_path = './model/model'



for e in range(epoch):

    epoch_list.append(e)
    total_cost = 0
    start_time = time.time()
    
    
    
    for i in range(total_batch):

        batch = dataset.next('Train', batch_size)
     
        sess.run(optimizer, feed_dict={X: batch[0],
                                       Y: batch[1],
                                       keep_prob1: 0.8,
                                       keep_prob2: 0.5,
                                       keep_prob3: 0.6,
                                       keep_prob4: 0.6}) # keep_prob4는 안씀

        
    train_accuracy = sess.run(accuracy, feed_dict={X: batch[0],
                                                   Y: batch[1],
                                                   keep_prob1: 1.0,
                                                   keep_prob2: 1.0,
                                                   keep_prob3: 1.0,
                                                   keep_prob4: 1.0})
    train_acc_list.append(train_accuracy*100)
    
    loss_print = sess.run(loss, feed_dict={X: batch[0],
                                           Y: batch[1],
                                           keep_prob1: 1.0,
                                           keep_prob2: 1.0,
                                           keep_prob3: 1.0,
                                           keep_prob4: 1.0})


    valid_batch = dataset.next('Validation', 2000)
    valid_accuracy = sess.run(accuracy, feed_dict={X: valid_batch[0],
                                                   Y: valid_batch[1],
                                                   keep_prob1: 1.0,
                                                   keep_prob2: 1.0,
                                                   keep_prob3: 1.0,
                                                   keep_prob4: 1.0})
    valid_acc_list.append(valid_accuracy*100)
    
    #if (valid_accuracy*100 > 70 and valid_accuracy*100 < 74):
      #learning_rate = 0.001
    #if (valid_accuracy*100 > 74):
      #learning_rate = 0.0005   
 
    
    print('epoch : ', e + 1, ' loss : ', loss_print)
    print('learning_rate = {0:.7f}'.format(learning_rate), 'train acc: ', train_accuracy * 100, 'Valdation acc : ', valid_accuracy * 100)
    print("--- %s seconds ---\n" % round(time.time() - start_time, 2))
    
    
    if(valid_accuracy*100 > 77 and max_valid_acc < valid_accuracy):
      max_valid_acc = valid_accuracy
      print("job`s ckpt files is save as : \n", model_path)
      saver.save(sess, model_path)



    
print("---total time : %s  ---" % round(time.time() - total_time, 2))



plt.plot(epoch_list, train_acc_list, label='train')
plt.plot(epoch_list, valid_acc_list, label='valid')
plt.axis([0, epoch + 5, 0, 110])
plt.xlabel('epoch')
plt.ylabel('acc')
plt.legend()
plt.grid(True)
plt.show()

# 필요 자료
# http://torch.ch/blog/2016/02/04/resnets.html
# https://m.blog.naver.com/PostView.nhn?blogId=laonple&logNo=220770760226&proxyReferer=https%3A%2F%2Fwww.google.com%2F