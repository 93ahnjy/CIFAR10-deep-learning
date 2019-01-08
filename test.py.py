import tensorflow as tf
import numpy as np
import math
import time
import os
import matplotlib.pyplot as plt
from CIFAR_10_augment import CIFAR_10
from google.colab import files
#from cfar10_test import CIFAR_10

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


correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))



sess = tf.Session()
sess.run(tf.global_variables_initializer())


saver = tf.train.Saver()
saver.restore(sess, './model/model')



test_batch = dataset.next('Validation', 2000)
test_accuracy = sess.run(accuracy, feed_dict={X: test_batch[0],
                                                   Y: test_batch[1],
                                                   keep_prob1: 1.0,
                                                   keep_prob2: 1.0,
                                                   keep_prob3: 1.0,
                                                   keep_prob4: 1.0})
print('Test acc : ', test_accuracy * 100)
    
