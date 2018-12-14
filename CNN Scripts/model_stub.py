# -*- coding: utf-8 -*-
"""
Created on Fri Mar  2 20:19:06 2018

@author: LaRu
"""
import tensorflow as tf


#Wrapper functions for cleaner code. 
#Taken from https://github.com/tensorflow/tensorflow/blob/r1.2/tensorflow/examples/tutorials/mnist/mnist_deep.py
#slightly modified
def conv2d(x, W, b,strides):
  """conv2d returns a 2d convolution layer with full stride."""
  return tf.nn.relu(tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME') + b)


def max_pool_2x2(x):
  """max_pool_2x2 downsamples a feature map by 2X."""
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')


def weight_variable(shape):
  """weight_variable generates a weight variable of a given shape."""
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)


def bias_variable(shape):
  """bias_variable generates a bias variable of a given shape."""
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)
#Wrapper functions end here


def build_model(input_data, label, train_mode, keep_prob, learning_rate, batch_size):

    #Initialize weights for the convolutions
    W1 = weight_variable([3,3,3,32])#dimensions wrong
    b1 = bias_variable([32])
    
    W2 = weight_variable([3,3,32,32])#dimensions wrong
    b2 = bias_variable([32])
    
    W3 = weight_variable([3,3,32,64])#dimensions wrong
    b3 = bias_variable([64])


    #Execute the first two convolutions
    h_conv1 = conv2d(input_data, W1, b1, 1)
    h_conv2 = conv2d(h_conv1, W2, b2, 1)#shape: (?,32,32,32)

    #Create a pooling layer
    h_pool1 = max_pool_2x2(h_conv2)#shape: (?,16,16,32)
    print(h_pool1.shape)

    #Thrid convolution
    h_conv3 = conv2d(h_pool1, W3, b3, 1)#shape: (?,16,16,64)

    #Another pooling layer
    h_pool2 = max_pool_2x2(h_conv3)#shape: (?,8,8,64)
    print(h_pool2.shape) 

    #Fully connected layers start here
    #Initialize Weights for fully connected layer
    W1_fc = weight_variable([8*8*64,512])
    b1_fc = bias_variable([512])

    W2_fc = weight_variable([512,10])
    b2_fc = bias_variable([10])

    #reshape h_pool2 to a vector
    h_pool2_vec = tf.reshape(h_pool2,[-1,8*8*64])

    #Fully connected layers fc1 and fc2
    #Maybe the activation function needs to be used again, not entirely sure though
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_vec,W1_fc) + b1_fc)

    #h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
    
    #Our final Predictions after the last layer
    y_fc2 = tf.matmul(h_fc1_drop, W2_fc) + b2_fc

    #Do we need to use softmax after the final layer?
    #This just calculates the cross entropy error
    loss = tf.losses.sparse_softmax_cross_entropy(label, y_fc2)

    #calculate the accuracy
    correct_prediction = tf.cast(tf.equal(tf.argmax(y_fc2, 1, output_type=tf.int32),label), tf.float32)
    accuracy = tf.reduce_mean(correct_prediction)

    
    
    ###our code is finished here###
    
    global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')
    if train_mode:
        # Optimization
        optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step)    
        #optimizer = tf.train.RMSPropOptimizer(lr,decay=1e-6).minimize(loss, global_step)
        tf.summary.scalar('loss', loss)
    tf.summary.scalar('accuracy', accuracy)#changed acc --> accuracy
    tf.summary.histogram('histogram loss', loss)
    summary_op = tf.summary.merge_all()
    if train_mode:
        return optimizer, global_step, loss, accuracy, summary_op
    else:
        return global_step, accuracy, summary_op
