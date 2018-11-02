# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 09:45:36 2018

@author: LaRu
"""

import numpy as np
import sklearn

import seaborn as sns
import tensorflow as tf
sns.set_style("whitegrid")
sns.set_context("poster")


from sklearn.datasets import load_boston
boston = load_boston()
print(boston.keys())
print(boston.data.shape)

#print(boston.feature_names)
#print(boston.DESCR)
X_ = boston.data
y_ = boston.target
#normalize features, features are in cols so take mean over each col-> axis = 0
mu = np.mean(X_,axis=0)#take mean over each col
sigma = np.std(X_,axis=0)
X_ = (X_ - mu)/sigma
#split X_ and y_ into training and test
#discomment and fill
X_train = X_[:400,:]
y_train = y_[:400]
#add column of ones in the first column
#check numpy ones and hstack
X_train = np.c_[np.ones(400), X_train]
print('data training shape: ', X_train.shape)
print('target training shape: ', y_train.shape)

#here again slice, add column of ones
X_test = X_[401:,:]
#print('test data shape: ', X_test.shape)

y_test = y_[401:]
#optimum solution for this model
opt = np.dot(np.linalg.inv(np.dot(X_train.T, X_train)), np.dot(X_train.T, y_train))
print(opt)

#######################end of a) b) c)
######start f)###################################

tf.reset_default_graph()

#define placeholder for computational graph as in the other exercise
X = tf.placeholder(tf.float32, shape=(400,14))
y = tf.placeholder(tf.float32, shape=(400,))

#define theta hat variable 
theta_hat= tf.Variable(np.random.rand(14,), dtype=tf.float32)

#calculate y_hat
y_hat = tf.tensordot(X,theta_hat,1)

#calculate error vector e (y - y_hat)
error_vec = tf.subtract(y_hat, y)

#calculate gradient vector
gradient_vec = tf.tensordot(tf.transpose(-X),error_vec,1)##Noch nicht richtig! Möglicherweise Fehler an anderer Stelle?

#update theta_hat
alpha = 0.0001
theta_hat = theta_hat.assign(theta_hat - alpha*gradient_vec)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(1000):
       
        e_vec, grad_vec, t_hat = sess.run([error_vec, gradient_vec, theta_hat], feed_dict={X:X_train, y:y_train})
        
        #print('iteration, error vec, grad vec, theta vec: ', epoch, e_vec, grad_vec, t_hat)
print('theta_hat iterative: ', t_hat)          
print('theta_hat normal equations: ', opt) 
     






