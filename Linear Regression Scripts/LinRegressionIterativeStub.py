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
#X_train = 
#y_train = 
#add column of ones in the first column
#check numpy ones and hstack

#print('data training shape: ', X_train.shape)
#print('target training shape: ', y_train.shape)

#here again slice, add column of ones
#X_test = 
#print('test data shape: ', X_test.shape)

#y_test = 
#optimum solution for this model
#opt = 
#print(opt)
#######################end of a) b) c)
######start d)###################################
tf.reset_default_graph()
#define placeholder for computational graph as in the other exercise
#X = tf.placeholder(TO DO)
#y = tf.placeholder(TO DO)
#define theta hat variable 
#theta_hat= 
#calculate y_hat
#y_hat = tf.tensordot(TO DO)
#calculate error vector e (y - y_hat)
#error_vec = 
#calculate gradient vector
#gradient_vec = tf.tensordot(TO DO)

#update theta_hat
#alpha = 0.0001
#theta_hat = theta_hat.assign(theta_hat - alpha*gradient_vec)
#with tf.Session() as sess:
#    sess.run(tf.global_variables_initializer())
#    for epoch in range(1000):
#        
#        e_vec, grad_vec, t_hat = sess.run([error_vec, gradient_vec, theta_hat], feed_dict={X:X_train, y:y_train})
#        
#        #print('iteration, error vec, grad vec, theta vec: ', epoch, e_vec, grad_vec, t_hat)
#print('theta_hat iterative: ', t_hat)          
#print('theta_hat normal equations: ', opt) 
     






