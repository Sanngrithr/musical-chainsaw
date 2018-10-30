# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 09:45:36 2018

@author: LaRu
"""

import numpy as np
import matplotlib.pyplot as plt
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
mu = np.mean(X_,axis=0)
sigma = np.std(X_,axis=0)
X_ = (X_ - mu)/sigma
print(y_.shape)
print(X_.shape)

#split X_ and y_ into training and test
#discomment and fill
X_train = X_[:400,:]
y_train = y_[:400]

#add column of ones in the first column
X_train = np.c_[np.ones(400), X_train]

print('data training shape: ', X_train.shape)
print('target training shape: ', y_train.shape)

#here again slice, add column of ones
X_test = X_[401:,:]
print('test data shape: ', X_test.shape)
y_test = y_[401:]


#optimum solution for this model
opt = np.dot(np.linalg.inv(np.dot(X_train.T, X_train)), np.dot(X_train.T, y_train))
print('Theta Values: ',opt)
print('Theta Size: ',opt.size)

#######################end of a) b) c)
######start d)###################################

tf.reset_default_graph()
#the input X, y is taken into account in the computational graph via a placeholder

#define placeholder for computational graph of shape [400,14] and type tf.float32
X = tf.Variable(X_train.shape, name="training_data", dtype=tf.float32)
#X = tf.convert_to_tensor(X_train)
print(X)

#define placeholder for target y with shape [400]
y = tf.Variable(y_train.shape, name="predictions", dtype=tf.float32)
print(y)

#the parameters that need to be learned are defined by means of variables
#define theta variable 
theta_hat= tf.Variable(opt, name="optimum_values", dtype=tf.float32)
print(theta_hat)
y_hat = tf.tensordot(X,theta_hat,1)

error_vec = tf.subtract(y_hat, y)
#check tf.subtract, tf.reduce_sum and tf.square
loss = tf.reduce_sum(tf.square(error_vec))

# define optimizer as gradient descent with learning rate of 0.0001 to minimize loss
#look for GradientDescentOptimizer in the tensorflow API
optimizer = tf.train.GradientDescentOptimizer(0.0001).minimize(loss)

#discomment when ready
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(1000):
        l, e_v, y_pred, t_hat, _ = sess.run([loss,error_vec,y_hat,theta_hat, optimizer], feed_dict={X:X_train, y:y_train})
        
        #print('epoch, y pred, loss, diff_vec: ', epoch, y_pred.shape, l, e_v.shape)
print('theta_hat iterative: ', t_hat)      
print('theta_hat normal equations: ', opt) 
     
#fill test target and predicted target
plt.scatter(y, y_hat)
plt.xlabel("Prices: $y_i$")
plt.ylabel("Predicted prices: $\hat{y}_i$")
plt.title("Prices vs Predicted prices: $y_i$ vs $\hat{y}_i$")




