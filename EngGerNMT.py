import numpy as np
import tensorflow as tf

import abc
import collections

import sys,os
#sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'nmt'))
import textToDict

#general variables
batch_size_training = 1
num_units = 64
learning_rate = 0.001

#turn our data into dictionaries to use as inputs
eng_dict, eng_inv, ger_dict, ger_inv = textToDict._create_dictionary('text2text')
inputData, outputData = textToDict._embed_sentence_data('text2text', 20, eng_inv, ger_inv)

encoder_inputs = tf.placeholder(tf.int32, shape = [batch_size_training, None], name='enc_in')
decoder_inputs = tf.placeholder(tf.int32, shape = [batch_size_training, None], name='dec_in')
decoder_outputs = tf.placeholder(tf.int32, shape = [batch_size_training, None], name='dec_out')
#target_weights = tf.ones(decoder_outputs.shape)

#encoder
encoder_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units)

encoder_outputs, encoder_state = tf.nn.dynamic_rnn(
    encoder_cell, encoder_inputs,
    sequence_length=20, time_major=False)


#decoder
decoder_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units)

#decoder helper
dec_helper= tf.contrib.seq2seq.TrainingHelper(
    decoder_inputs, 20)

decoder = tf.contrib.seq2seq.BasicDecoder(
    decoder_cell, dec_helper, encoder_state)

outputs, _ = tf.contrib.seq2seq.dynamic_decode(decoder)
logits = outputs.rnn_output

#loss
cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=decoder_outputs, logits=logits)
train_loss = (tf.reduce_sum(cross_entropy)) #* #target_weights) /
    #batch_size_training)


#gradient optimization
params = tf.trainable_variables()
gradients = tf.gradients(train_loss, params)
clipped_gradients, _ = tf.clip_by_global_norm(
    gradients, 5.0)

optimizer = tf.train.AdamOptimizer(learning_rate)
update_step = optimizer.apply_gradients(
    zip(clipped_gradients, params))


#Inference (generate Translations)


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer, feed_dict={encoder_inputs:inputData, decoder_inputs:encoder_outputs, decoder_outputs:outputData})