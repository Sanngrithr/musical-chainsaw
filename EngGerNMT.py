import numpy as np
import tensorflow as tf

import abc
import collections

import sys,os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'nmt'))
import textToDict

#general variables
batch_size = 10

#turn our data into dictionaries to use as inputs
eng_dict, eng_inv, ger_dict, ger_inv = textToDict._create_dictionary('text2text')
inputData, outputData = textToDict._create_sentence_data('text2text', 20, eng_inv, ger_inv)

#placeholders for feeding in the source sentence words and target sentence words
encoder_train_inputs,decoder_train_inputs = [],[]
decoder_train_labels, decoder_label_masks=[],[]
 
# Defining unrolled training inptus for encoder and decoder
for i in range(20):
    encoder_train_inputs.append(tf.placeholder(tf.int32, shape=[batch_size]))
    decoder_train_inputs.append(tf.placeholder(tf.int32, shape=[batch_size]))
    decoder_train_labels.append(tf.placeholder(tf.int32, shape=[batch_size]))
    decoder_label_masks.append(tf.placeholder(tf.float32, shape=[batch_size]))

for i in range(30):
    encoder_train_inputs[i] = tf.Variable(inputData[i])
    decoder_train_inputs[i] = tf.Variable(outputData[i])
    





#encoder
encoder_cell = tf.nn.rnn_cell.BasicLSTMCell(20)

#decoder

#loss

#gradient potimization

#Inference (generate Translations)

sess = tf.Session()
sess.run(tf.initialize_all_variables())
