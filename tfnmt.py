import numpy as np
import tensorflow as tf

#Test file for
#http://www.thushv.com/natural_language_processing/neural-machine-translator-with-50-lines-of-code-using-tensorflow-seq2seq/

###Defining Parameters
batch_size = 10
source_sequence_length = 8
target_sequence_length = 8
decoder_type = 'basic'

###Our Model:

enc_train_inputs,dec_train_inputs = [],[]
 
# Defining unrolled training inputs for encoder
for ui in range(source_sequence_length):
    enc_train_inputs.append(tf.placeholder(tf.int32, shape=[batch_size],name='enc_train_inputs_%d'%ui))
 
dec_train_labels, dec_label_masks=[],[]
 
# Defining unrolled training inptus for decoder
for ui in range(target_sequence_length):
    dec_train_inputs.append(tf.placeholder(tf.int32, shape=[batch_size],name='dec_train_inputs_%d'%ui))
    dec_train_labels.append(tf.placeholder(tf.int32, shape=[batch_size],name='dec-train_outputs_%d'%ui))
    dec_label_masks.append(tf.placeholder(tf.float32, shape=[batch_size],name='dec-label_masks_%d'%ui))

# Need to use pre-trained word embeddings
encoder_emb_layer = tf.convert_to_tensor(np.load('de-embeddings.npy'))
decoder_emb_layer = tf.convert_to_tensor(np.load('en-embeddings.npy'))
 
# looking up embeddings for encoder inputs
encoder_emb_inp = [tf.nn.embedding_lookup(encoder_emb_layer, src) for src in enc_train_inputs]
encoder_emb_inp = tf.stack(encoder_emb_inp)
 
# looking up embeddings for decoder inputs
decoder_emb_inp = [tf.nn.embedding_lookup(decoder_emb_layer, src) for src in dec_train_inputs]
decoder_emb_inp = tf.stack(decoder_emb_inp)
 
# to contain the sentence length for each sentence in the batch
enc_train_inp_lengths = tf.placeholder(tf.int32, shape=[batch_size],name='train_input_lengths')

#defining the encoder
encoder_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units)
 
initial_state = encoder_cell.zero_state(batch_size, dtype=tf.float32)
 
encoder_outputs, encoder_state = tf.nn.dynamic_rnn(
    encoder_cell, encoder_emb_inp, initial_state=initial_state,
    sequence_length=enc_train_inp_lengths, 
    time_major=True, swap_memory=True)

#defining the decoder
decoder_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units)
 
projection_layer = Dense(units=vocab_size, use_bias=True)
 
# Helper
helper = tf.contrib.seq2seq.TrainingHelper(
    decoder_emb_inp, [tgt_max_sent_length-1 for _ in range(batch_size)], time_major=True)
 
# Decoder
if decoder_type == 'basic':
    decoder = tf.contrib.seq2seq.BasicDecoder(
        decoder_cell, helper, encoder_state,
        output_layer=projection_layer)
     
elif decoder_type == 'attention':
    decoder = tf.contrib.seq2seq.BahdanauAttention(
        decoder_cell, helper, encoder_state,
        output_layer=projection_layer)
     
# Dynamic decoding
outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(
    decoder, output_time_major=True,
    swap_memory=True)

#loss functions
logits = outputs.rnn_output
 
crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(
    labels=dec_train_labels, logits=logits)
loss = (tf.reduce_sum(crossent*tf.stack(dec_label_masks)) / (batch_size*target_sequence_length))

train_prediction = outputs.sample_id

#Optimizer, probably need to switch to a different one after a lot of iterations
adam_gradients, v = zip(*adam_optimizer.compute_gradients(loss))
adam_gradients, _ = tf.clip_by_global_norm(adam_gradients, 25.0)
adam_optimize = adam_optimizer.apply_gradients(zip(adam_gradients, v))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(100):
        _crossentropy,_loss,_ = sess.run([crossent, loss, adam_optimize])
        print(_crossentropy, _loss)
    print(train_prediction)
sess.close()
