from __future__ import division
import tensorflow as tf
import numpy as np
import os
import cPickle as pickle
import pandas as pd

import read_input_tf
import defs
import utils

FLAGS = tf.app.flags.FLAGS

#read file to test
input_filename = os.path.join(FLAGS.data_dir,'dev.conll')
#read word_dict saved during training
with open(os.path.join(FLAGS.data_dir,'word_dict_train.pkl'),'rb') as fptr:
   word_dict = pickle.load(fptr)

line_batch,word_dict = read_input_tf.get_input(input_filename,FLAGS.batch_size, FLAGS.num_examples_test, False, word_dict)

#print 'Testing: word_dict:'
#for k,v in word_dict.iteritems():
#  print k,v

window_size = defs.WINDOW_SIZE
no_of_words_per_window =  2*window_size+1
stacked_window_dim = no_of_words_per_window*FLAGS.embedding_dimension

Word_Vectors = tf.get_variable(name='Word_Embedding_Matrix',shape=[FLAGS.vocabulary_size,FLAGS.embedding_dimension])
W_matrix = utils.get_weights(name='W_Matrix',shape=[stacked_window_dim,FLAGS.hidden_dim])
W_biases = utils.get_weights('hidden_biases',shape=[FLAGS.batch_size,FLAGS.hidden_dim])

U_matrix = tf.get_variable('U_Matrix',shape=[FLAGS.hidden_dim,FLAGS.number_of_classes])
U_biases = tf.get_variable('output_biases',shape=[FLAGS.batch_size,FLAGS.number_of_classes])

window = line_batch[0]
labels = line_batch[1]

one_hot_words = tf.one_hot(window,depth=FLAGS.vocabulary_size,dtype=tf.float32)
words_reshaped = tf.reshape(one_hot_words,[-1,FLAGS.vocabulary_size])

word_embeddings = tf.matmul(words_reshaped,Word_Vectors)
word_embeddings_stacked_per_window = tf.reshape(word_embeddings,shape=[-1,stacked_window_dim]) # shape = [batch_size x stacked_window_dim]

hidden_layer = tf.nn.relu(tf.matmul(word_embeddings_stacked_per_window,W_matrix) + W_biases,name='hidden_layer')
logits = tf.nn.softmax(tf.matmul(hidden_layer,U_matrix) + U_biases,name='logits')  # should be of shape = [batch_size x no.of.classes] 

#loss
loss = tf.reduce_mean((tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels, name="cross_entropy")))

#predicted NER labels
predicted_labels = tf.argmax(logits,axis=1)


saver = tf.train.Saver()
with tf.Session() as sess:
     coord = tf.train.Coordinator()
     threads = tf.train.start_queue_runners(sess=sess,coord=coord)
     saver.restore(sess, FLAGS.model_dir+'model_9000.ckpt')

     pr_labels = []
     act_labels = []
     for step in range(5000):
        loss_v,predictions,actual = sess.run([loss,predicted_labels,labels])
        pr,re = utils.precision_recall(predictions,actual)
        f1 = 2*pr*re/(pr + re + 1e-10)
        pr_labels += list(predictions)
        act_labels +=  list(actual)
        print step,' loss=',loss_v,' pr=',pr,' re=',re,' F1=',f1
        #print zip(predictions,actual)
        print '----------------------------------------'

     pr,re = utils.precision_recall(predictions,actual)
     f1 = 2*pr*re/(pr + re + 1e-10)
     cf = pd.crosstab(pd.Series(act_labels), pd.Series(pr_labels), rownames=['True'], colnames=['Predicted'], margins=True)
     print 'precision = ',pr,' recall = ',re, ' f1=',f1
     print 'Confusion Matrix = ',cf
     coord.request_stop()
     coord.join(threads)
