from __future__ import division
import tensorflow as tf
import numpy as np
import os
import cPickle as pickle

import read_input_tf
import defs
import utils

FLAGS = tf.app.flags.FLAGS

input_filename = os.path.join(FLAGS.data_dir,'train.conll')

#read the vocabulary file and pre-trained embedding file on some of the words.
vocab_file = os.path.join(FLAGS.data_dir,'vocab.txt')
embedding_file = os.path.join(FLAGS.data_dir,'wordVectors.txt')
line_batch,word_dict = read_input_tf.get_input(input_filename,FLAGS.batch_size, FLAGS.num_examples_train, True)

#print('Training : word_dict :')
#for k,v in word_dict.iteritems():
#  print k,v

#save the dictionary to use during testing
with open(os.path.join(FLAGS.data_dir,'word_dict_train.pkl'),'wb') as fptr:
   pickle.dump(word_dict,fptr)

embeddings = utils.load_word_embeddings(word_dict,vocab_file,embedding_file)

window_size = defs.WINDOW_SIZE
no_of_words_per_window =  2*window_size+1
stacked_window_dim = no_of_words_per_window*FLAGS.embedding_dimension

Word_Vectors = tf.Variable(embeddings,name='Word_Embedding_Matrix')
W_matrix = utils.get_weights(name='W_Matrix',shape=[stacked_window_dim,FLAGS.hidden_dim])
W_biases = utils.get_weights('hidden_biases',shape=[FLAGS.batch_size,FLAGS.hidden_dim])

U_matrix = utils.get_weights('U_Matrix',shape=[FLAGS.hidden_dim,FLAGS.number_of_classes])
U_biases = utils.get_weights('output_biases',shape=[FLAGS.batch_size,FLAGS.number_of_classes])

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
train_step = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(loss)

#predicted NER labels
predicted_labels = tf.argmax(logits,axis=1)

#accuracy,precision,recall etc.
accuracy = tf.metrics.accuracy(labels,predicted_labels)
CF = tf.confusion_matrix(tf.cast(labels,tf.int32),tf.cast(predicted_labels,tf.int32),num_classes=FLAGS.number_of_classes)

saver = tf.train.Saver(max_to_keep=FLAGS.max_to_keep)

init_l = tf.local_variables_initializer()
with tf.Session() as sess:
     init = tf.initialize_all_variables()
     sess = tf.Session()
     sess.run(init)
     sess.run(init_l)
     coord = tf.train.Coordinator()
     threads = tf.train.start_queue_runners(sess=sess,coord=coord)

     for step in range(10000):
        _,loss_v,acc_v,predictions,actual,CF_v = sess.run([train_step,loss,accuracy,predicted_labels,labels,CF])
        pr,re = utils.precision_recall(predictions,actual)
        f1 = 2*pr*re/(pr + re + 1e-10)
        print step,' loss=',loss_v,' acc=',acc_v[0],' pr=',pr,' re=',re,' F1=',f1
        #print zip(predictions,actual)
        print CF_v
        print '----------------------------------------'

        if step % 500 == 0 and step>=500:
           #save the model
           ckptfile = FLAGS.model_dir+'model_'+str(step)+'.ckpt'
           ckptfile = saver.save(sess, ckptfile)

     coord.request_stop()
     coord.join(threads)

