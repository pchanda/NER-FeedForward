from __future__ import division
import tensorflow as tf
import numpy as np
import read_input_tf
import defs

FLAGS = tf.app.flags.FLAGS

def precision_recall(predicted,actual):
   no_correct_nonnull_predictions = np.sum([x==y and x!=defs.LBLS.index('O') for (x,y) in zip(predicted,actual)])
   no_nonnull_predictions = np.sum([x!=defs.LBLS.index('O') for x in predicted])
   no_nonnull_predictions = (no_nonnull_predictions+1) if no_nonnull_predictions==0 else no_nonnull_predictions
   no_nonnull_labels = np.sum([y!=defs.LBLS.index('O') for y in actual])
   no_nonnull_labels = (no_nonnull_labels+1) if no_nonnull_labels==0 else no_nonnull_labels
   precision = no_correct_nonnull_predictions/no_nonnull_predictions
   recall = no_correct_nonnull_predictions/no_nonnull_labels
   return precision,recall

def load_word_embeddings(word_dict,vocab_file,embedding_file):
   #read the embeddings
   embeddings = np.array(np.random.randn(FLAGS.vocabulary_size, FLAGS.embedding_dimension), dtype=np.float32)
   fstream_1 = open(vocab_file,'rb')
   fstream_2 = open(embedding_file,'rb')
   for word,vector in zip(fstream_1,fstream_2):
      word = word.strip()
      if word.isdigit(): word = defs.NUM
      else: word = word.lower()
      vector = np.array(list(map(float,vector.strip().split(" "))))
      if word in word_dict:
          embeddings[word_dict[word]] = vector
   fstream_1.close()
   fstream_2.close()   
   print 'load_word_embeddings : vocabulary size = ',FLAGS.vocabulary_size, ' embedding_shape = ',embeddings.shape
   return embeddings


def get_weights(name, shape, stddev=0.02, dtype=tf.float32, initializer=None):
    if initializer is None:
      initializer = tf.truncated_normal_initializer(stddev = stddev, dtype=dtype)
    return tf.get_variable(name, shape, initializer = initializer, dtype=dtype)


def get_biases(name, shape, val=0.0, dtype=tf.float32, initializer=None):
    if initializer is None:
       initializer = tf.constant_initializer(val)
    return tf.get_variable(name, shape, initializer = initializer, dtype=dtype)
