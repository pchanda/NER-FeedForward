from __future__ import division
from __future__ import print_function
import tensorflow as tf
import os
import defs
import process_vocabulary
FLAGS = tf.app.flags.FLAGS

def _generate_batch(part_0,part_1, min_queue_examples,batch_size, shuffle):
  num_preprocess_threads = 16
  if shuffle:
    line_value_batch = tf.train.shuffle_batch(
        [part_0,part_1],
        batch_size=batch_size,
        num_threads=num_preprocess_threads,
        capacity=min_queue_examples + 3 * batch_size,
        min_after_dequeue=min_queue_examples)
  else:
    line_value_batch = tf.train.batch(
        [part_0,part_1],
        batch_size=batch_size,
        num_threads=num_preprocess_threads,
        capacity=min_queue_examples + 3 * batch_size)
  return line_value_batch



def get_input(data_filename, batch_size, num_examples, shuffle, word_dict=None):
  # word_dict should be populated during testing to use the prebuilt dictionary made during testing.
  vocab_fstream = open(data_filename,'rb')
  data = process_vocabulary.read_conll_file(vocab_fstream)
  vocab_fstream.close()
  word_dict,windowed_data_string = process_vocabulary.process_sentences_and_labels(data,defs.WINDOW_SIZE,word_dict)

  windowed_data_tensor = tf.convert_to_tensor(windowed_data_string, dtype=tf.string)
  input_queue = tf.train.slice_input_producer([windowed_data_tensor],shuffle=shuffle)

  line_value = input_queue[0]
  line_value_parts = tf.decode_csv(line_value, record_defaults=[['NA']]*2,field_delim=";")
  part_0 = tf.decode_csv(line_value_parts[0], record_defaults=[['.']]*(2*defs.WINDOW_SIZE+1),field_delim=",")
  part_1 = line_value_parts[1]

  part_0 = tf.string_to_number(part_0,out_type=tf.int64)
  part_1 = tf.string_to_number(part_1,out_type=tf.int64)

  # Ensure that the random shuffling has good mixing properties.
  min_fraction_of_examples_in_queue = 0.4
  min_queue_examples = int(num_examples * min_fraction_of_examples_in_queue)

  # Generate a batch by building up a queue of examples.
  batch =  _generate_batch(part_0,part_1,min_queue_examples, batch_size,shuffle=shuffle)
  return batch,word_dict
