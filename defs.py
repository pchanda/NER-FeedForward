import tensorflow as tf
import os

base_path = '/home/u588401/deep_learning/tensorflow/experiments/NLP/word_classification_1/'

# Basic model parameters.
tf.app.flags.DEFINE_integer('batch_size', 100,"Number of windows to process in a batch.")
tf.app.flags.DEFINE_string('data_dir', os.path.join(base_path,'real_data/'),"""Path to data directory""")
tf.app.flags.DEFINE_string('model_dir', os.path.join(base_path,'saved_models/'),"""Path to saved models during training""")
tf.app.flags.DEFINE_integer('num_examples_train', 204000,"Total number of pairs per epoch for training")
tf.app.flags.DEFINE_integer('num_examples_test', 51362,"Total number of pairs per epoch for testing")
tf.app.flags.DEFINE_integer('vocabulary_size',50000,"vocabulary size")
tf.app.flags.DEFINE_integer('embedding_dimension', 50,"dimension of the word vectors")
tf.app.flags.DEFINE_integer('hidden_dim', 200,"dimension of hidden layers")
tf.app.flags.DEFINE_integer('number_of_classes', 5,"no of NER classes (including null or O)")
tf.app.flags.DEFINE_integer('max_to_keep', 50,"max models to retain")
tf.app.flags.DEFINE_integer('learning_rate', 0.001,"optimizer learning rate")

START_TOKEN = "<s>"
END_TOKEN = "</s>"
P_CASE = "CASE:"
CASES = ["aa", "AA", "Aa", "aA"]
NUM = "nnnummm"
UNK = "uuunkkk"
LBLS = ["PER","ORG","LOC","MISC","O"]

WINDOW_SIZE = 2
