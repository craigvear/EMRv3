# import tensorflow_datasets as tfds
import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense, LayerNormalization, Embedding, Dropout
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers.schedules import LearningRateSchedule
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.metrics import Mean, SparseCategoricalAccuracy

# from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt


class DataHandler(object):
    def __init__(self, word_max_length=30, batch_size=64, buffer_size=20000):
        train_data, test_data = self._load_data()

        self.tokenizer_ru = tfds.features.text.SubwordTextEncoder.build_from_corpus(
            (ru.numpy() for ru, en in train_data), target_vocab_size=2 ** 13)
        self.tokenizer_en = tfds.features.text.SubwordTextEncoder.build_from_corpus(
            (en.numpy() for ru, en in train_data), target_vocab_size=2 ** 13)

        self.train_data = self._prepare_training_data(train_data, word_max_length, batch_size, buffer_size)
        self.test_data = self._prepare_testing_data(test_data, word_max_length, batch_size)

    def _load_data(self):
        data, info = tfds.load('ted_hrlr_translate/ru_to_en', with_info=True, as_supervised=True)
        return data['train'], data['validation']

    def _prepare_training_data(self, data, word_max_length, batch_size, buffer_size):
        data = data.map(self._encode_tf_wrapper)
        data.filter(lambda x, y: tf.logical_and(tf.size(x) <= word_max_length, tf.size(y) <= word_max_length))
        data = data.cache()
        data = data.shuffle(buffer_size).padded_batch(batch_size, padded_shapes=([–1], [–1]))
        data = data.prefetch(tf.data.experimental.AUTOTUNE)
        return data

    def _prepare_testing_data(self, data, word_max_length, batch_size):
        data = data.map(self._encode_tf_wrapper)
        data = data.filter(
            lambda x, y: tf.logical_and(tf.size(x) <= word_max_length, tf.size(y) <= word_max_length)).padded_batch(
            batch_size, padded_shapes=([–1], [–1]))

    def _encode(self, english, russian):
        russian = [self.tokenizer_ru.vocab_size] + self.tokenizer_ru.encode(russian.numpy()) + [
            self.tokenizer_ru.vocab_size + 1]
        english = [self.tokenizer_en.vocab_size] + self.tokenizer_en.encode(english.numpy()) + [
            self.tokenizer_en.vocab_size + 1]

        return russian, english

    def _encode_tf_wrapper(self, pt, en):
        return tf.py_function(self._encode, [pt, en], [tf.int64, tf.int64])