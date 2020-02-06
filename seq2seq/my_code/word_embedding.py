import tensorflow as tf
from my_code.util.vocabulary import read_vocabulary
from dependency.opennmt_v2.utils import compat
import numpy as np
import json
import os


class MyWordEmbedding(tf.keras.layers.Layer):

    """wrap the word embedding functionality. supports both pre-trained and the random initialized embeddings"""

    def __init__(self, json_str):
        """
        initialize MyWordEmbedding
        :param json_str: json string for the configuration. the json contains the following fields:
            vocabulary_file_name: string, vocabulary file name
            case_insensitive_embeddings: bool, case sensitive or insensitive string match
            word_embedding_file_name: string, pretrained word embedding file
            trainable: bool, whether the word embeddings are trainable
            num_oov_buckets: positive int, number of buckets for oov
            with_header: bool, whether the word embedding file has a header line
            word_embedding_size: none or positive number; the word embedding size
        """
        super(MyWordEmbedding, self).__init__()
        self.config = json.loads(json_str)

        word_to_ids = self._read_vocabulary()
        if self.config["word_embedding_file_name"] is not None and os.path.exists(self.config["word_embedding_file_name"]):
            self._load_pretrained_embeddings(word_to_ids)
        else:
            self._init_embeddings()

    def _read_vocabulary(self):
        word_to_ids, self._voc_size, self._known_voc_size, self._vocabulary = read_vocabulary(
            self.config["vocabulary_file_name"],
            self.config["case_insensitive_embeddings"],
            self.config["num_oov_buckets"])
        return word_to_ids

    def _load_pretrained_embeddings(self, word_to_ids):
        with compat.gfile_open(self.config["word_embedding_file_name"], mode="rb") as embedding_f:
            self.pretrained_embeddings = None

            if self.config["with_header"]:
                next(embedding_f)

            for line in embedding_f:
                fields = line.strip().split()
                word = fields[0]

                if self.pretrained_embeddings is None:
                    assert self.config["word_embedding_size"] is None \
                           or self.config["word_embedding_size"] == len(fields) - 1
                    if self.config["word_embedding_size"] is None:
                        self.config["word_embedding_size"] = len(fields) - 1
                    self.pretrained_embeddings = np.random.normal(size=(self._voc_size, len(fields) - 1))

                # Lookup word in the vocabulary.
                if word in word_to_ids:
                    id = word_to_ids[word]
                    self.pretrained_embeddings[id] = np.asarray(fields[1:])

            initializer = tf.constant_initializer(
                        value=self.pretrained_embeddings)
            self.embedding = tf.Variable(
                    initial_value=lambda:initializer(
                        [self._voc_size, self.config["word_embedding_size"]],
                        dtype=tf.float32),
                    trainable=self.config["trainable"])

    def _init_embeddings(self):
        initializer = tf.initializers.GlorotUniform()
        self.embedding = tf.Variable(
            initial_value=lambda: initializer(
                [self._voc_size, self.config["word_embedding_size"]],
                dtype=tf.float32),
            trainable=self.config["trainable"])

    def _lookup(self, ids):
        return tf.nn.embedding_lookup(self.embedding, ids)

    def call(self, inputs):
        """
        run word embedding
        :param inputs: a tensor list of size 1; input tensor shape [batch_size, dim_0, ..., dim_n]
        :return: a tensor list of size 1; output tensor shape [batch_size, dim_0, ..., dim_n, word_embedding_size]
        """
        if not self.config["is_ids"]:
            ids = self._vocabulary.lookup(inputs)
        else:
            ids = inputs
        return self._lookup(ids)

    def _matmul(self, inputs):
        return tf.matmul(inputs, self.embedding, transpose_b=True)

    def matmul(self, inputs):
        return self._matmul(inputs)
