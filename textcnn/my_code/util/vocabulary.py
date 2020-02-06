import tensorflow as tf
from tensorflow.python.ops import lookup_ops as lookup
import dependency.opennmt_v2.constants
import os


def read_vocabulary(vocabulary_file_name, case_insensitive, num_oov_buckets):
    word_to_ids = dict()
    if not os.path.exists(vocabulary_file_name):
        raise Exception(vocabulary_file_name + " does not exists!")
    with tf.io.gfile.GFile(vocabulary_file_name, mode="rb") as vocabulary:
        count = 0
        for word in vocabulary:
            word = word.strip()
            if case_insensitive:
                word = word.lower()
            if word not in word_to_ids:
                word_to_ids.update({word: count})
                count += 1
    assert word_to_ids.get(dependency.opennmt_v2.constants.PADDING_TOKEN.encode()) == dependency.opennmt_v2.constants.PADDING_ID
    assert word_to_ids.get(
        dependency.opennmt_v2.constants.START_OF_SENTENCE_TOKEN.encode()) == dependency.opennmt_v2.constants.START_OF_SENTENCE_ID
    assert word_to_ids.get(
        dependency.opennmt_v2.constants.END_OF_SENTENCE_TOKEN.encode()) == dependency.opennmt_v2.constants.END_OF_SENTENCE_ID
    if num_oov_buckets == 1:
        assert word_to_ids.get(dependency.opennmt_v2.constants.UNKNOWN_TOKEN) is None
        word_to_ids.update({dependency.opennmt_v2.constants.UNKNOWN_TOKEN: len(word_to_ids)})
    else:
        for i in range(0, num_oov_buckets):
            assert word_to_ids.get(dependency.opennmt_v2.constants.UNKNOWN_TOKEN[0: -1] + str(i) + ">") is None
            word_to_ids.update({dependency.opennmt_v2.constants.UNKNOWN_TOKEN[0: -1] + str(i) + ">": len(word_to_ids)})
    voc_size = len(word_to_ids)
    known_voc_size = voc_size - num_oov_buckets

    index_table = lookup.index_table_from_file(vocabulary_file_name, num_oov_buckets, known_voc_size)

    return word_to_ids, voc_size, known_voc_size, index_table
