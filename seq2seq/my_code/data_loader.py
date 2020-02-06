import tensorflow as tf
from my_code.util.io_util import from_path_patterns
import my_code.util.str_util


def get_data_count(file_name):
    lines = open(file_name, "r", encoding="utf8").readlines()
    return len(lines)


def map_zip_text_files_tokenized(file_names, tokenizers, batch_size, max_length, padding_token):
    datasets = ()
    for i in range(0, len(file_names)):
        dataset = tf.data.TextLineDataset(file_names[i])
        if tokenizers is not None and tokenizers[i] is not None:
            dataset = dataset.map(lambda x: tokenizers[i](x))
        padded_shapes = (tf.TensorShape([max_length]))
        padding_values = (padding_token.encode())
        dataset = dataset.padded_batch(batch_size, padded_shapes=padded_shapes, padding_values=padding_values)
        datasets = datasets + (dataset, )
    return tf.data.Dataset.zip(datasets)


def train_dataset():

    instance_file_names = from_path_patterns(["./data/train_src.txt"])
    target_file_names = from_path_patterns(["./data/train_tgt.txt"])
    tokenizers_input = [my_code.util.str_util.split_string_by_space]
    pad_word = "<blank>"
    tokenizers_target = [my_code.util.str_util.split_string_by_space]
    pad_tag = "0"

    input_features = ["f0"]
    input_targets = ["l0"]

    repeat = 10
    need_shuffle = True
    max_sequence_length = 21
    batch_size = 128
    pre_batch_size = 10
    buffer_size = 1000

    dataset_size = get_data_count(instance_file_names[0][0])
    steps_per_epoch = dataset_size // batch_size

    dataset_input = map_zip_text_files_tokenized(instance_file_names, tokenizers_input, batch_size, max_sequence_length, pad_word)
    dataset_target = map_zip_text_files_tokenized(target_file_names, tokenizers_target, batch_size, max_sequence_length, pad_tag)
    dataset = tf.data.Dataset.zip((dataset_input, dataset_target))

    if need_shuffle:
        dataset = tf.data.Dataset.shuffle(dataset, buffer_size)
    dataset = tf.data.Dataset.repeat(dataset, repeat)

    dataset = dataset.prefetch(pre_batch_size)

    def dataset_iter():
        for one_item in dataset:
            features = dict()
            targets = dict()
            for i in range(0, len(one_item[0])):
                features.update({input_features[i]: one_item[0][i]})
            for i in range(0, len(one_item[1])):
                targets.update({input_targets[i]: one_item[1][i]})
            yield features, targets

    return dataset_iter, steps_per_epoch


def dev_dataset():

    instance_file_names = from_path_patterns(["./data/dev_src.txt"])
    target_file_names = from_path_patterns(["./data/dev_tgt.txt"])
    tokenizers_input = [my_code.util.str_util.split_string_by_space]
    pad_word = "<blank>"
    tokenizers_target = [my_code.util.str_util.split_string_by_space]
    pad_tag = "0"

    input_features = ["f0"]
    input_targets = ["l0"]

    repeat = 10
    need_shuffle = True
    max_sequence_length = 21
    batch_size = 128
    pre_batch_size = 1
    buffer_size = 1

    dataset_size = get_data_count(instance_file_names[0][0])
    steps_per_epoch = dataset_size // batch_size

    dataset_input = map_zip_text_files_tokenized(instance_file_names, tokenizers_input, batch_size, max_sequence_length, pad_word)
    dataset_target = map_zip_text_files_tokenized(target_file_names, tokenizers_target, batch_size, max_sequence_length, pad_tag)
    dataset = tf.data.Dataset.zip((dataset_input, dataset_target))

    if need_shuffle:
        dataset = tf.data.Dataset.shuffle(dataset, buffer_size)
    dataset = tf.data.Dataset.repeat(dataset, repeat)

    dataset = dataset.prefetch(pre_batch_size)

    def dataset_iter():
        for one_item in dataset:
            features = dict()
            targets = dict()
            for i in range(0, len(one_item[0])):
                features.update({input_features[i]: one_item[0][i]})
            for i in range(0, len(one_item[1])):
                targets.update({input_targets[i]: one_item[1][i]})
            yield features, targets

    return dataset_iter, steps_per_epoch


def test_dataset():

    instance_file_names = from_path_patterns(["./data/debug_src.txt"])
    tokenizers_input = [my_code.util.str_util.split_string_by_space]
    pad_word = "<blank>"
    input_features = ["f0"]

    repeat = 1
    need_shuffle = False
    max_sequence_length = 21
    batch_size = 128
    pre_batch_size = 1
    buffer_size = 1

    dataset_size = get_data_count(instance_file_names[0][0])
    steps_per_epoch = dataset_size // batch_size if dataset_size % batch_size == 0 else dataset_size // batch_size + 1

    dataset = map_zip_text_files_tokenized(instance_file_names, tokenizers_input, batch_size, max_sequence_length, pad_word)
    if need_shuffle:
        dataset = tf.data.Dataset.shuffle(dataset, buffer_size)
    dataset = tf.data.Dataset.repeat(dataset, repeat)

    dataset = dataset.prefetch(pre_batch_size)

    def dataset_iter():
        for one_item in dataset:
            features = dict()
            for i in range(0, len(one_item)):
                features.update({input_features[i]: one_item[i]})
            yield features, None

    return dataset_iter, steps_per_epoch
