import tensorflow as tf
import random
from my_code.my_tensor import MyTensor


def generate_random_tensor_by_lengths(length_tensor, max_sequence_length, embedding_size):
    mask = tf.sequence_mask(length_tensor, max_sequence_length)
    mask_float = tf.cast(mask, tf.float32)
    shape = tf.shape(mask).numpy()
    shape = list(shape)
    shape.append(embedding_size)
    out_tensor = tf.Variable(tf.random.uniform(shape, 0, 1.0))
    out_tensor = tf.math.multiply(out_tensor, tf.expand_dims(mask_float, axis=-1))
    return out_tensor


def generate_random_lengths(max_length, shape):
    if len(shape) == 0:
        return None
    results = []
    for _ in range(0, shape[0]):
        if len(shape) == 1:
            nonzero = random.randrange(0, max_length + 1)
            results.append(nonzero)
        else:
            results.append(generate_random_lengths(max_length, shape[1:]))
    return results


def generate_tensor_with_padding(max_length, max_sequence_length, length_shape, embedding_size):
    lengths = generate_random_lengths(max_length, length_shape)
    return generate_random_tensor_by_lengths(lengths, max_sequence_length, embedding_size)


def generate_tensor_with_padding_int(max_length, max_sequence_length, length_shape, voc_size):
    float_tensor = generate_tensor_with_padding(max_length, max_sequence_length, length_shape, 1)
    my_tensor = MyTensor(float_tensor)
    mask = my_tensor.mask
    float_tensor = tf.squeeze(float_tensor, -1)
    float_tensor = tf.math.add(float_tensor, 1.0)
    float_tensor = tf.math.multiply(float_tensor, 0.5)
    return tf.cast(tf.math.multiply(float_tensor, voc_size), tf.int32), mask


if __name__ == "__main__":
    random.seed(3)
    tensor = generate_tensor_with_padding(5, 8, [4], 16)
