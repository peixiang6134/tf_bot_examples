import tensorflow as tf


"""this file contains some commonly used computation related to time step masks. These computaion are needed
   for most of the text processing tasks since a sequence of tokens is a basic representation of text.
"""


def compute_2d_mask_padding_token(input_tensor, padding_token):
    """
    given a 2D input tensor, compute the corresponding mask by checking the padding token
    :param input: str tensor, shape: [batch, time_steps]
    :param padding_token: str, the token used for padding, e.g. in OpenNMT, people use <blank>.
    :return: bool tensor, shape [batch, time_steps], True if the time step is for real, False if the
            time step is a padding
    """
    return tf.not_equal(input_tensor, padding_token)


def compute_2d_mask_toke_id(input_tensor, token_id):
    return tf.not_equal(input_tensor, token_id)


def compute_mask_embedding(input_tensor):
    emb_sum = tf.reduce_sum(tf.abs(input_tensor), -1)
    return tf.greater(emb_sum, 0.0)


def run_mask_embedding(input_tensor, mask):
    mask_float = tf.cast(mask, tf.float32)
    mask_float = tf.expand_dims(mask_float, -1)
    return tf.math.multiply(input_tensor, mask_float)


def run_mask_id(input_tensor, mask):
    mask_int = tf.cast(mask, tf.int32)
    return tf.math.multiply(input_tensor, mask_int)
