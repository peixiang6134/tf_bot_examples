import tensorflow as  tf


def dropout(x, rate, training=None):
    if not training or rate == 0:
        return x
    return tf.nn.dropout(x, rate)


def select_by_ids(x, y):
    """
    given a tensor representing scores for a whole vocabulary, and the target words, compute the
    scores of the target words
    :param x: float tensor representing scores of a whole vocabulary, the typical shape is like
        [batch, time_step, voc_size]
    :param y: int tensor representing target word ids. Its typical shape is like [batch, time_step]
    :return: the scores for the target outputs. Its shape will like [batch, time_step]
    """
    z = tf.zeros(tf.shape(x))
    z = tf.add(z, tf.expand_dims(y, axis=-1))
    z = tf.cast(tf.not_equal(z, 0), tf.float32)
    z = tf.multiply(x, z)
    z = tf.reduce_sum(z, axis=-1)
    return z
