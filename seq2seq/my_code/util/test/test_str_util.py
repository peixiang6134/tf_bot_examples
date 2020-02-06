from my_code.util.str_util import split_string_to_chars, find_first, parse_shapes
import tensorflow as tf
import numpy as np


def test():
    offset = find_first("adbdd", ["xd", "xbdd"])
    print(offset)
    assert tf.executing_eagerly()
    strings = np.asarray(["I am", "hi"], np.str)
    string_tensor = tf.convert_to_tensor(strings)
    char_tensor = split_string_to_chars(string_tensor, "<pad>")
    print(char_tensor)


if __name__ == "__main__":
    test()
