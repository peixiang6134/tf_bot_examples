import tensorflow as tf


def is_1d_list(l, t):
    if not isinstance(l, list):
        return False
    for value in l:
        if type(value) != t:
            return False
    return True


def is_nd_list(l, n, t):
    if n == 1:
        return is_1d_list(l, t)
    if not isinstance(l, list):
        return False
    for value in l:
        if not is_nd_list(value, n-1, t):
            return False
    return True


def check_format_number_range(x, dtype, min_value, max_value, argument_name):
    if type(x) != dtype:
        return False, "{} should be a {} within [{}, {}]!".foramt(
            argument_name, str(dtype), str(min_value), str(max_value))
    if x < min_value or x > max_value:
        return False, "{} should be a {} within [{}, {}]!".foramt(
            argument_name, str(dtype), str(min_value), str(max_value))
    return True, None


def check_format_int(value, argument_name):
    if type(value) != int:
        return False, argument_name + " should be a int!"
    return True, ""


def check_format_axis_shape_compatible(axis, input_shape):
    if axis is not None:
        if axis >= len(input_shape) or axis < (-1) * len(input_shape):
            return False, "axis and input shape does not match!"
    return True, ""


def check_format_type(value, one_type, argument_name):
    if value is None or type(value) is not one_type:
        return False, argument_name + " should be in type of " + str(one_type)
    return True, ""


def check_format_in_set(value, a_set, argument_name):
    if value not in a_set:
        return False, argument_name + " is not recognized!"
    return True, ""


def check_format_throw_exception(result):
    if not result[0]:
        raise Exception(result[1])
    return result


def check_format_argument_exists(config, key_value_types):
    for k, t in key_value_types.items():
        v = config.get(k)
        if v is None:
            return False, "cannot find key " + k + " in the config!"
        if not isinstance(v, t):
            return False, "key " + k + " should be in type of " + str(t)
    return True, ""


def check_format_optional(config, key_value_types):
    for k, t in key_value_types.items():
        v = config.get(k)
        if v is not None:
            if not isinstance(v, t):
                return False, "key " + k + " should be in type of " + str(t)
    return True, ""


def to_rnn_type(type_str):
    if type_str == "LSTM":
        return tf.keras.layers.LSTMCell
    if type_str == "GRU":
        return tf.keras.layers.GRUCell
    if type_str == "Vanilla" or type_str == "Simple":
        return tf.keras.layers.SimpleRNNCell
    return None


def shape_compatible(shape_0, shape_1):
    if len(shape_0) != len(shape_1):
        return False
    index = [i for i in range(0, len(shape_0)) if shape_0[i] is not None and shape_0[i] >= 0 and shape_1[i] is not None
             and shape_1[i] >= 0 and not tf.equal(shape_0[i], shape_1[i])]
    return len(index) == 0
