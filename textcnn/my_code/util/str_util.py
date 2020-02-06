import tensorflow as tf
import dependency.opennmt_v2.constants
from tensorflow.python.ops.ragged import ragged_tensor
import random
import string


def random_string(string_length=5):
    """Generate a random string of fixed length """
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for _ in range(string_length))


def print_self_str_list(str_list, output_as_list=True):
    assert type(str_list) is list
    a_str = ""
    for x in str_list:
        if len(a_str) > 0:
            a_str += ", "
        a_str += "\"" + x + "\""
    if output_as_list:
        a_str = "[" + a_str + "]"
    return a_str


def print_self_var_list_s(var_list, output_as_list=True):
    assert type(var_list) is list
    a_str = ""
    for x in var_list:
        if len(a_str) > 0:
            a_str += ", "
        if type(x) is list:
            a_str += print_self_var_list(x)
        else:
            a_str += str(x)
    if len(var_list) <= 1:
        return a_str
    if output_as_list:
        a_str = "[" + a_str + "]"
    return a_str


def print_self_var_list(var_list, *args):
    assert type(var_list) is list
    a_str = ""
    for x in var_list:
        if len(a_str) > 0:
            a_str += ", "
        if type(x) is list:
            a_str += print_self_var_list(x)
        else:
            a_str += str(x)
    a_str = "[" + a_str + "]"
    return a_str


def print_self_func_list(str_list):
    assert type(str_list) is list
    a_str = ""
    for x in str_list:
        if len(a_str) > 0:
            a_str += ", "
        a_str += x
    a_str = "[" + a_str + "]"
    return a_str


def print_self_func_list_2d(str_list_2d):
    a_str = ""
    for x in str_list_2d:
        if len(a_str) > 0:
            a_str += ", "
        a_str += print_self_func_list(x)
    a_str = "[" + a_str + "]"
    return a_str


def split_string_to_chars(input_str_tensor, padding_value):
    ragged = tf.strings.unicode_split(input_str_tensor, "UTF-8")
    chars = ragged.to_tensor(default_value=padding_value)
    lengths = ragged.row_lengths()
    return chars, lengths


def split_string_by_space(input_str_tensor, padding_value=dependency.opennmt_v2.constants.PADDING_TOKEN):
    output = tf.strings.split(input_str_tensor, " ")
    if isinstance(output, ragged_tensor.RaggedTensor):
        output = output.to_tensor(default_value=padding_value)
    return output


def split_in_middle(a_string, delimiter, allow_no_delimiter=False, allow_partial_empty=False):
    """
    given a delimter, return the left and right substrings
    :param a_string: input string
    :param delimiter: input delimiter
    :param allow_no_delimiter: if False, exception will be thrown if no delimiter is found
    :param allow_partial_empty: if False, exception will be thrown if the left or right side is an empty string after
    stripping
    :return: left and right substrings after stripping
    """
    a_string = a_string.strip()
    offset = a_string.find(delimiter)
    if offset < 0:
        if not allow_no_delimiter:
            raise Exception("no delimiter is found.")
        else:
            return a_string, ""
    x, y = a_string[0:offset].strip(), a_string[offset+1:].strip()
    if not allow_partial_empty:
        if len(x) == 0 or len(y) == 0:
            raise Exception("left or right part is empty.")
    return x, y


def split_and_strip(a_string, delimiter):
    """
    get all the substrings separated by the delimiter
    throws exception for empty substrings or empty final result
    """
    elements = a_string.split(delimiter)
    output = []
    for element in elements:
        element = element.strip()
        if len(element) > 0:
            output.append(element)
        else:
            raise Exception("empty substring exists between delimiters.")
    if len(output) == 0:
        raise Exception()
    return output


def skip_starting_empty_lines(lines):
    lines_left = []
    left = False
    for line in lines:
        if left:
            lines_left.append(line)
        elif len(line.strip()) > 0:
            lines_left.append(line)
            left = True
    return lines_left


def parse_int(s):
    try:
        return int(s)
    except ValueError:
        return None


def parse_float(s):
    try:
        return float(s)
    except ValueError:
        return None


def parse_num(s):
    x = parse_int(s)
    if x is not None:
        return x
    return parse_float(s)


def is_empty_after_stripping(s):
    if s is None:
        return True
    return len(s.strip()) == 0


def remove_suffix(s, delimiter):
    offset = s.rfind(delimiter)
    if offset >= 0:
        return s[0: offset]
    return s


def find_return_positive(s, token):
    o = s.find(token)
    if o < 0:
        return len(s)
    return o


def find_first(s, keywords):
    o = min(find_return_positive(s, kw) for _, kw in enumerate(keywords))
    if o == len(s):
        return -1
    return o


def find_first_return_positive(s, keywords):
    return min(find_return_positive(s, kw) for _, kw in enumerate(keywords))


def combine_python_code_multiple_lines(lines):
    new_lines = []
    current_line = ""
    for line in lines:
        current_line += line.rstrip()
        if str.endswith(current_line, "\\"):
            current_line = current_line[0:-1] + " "
        else:
            new_lines.append(current_line.rstrip())
    if len(current_line) > 0:
        new_lines.append(current_line.rstrip())
    return new_lines


def to_tensor_shape(shape_str: str):
    shape_str = shape_str.strip()
    if not str.startswith(shape_str, "[") or not str.endswith(shape_str, "]"):
        raise Exception(shape_str + " is not in a legal tensor shape format!")
    shape_str = shape_str.lstrip("[").rstrip("]")
    shape_strs = shape_str.split(",")
    if len(shape_strs) == 0:
        raise Exception(shape_str + " is not in a legal tensor shape format!")
    try:
        shape_list = [int(i.strip()) for i in shape_strs]
    except:
        raise Exception(shape_str + " is not in a legal tensor shape format!")
    return tf.TensorShape(shape_list)


def to_bool(s):
    s = s.lower()
    if s == "true" or s == "yes" or s == "是" or s == "ok":
        return True
    elif s == "false" or s == "no" or s == "否" or s == "不是":
        return False
    assert False


def get_function_name(ss):
    return ss[ss.find("=") + 1: ss.find("(")].strip()
