from my_code.util.str_util import split_in_middle, skip_starting_empty_lines
from my_code.util.str_util import to_bool


def parse_dict(lines, func):
    """
    read dict object from lines of text
    the empty line is used as the mark of the end

    :param lines: input lines of text
    :param func: a function being applied to every value before saving into dict
    :returns:
     dict object and lines left
    """

    kvs = {}
    lines_left = []
    lines = skip_starting_empty_lines(lines)
    done = False
    for line in lines:
        if done:
            lines_left.append(line)
            continue
        line = line.strip()
        if len(line) == 0:
            if not done:
                done = True
            continue
        key, value = split_in_middle(line, "=")
        kvs.update({key: func(value)})
    return kvs, lines_left


def inc(a_dict, k, increase, v0=0):
    """
    increase the value corresponding to k by increase. the initial value is v0
    :param a_dict:
    :param k: the input key
    :param increase: the increase of the value
    :param v0: the initial value
    :return: no return
    """

    if k not in a_dict:
        a_dict.update({k: v0 + increase})
    else:
        a_dict[k] += increase


def len_none_as_empty(a_dict, k):
    values = a_dict.get(k)
    if values is None:
        return 0
    return len(values)


def get_value_with_default(a_dict, k, v0):
    v = a_dict.get(k)
    if v is None:
        return v0
    return v


def get_value_or_set_default(a_dict, k, v0):
    if k not in a_dict:
        a_dict.update({k: v0})
    return a_dict.get(k)


def get_value_converted(a_dict, k, conversion, v0):
    v = a_dict.get(k)
    if v in conversion:
        return conversion.get(v)
    return v0


def get_value_or_return_error(a_dict, k, a_type, err):
    v = a_dict.get(k)
    if v is None:
        return None, err
    try:
        if a_type is bool:
            v = to_bool(v)
        else:
            v = a_type(v)
        return v, None
    except:
        return None, err


def get_total_number(list_dict):
    num = 0
    for _, l in list_dict.items():
        num += len(l) if l is not None else 0
    return num
