def dedup(a_list):
    a_set = set()
    def f(x): return [a_set.add(x), x]
    return [f(x)[1] for x in a_list if x not in a_set]


def append_lines(lines_0, lines_1, space_num):
    [lines_0.append(" " * space_num + line) for line in lines_1]
