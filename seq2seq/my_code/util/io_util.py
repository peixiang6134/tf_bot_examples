import os
import glob
import shutil


def get_dir(file_path):
    return os.path.dirname(os.path.abspath(file_path))


def listdir_complete_path(dir_path):
    dir_path = os.path.abspath(dir_path)
    return [os.path.join(dir_path, x) for x in os.listdir(dir_path) if x != "." and x != ".."]


def from_path_patterns(path_patterns):
    if isinstance(path_patterns, list):
        path_list = list()
        for path_pattern in path_patterns:
            path_list.append(from_path_patterns(path_pattern))
        return path_list
    else:
        return glob.glob(path_patterns)


def get_parent_current_dir(dir_name):
    return os.path.split(os.path.abspath(dir_name))


def get_all_parent_dirs(dir_file_name):
    dir_file_name = os.path.abspath(dir_file_name)
    dirs = []
    while True:
        parent, current = get_parent_current_dir(dir_file_name)
        dirs.insert(0, current)
        if len(parent) == 0:
            break
        dir_file_name = parent
    if os.path.isfile(dir_file_name):
        return dirs[0: -1], dirs[-1]
    return dirs, ""


def get_path(dir_list):
    if dir_list is None or len(dir_list) == 0:
        return None
    if len(dir_list) == 1:
        return dir_list[0]
    path_str = dir_list[0]
    for i in range(1, len(dir_list)):
        path_str = os.path.join(path_str, dir_list[i])
    return path_str


def getlines_with_rstrip(input_file):
    lines = []
    while True:
        line = input_file.getline()
        if line is None:
            break
        lines.append(line.rstrip())
    return lines


def copy_file(src_file, tar_file):
    src_file = os.path.abspath(src_file)
    tar_file = os.path.abspath(tar_file)
    directory = os.path.dirname(tar_file)
    if not os.path.exists(directory):
        os.mkdir(directory)
    shutil.copy(src_file, tar_file)


def in_dir(dir_path_0, path_1):
    dir_path_0 = os.path.abspath(dir_path_0)
    path_1 = os.path.abspath(path_1)
    dir_paths_0, file_name = get_all_parent_dirs(dir_path_0)
    assert file_name is None
    dir_paths_1, file_name = get_all_parent_dirs(path_1)
    if len(dir_paths_0) < len(dir_paths_1):
        return False, None, None
    for i in range(0, len(dir_paths_0)):
        if dir_paths_0[0] != dir_paths_1[i]:
            return False, None, None
    return True, dir_paths_1[len(dir_paths_0):], file_name


def output_lines(file, lines, num_spaces):
    [file.write(" " * num_spaces + line + "\n") for line in lines]
