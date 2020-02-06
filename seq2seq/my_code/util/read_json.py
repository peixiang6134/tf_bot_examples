import json


def read_one_json(file):
    json_str = ""
    is_json = False
    while True:
        line = file.readline()
        if line is None or len(line) == 0:
            return None
        line = line.rstrip()
        if len(line) == 0:
            continue
        if not is_json and not str.startswith(line, "{") and not str.startswith(line, "["):
            return line
        json_str += line
        is_json = True
        try:
            return json.loads(json_str)
        except:
            continue
