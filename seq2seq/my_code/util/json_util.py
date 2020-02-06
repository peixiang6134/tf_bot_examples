import json


def get_map_array(json_str, keys):
    jarray = json.loads(json_str)
    for i in range(0, len(keys)):
        assert jarray[i].get(keys[i]) is not None
    return jarray

