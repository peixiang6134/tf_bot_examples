class JsonFormatCheck:

    @staticmethod
    def has_typed_good_kv(json_obj, key, expected_type, f=None):
        value = json_obj.get(key)
        if value is None:
            return False, key + " cannot be found!"
        if type(value) != expected_type:
            return False, key + " 's value is not with type of " + expected_type + "!"
        if f is not None:
            if not f(value):
                return False, key + " 's value is not in good format!"
        return True, None
