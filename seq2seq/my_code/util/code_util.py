import ast
import symtable


def get_obj_public_attributes(obj):
    return [a for a in dir(obj) if not a.startswith('_') and not callable(getattr(obj, a))]


def get_import_lines(lines):
    import_lines = []
    for line in lines:
        line = line.strip()
        if str.startswith(line, "import ") or (str.startswith(line, "from ") and line.find(" import ") > 0):
            import_lines.append(line)
    return import_lines


def get_function_code(lines, function_name):
    function_body = ""
    start = False
    empty_line = 0
    for line in lines:
        tmp = line.strip()
        if tmp.startswith("def " + function_name):
            start = True
        if start:
            if len(line.strip()) > 0:
                function_body += line + "\n"
                empty_line = 0
            else:
                empty_line += 1
                if empty_line >= 2:
                    break
                else:
                    function_body += line + "\n"
    return function_body


class CallCollector(ast.NodeVisitor):
    def __init__(self):
        self.calls = []
        self._current = []
        self._in_call = False
        self.names = set()
        self._currrent_name = []

    def visit_Call(self, node):
        self._current = []
        self._in_call = True
        self.generic_visit(node)

    def visit_Attribute(self, node):
        self._currrent_name.append(node.attr)
        if self._in_call:
            self._current.append(node.attr)
        self.generic_visit(node)

    def visit_Name(self, node):
        self._currrent_name.append(node.id)
        self.names.add('.'.join(self._currrent_name[::-1]))
        self._currrent_name = []
        if self._in_call:
            self._current.append(node.id)
            self.calls.append('.'.join(self._current[::-1]))
            # Reset the state
            self._current = []
            self._in_call = False
        self.generic_visit(node)


def get_functions_being_called(source_code):
    tree = ast.parse(source_code)
    cc = CallCollector()
    cc.visit(tree)
    return cc.calls, cc.names


if __name__ == "__main__":
    with open("C:/Users/niucheng/Desktop/test.py") as in_f:
        content = in_f.read()
        calls, names = get_functions_being_called(content)
        print(calls)
    table = symtable.symtable(content, "test", "exec")
    children = table.get_children()
    symbols = table.get_symbols()
    for s in symbols:
        x = s.get_name()
        y = s.is_referenced()
        z = s.is_imported()
        g = s.is_global()
        l = s.is_local()
        print("here")
    for child in children:
        x = child.get



