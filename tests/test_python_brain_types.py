import unittest

from rafig.python_brain import PythonAnalyzer


class PythonBrainTypeInferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.analyzer = PythonAnalyzer()

    def infer_vars(self, source: str) -> dict:
        result = self.analyzer.analyze(source)
        return {var.name: var.inferred_type for var in result.variables}

    # -- literals ------------------------------------------------------------

    def test_literal_types(self) -> None:
        types = self.infer_vars(
            "a = 5\nb = 3.14\nc = 'text'\nd = True\ne = None\nf = [1, 2]\n"
            "g = {}\nh = (1, 2)\ni = b'bytes'\nj = {1, 2}\n"
        )
        self.assertEqual(types["a"], "int")
        self.assertEqual(types["b"], "float")
        self.assertEqual(types["c"], "str")
        self.assertEqual(types["d"], "bool")
        self.assertEqual(types["e"], "NoneType")
        self.assertEqual(types["f"], "list")
        self.assertEqual(types["g"], "dict")
        self.assertEqual(types["h"], "tuple")
        self.assertEqual(types["i"], "bytes")
        self.assertEqual(types["j"], "set")

    # -- annotations & parameters ---------------------------------------------

    def test_annotation_wins_over_literal(self) -> None:
        types = self.infer_vars("x: int = 'not really'\n")
        self.assertEqual(types["x"], "int")

    def test_parameter_annotation(self) -> None:
        result = self.analyzer.analyze("def f(name: str) -> int:\n    return 1\n")
        fn = result.functions[0]
        self.assertEqual(fn.parameters[0].annotation, "str")
        self.assertEqual(fn.return_annotation, "int")

    def test_parameter_default_infers_type(self) -> None:
        types = self.infer_vars("def f(limit=10, label='x'):\n    return limit\n")
        self.assertEqual(types["limit"], "int")
        self.assertEqual(types["label"], "str")

    # -- operations ------------------------------------------------------------

    def test_arithmetic_operations(self) -> None:
        types = self.infer_vars(
            "x = 1 + 2\ny = x / 2\nz = 3 * 4\nw = 10 // 3\nm = 7 % 2\n"
            "text = 'a' + 'b'\nrepeated = 'ab' * 2\n"
        )
        self.assertEqual(types["x"], "int")
        self.assertEqual(types["y"], "float")
        self.assertEqual(types["z"], "int")
        self.assertEqual(types["w"], "int")
        self.assertEqual(types["m"], "int")
        self.assertEqual(types["text"], "str")
        self.assertEqual(types["repeated"], "str")

    def test_comparisons_and_bools(self) -> None:
        types = self.infer_vars(
            "a = 1 > 0\nb = 1 < 0 or 2 > 1\nc = not a\n"
        )
        self.assertEqual(types["a"], "bool")
        self.assertEqual(types["b"], "bool")
        self.assertEqual(types["c"], "bool")

    # -- builtin calls ------------------------------------------------------------

    def test_builtin_call_types(self) -> None:
        types = self.infer_vars(
            "a = int('5')\nb = str(5)\nc = len([1, 2])\nd = input()\ne = open('f.txt')\n"
            "f = list(range(3))\ng = float(1)\nh = sorted([3, 1])\ni = isinstance(1, int)\n"
        )
        self.assertEqual(types["a"], "int")
        self.assertEqual(types["b"], "str")
        self.assertEqual(types["c"], "int")
        self.assertEqual(types["d"], "str")
        self.assertEqual(types["e"], "file")
        self.assertEqual(types["f"], "list")
        self.assertEqual(types["g"], "float")
        self.assertEqual(types["h"], "list")
        self.assertEqual(types["i"], "bool")

    # -- methods on builtin types -------------------------------------------------

    def test_string_methods(self) -> None:
        types = self.infer_vars(
            "text = 'hello'\nupper = text.upper()\nparts = text.split(',')\n"
            "joined = '|'.join(parts)\nfound = text.find('e')\nstarts = text.startswith('h')\n"
        )
        self.assertEqual(types["upper"], "str")
        self.assertEqual(types["parts"], "list")
        self.assertEqual(types["joined"], "str")
        self.assertEqual(types["found"], "int")
        self.assertEqual(types["starts"], "bool")

    def test_list_and_dict_methods(self) -> None:
        types = self.infer_vars(
            "data = [1, 2]\nresult = data.append(3)\nsize = data.count(1)\n"
            "mapping = {'a': 1}\nkeys = mapping.keys()\ncopy = mapping.copy()\n"
        )
        self.assertEqual(types["result"], "NoneType")
        self.assertEqual(types["size"], "int")
        self.assertEqual(types["keys"], "list")
        self.assertEqual(types["copy"], "dict")

    # -- comprehensions ------------------------------------------------------------

    def test_comprehension_types(self) -> None:
        types = self.infer_vars(
            "squares = [n * n for n in range(5)]\n"
            "pairs = {k: 1 for k in range(3)}\n"
            "unique = {x for x in range(3)}\n"
            "stream = (x for x in range(3))\n"
        )
        self.assertEqual(types["squares"], "list")
        self.assertEqual(types["pairs"], "dict")
        self.assertEqual(types["unique"], "set")
        self.assertEqual(types["stream"], "generator")

    # -- function return aggregation ----------------------------------------------

    def test_function_return_aggregation(self) -> None:
        result = self.analyzer.analyze(
            "def build():\n"
            "    return {'key': 1}\n\n"
            "def add(a: int, b: int):\n"
            "    return a + b\n\n"
            "def nothing():\n"
            "    pass\n\n"
            "def maybe(flag):\n"
            "    if flag:\n"
            "        return 1\n"
            "    return 'text'\n"
        )
        by_name = {fn.name: fn.returns for fn in result.functions}
        self.assertEqual(by_name["build"], ["dict"])
        self.assertEqual(by_name["add"], ["int"])
        self.assertEqual(by_name["nothing"], ["NoneType"])
        self.assertIn("int", by_name["maybe"])
        self.assertIn("str", by_name["maybe"])

    def test_call_to_user_function_infers_return(self) -> None:
        types = self.infer_vars(
            "def build():\n    return [1]\n\nresult = build()\n"
        )
        self.assertEqual(types["result"], "list")

    # -- instance & class attributes ------------------------------------------------

    def test_self_attribute_types(self) -> None:
        source = (
            "class Counter:\n"
            "    def __init__(self):\n"
            "        self.count = 0\n"
            "        self.name = 'counter'\n\n"
            "    def increment(self):\n"
            "        self.count += 1\n"
            "        return self.count\n\n"
            "    def label(self):\n"
            "        return self.name\n"
        )
        result = self.analyzer.analyze(source)
        cls = result.classes[0]
        by_name = {var.name: var for var in cls.instance_variables}
        self.assertEqual(by_name["count"].inferred_type, "int")
        self.assertEqual(by_name["name"].inferred_type, "str")
        fn_returns = {fn.name: fn.returns for fn in result.functions}
        self.assertEqual(fn_returns["increment"], ["int"])
        self.assertEqual(fn_returns["label"], ["str"])

    def test_class_variable_types(self) -> None:
        result = self.analyzer.analyze(
            "class Config:\n    version = '1.0'\n    retries = 3\n"
        )
        cls = result.classes[0]
        by_name = {var.name: var for var in cls.class_variables}
        self.assertEqual(by_name["version"].inferred_type, "str")
        self.assertEqual(by_name["retries"].inferred_type, "int")

    # -- misc ----------------------------------------------------------------------

    def test_exception_variable_type(self) -> None:
        result = self.analyzer.analyze(
            "def f():\n    try:\n        risky()\n    except ValueError as exc:\n        return exc\n"
        )
        exc = [var for var in result.variables if var.name == "exc"][0]
        self.assertEqual(exc.inferred_type, "ValueError")

    def test_global_variable_type(self) -> None:
        result = self.analyzer.analyze(
            "counter = 0\ndef bump():\n    global counter\n    counter += 1\n    return counter\n"
        )
        counter = [var for var in result.variables if var.name == "counter"][0]
        self.assertEqual(counter.inferred_type, "int")

    def test_unknown_is_not_guessed(self) -> None:
        result = self.analyzer.analyze("def f(x):\n    return x\n")
        fn = result.functions[0]
        self.assertEqual(fn.returns, ["unknown"])

    def test_arabic_variables(self) -> None:
        types = self.infer_vars("عدد = 10\nنص = 'مرحبا'\n")
        self.assertEqual(types["عدد"], "int")
        self.assertEqual(types["نص"], "str")


if __name__ == "__main__":
    unittest.main()
