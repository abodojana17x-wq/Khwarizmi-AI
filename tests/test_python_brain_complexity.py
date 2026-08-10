import unittest

from rafig.python_brain import ComplexityInfo, PythonAnalyzer
from rafig.python_brain.complexity import measure_function
from rafig.python_brain.parser import PythonParser


class PythonBrainComplexityTests(unittest.TestCase):
    def test_simple_function(self) -> None:
        tree = PythonParser.parse_strict("def f():\n    return 1\n")
        cyclomatic, nesting, statements = measure_function(tree.body[0])
        self.assertEqual(cyclomatic, 1)
        self.assertEqual(nesting, 0)
        self.assertEqual(statements, 1)

    def test_if_elif_else_function(self) -> None:
        source = (
            "def decision(x):\n"
            "    if x > 0:\n"
            "        if x > 10:\n"
            "            return 'big'\n"
            "        elif x > 5:\n"
            "            return 'medium'\n"
            "    else:\n"
            "        return 'small'\n"
            "    return 'unknown'\n"
        )
        tree = PythonParser.parse_strict(source)
        cyclomatic, nesting, statements = measure_function(tree.body[0])
        self.assertEqual(cyclomatic, 4)  # 1 + outer if + inner if + elif
        self.assertEqual(nesting, 2)

    def test_loops_and_boolean_operators(self) -> None:
        source = (
            "def loops(n):\n"
            "    total = 0\n"
            "    for i in range(n):\n"
            "        if i % 2 == 0 and n > 0:\n"
            "            total += i\n"
            "    while total < 100:\n"
            "        total *= 2\n"
            "    return total\n"
        )
        tree = PythonParser.parse_strict(source)
        cyclomatic, _, _ = measure_function(tree.body[0])
        self.assertEqual(cyclomatic, 5)  # 1 + for + if + 'and' + while

    def test_comprehension_counts_as_decision(self) -> None:
        source = "def f():\n    return [x for x in range(5) if x > 0]\n"
        tree = PythonParser.parse_strict(source)
        cyclomatic, _, _ = measure_function(tree.body[0])
        self.assertEqual(cyclomatic, 3)  # 1 + comprehension + filter if

    def test_module_complexity(self) -> None:
        result = PythonAnalyzer().analyze(
            "import os\n\ndef f(x):\n    if x:\n        return 1\n    return 0\n"
        )
        self.assertEqual(result.complexity.cyclomatic, 2)  # 1 + the if
        self.assertGreater(result.complexity.statement_count, 0)
        self.assertGreater(result.complexity.lines_of_code, 0)

    def test_complexity_per_function_attached(self) -> None:
        result = PythonAnalyzer().analyze(
            "def simple():\n    return 1\n\ndef complex_fn(x):\n"
            "    for i in range(10):\n        if i == x:\n            return i\n"
            "    if x > 5 and x < 20:\n        return x\n    return 0\n"
        )
        by_name = {fn.name: fn.complexity for fn in result.functions}
        self.assertEqual(by_name["simple"], 1)
        self.assertEqual(by_name["complex_fn"], 5)  # 1 + for + if + if + and

    def test_nesting_depth_attached(self) -> None:
        result = PythonAnalyzer().analyze(
            "def deep(x):\n"
            "    if x:\n"
            "        for i in range(3):\n"
            "            while i:\n"
            "                return i\n"
            "    return 0\n"
        )
        fn = result.functions[0]
        self.assertEqual(fn.nesting_depth, 3)

    def test_label_mapping(self) -> None:
        self.assertEqual(ComplexityInfo.label_for(4), "simple")
        self.assertEqual(ComplexityInfo.label_for(5), "moderate")
        self.assertEqual(ComplexityInfo.label_for(9), "complex")
        self.assertEqual(ComplexityInfo.label_for(16), "very complex")


if __name__ == "__main__":
    unittest.main()
