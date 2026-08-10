import tempfile
import unittest
from pathlib import Path

from rafig.python_brain.parser import PythonParseError, PythonParser


class PythonParserTests(unittest.TestCase):
    def test_parse_valid_source(self) -> None:
        tree, error = PythonParser.parse("x = 1\ndef f():\n    return x\n")
        self.assertIsNone(error)
        self.assertIsNotNone(tree)

    def test_parse_invalid_source_returns_syntax_issue(self) -> None:
        tree, error = PythonParser.parse("def broken(:\n    pass\n")
        self.assertIsNone(tree)
        self.assertIsNotNone(error)
        self.assertEqual(error.error_type, "SyntaxError")
        self.assertGreaterEqual(error.line, 1)
        self.assertGreaterEqual(error.column, 1)
        self.assertIn("invalid syntax", error.message)

    def test_parse_strict_success(self) -> None:
        tree = PythonParser.parse_strict("value = 42")
        self.assertEqual(tree.body[0].targets[0].id, "value")

    def test_parse_strict_raises_python_parse_error(self) -> None:
        with self.assertRaises(PythonParseError) as context:
            PythonParser.parse_strict("if True print('x')")
        self.assertIn("invalid syntax", str(context.exception))
        self.assertGreaterEqual(context.exception.line, 1)

    def test_parse_file(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "sample.py"
            path.write_text("def hello():\n    return 'hi'\n", encoding="utf-8")
            tree, error = PythonParser.parse_file(path)
            self.assertIsNone(error)
            self.assertIsNotNone(tree)

    def test_inspect_returns_ast_dump(self) -> None:
        tree = PythonParser.parse_strict("x = [1, 2]")
        dump = PythonParser.inspect(tree)
        self.assertIn("Module", dump)
        self.assertIn("Assign", dump)

    def test_empty_source_parses(self) -> None:
        tree, error = PythonParser.parse("")
        self.assertIsNone(error)
        self.assertIsNotNone(tree)
        self.assertEqual(len(tree.body), 0)


if __name__ == "__main__":
    unittest.main()
