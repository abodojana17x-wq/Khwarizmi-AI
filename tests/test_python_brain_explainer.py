import unittest

from rafig.python_brain import PythonAnalyzer, PythonExplainer, __version__

SAMPLE = '''\
import json


def organize(directory):
    """Group files."""
    groups = {}
    for item in directory:
        if item:
            groups[item] = 1
    return groups


class Organizer:
    """Organize things."""

    def __init__(self):
        self.items = {}

    def scan(self):
        return self.items


if __name__ == "__main__":
    organizer = Organizer()
    print(organizer.scan())
'''


class PythonBrainExplainerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.analyzer = PythonAnalyzer()
        cls.explainer = PythonExplainer()

    def test_explain_valid_program(self) -> None:
        result = self.analyzer.analyze(SAMPLE)
        text = self.explainer.explain(result)
        self.assertIn("Function 'organize'", text)
        self.assertIn("Class 'Organizer'", text)
        self.assertIn("imports 1 module(s)", text)
        self.assertIn("complexity", text)
        self.assertIn("Imported modules: json", text)

    def test_explain_function_metadata(self) -> None:
        result = self.analyzer.analyze(SAMPLE)
        text = self.explainer.explain(result)
        self.assertIn("accepts directory", text)
        self.assertIn("returns dict", text)

    def test_explain_syntax_error(self) -> None:
        result = self.analyzer.analyze("def (")
        text = self.explainer.explain(result)
        self.assertIn("could not be parsed", text)
        self.assertIn("SyntaxError", text)

    def test_explain_no_issues(self) -> None:
        result = self.analyzer.analyze("x = 1\nprint(x)\n")
        self.assertEqual(self.explainer.explain_issues(result), "No issues detected.")

    def test_explain_issues_lists_detected_problems(self) -> None:
        result = self.analyzer.analyze("print(missing)\n")
        text = self.explainer.explain_issues(result)
        self.assertIn("[error]", text)
        self.assertIn("missing", text)

    def test_explain_class_methods(self) -> None:
        result = self.analyzer.analyze(SAMPLE)
        text = self.explainer.explain(result)
        self.assertIn("methods: __init__, scan", text)

    def test_explain_via_analyzer_convenience(self) -> None:
        text = self.analyzer.explain("def hello():\n    return 'hi'\n")
        self.assertIn("Function 'hello'", text)

    def test_package_version(self) -> None:
        self.assertEqual(__version__, "0.7.0")


class PythonBrainDiagnosticsTests(unittest.TestCase):
    def test_diagnostics_keys(self) -> None:
        diagnostics = PythonAnalyzer().diagnostics("def f():\n    return 1\n")
        for key in (
            "parse_successful",
            "python_version",
            "functions",
            "classes",
            "imports",
            "loops",
            "cyclomatic_complexity",
            "undefined_names",
            "issues",
        ):
            self.assertIn(key, diagnostics)

    def test_diagnostics_undefined_names(self) -> None:
        diagnostics = PythonAnalyzer().diagnostics("print(missing)\n")
        self.assertEqual(diagnostics["undefined_names"], ["missing"])
        self.assertEqual(diagnostics["issues"]["error"], 1)

    def test_issue_severity_helpers(self) -> None:
        result = PythonAnalyzer().analyze(
            "def f():\n    print(missing)\n    while True:\n        pass\n"
        )
        self.assertEqual(len(result.error_issues), 1)
        self.assertGreaterEqual(len(result.warning_issues), 1)
        self.assertTrue(result.issues_of_kind("undefined-name"))
        self.assertFalse(result.is_valid)


if __name__ == "__main__":
    unittest.main()
