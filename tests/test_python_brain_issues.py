import unittest

from rafig.python_brain import IssueSeverity, PythonAnalyzer

BAD_PROGRAM = '''\
def broken():
    print(missing_var)
    return 5
    unused_after_return = 1


def other(items=[]):
    if items == None:
        pass
    while True:
        pass
'''


class PythonBrainIssueTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.analyzer = PythonAnalyzer()

    # -- undefined names ----------------------------------------------------

    def test_undefined_name_detection(self) -> None:
        result = self.analyzer.analyze("print(missing_var)\n")
        issues = result.issues_of_kind("undefined-name")
        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0].name, "missing_var")
        self.assertEqual(issues[0].severity, IssueSeverity.ERROR)

    def test_builtin_names_are_defined(self) -> None:
        result = self.analyzer.analyze("length = len([1, 2, 3])\n")
        self.assertEqual(result.issues_of_kind("undefined-name"), [])

    def test_imported_names_are_defined(self) -> None:
        result = self.analyzer.analyze("import json\ndata = json.dumps({})\n")
        self.assertEqual(result.issues_of_kind("undefined-name"), [])

    def test_names_defined_later_in_module_resolve(self) -> None:
        result = self.analyzer.analyze("def f():\n    return helper()\n\ndef helper():\n    return 1\n")
        self.assertEqual(result.issues_of_kind("undefined-name"), [])

    def test_closure_reference_resolves(self) -> None:
        source = "def outer():\n    x = 1\n    def inner():\n        return x\n    return inner\n"
        result = self.analyzer.analyze(source)
        self.assertEqual(result.issues_of_kind("undefined-name"), [])

    def test_method_cannot_see_class_scope(self) -> None:
        source = "class C:\n    y = 1\n    def m(self):\n        return y\n"
        result = self.analyzer.analyze(source)
        issues = result.issues_of_kind("undefined-name")
        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0].name, "y")

    def test_class_body_can_use_class_variables(self) -> None:
        source = "class C:\n    y = 1\n    z = y + 1\n"
        result = self.analyzer.analyze(source)
        self.assertEqual(result.issues_of_kind("undefined-name"), [])

    def test_star_import_suppresses_undefined_names(self) -> None:
        result = self.analyzer.analyze("from somewhere import *\nprint(mystery)\n")
        self.assertEqual(result.issues_of_kind("undefined-name"), [])

    def test_global_declaration_resolves(self) -> None:
        source = "counter = 0\ndef bump():\n    global counter\n    counter += 1\n    return counter\n"
        result = self.analyzer.analyze(source)
        self.assertEqual(result.issues_of_kind("undefined-name"), [])

    # -- use before assignment ------------------------------------------------

    def test_use_before_assignment(self) -> None:
        source = "def f():\n    print(value)\n    value = 10\n"
        result = self.analyzer.analyze(source)
        issues = result.issues_of_kind("used-before-assignment")
        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0].name, "value")
        self.assertEqual(issues[0].severity, IssueSeverity.WARNING)

    def test_no_false_use_before_assignment_for_closures(self) -> None:
        source = "def outer():\n    x = 1\n    def inner():\n        return x\n    return inner\n"
        result = self.analyzer.analyze(source)
        self.assertEqual(result.issues_of_kind("used-before-assignment"), [])

    # -- unreachable code ------------------------------------------------------

    def test_unreachable_after_return(self) -> None:
        result = self.analyzer.analyze("def f():\n    return 1\n    x = 2\n")
        issues = result.issues_of_kind("unreachable-code")
        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0].line, 3)

    def test_unreachable_after_raise(self) -> None:
        result = self.analyzer.analyze("def f():\n    raise ValueError()\n    x = 2\n")
        self.assertEqual(len(result.issues_of_kind("unreachable-code")), 1)

    def test_unreachable_after_break_in_loop(self) -> None:
        result = self.analyzer.analyze(
            "def f():\n    for i in range(3):\n        break\n        print(i)\n"
        )
        self.assertEqual(len(result.issues_of_kind("unreachable-code")), 1)

    def test_code_after_if_return_is_reachable(self) -> None:
        source = "def f(x):\n    if x:\n        return 1\n    return 2\n"
        result = self.analyzer.analyze(source)
        self.assertEqual(result.issues_of_kind("unreachable-code"), [])

    # -- suspicious constructs ---------------------------------------------------

    def test_mutable_default_argument(self) -> None:
        result = self.analyzer.analyze("def f(items=[]):\n    items.append(1)\n")
        issues = result.issues_of_kind("mutable-default-arg")
        self.assertEqual(len(issues), 1)

    def test_none_comparison(self) -> None:
        result = self.analyzer.analyze("def f(x):\n    return x == None\n")
        issues = result.issues_of_kind("none-comparison")
        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0].severity, IssueSeverity.WARNING)

    def test_identity_comparison_with_literal(self) -> None:
        result = self.analyzer.analyze("def f(x):\n    return x is 5\n")
        issues = result.issues_of_kind("identity-comparison")
        self.assertEqual(len(issues), 1)

    def test_bare_except_and_swallowed_exception(self) -> None:
        source = "def f():\n    try:\n        risky()\n    except:\n        pass\n"
        result = self.analyzer.analyze(source)
        self.assertEqual(len(result.issues_of_kind("bare-except")), 1)
        self.assertEqual(len(result.issues_of_kind("exception-swallowed")), 1)

    def test_shadowed_builtin_function(self) -> None:
        result = self.analyzer.analyze("def print():\n    return 1\n")
        issues = result.issues_of_kind("shadowed-builtin")
        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0].severity, IssueSeverity.WARNING)

    def test_shadowed_builtin_module_variable(self) -> None:
        result = self.analyzer.analyze("list = [1, 2, 3]\n")
        issues = result.issues_of_kind("shadowed-builtin")
        self.assertEqual(len(issues), 1)

    def test_shadowed_builtin_import(self) -> None:
        result = self.analyzer.analyze("import json as input\n")
        issues = result.issues_of_kind("shadowed-builtin")
        self.assertGreaterEqual(len(issues), 1)

    def test_shadowed_builtin_parameter_is_info(self) -> None:
        result = self.analyzer.analyze("def f(str):\n    return str\n")
        issues = result.issues_of_kind("shadowed-builtin")
        self.assertTrue(issues)
        self.assertEqual(issues[0].severity, IssueSeverity.INFO)

    def test_infinite_while_true(self) -> None:
        result = self.analyzer.analyze("def f():\n    while True:\n        pass\n")
        self.assertEqual(len(result.issues_of_kind("infinite-loop")), 1)

    def test_while_true_with_break_is_fine(self) -> None:
        source = "def f():\n    while True:\n        if stop():\n            break\n"
        result = self.analyzer.analyze(source)
        self.assertEqual(result.issues_of_kind("infinite-loop"), [])

    def test_duplicate_dict_key(self) -> None:
        result = self.analyzer.analyze('data = {"a": 1, "a": 2}\n')
        self.assertEqual(len(result.issues_of_kind("duplicate-dict-key")), 1)

    def test_empty_body_info(self) -> None:
        result = self.analyzer.analyze("def f():\n    pass\n")
        issues = result.issues_of_kind("empty-body")
        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0].severity, IssueSeverity.INFO)

    def test_literal_condition_info(self) -> None:
        result = self.analyzer.analyze("if True:\n    print('always')\n")
        issues = result.issues_of_kind("literal-condition")
        self.assertGreaterEqual(len(issues), 1)
        self.assertEqual(issues[0].severity, IssueSeverity.INFO)

    # -- unused symbols ----------------------------------------------------------

    def test_unused_variable(self) -> None:
        result = self.analyzer.analyze("def f():\n    unused = 1\n    return 2\n")
        issues = result.issues_of_kind("unused-variable")
        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0].name, "unused")

    def test_unused_import(self) -> None:
        result = self.analyzer.analyze("import os\n")
        issues = result.issues_of_kind("unused-variable")
        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0].name, "os")

    def test_used_import_in_function_not_flagged(self) -> None:
        source = "import json\ndef f():\n    return json.dumps({})\n"
        result = self.analyzer.analyze(source)
        self.assertEqual(result.issues_of_kind("unused-variable"), [])

    def test_unused_parameter_is_info(self) -> None:
        result = self.analyzer.analyze("def f(unused):\n    return 1\n")
        issues = result.issues_of_kind("unused-parameter")
        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0].severity, IssueSeverity.INFO)

    def test_underscore_names_are_not_flagged(self) -> None:
        result = self.analyzer.analyze("_ = 1\n")
        self.assertEqual(result.issues_of_kind("unused-variable"), [])

    # -- combined program ---------------------------------------------------------

    def test_bad_program_detects_expected_kinds(self) -> None:
        result = self.analyzer.analyze(BAD_PROGRAM)
        kinds = {issue.kind for issue in result.issues}
        self.assertIn("undefined-name", kinds)
        self.assertIn("unreachable-code", kinds)
        self.assertIn("mutable-default-arg", kinds)
        self.assertIn("none-comparison", kinds)
        self.assertIn("infinite-loop", kinds)
        self.assertFalse(result.is_valid)

    def test_clean_program_has_no_errors(self) -> None:
        source = "import json\n\ndef build():\n    return json.dumps({'ok': True})\n"
        result = self.analyzer.analyze(source)
        self.assertTrue(result.is_valid)
        self.assertEqual(result.error_issues, [])


if __name__ == "__main__":
    unittest.main()
