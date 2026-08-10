import unittest

from rafig.python_brain import PythonAnalyzer

SAMPLE = '''\
import os
import json
from pathlib import Path


def organize_by_extension(directory: str) -> dict:
    """Group files in a directory by their extension."""
    groups = {}
    for item in Path(directory).iterdir():
        if item.is_file():
            ext = item.suffix.lower()
            groups.setdefault(ext, []).append(str(item))
    return groups


class FileOrganizer:
    """Organize files into groups by extension."""

    def __init__(self, root="."):
        self.root = root
        self.groups = {}

    def scan(self):
        self.groups = organize_by_extension(self.root)
        return self.groups

    def save_report(self, target):
        with open(target, "w", encoding="utf-8") as handle:
            json.dump(self.groups, handle, indent=2)
        return target


if __name__ == "__main__":
    organizer = FileOrganizer()
    report = organizer.save_report("report.json")
    print(report)
'''


class PythonBrainAnalyzerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.result = PythonAnalyzer().analyze(SAMPLE)

    def test_parse_successful(self) -> None:
        self.assertTrue(self.result.parse_successful)
        self.assertIsNone(self.result.syntax_error)
        self.assertTrue(self.result.is_valid)

    # -- functions ---------------------------------------------------------

    def test_function_discovery(self) -> None:
        names = [fn.name for fn in self.result.functions]
        self.assertEqual(names, ["organize_by_extension", "__init__", "scan", "save_report"])

    def test_function_metadata(self) -> None:
        fn = self.result.functions[0]
        self.assertFalse(fn.is_method)
        self.assertIsNone(fn.class_name)
        self.assertEqual(fn.scope_name, "organize_by_extension")
        self.assertEqual(fn.docstring, "Group files in a directory by their extension.")
        self.assertEqual(fn.return_annotation, "dict")
        self.assertEqual([p.name for p in fn.parameters], ["directory"])
        self.assertEqual(fn.parameters[0].annotation, "str")
        self.assertEqual(fn.parameters[0].has_default, False)
        self.assertEqual(fn.complexity, 3)

    def test_method_metadata(self) -> None:
        scan = self.result.functions[2]
        self.assertTrue(scan.is_method)
        self.assertEqual(scan.class_name, "FileOrganizer")
        self.assertEqual(scan.scope_name, "FileOrganizer.scan")

    def test_init_method_with_default_parameter(self) -> None:
        init = self.result.functions[1]
        self.assertTrue(init.is_method)
        self.assertEqual(init.parameters[0].name, "self")
        self.assertEqual(init.parameters[1].name, "root")
        self.assertTrue(init.parameters[1].has_default)
        self.assertEqual(init.parameters[1].default, "'.'")
        self.assertEqual(init.returns, ["NoneType"])

    # -- classes ------------------------------------------------------------

    def test_class_discovery(self) -> None:
        self.assertEqual(len(self.result.classes), 1)
        cls = self.result.classes[0]
        self.assertEqual(cls.name, "FileOrganizer")
        self.assertEqual([m.name for m in cls.methods], ["__init__", "scan", "save_report"])
        self.assertFalse(cls.is_dataclass)
        self.assertEqual(cls.bases, [])

    def test_instance_variables(self) -> None:
        cls = self.result.classes[0]
        names = sorted(var.name for var in cls.instance_variables)
        self.assertEqual(names, ["groups", "root"])
        by_name = {var.name: var for var in cls.instance_variables}
        self.assertEqual(by_name["root"].inferred_type, "str")
        self.assertEqual(by_name["groups"].inferred_type, "dict")

    # -- imports --------------------------------------------------------------

    def test_import_discovery(self) -> None:
        self.assertEqual(len(self.result.imports), 3)
        modules = [(imp.module, imp.is_from) for imp in self.result.imports]
        self.assertIn(("os", False), modules)
        self.assertIn(("json", False), modules)
        self.assertIn(("pathlib", True), modules)
        from_import = [imp for imp in self.result.imports if imp.is_from][0]
        self.assertEqual(from_import.names, ["Path"])

    # -- scopes & symbol table ------------------------------------------------

    def test_scope_chain(self) -> None:
        names = [scope.name for scope in self.result.scopes]
        self.assertIn("<module>", names)
        self.assertIn("organize_by_extension", names)
        self.assertIn("FileOrganizer", names)
        self.assertIn("FileOrganizer.scan", names)

    def test_method_scope_skips_class_scope(self) -> None:
        scopes = {scope.name: scope for scope in self.result.scopes}
        self.assertEqual(scopes["FileOrganizer.scan"].parent, "<module>")
        self.assertEqual(scopes["FileOrganizer"].parent, "<module>")

    def test_symbol_table_resolution(self) -> None:
        table = self.result.symbol_table
        self.assertIsNotNone(table.resolve("Path", "organize_by_extension"))
        self.assertIsNotNone(table.resolve("json", "FileOrganizer.save_report"))
        self.assertIsNotNone(table.resolve("len", "FileOrganizer.save_report"))
        entry = table.resolve("len", "FileOrganizer.save_report")
        self.assertEqual(entry.kind, "builtin")
        self.assertIsNone(table.resolve("totally_missing_name", "<module>"))

    def test_symbol_resolution_in_method_body(self) -> None:
        # module-level function used inside a method body must resolve
        table = self.result.symbol_table
        entry = table.resolve("organize_by_extension", "FileOrganizer.scan")
        self.assertIsNotNone(entry)
        self.assertEqual(entry.kind, "function")

    # -- control flow -----------------------------------------------------------

    def test_loop_discovery(self) -> None:
        self.assertEqual(len(self.result.loops), 1)
        loop = self.result.loops[0]
        self.assertEqual(loop.kind, "for")
        self.assertEqual(loop.variables, ["item"])
        self.assertIn("iterdir", loop.iterable or "")
        self.assertFalse(loop.has_break)
        self.assertFalse(loop.has_continue)

    def test_control_flow_nodes(self) -> None:
        kinds = [node.kind for node in self.result.control_flow]
        self.assertIn("if", kinds)
        self.assertIn("return", kinds)
        self.assertIn("with", kinds)
        branches = [node.branches for node in self.result.control_flow if node.kind == "if"]
        self.assertEqual(set(branches), {1})  # simple if + the __main__ guard

    # -- calls & returns --------------------------------------------------------

    def test_call_discovery(self) -> None:
        names = [call.name for call in self.result.calls]
        self.assertTrue(any(name.endswith(".iterdir") for name in names))
        self.assertIn("groups.setdefault", names)
        self.assertIn("open", names)
        self.assertIn("json.dump", names)
        self.assertIn("organizer.save_report", names)

    def test_calls_attached_to_function(self) -> None:
        fn = self.result.functions[0]
        self.assertGreaterEqual(len(fn.calls), 3)

    def test_return_values(self) -> None:
        returns = [(r.value, r.scope) for r in self.result.returns]
        self.assertIn(("groups", "organize_by_extension"), returns)
        self.assertIn(("self.groups", "FileOrganizer.scan"), returns)
        self.assertIn(("target", "FileOrganizer.save_report"), returns)

    def test_inferred_return_types(self) -> None:
        by_name = {fn.name: fn.returns for fn in self.result.functions}
        self.assertEqual(by_name["organize_by_extension"], ["dict"])
        self.assertEqual(by_name["scan"], ["dict"])
        self.assertEqual(by_name["__init__"], ["NoneType"])
        self.assertEqual(by_name["save_report"], ["unknown"])

    # -- program structure --------------------------------------------------------

    def test_structure(self) -> None:
        structure = self.result.structure
        self.assertEqual(structure.module_docstring, None)
        self.assertGreater(structure.lines_of_code, 20)
        self.assertEqual(structure.modules_imported, ["os", "json", "pathlib"])
        self.assertTrue(structure.has_main_guard)
        self.assertEqual(structure.total_functions, 4)
        self.assertEqual(structure.total_classes, 1)
        self.assertEqual(structure.total_imports, 3)
        kinds = [stmt.kind for stmt in structure.top_level_statements]
        self.assertIn("Import", kinds)
        self.assertIn("FunctionDef", kinds)
        self.assertIn("ClassDef", kinds)
        self.assertIn("If", kinds)

    def test_entry_points(self) -> None:
        self.assertEqual(
            sorted(self.result.structure.entry_points),
            sorted(["FileOrganizer", "organizer.save_report", "print"]),
        )

    def test_statement_counts(self) -> None:
        counts = self.result.structure.statement_counts
        self.assertGreaterEqual(counts.get("FunctionDef", 0), 3)
        self.assertGreaterEqual(counts.get("For", 0), 1)

    # -- diagnostics & inspection -------------------------------------------------

    def test_diagnostics_report(self) -> None:
        diagnostics = self.result.diagnostics()
        self.assertTrue(diagnostics["parse_successful"])
        self.assertEqual(diagnostics["functions"], 4)
        self.assertEqual(diagnostics["classes"], 1)
        self.assertIn("python_version", diagnostics)
        self.assertIn("issues", diagnostics)

    def test_ast_summary(self) -> None:
        summary = self.result.ast_summary()
        self.assertIn("Module", summary)
        self.assertIn("FunctionDef", summary)

    def test_issues_of_kind_helper(self) -> None:
        kinds = {issue.kind for issue in self.result.issues}
        self.assertIn("unused-variable", kinds)  # os import is unused in the sample


class PythonBrainAnalyzerEdgeCases(unittest.TestCase):
    def test_analyze_empty_source(self) -> None:
        result = PythonAnalyzer().analyze("")
        self.assertTrue(result.parse_successful)
        self.assertEqual(result.structure.lines_of_code, 0)
        self.assertEqual(result.functions, [])

    def test_analyze_syntax_error(self) -> None:
        result = PythonAnalyzer().analyze("def (\n")
        self.assertFalse(result.parse_successful)
        self.assertIsNotNone(result.syntax_error)
        self.assertFalse(result.is_valid)

    def test_arabic_identifiers(self) -> None:
        result = PythonAnalyzer().analyze("اسم = 5\nresult = اسم + 1\n")
        by_name = {var.name: var for var in result.variables}
        self.assertEqual(by_name["اسم"].inferred_type, "int")
        self.assertEqual(by_name["result"].inferred_type, "int")

    def test_async_function(self) -> None:
        result = PythonAnalyzer().analyze("async def fetch(url):\n    return url\n")
        fn = result.functions[0]
        self.assertTrue(fn.is_async)
        self.assertEqual(fn.scope_name, "fetch")

    def test_dataclass_detection(self) -> None:
        result = PythonAnalyzer().analyze(
            "@dataclass\nclass Point:\n    x: int\n    y: int\n"
        )
        self.assertEqual(len(result.classes), 1)
        self.assertTrue(result.classes[0].is_dataclass)
        kinds = {var.kind for var in result.variables if var.scope == "Point"}
        self.assertEqual(kinds, {"class"})

    def test_lambda_scope(self) -> None:
        result = PythonAnalyzer().analyze("f = lambda x: x + 1\n")
        names = [scope.name for scope in result.scopes]
        self.assertTrue(any("<lambda>" in name for name in names))

    def test_comprehension_scope_does_not_leak(self) -> None:
        result = PythonAnalyzer().analyze("vals = [i for i in range(3)]\n")
        kinds = {var.kind for var in result.variables if var.name == "i"}
        self.assertEqual(kinds, {"comprehension"})

    def test_parameters_kinds(self) -> None:
        result = PythonAnalyzer().analyze(
            "def f(a, b=2, *args, c=3, **kwargs):\n    return a\n"
        )
        fn = result.functions[0]
        kinds = {param.name: param.kind for param in fn.parameters}
        self.assertEqual(kinds["a"], "positional")
        self.assertEqual(kinds["b"], "positional")
        self.assertEqual(kinds["args"], "varargs")
        self.assertEqual(kinds["c"], "keyword_only")
        self.assertEqual(kinds["kwargs"], "varkw")
        by_name = {param.name: param for param in fn.parameters}
        self.assertTrue(by_name["b"].has_default)
        self.assertEqual(by_name["b"].default, "2")
        self.assertEqual(by_name["c"].default, "3")

    def test_match_statement_bindings(self) -> None:
        result = PythonAnalyzer().analyze(
            "def f(v):\n"
            "    match v:\n"
            "        case [a, b]:\n"
            "            return a\n"
            "        case _:\n"
            "            return 0\n"
        )
        bound = {var.name for var in result.variables if var.scope == "f"}
        self.assertIn("a", bound)
        self.assertIn("b", bound)
        flow = [node.kind for node in result.control_flow]
        self.assertIn("match", flow)


if __name__ == "__main__":
    unittest.main()
