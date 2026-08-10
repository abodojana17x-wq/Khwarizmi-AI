import unittest

from rafig.language.semantic_representation import SemanticAnalyzer


class SemanticRepresentationTests(unittest.TestCase):
    def test_arabic_repair_request_maps_to_code_repair(self) -> None:
        analyzer = SemanticAnalyzer()
        rep = analyzer.analyze("عايزك تصلحلي الكود")
        self.assertEqual(rep.intent, "code_repair")
        self.assertEqual(rep.action, "repair")
        self.assertIn(rep.object, {"code", "python_code"})

    def test_franco_repair_request_maps_to_same_intent(self) -> None:
        analyzer = SemanticAnalyzer()
        rep = analyzer.analyze("3ayezak tesala7 el code")
        self.assertEqual(rep.intent, "code_repair")
        self.assertEqual(rep.action, "repair")

    def test_english_python_repair_request_maps_to_same_intent(self) -> None:
        analyzer = SemanticAnalyzer()
        rep = analyzer.analyze("Fix this Python code")
        self.assertEqual(rep.intent, "code_repair")
        self.assertEqual(rep.action, "repair")
        self.assertEqual(rep.object, "python_code")

    def test_semantic_representation_includes_constraints_and_relationships(self) -> None:
        analyzer = SemanticAnalyzer()
        rep = analyzer.analyze("Create a Python script for sorting files")
        self.assertEqual(rep.intent, "code_generation")
        self.assertEqual(rep.action, "create")
        self.assertEqual(rep.object, "python_code")
        self.assertGreaterEqual(len(rep.constraints), 1)
        self.assertGreaterEqual(len(rep.relationships), 1)


if __name__ == "__main__":
    unittest.main()
