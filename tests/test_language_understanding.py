import unittest

from rafig.language.language_understanding import LanguageAnalyzer


class LanguageUnderstandingTests(unittest.TestCase):
    def test_arabic_request_detects_arabic_and_fix_intent(self) -> None:
        analyzer = LanguageAnalyzer()
        result = analyzer.analyze("عايز تصلح الكود")
        self.assertEqual(result.detected_language, "ar")
        self.assertIn("repair", result.intent_candidates)

    def test_franco_request_detects_franco_and_fix_intent(self) -> None:
        analyzer = LanguageAnalyzer()
        result = analyzer.analyze("3ayezak tesala7 el code")
        self.assertIn(result.detected_language, {"franco", "mixed", "en"})
        self.assertIn("repair", result.intent_candidates)

    def test_english_request_detects_english(self) -> None:
        analyzer = LanguageAnalyzer()
        result = analyzer.analyze("I want you to fix the code")
        self.assertEqual(result.detected_language, "en")
        self.assertIn("repair", result.intent_candidates)

    def test_mixed_request_has_mixed_language(self) -> None:
        analyzer = LanguageAnalyzer()
        result = analyzer.analyze("عايز أعمل Python program")
        self.assertGreaterEqual(result.language_mix["ar"], 1)
        self.assertIn("python", result.entities)

    def test_code_snippet_is_marked_as_code(self) -> None:
        analyzer = LanguageAnalyzer()
        result = analyzer.analyze("def hello(name): return name")
        self.assertTrue(result.is_code_like)
        self.assertIn("function", result.entities)


if __name__ == "__main__":
    unittest.main()
