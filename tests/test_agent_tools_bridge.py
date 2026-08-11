"""
Comprehensive CPU Unit Tests for Khwarizmi Layered Agent and Tool Bridge.

Tests:
    - Input sanitization and multi-lingual modality detection ("ar", "en", "code", "mixed").
    - End-to-end AgentLoop orchestration linking Input -> Router -> Neural Core -> Tools.
    - Python Brain AST static analyzer optional tool verification.
    - Project Planner DAG symbolic tool verification.
    - Zero-overhead verification skipping on fast-path / high confidence.
"""

import unittest

from khwarizmi.config import get_tiny_test_config
from khwarizmi.agent import InputSanitizer, KhwarizmiAgentLoop
from khwarizmi.tools import (
    PythonAnalysisTool,
    ProjectPlannerTool,
    SelectiveVerifier,
    ToolVerificationRequest,
)


class TestAgentToolsBridge(unittest.TestCase):
    def setUp(self) -> None:
        self.config = get_tiny_test_config()
        self.agent = KhwarizmiAgentLoop(self.config)

    def test_agent_loop_sanitizes_multilingual_input(self) -> None:
        ar_input = "مرحبا بك في خوارزمي"
        en_input = "Hello Khwarizmi AI"
        code_input = "def factorial(n):\n    return 1 if n <= 1 else n * factorial(n - 1)"
        mixed_input = "اكتب دالة in Python لحساب المضروب"

        f_ar = InputSanitizer.sanitize(ar_input)
        f_en = InputSanitizer.sanitize(en_input)
        f_code = InputSanitizer.sanitize(code_input)
        f_mixed = InputSanitizer.sanitize(mixed_input)

        self.assertEqual(f_ar.detected_language, "ar")
        self.assertEqual(f_en.detected_language, "en")
        self.assertTrue(f_code.has_code_payload)
        self.assertEqual(f_mixed.detected_language, "mixed")

    def test_agent_loop_end_to_end_orchestration(self) -> None:
        prompt = "def foo(a, b):\n    return a + b\n"
        res = self.agent.process_request(prompt, deterministic_router=True)

        self.assertIsNotNone(res.neural_output)
        self.assertIn(res.selected_pathway, ["FAST", "CODING", "REASONING", "PROJECT_PLAN", "VERIFICATION"])
        self.assertGreaterEqual(res.confidence_score, 0.0)
        self.assertLessEqual(res.confidence_score, 1.0)
        self.assertIn("step_counter", res.diagnostics)

    def test_selective_verifier_python_brain_integration(self) -> None:
        code = "def valid_func(x):\n    return x * 2\n"
        res = PythonAnalysisTool.verify_code(code)

        self.assertTrue(res.success)
        self.assertEqual(res.tool_name, "python_brain")
        self.assertEqual(res.diagnostics["error_count"], 0)

    def test_selective_verifier_project_planner_integration(self) -> None:
        plan_text = "Create a Python project that downloads images, processes them, and saves reports."
        res = ProjectPlannerTool.verify_dag_plan(plan_text)

        self.assertTrue(res.success)
        self.assertEqual(res.tool_name, "project_planner")
        self.assertGreaterEqual(res.diagnostics["task_count"], 1)

    def test_selective_verifier_skip_on_high_confidence(self) -> None:
        req = ToolVerificationRequest(
            tool_name="python_brain",
            payload="def test(): pass",
        )
        res = SelectiveVerifier.verify(req, needs_verification=False)

        self.assertTrue(res.success)
        self.assertEqual(res.tool_name, "skipped")
        self.assertEqual(res.execution_overhead_ms, 0.0)


if __name__ == "__main__":
    unittest.main()
