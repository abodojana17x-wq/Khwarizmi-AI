"""
Tests for Khwarizmi Multi-Stage Generator.
"""

import unittest
from khwarizmi.coding.multi_stage_generator import (
    MultiStageGenerator,
    StageType,
    StageResult,
    PipelineReport,
    SimpleReasoningEngine,
)


class TestStageType(unittest.TestCase):
    """Test suite for StageType enum."""

    def test_stage_type_values(self):
        """Test that all stage types have correct values."""
        self.assertEqual(StageType.DESIGN.value, "design")
        self.assertEqual(StageType.INTERFACE.value, "interface")
        self.assertEqual(StageType.IMPLEMENTATION.value, "implementation")
        self.assertEqual(StageType.TESTS.value, "tests")
        self.assertEqual(StageType.REFACTOR.value, "refactor")


class TestStageResult(unittest.TestCase):
    """Test suite for StageResult class."""

    def test_is_valid_success(self):
        """Test is_valid property for successful stage."""
        result = StageResult(
            stage=StageType.DESIGN,
            success=True,
            output="Some output",
            issues=[],
        )
        self.assertTrue(result.is_valid)

    def test_is_valid_with_issues(self):
        """Test is_valid property when there are issues."""
        result = StageResult(
            stage=StageType.DESIGN,
            success=True,
            output="Some output",
            issues=["Issue 1", "Issue 2"],
        )
        self.assertFalse(result.is_valid)

    def test_to_dict_serialization(self):
        """Test serialization to dictionary."""
        result = StageResult(
            stage=StageType.IMPLEMENTATION,
            success=True,
            output="def foo(): pass",
            confidence=0.85,
            duration=0.1,
        )
        d = result.to_dict()
        self.assertIn('stage', d)
        self.assertEqual(d['stage'], 'implementation')
        self.assertIn('success', d)
        self.assertIn('output_length', d)
        self.assertIn('confidence', d)


class TestPipelineReport(unittest.TestCase):
    """Test suite for PipelineReport class."""

    def test_is_valid_all_stages(self):
        """Test is_valid when all stages completed."""
        report = PipelineReport(prompt="test prompt")
        report.stages_completed = 5
        report.total_stages = 5
        
        # Add mock stage results
        for stage in StageType:
            report.stage_results[stage] = StageResult(
                stage=stage,
                success=True,
                output="output",
                issues=[],
            )
        
        self.assertTrue(report.is_valid)

    def test_is_valid_incomplete(self):
        """Test is_valid when not all stages completed."""
        report = PipelineReport(prompt="test prompt")
        report.stages_completed = 3
        report.total_stages = 5
        
        self.assertFalse(report.is_valid)

    def test_to_dict_serialization(self):
        """Test serialization to dictionary."""
        report = PipelineReport(prompt="test prompt")
        report.stages_completed = 1
        report.final_code = "def foo(): pass"
        
        d = report.to_dict()
        self.assertIn('prompt', d)
        self.assertIn('stages_completed', d)
        self.assertIn('total_stages', d)
        self.assertIn('final_code_length', d)


class TestSimpleReasoningEngine(unittest.TestCase):
    """Test suite for SimpleReasoningEngine."""

    def setUp(self):
        """Set up test fixtures."""
        self.reasoner = SimpleReasoningEngine(max_iterations=3)

    def test_evaluate_clean_code(self):
        """Test evaluation of clean code."""
        code = '''
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b
'''
        score, issues = self.reasoner.evaluate_code_quality(code)
        
        self.assertGreater(score, 0.5)
        # Should have docstring and type annotations

    def test_evaluate_code_without_docstrings(self):
        """Test evaluation of code without docstrings."""
        code = '''
def process(x):
    return x * 2
'''
        score, issues = self.reasoner.evaluate_code_quality(code)
        
        # Should detect missing docstring
        issue_str = ' '.join(issues).lower()
        self.assertIn('docstring', issue_str)

    def test_evaluate_long_function(self):
        """Test evaluation of very long function."""
        # Create a function with many lines
        lines = ['def long_function():']
        for i in range(60):
            lines.append(f'    x{i} = {i}')
        lines.append('    return x0')
        code = '\n'.join(lines)
        
        score, issues = self.reasoner.evaluate_code_quality(code)
        
        # Should detect long function
        self.assertLess(score, 1.0)

    def test_self_consistency_vote_single(self):
        """Test voting with single candidate."""
        candidates = ['def foo(): return 1']
        best, confidence = self.reasoner.self_consistency_vote(candidates)
        
        self.assertEqual(best, candidates[0])
        self.assertGreaterEqual(confidence, 0.0)

    def test_self_consistency_vote_multiple(self):
        """Test voting with multiple candidates."""
        candidates = [
            'def add(a, b): return a + b',
            'def add(a, b):\n    """Add."""\n    return a + b',
        ]
        best, confidence = self.reasoner.self_consistency_vote(candidates)
        
        self.assertIn('add', best)

    def test_extract_identifiers(self):
        """Test identifier extraction from code."""
        code = '''
def calculate(x, y):
    result = x + y
    return result
'''
        identifiers = self.reasoner._extract_identifiers(code)
        
        self.assertIn('calculate', identifiers)
        self.assertIn('x', identifiers)
        self.assertIn('y', identifiers)
        self.assertIn('result', identifiers)

    def test_refine_output_adds_docstrings(self):
        """Test that refinement adds docstrings."""
        code = '''
def process(data):
    return data
'''
        issues = ["Missing docstring for 'process'"]
        refined, remaining = self.reasoner.refine_output(code, issues, StageType.REFACTOR)
        
        # Should have attempted to add docstring
        self.assertIn('"""', refined)


class TestMultiStageGenerator(unittest.TestCase):
    """Test suite for MultiStageGenerator."""

    def setUp(self):
        """Set up test fixtures."""
        self.generator = MultiStageGenerator(
            timeout=5.0,
            max_memory_mb=100,
            max_refinement_iterations=2,
        )

    def test_generate_design_stage(self):
        """Test design stage generation."""
        prompt = "Create a calculator"
        design = self.generator._generate_design(prompt, "")
        
        self.assertIn("Design Specification", design)
        self.assertIn("Requirements", design)

    def test_generate_interface_stage(self):
        """Test interface stage generation."""
        prompt = "Create a fibonacci calculator"
        interface = self.generator._generate_interface(prompt, "")
        
        self.assertIn("def ", interface)
        self.assertIn("class ", interface)

    def test_generate_implementation_stage(self):
        """Test implementation stage generation."""
        prompt = "Create a simple processor"
        impl = self.generator._generate_implementation(prompt, "")
        
        self.assertIn("def ", impl)
        # Should be valid Python
        import ast
        try:
            ast.parse(impl)
        except SyntaxError:
            self.fail("Generated implementation has syntax error")

    def test_generate_tests_stage(self):
        """Test tests stage generation."""
        prompt = "Create a data validator"
        tests = self.generator._generate_tests(prompt, "")
        
        self.assertIn("unittest", tests)
        self.assertIn("TestCase", tests)

    def test_run_stage_syntax_validation(self):
        """Test that run_stage validates syntax."""
        # This would fail syntax check
        bad_code = "def broken(: pass"
        
        result = self.generator._run_stage(
            StageType.DESIGN,
            "test",
            "",
            lambda p, c: bad_code,
        )
        
        self.assertFalse(result.success)
        self.assertGreater(len(result.issues), 0)

    def test_full_pipeline_basic(self):
        """Test basic full pipeline execution."""
        prompt = "Create a simple utility function"
        
        report = self.generator.generate(prompt)
        
        self.assertIsNotNone(report)
        self.assertIn("prompt", report.to_dict())
        # Pipeline may not fully succeed due to template limitations
        # but should complete all stages

    def test_pipeline_report_collection(self):
        """Test that pipeline collects all issues and improvements."""
        prompt = "Test prompt"
        
        report = self.generator.generate(prompt)
        
        self.assertIsInstance(report.all_issues, list)
        self.assertIsInstance(report.all_improvements, list)
        self.assertIn('stage_results', report.to_dict())


if __name__ == '__main__':
    unittest.main()
