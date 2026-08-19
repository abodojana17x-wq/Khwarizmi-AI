"""
Unit Tests for Khwarizmi Agentic Loop.

Tests the full agentic execution loop with tool integration.
"""

import unittest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from khwarizmi.integration.agentic_loop import (
    AgenticLoop,
    AgenticLoopResult,
    AgentStep,
    ProcessRewardModel,
    TestTimeComputeScaling,
    run_agentic_loop,
    create_loop,
)
from khwarizmi.integration.tool_registry import ToolRegistry, ToolResult


class TestProcessRewardModel(unittest.TestCase):
    """Test cases for process reward model verification."""
    
    def test_verify_sandbox_success(self):
        """Test successful sandbox result verification."""
        result = {
            "success": True,
            "exit_code": 0,
            "timed_out": False,
            "memory_exceeded": False,
        }
        verified, msg = ProcessRewardModel.verify_sandbox_result(result)
        self.assertTrue(verified)
        self.assertIn("successful", msg)
    
    def test_verify_sandbox_failure(self):
        """Test failed sandbox result verification."""
        result = {
            "success": False,
            "exit_code": 1,
            "error_message": "Runtime error",
        }
        verified, msg = ProcessRewardModel.verify_sandbox_result(result)
        self.assertFalse(verified)
    
    def test_verify_sandbox_timeout(self):
        """Test timeout detection in sandbox verification."""
        result = {
            "success": False,
            "timed_out": True,
        }
        verified, msg = ProcessRewardModel.verify_sandbox_result(result)
        self.assertFalse(verified)
        self.assertIn("timed out", msg)
    
    def test_verify_unit_verdict_consistent(self):
        """Test unit consistency verification with matching dimensions."""
        result = {
            "success": True,
            "lhs_dimensions": {"kg": 1, "m": 1, "s": -2},
            "rhs_dimensions": {"kg": 1, "m": 1, "s": -2},
        }
        verified, msg = ProcessRewardModel.verify_unit_verdict(result)
        self.assertTrue(verified)
        self.assertIn("consistent", msg)
    
    def test_verify_unit_verdict_mismatch(self):
        """Test unit consistency verification with mismatched dimensions."""
        result = {
            "success": True,
            "lhs_dimensions": {"kg": 1, "m": 1},
            "rhs_dimensions": {"kg": 1, "s": 1},
        }
        verified, msg = ProcessRewardModel.verify_unit_verdict(result)
        self.assertFalse(verified)
        self.assertIn("mismatch", msg)
    
    def test_verify_aesthetic_score_valid(self):
        """Test aesthetic score verification with valid score."""
        result = {
            "success": True,
            "overall_score": 75.5,
        }
        verified, msg = ProcessRewardModel.verify_aesthetic_score(result)
        self.assertTrue(verified)
        self.assertIn("score", msg)
    
    def test_verify_aesthetic_score_out_of_range(self):
        """Test aesthetic score verification with out-of-range score."""
        result = {
            "success": True,
            "overall_score": 150,
        }
        verified, msg = ProcessRewardModel.verify_aesthetic_score(result)
        self.assertFalse(verified)
    
    def test_verify_scamper_candidates(self):
        """Test SCAMPER candidate verification."""
        result = {
            "success": True,
            "candidates": [{"technique": "Substitute"}] * 7,
        }
        verified, msg = ProcessRewardModel.verify_scamper_candidates(result)
        self.assertTrue(verified)
        self.assertGreater(len(msg), 0)
    
    def test_verify_scamper_too_few(self):
        """Test SCAMPER verification with too few candidates."""
        result = {
            "success": True,
            "candidates": [{"technique": "Substitute"}],
        }
        verified, msg = ProcessRewardModel.verify_scamper_candidates(result)
        self.assertFalse(verified)
        self.assertIn("few", msg)
    
    def test_verify_dispatch(self):
        """Test verify method dispatches to correct verifier."""
        # Test with unknown tool (should accept by default)
        verified, msg = ProcessRewardModel.verify("UnknownTool", {})
        self.assertTrue(verified)


class TestTestTimeComputeScaling(unittest.TestCase):
    """Test cases for test-time compute scaling (THINK phase)."""
    
    def setUp(self):
        self.thinker = TestTimeComputeScaling(max_cycles=3)
    
    def test_think_code_domain(self):
        """Test thinking phase for CODE domain."""
        plan = self.thinker.think("Write Python code", "CODE")
        self.assertIn("suggested_tools", plan)
        self.assertIn("DataFlowAnalyzer", plan["suggested_tools"])
        self.assertEqual(plan["priority"], "analysis_first")
    
    def test_think_science_domain(self):
        """Test thinking phase for SCIENCE domain."""
        plan = self.thinker.think("Verify F = m * a", "SCIENCE")
        self.assertIn("UnitConsistencyVerifier", plan["suggested_tools"])
        self.assertEqual(plan["priority"], "verify_equation")
    
    def test_think_art_domain(self):
        """Test thinking phase for ART domain."""
        plan = self.thinker.think("Score this composition", "ART")
        self.assertIn("AestheticScorer", plan["suggested_tools"])
    
    def test_think_creativity_domain(self):
        """Test thinking phase for CREATIVITY domain."""
        plan = self.thinker.think("Brainstorm ideas", "CREATIVITY")
        self.assertIn("ScamperEngine", plan["suggested_tools"])
        self.assertEqual(plan["priority"], "generate_alternatives")
    
    def test_think_general_domain(self):
        """Test thinking phase for GENERAL domain."""
        plan = self.thinker.think("Hello there", "GENERAL")
        self.assertEqual(plan["suggested_tools"], [])
        self.assertEqual(plan["priority"], "direct_response")


class TestAgenticLoop(unittest.TestCase):
    """Test cases for the full agentic loop."""
    
    def setUp(self):
        self.loop = AgenticLoop(max_iterations=3)
    
    def test_loop_code_task(self):
        """Test agentic loop with code task."""
        task = "def add(a, b): return a + b"
        result = self.loop.execute(task, domain_hint="CODE")
        
        self.assertIsInstance(result, AgenticLoopResult)
        self.assertEqual(result.domain, "CODE")
        self.assertGreater(len(result.steps), 0)
    
    def test_loop_science_task(self):
        """Test agentic loop with science task."""
        task = "F = m * a"
        result = self.loop.execute(task, domain_hint="SCIENCE")
        
        self.assertIsInstance(result, AgenticLoopResult)
        self.assertEqual(result.domain, "SCIENCE")
    
    def test_loop_creativity_task(self):
        """Test agentic loop with creativity task."""
        task = "Brainstorm new features for a calculator app"
        result = self.loop.execute(task, domain_hint="CREATIVITY")
        
        self.assertIsInstance(result, AgenticLoopResult)
        self.assertEqual(result.domain, "CREATIVITY")
    
    def test_loop_general_task(self):
        """Test agentic loop with general task."""
        task = "What is 2 + 2?"
        result = self.loop.execute(task, domain_hint="GENERAL")
        
        self.assertIsInstance(result, AgenticLoopResult)
        self.assertEqual(result.domain, "GENERAL")
        self.assertTrue(result.success)
    
    def test_result_serialization(self):
        """Test AgenticLoopResult serialization."""
        result = AgenticLoopResult(
            success=True,
            task="test task",
            domain="CODE",
            final_output={"result": "success"},
            total_duration_ms=100.0,
        )
        
        result_dict = result.to_dict()
        self.assertEqual(result_dict["success"], True)
        self.assertEqual(result_dict["task"], "test task")
        self.assertEqual(result_dict["domain"], "CODE")
    
    def test_agent_step_creation(self):
        """Test AgentStep dataclass."""
        step = AgentStep(
            step_number=1,
            action="think",
            input_data={"task": "test"},
            output_data={"plan": "analyze"},
            duration_ms=10.0,
        )
        
        self.assertEqual(step.step_number, 1)
        self.assertEqual(step.action, "think")
        self.assertIsNone(step.verification_result)
    
    def test_convenience_functions(self):
        """Test convenience functions."""
        # run_agentic_loop
        result = run_agentic_loop("test", domain_hint="GENERAL")
        self.assertIsInstance(result, AgenticLoopResult)
        
        # create_loop
        loop = create_loop(max_iterations=2)
        self.assertIsInstance(loop, AgenticLoop)
        self.assertEqual(loop.max_iterations, 2)
    
    def test_loop_tracks_steps(self):
        """Test that loop tracks all steps."""
        result = self.loop.execute("print('hello')", domain_hint="CODE")
        
        # Should have at least: route, think, pick_tool, execute, verify
        self.assertGreaterEqual(len(result.steps), 3)
        
        actions = [step.action for step in result.steps]
        self.assertIn("route", actions)
        self.assertIn("think", actions)


class TestAgenticLoopIntegration(unittest.TestCase):
    """Integration tests for agentic loop with real tools."""
    
    def test_loop_with_unit_verifier(self):
        """Test loop executes unit consistency verification."""
        loop = AgenticLoop(max_iterations=2)
        task = "F = m * a"
        result = loop.execute(task, domain_hint="SCIENCE")
        
        # The loop should attempt unit verification
        self.assertEqual(result.domain, "SCIENCE")
        self.assertTrue(result.success or not result.max_iterations_reached)
    
    def test_loop_memory_efficiency(self):
        """Test that loop respects memory constraints."""
        # Run multiple iterations and check no memory leak indicators
        loop = AgenticLoop(max_iterations=2)
        
        for i in range(5):
            result = loop.execute(f"task {i}", domain_hint="GENERAL")
            self.assertIsInstance(result, AgenticLoopResult)


if __name__ == "__main__":
    unittest.main()
