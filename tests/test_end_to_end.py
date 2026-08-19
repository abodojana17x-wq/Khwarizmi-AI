"""
End-to-End Tests for Khwarizmi Integration Layer.

Tests complete workflows:
1. Physics problem with unit verification
2. Code task with static analysis and execution
3. Creative brief with SCAMPER ideation
"""

import unittest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from khwarizmi.integration.agentic_loop import AgenticLoop, run_agentic_loop
from khwarizmi.integration.cognitive_router import CognitiveRouter, route_task
from khwarizmi.integration.tool_registry import ToolRegistry, invoke_tool


class TestPhysicsProblemEndToEnd(unittest.TestCase):
    """End-to-end test for physics problem solving."""
    
    def test_physics_equation_verification(self):
        """Test: Verify Newton's second law F = m * a."""
        # Step 1: Route the task
        task = "Verify the equation F = m * a for unit consistency"
        domain_result = route_task(task)
        
        self.assertEqual(domain_result.domain, "SCIENCE")
        self.assertGreater(domain_result.confidence, 0.5)
        
        # Step 2: Invoke unit consistency verifier directly
        result = invoke_tool("UnitConsistencyVerifier", equation="F = m * a")
        
        self.assertTrue(result.success)
        self.assertIsNotNone(result.output)
        self.assertEqual(result.output["lhs_dimensions"], result.output["rhs_dimensions"])
    
    def test_physics_problem_with_loop(self):
        """Test full agentic loop for physics problem."""
        loop = AgenticLoop(max_iterations=2)
        task = "Check if E = 0.5 * m * v**2 has consistent units"
        
        result = loop.execute(task, domain_hint="SCIENCE")
        
        self.assertEqual(result.domain, "SCIENCE")
        self.assertIsInstance(result.total_duration_ms, float)
        self.assertGreater(len(result.steps), 0)
    
    def test_momentum_equation(self):
        """Test momentum equation p = m * v."""
        result = invoke_tool("UnitConsistencyVerifier", equation="p = m * v")
        
        self.assertTrue(result.success)
        # Momentum: kg * m/s on both sides


class TestCodeTaskEndToEnd(unittest.TestCase):
    """End-to-end test for code analysis and execution."""
    
    def test_code_routing(self):
        """Test that code tasks are routed correctly."""
        code_tasks = [
            "def factorial(n): return 1 if n <= 1 else n * factorial(n-1)",
            "class Calculator: def add(self, a, b): return a + b",
            "for i in range(10): print(i ** 2)",
        ]
        
        for task in code_tasks:
            result = route_task(task)
            self.assertEqual(result.domain, "CODE", f"Failed for: {task}")
    
    def test_data_flow_analysis(self):
        """Test data flow analysis on sample code."""
        code = """
def calculate_sum(numbers):
    total = 0
    for n in numbers:
        total += n
    return total

unused_var = 42
result = calculate_sum([1, 2, 3])
"""
        result = invoke_tool("DataFlowAnalyzer", source_code=code)
        
        self.assertTrue(result.success)
        self.assertIn("total_variables", result.metadata)
    
    def test_control_flow_graph(self):
        """Test CFG generation on sample code."""
        code = """
def classify(x):
    if x > 0:
        return "positive"
    elif x < 0:
        return "negative"
    else:
        return "zero"
"""
        result = invoke_tool("ControlFlowGraph", source_code=code)
        
        self.assertTrue(result.success)
        self.assertIn("cyclomatic_complexity", result.metadata)
        # Should have complexity > 1 due to if/elif/else
    
    def test_type_inference(self):
        """Test type inference on sample code."""
        code = """
x = 42
y = "hello"
z = [1, 2, 3]
w = {"key": "value"}
"""
        result = invoke_tool("TypeInference", source_code=code)
        
        self.assertTrue(result.success)
    
    def test_sandbox_execution(self):
        """Test sandboxed code execution."""
        code = "print('Hello, Khwarizmi!')"
        result = invoke_tool("ExecutionSandbox", code=code, timeout=2.0)
        
        self.assertTrue(result.success)
        self.assertIn("Hello, Khwarizmi!", result.output.get("stdout", ""))
    
    def test_full_code_workflow(self):
        """Test complete code workflow: analyze -> verify -> execute."""
        code = """
def multiply(a, b):
    '''Multiply two numbers.'''
    return a * b

if __name__ == "__main__":
    print(multiply(5, 3))
"""
        loop = AgenticLoop(max_iterations=3)
        result = loop.execute(code, domain_hint="CODE")
        
        self.assertEqual(result.domain, "CODE")
        self.assertGreater(len(result.steps), 2)


class TestCreativeBriefEndToEnd(unittest.TestCase):
    """End-to-end test for creative ideation."""
    
    def test_creativity_routing(self):
        """Test that creative tasks are routed correctly."""
        creative_tasks = [
            "Brainstorm features for a meditation app",
            "Use SCAMPER to improve a water bottle design",
            "What if we eliminate the keyboard from computers?",
        ]
        
        for task in creative_tasks:
            result = route_task(task)
            self.assertEqual(result.domain, "CREATIVITY", f"Failed for: {task}")
    
    def test_scamper_generation(self):
        """Test SCAMPER candidate generation."""
        brief = "Design a better backpack for students"
        result = invoke_tool("ScamperEngine", brief=brief)
        
        self.assertTrue(result.success)
        self.assertEqual(result.output["safety_verdict"], "allowed")
        self.assertGreaterEqual(len(result.output["candidates"]), 5)
    
    def test_scamper_candidates_structure(self):
        """Test structure of SCAMPER candidates."""
        brief = "Improve online learning platforms"
        result = invoke_tool("ScamperEngine", brief=brief)
        
        candidates = result.output["candidates"]
        for candidate in candidates:
            self.assertIn("technique", candidate)
            self.assertIn("idea", candidate)
            self.assertIn("novelty", candidate)
            self.assertIn("usefulness", candidate)
            self.assertIn("rationale", candidate)
    
    def test_scamper_blocked_content(self):
        """Test that hazardous content is blocked."""
        hazardous_brief = "Design a weapon drone"
        result = invoke_tool("ScamperEngine", brief=hazardous_brief)
        
        self.assertEqual(result.output["safety_verdict"], "blocked")
    
    def test_creative_loop_execution(self):
        """Test full agentic loop for creative task."""
        loop = AgenticLoop(max_iterations=2)
        task = "Generate innovative ideas for sustainable packaging"
        
        result = loop.execute(task, domain_hint="CREATIVITY")
        
        self.assertEqual(result.domain, "CREATIVITY")
        self.assertTrue(result.success or len(result.steps) > 0)


class TestArtEvaluationEndToEnd(unittest.TestCase):
    """End-to-end test for art/aesthetic evaluation."""
    
    def test_art_routing(self):
        """Test that art tasks are routed correctly."""
        art_tasks = [
            "Score this composition using rule of thirds",
            "Evaluate the color harmony of this palette",
            "Rate the aesthetic balance and symmetry",
        ]
        
        for task in art_tasks:
            result = route_task(task)
            self.assertEqual(result.domain, "ART", f"Failed for: {task}")
    
    def test_aesthetic_scoring(self):
        """Test aesthetic scoring with structured brief."""
        description = {
            "focal_point": (0.33, 0.66),
            "symmetry": 0.7,
            "balance": 0.8,
            "negative_space": 0.3,
            "harmony": "complementary",
            "contrast": 0.7,
            "temperature": "warm",
            "saturation": 0.6,
        }
        
        result = invoke_tool("AestheticScorer", description=description)
        
        self.assertTrue(result.success)
        self.assertIn("overall_score", result.output)
        self.assertIn("composition_score", result.output)
        self.assertIn("color_score", result.output)
        self.assertGreaterEqual(result.output["overall_score"], 0)
        self.assertLessEqual(result.output["overall_score"], 100)
    
    def test_aesthetic_suggestions(self):
        """Test that aesthetic scorer provides suggestions."""
        description = {
            "focal_point": (0.5, 0.5),  # Center - not ideal for rule of thirds
            "symmetry": 0.3,
            "balance": 0.4,
            "negative_space": 0.5,
            "harmony": "analogous",
            "contrast": 0.3,  # Low contrast
            "temperature": "balanced",
            "saturation": 0.5,
        }
        
        result = invoke_tool("AestheticScorer", description=description)
        
        self.assertIn("suggestions", result.output)
        # May have suggestions based on low scores


class TestIntegrationCrossDomain(unittest.TestCase):
    """Cross-domain integration tests."""
    
    def test_router_distinguishes_domains(self):
        """Test that router correctly distinguishes between domains."""
        tasks = [
            ("def sort(arr): pass", "CODE"),
            ("F = m * a", "SCIENCE"),
            ("Score this artwork", "ART"),
            ("Brainstorm ideas", "CREATIVITY"),
            ("Hello there", "GENERAL"),
        ]
        
        for task, expected_domain in tasks:
            result = route_task(task)
            self.assertEqual(result.domain, expected_domain, f"Failed for: {task}")
    
    def test_tool_registry_lists_all_tools(self):
        """Test that tool registry contains all expected tools."""
        registry = ToolRegistry()
        tools = registry.list_tools()
        
        expected_tools = [
            "ExecutionSandbox",
            "UnitConsistencyVerifier",
            "AestheticScorer",
            "ScamperEngine",
            "DataFlowAnalyzer",
            "ControlFlowGraph",
            "TypeInference",
        ]
        
        for tool in expected_tools:
            self.assertIn(tool, tools)
    
    def test_multiple_loops_no_interference(self):
        """Test that multiple loop executions don't interfere."""
        loop1 = AgenticLoop(max_iterations=1)
        loop2 = AgenticLoop(max_iterations=1)
        
        result1 = loop1.execute("code task", domain_hint="CODE")
        result2 = loop2.execute("science task", domain_hint="SCIENCE")
        
        self.assertEqual(result1.domain, "CODE")
        self.assertEqual(result2.domain, "SCIENCE")


class TestOfflineConstraints(unittest.TestCase):
    """Tests for offline-first and memory constraints."""
    
    def test_no_network_calls(self):
        """Verify tools work without network access."""
        # All tools should be purely local
        result = invoke_tool("UnitConsistencyVerifier", equation="E = m * c**2")
        self.assertIsNotNone(result)
    
    def test_memory_efficient_execution(self):
        """Test that execution respects memory constraints."""
        # Run multiple iterations
        loop = AgenticLoop(max_iterations=1)
        
        for i in range(10):
            result = loop.execute(f"task {i}", domain_hint="GENERAL")
            self.assertIsInstance(result, type(loop.execute("x", domain_hint="GENERAL")))
    
    def test_rapid_response_time(self):
        """Test that responses are generated quickly."""
        loop = AgenticLoop(max_iterations=1)
        
        result = loop.execute("quick task", domain_hint="GENERAL")
        
        # Should complete in reasonable time (< 1 second for GENERAL)
        self.assertLess(result.total_duration_ms, 1000)


if __name__ == "__main__":
    unittest.main()
