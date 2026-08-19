"""
Unit Tests for Khwarizmi Cognitive Router.

Tests domain classification for CODE, SCIENCE, ART, CREATIVITY, and GENERAL domains.
"""

import unittest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from khwarizmi.integration.cognitive_router import (
    CognitiveRouter,
    DomainResult,
    get_router,
    route_task,
)


class TestCognitiveRouter(unittest.TestCase):
    """Test cases for cognitive router domain classification."""
    
    def setUp(self):
        self.router = CognitiveRouter()
    
    def test_code_domain_detection(self):
        """Test that code-related tasks are routed to CODE domain."""
        code_tasks = [
            "Write a Python function to calculate factorial",
            "Debug this code: for i in range(10): print(i)",
            "Implement a binary search algorithm",
            "def hello(): return 'world'",
            "Fix the syntax error in my class definition",
        ]
        
        for task in code_tasks:
            result = self.router.route(task)
            self.assertEqual(result.domain, "CODE", f"Task '{task}' should be CODE domain")
            self.assertGreater(result.confidence, 0.3, f"Low confidence for task: {task}")
    
    def test_science_domain_detection(self):
        """Test that physics/math tasks are routed to SCIENCE domain."""
        science_tasks = [
            "Verify the equation F = m * a",
            "Calculate the kinetic energy: E = 0.5 * m * v**2",
            "Physics problem: find the velocity after 5 seconds",
            "Check unit consistency for momentum p = m * v",
            "Solve this algebra equation for x",
        ]
        
        for task in science_tasks:
            result = self.router.route(task)
            self.assertEqual(result.domain, "SCIENCE", f"Task '{task}' should be SCIENCE domain")
    
    def test_art_domain_detection(self):
        """Test that art/aesthetic tasks are routed to ART domain."""
        art_tasks = [
            "Score this composition for aesthetic quality",
            "Evaluate the color harmony and balance of this design",
            "Rate the visual appeal using rule of thirds",
            "Critique the symmetry and focal point placement",
        ]
        
        for task in art_tasks:
            result = self.router.route(task)
            self.assertEqual(result.domain, "ART", f"Task '{task}' should be ART domain")
    
    def test_creativity_domain_detection(self):
        """Test that creative ideation tasks are routed to CREATIVITY domain."""
        creativity_tasks = [
            "Brainstorm ideas for a new educational app",
            "Use SCAMPER to generate alternative designs",
            "What if we reverse the usual workflow?",
            "Generate creative alternatives for this problem",
        ]
        
        for task in creativity_tasks:
            result = self.router.route(task)
            self.assertEqual(result.domain, "CREATIVITY", f"Task '{task}' should be CREATIVITY domain")
    
    def test_general_domain_detection(self):
        """Test that generic tasks are routed to GENERAL domain."""
        general_tasks = [
            "Hello, how are you?",
            "What is the capital of France?",
            "Tell me a joke",
            "Summarize this article",
        ]
        
        for task in general_tasks:
            result = self.router.route(task)
            self.assertEqual(result.domain, "GENERAL", f"Task '{task}' should be GENERAL domain")
    
    def test_confidence_threshold(self):
        """Test that confident predictions have high confidence scores."""
        strong_code_task = "def fibonacci(n): return n if n <= 1 else fibonacci(n-1) + fibonacci(n-2)"
        result = self.router.route(strong_code_task)
        self.assertTrue(result.is_confident, "Strong code signal should be confident")
        self.assertGreaterEqual(result.confidence, 0.7)
    
    def test_domain_result_to_dict(self):
        """Test DomainResult serialization."""
        result = DomainResult(
            domain="CODE",
            confidence=0.85,
            features={"code_score": 0.9},
            reasoning="Strong code patterns detected",
        )
        
        result_dict = result.to_dict()
        self.assertEqual(result_dict["domain"], "CODE")
        self.assertEqual(result_dict["confidence"], 0.85)
        self.assertIn("code_score", result_dict["features"])
        self.assertIn("reasoning", result_dict)
    
    def test_router_singleton(self):
        """Test that get_router returns singleton instance."""
        router1 = get_router()
        router2 = get_router()
        self.assertIs(router1, router2, "get_router should return singleton")
    
    def test_route_task_convenience_function(self):
        """Test the route_task convenience function."""
        result = route_task("Write Python code")
        self.assertIsInstance(result, DomainResult)
        self.assertEqual(result.domain, "CODE")
    
    def test_core_confidence_integration(self):
        """Test that core confidence affects routing."""
        task = "Calculate the force"
        
        # Without core confidence
        result1 = self.router.route(task, core_confidence=None)
        
        # With high core confidence
        result2 = self.router.route(task, core_confidence=0.9)
        
        # Features should be adjusted
        self.assertIn("science_score", result2.features)
    
    def test_get_domain_for_tool(self):
        """Test tool-to-domain mapping."""
        tool_domains = {
            "ExecutionSandbox": "CODE",
            "UnitConsistencyVerifier": "SCIENCE",
            "AestheticScorer": "ART",
            "ScamperEngine": "CREATIVITY",
        }
        
        for tool, expected_domain in tool_domains.items():
            domain = self.router.get_domain_for_tool(tool)
            self.assertEqual(domain, expected_domain, f"{tool} should map to {expected_domain}")
    
    def test_routing_history(self):
        """Test that routing history is tracked."""
        self.router.clear_history()
        
        tasks = ["code task", "science task", "art task"]
        for task in tasks:
            self.router.route(task)
        
        history = self.router.get_history()
        self.assertEqual(len(history), len(tasks))
    
    def test_empty_input_handling(self):
        """Test handling of empty or minimal input."""
        result = self.router.route("")
        self.assertEqual(result.domain, "GENERAL")


if __name__ == "__main__":
    unittest.main()
