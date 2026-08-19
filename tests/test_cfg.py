"""
Tests for Khwarizmi Control Flow Graph.
"""

import unittest
from khwarizmi.coding.control_flow_graph import (
    ControlFlowGraph,
    CFGNode,
    CFGEdge,
    CFGReport,
)


class TestControlFlowGraph(unittest.TestCase):
    """Test suite for ControlFlowGraph."""

    def test_simple_sequence(self):
        """Test CFG for simple sequential code."""
        code = """
x = 10
y = 20
z = x + y
print(z)
"""
        cfg_builder = ControlFlowGraph()
        report = cfg_builder.build(code)

        self.assertTrue(report.parse_successful)
        self.assertGreater(report.num_nodes, 0)
        self.assertEqual(report.cyclomatic_complexity, 1)  # No decisions

    def test_if_statement_branches(self):
        """Test CFG for if statement creates branches."""
        code = """
def check(x):
    if x > 0:
        return "positive"
    else:
        return "non-positive"
"""
        cfg_builder = ControlFlowGraph()
        report = cfg_builder.build(code)

        self.assertTrue(report.parse_successful)
        self.assertGreater(report.cyclomatic_complexity, 1)  # Has decision
        self.assertGreater(report.num_decision_points, 0)

    def test_for_loop_cfg(self):
        """Test CFG for for loop."""
        code = """
def sum_list(items):
    total = 0
    for item in items:
        total += item
    return total
"""
        cfg_builder = ControlFlowGraph()
        report = cfg_builder.build(code)

        self.assertTrue(report.parse_successful)
        self.assertGreater(report.num_nodes, 0)
        # For loop adds a decision point
        self.assertGreater(report.num_decision_points, 0)

    def test_while_loop_cfg(self):
        """Test CFG for while loop."""
        code = """
def countdown(n):
    while n > 0:
        print(n)
        n -= 1
    return 0
"""
        cfg_builder = ControlFlowGraph()
        report = cfg_builder.build(code)

        self.assertTrue(report.parse_successful)
        self.assertGreater(report.num_decision_points, 0)

    def test_nested_if_statements(self):
        """Test CFG for nested if statements."""
        code = """
def classify(x, y):
    if x > 0:
        if y > 0:
            return "both positive"
        else:
            return "x positive, y not"
    else:
        return "x not positive"
"""
        cfg_builder = ControlFlowGraph()
        report = cfg_builder.build(code)

        self.assertTrue(report.parse_successful)
        # Multiple decision points from nested ifs
        self.assertGreater(report.num_decision_points, 1)

    def test_try_except_cfg(self):
        """Test CFG for try/except blocks."""
        code = """
def safe_divide(a, b):
    try:
        result = a / b
        return result
    except ZeroDivisionError:
        return None
"""
        cfg_builder = ControlFlowGraph()
        report = cfg_builder.build(code)

        self.assertTrue(report.parse_successful)
        self.assertGreater(report.num_nodes, 0)

    def test_break_continue_cfg(self):
        """Test CFG with break and continue statements."""
        code = """
def process(items):
    for item in items:
        if item == 0:
            continue
        if item < 0:
            break
        print(item)
"""
        cfg_builder = ControlFlowGraph()
        report = cfg_builder.build(code)

        self.assertTrue(report.parse_successful)
        # Should have edges for break and continue paths

    def test_return_statement(self):
        """Test CFG with early return."""
        code = """
def find_first(items, target):
    for item in items:
        if item == target:
            return True
    return False
"""
        cfg_builder = ControlFlowGraph()
        report = cfg_builder.build(code)

        self.assertTrue(report.parse_successful)
        # Multiple return paths

    def test_adjacency_dict_export(self):
        """Test adjacency dictionary export."""
        code = """
x = 10
if x > 0:
    print("positive")
else:
    print("non-positive")
"""
        cfg_builder = ControlFlowGraph()
        report = cfg_builder.build(code)

        adj_dict = report.adjacency_dict
        self.assertIsInstance(adj_dict, dict)
        self.assertGreater(len(adj_dict), 0)

    def test_visualization_output(self):
        """Test text visualization generation."""
        code = """
def example():
    x = 10
    if x > 0:
        return x
    return 0
"""
        cfg_builder = ControlFlowGraph()
        report = cfg_builder.build(code)

        viz = report.visualize()
        self.assertIsInstance(viz, str)
        self.assertIn("CONTROL FLOW GRAPH", viz)
        self.assertIn("Cyclomatic Complexity", viz)

    def test_syntax_error_handling(self):
        """Test handling of syntax errors."""
        code = """
def broken(:
    pass
"""
        cfg_builder = ControlFlowGraph()
        report = cfg_builder.build(code)

        self.assertFalse(report.parse_successful)
        self.assertIsNotNone(report.syntax_error)

    def test_to_dict_serialization(self):
        """Test serialization to dictionary."""
        code = """
x = 10
"""
        cfg_builder = ControlFlowGraph()
        report = cfg_builder.build(code)

        result_dict = report.to_dict()
        self.assertIn('parse_successful', result_dict)
        self.assertIn('cyclomatic_complexity', result_dict)
        self.assertIn('nodes', result_dict)
        self.assertIn('edges', result_dict)

    def test_cfg_node_properties(self):
        """Test CFGNode properties."""
        node = CFGNode(
            node_id=1,
            node_type="statement",
            start_line=5,
            end_line=5,
            source_code="x = 10"
        )
        
        self.assertEqual(node.node_id, 1)
        self.assertEqual(node.node_type, "statement")
        self.assertEqual(node.start_line, 5)
        self.assertIn("CFGNode", str(node))

    def test_cfg_edge_properties(self):
        """Test CFGEdge properties."""
        edge = CFGEdge(source=1, target=2, edge_type="flow")
        
        self.assertEqual(edge.source, 1)
        self.assertEqual(edge.target, 2)
        self.assertEqual(edge.edge_type, "flow")
        self.assertIn("CFGEdge", str(edge))

    def test_complex_function_cfg(self):
        """Test CFG for complex function with multiple control flows."""
        code = """
def complex_example(data, threshold):
    result = []
    
    for item in data:
        if item is None:
            continue
        
        if item > threshold:
            try:
                processed = item * 2
                result.append(processed)
            except Exception:
                pass
        elif item == threshold:
            result.append(0)
        else:
            break
    
    return result if result else None
"""
        cfg_builder = ControlFlowGraph()
        report = cfg_builder.build(code)

        self.assertTrue(report.parse_successful)
        self.assertGreater(report.num_nodes, 5)
        self.assertGreater(report.num_decision_points, 2)


if __name__ == '__main__':
    unittest.main()
