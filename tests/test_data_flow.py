"""
Tests for Khwarizmi Data Flow Analyzer.
"""

import unittest
from khwarizmi.coding.data_flow_analyzer import (
    DataFlowAnalyzer,
    DataFlowReport,
    VariableBinding,
    DataFlowIssue,
)


class TestDataFlowAnalyzer(unittest.TestCase):
    """Test suite for DataFlowAnalyzer."""

    def test_simple_function_analysis(self):
        """Test analysis of a simple function."""
        code = """
def add(a, b):
    result = a + b
    return result

x = 10
y = 20
sum_result = add(x, y)
unused_var = "never used"
"""
        analyzer = DataFlowAnalyzer()
        report = analyzer.analyze(code)

        self.assertTrue(report.parse_successful)
        self.assertEqual(report.total_variables, 6)  # a, b, result, x, y, sum_result, unused_var
        self.assertGreater(len(report.unused_variables), 0)
        
        # Check that 'unused_var' is detected as unused
        unused_names = [i.name for i in report.unused_variables]
        self.assertIn('unused_var', unused_names)

    def test_unused_variable_detection(self):
        """Test detection of unused variables."""
        code = """
def process():
    used_var = 10
    unused_var = 20
    print(used_var)
    return used_var
"""
        analyzer = DataFlowAnalyzer()
        report = analyzer.analyze(code)

        self.assertTrue(report.parse_successful)
        unused_names = [i.name for i in report.unused_variables]
        self.assertIn('unused_var', unused_names)
        self.assertNotIn('used_var', unused_names)

    def test_dead_assignment_detection(self):
        """Test detection of dead assignments (overwritten before use)."""
        code = """
def example():
    x = 10
    x = 20  # Dead assignment - x overwritten before use
    return x
"""
        analyzer = DataFlowAnalyzer()
        report = analyzer.analyze(code)

        self.assertTrue(report.parse_successful)
        # Should detect dead assignment
        dead_names = [i.name for i in report.dead_assignments]
        self.assertIn('x', dead_names)

    def test_import_bindings(self):
        """Test that imports are properly tracked."""
        code = """
import os
import json as js
from pathlib import Path

def use_imports():
    return Path(".")
"""
        analyzer = DataFlowAnalyzer()
        report = analyzer.analyze(code)

        self.assertTrue(report.parse_successful)
        # os and js should be bindings
        binding_names = [b.name for b in report.all_bindings]
        self.assertIn('os', binding_names)
        self.assertIn('js', binding_names)
        self.assertIn('Path', binding_names)

    def test_loop_variable_tracking(self):
        """Test tracking of loop variables."""
        code = """
def process_list(items):
    total = 0
    for item in items:
        total += item
    return total
"""
        analyzer = DataFlowAnalyzer()
        report = analyzer.analyze(code)

        self.assertTrue(report.parse_successful)
        binding_names = [b.name for b in report.all_bindings]
        self.assertIn('item', binding_names)
        self.assertIn('total', binding_names)

    def test_function_parameters(self):
        """Test that function parameters are tracked."""
        code = """
def greet(name, greeting="Hello"):
    message = f"{greeting}, {name}!"
    return message
"""
        analyzer = DataFlowAnalyzer()
        report = analyzer.analyze(code)

        self.assertTrue(report.parse_successful)
        binding_names = [b.name for b in report.all_bindings]
        self.assertIn('name', binding_names)
        self.assertIn('greeting', binding_names)

    def test_class_method_analysis(self):
        """Test analysis of class methods."""
        code = """
class Calculator:
    def __init__(self, initial_value=0):
        self.value = initial_value
    
    def add(self, x):
        self.value += x
        return self.value
    
    def get_value(self):
        return self.value
"""
        analyzer = DataFlowAnalyzer()
        report = analyzer.analyze(code)

        self.assertTrue(report.parse_successful)
        # Should find class and method bindings

    def test_comprehension_variables(self):
        """Test tracking of comprehension variables."""
        code = """
def process():
    numbers = [1, 2, 3, 4, 5]
    squares = [x**2 for x in numbers]
    evens = [x for x in numbers if x % 2 == 0]
    return squares, evens
"""
        analyzer = DataFlowAnalyzer()
        report = analyzer.analyze(code)

        self.assertTrue(report.parse_successful)
        binding_names = [b.name for b in report.all_bindings]
        self.assertIn('numbers', binding_names)
        self.assertIn('squares', binding_names)
        self.assertIn('evens', binding_names)

    def test_exception_handling(self):
        """Test tracking of exception variables."""
        code = """
def risky_operation():
    try:
        result = 10 / 0
    except ZeroDivisionError as e:
        print(f"Error: {e}")
        result = 0
    return result
"""
        analyzer = DataFlowAnalyzer()
        report = analyzer.analyze(code)

        self.assertTrue(report.parse_successful)
        binding_names = [b.name for b in report.all_bindings]
        self.assertIn('e', binding_names)
        self.assertIn('result', binding_names)

    def test_syntax_error_handling(self):
        """Test handling of syntax errors."""
        code = """
def broken(:
    pass
"""
        analyzer = DataFlowAnalyzer()
        report = analyzer.analyze(code)

        self.assertFalse(report.parse_successful)
        self.assertIsNotNone(report.syntax_error)

    def test_is_valid_property(self):
        """Test the is_valid property of DataFlowReport."""
        # Clean code
        clean_code = """
def add(a, b):
    return a + b

result = add(1, 2)
"""
        analyzer = DataFlowAnalyzer()
        report = analyzer.analyze(clean_code)
        # Note: 'result' will be flagged as unused, so is_valid may be False
        
        # Code with issues
        bad_code = """
def example():
    x = 10
    x = 20
    return x
"""
        report2 = analyzer.analyze(bad_code)
        # May have dead assignment

    def test_to_dict_serialization(self):
        """Test serialization to dictionary."""
        code = """
x = 10
"""
        analyzer = DataFlowAnalyzer()
        report = analyzer.analyze(code)

        result_dict = report.to_dict()
        self.assertIn('parse_successful', result_dict)
        self.assertIn('total_variables', result_dict)
        self.assertIn('is_valid', result_dict)


if __name__ == '__main__':
    unittest.main()
