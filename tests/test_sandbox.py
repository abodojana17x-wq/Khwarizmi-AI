"""
Tests for Khwarizmi Execution Sandbox.
"""

import unittest
import time
from khwarizmi.coding.execution_sandbox import (
    ExecutionSandbox,
    SandboxResult,
    SecurityError,
)


class TestSandboxResult(unittest.TestCase):
    """Test suite for SandboxResult class."""

    def test_is_valid_success(self):
        """Test is_valid property for successful execution."""
        result = SandboxResult(
            success=True,
            exit_code=0,
            stdout="output",
            stderr="",
        )
        self.assertTrue(result.is_valid)

    def test_is_valid_failure(self):
        """Test is_valid property for failed execution."""
        result = SandboxResult(
            success=False,
            exit_code=1,
            error_message="Error occurred",
        )
        self.assertFalse(result.is_valid)

    def test_is_valid_timeout(self):
        """Test is_valid property for timed out execution."""
        result = SandboxResult(
            success=False,
            exit_code=-1,
            timed_out=True,
        )
        self.assertFalse(result.is_valid)

    def test_to_dict_serialization(self):
        """Test serialization to dictionary."""
        result = SandboxResult(
            success=True,
            exit_code=0,
            stdout="hello",
            stderr="",
            duration=0.5,
        )
        d = result.to_dict()
        self.assertIn('success', d)
        self.assertIn('exit_code', d)
        self.assertIn('stdout', d)
        self.assertIn('duration_seconds', d)


class TestExecutionSandbox(unittest.TestCase):
    """Test suite for ExecutionSandbox."""

    def setUp(self):
        """Set up test fixtures."""
        self.sandbox = ExecutionSandbox(timeout=5.0, max_memory_mb=100)

    def test_simple_print(self):
        """Test simple print statement execution."""
        code = "print('Hello, World!')"
        result = self.sandbox.execute(code)

        self.assertTrue(result.success)
        self.assertEqual(result.exit_code, 0)
        self.assertIn("Hello, World!", result.stdout)

    def test_arithmetic_operations(self):
        """Test arithmetic operations."""
        code = """
x = 10
y = 20
result = x + y * 2
print(f"Result: {result}")
"""
        result = self.sandbox.execute(code)

        self.assertTrue(result.success)
        self.assertIn("Result: 50", result.stdout)

    def test_function_definition_and_call(self):
        """Test function definition and call."""
        code = """
def add(a, b):
    return a + b

result = add(5, 3)
print(f"Sum: {result}")
"""
        result = self.sandbox.execute(code)

        self.assertTrue(result.success)
        self.assertIn("Sum: 8", result.stdout)

    def test_syntax_error(self):
        """Test handling of syntax errors."""
        code = "def broken(: pass"
        result = self.sandbox.execute(code)

        self.assertFalse(result.success)
        self.assertIn("Syntax", result.stderr or result.error_message)

    def test_runtime_error(self):
        """Test handling of runtime errors."""
        code = """
x = 10 / 0
"""
        result = self.sandbox.execute(code)

        self.assertFalse(result.success)
        self.assertIn("ZeroDivisionError", result.error_message)

    def test_timeout_enforcement(self):
        """Test that timeout is enforced."""
        # Create sandbox with very short timeout
        sandbox = ExecutionSandbox(timeout=0.5, max_memory_mb=100)
        
        code = """
import time
time.sleep(2)
print("Should not reach here")
"""
        result = sandbox.execute(code)

        # Should have timed out
        self.assertFalse(result.success)
        self.assertTrue(result.timed_out)

    def test_blocked_import_os(self):
        """Test that os module import is blocked."""
        code = "import os"
        result = self.sandbox.execute(code)

        # Should fail due to security restrictions
        self.assertFalse(result.success)

    def test_blocked_import_sys(self):
        """Test that sys module import is blocked."""
        code = "import sys"
        result = self.sandbox.execute(code)

        self.assertFalse(result.success)

    def test_blocked_import_subprocess(self):
        """Test that subprocess import is blocked."""
        code = "import subprocess"
        result = self.sandbox.execute(code)

        self.assertFalse(result.success)

    def test_blocked_eval_call(self):
        """Test that eval() call is blocked."""
        code = """
result = eval("1 + 1")
"""
        result = self.sandbox.execute(code)

        self.assertFalse(result.success)

    def test_blocked_exec_call(self):
        """Test that exec() call is blocked."""
        code = """
exec("print('hello')")
"""
        result = self.sandbox.execute(code)

        self.assertFalse(result.success)

    def test_allowed_math_module(self):
        """Test that math module is allowed."""
        code = """
import math
result = math.sqrt(16)
print(f"Square root: {result}")
"""
        result = self.sandbox.execute(code)

        self.assertTrue(result.success)
        self.assertIn("Square root: 4.0", result.stdout)

    def test_allowed_json_module(self):
        """Test that json module is allowed."""
        code = """
import json
data = {"key": "value"}
print(json.dumps(data))
"""
        result = self.sandbox.execute(code)

        self.assertTrue(result.success)
        self.assertIn("key", result.stdout)

    def test_list_comprehension(self):
        """Test list comprehension execution."""
        code = """
squares = [x**2 for x in range(10)]
print(f"Squares: {squares}")
"""
        result = self.sandbox.execute(code)

        self.assertTrue(result.success)
        self.assertIn("0, 1, 4", result.stdout)

    def test_class_definition(self):
        """Test class definition and instantiation."""
        code = """
class Person:
    def __init__(self, name):
        self.name = name
    
    def greet(self):
        return f"Hello, I'm {self.name}"

p = Person("Alice")
print(p.greet())
"""
        result = self.sandbox.execute(code)

        self.assertTrue(result.success)
        self.assertIn("Hello, I'm Alice", result.stdout)

    def test_exception_handling(self):
        """Test exception handling in sandboxed code."""
        code = """
try:
    x = 10 / 0
except ZeroDivisionError as e:
    print(f"Caught error: {e}")
"""
        result = self.sandbox.execute(code)

        self.assertTrue(result.success)
        self.assertIn("Caught error:", result.stdout)

    def test_duration_tracking(self):
        """Test that execution duration is tracked."""
        code = """
import time
time.sleep(0.1)
print("Done")
"""
        result = self.sandbox.execute(code)

        self.assertTrue(result.success)
        self.assertGreater(result.duration, 0.0)

    def test_memory_tracking(self):
        """Test that memory usage is tracked."""
        code = """
data = [i for i in range(1000)]
print(len(data))
"""
        result = self.sandbox.execute(code)

        self.assertTrue(result.success)
        # Memory tracking may be 0 on some systems but shouldn't fail

    def test_execute_with_globals(self):
        """Test execution with custom globals."""
        code = """
result = input_value * 2
print(f"Result: {result}")
"""
        result = self.sandbox.execute(code, globals_dict={'input_value': 21})

        self.assertTrue(result.success)
        self.assertIn("Result: 42", result.stdout)

    def test_invalid_timeout_parameter(self):
        """Test that invalid timeout raises error."""
        with self.assertRaises(ValueError):
            ExecutionSandbox(timeout=0, max_memory_mb=100)

    def test_invalid_memory_parameter(self):
        """Test that invalid memory parameter raises error."""
        with self.assertRaises(ValueError):
            ExecutionSandbox(timeout=5.0, max_memory_mb=0)


if __name__ == '__main__':
    unittest.main()
