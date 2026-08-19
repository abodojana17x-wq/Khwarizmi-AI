"""
Tests for Khwarizmi Type Inference.
"""

import unittest
from khwarizmi.coding.type_inference import (
    TypeInference,
    InferredType,
    TypeBinding,
    TypeInferenceReport,
)


class TestInferredType(unittest.TestCase):
    """Test suite for InferredType class."""

    def test_from_literal_int(self):
        """Test type inference from integer literal."""
        t = InferredType.from_literal(42)
        self.assertEqual(t.type_name, "int")
        self.assertEqual(t.confidence, 1.0)

    def test_from_literal_float(self):
        """Test type inference from float literal."""
        t = InferredType.from_literal(3.14)
        self.assertEqual(t.type_name, "float")

    def test_from_literal_str(self):
        """Test type inference from string literal."""
        t = InferredType.from_literal("hello")
        self.assertEqual(t.type_name, "str")

    def test_from_literal_bool(self):
        """Test type inference from boolean literal."""
        t = InferredType.from_literal(True)
        self.assertEqual(t.type_name, "bool")

    def test_from_literal_none(self):
        """Test type inference from None."""
        t = InferredType.from_literal(None)
        self.assertEqual(t.type_name, "None")

    def test_from_literal_list(self):
        """Test type inference from list literal."""
        t = InferredType.from_literal([1, 2, 3])
        self.assertEqual(t.type_name, "list")
        self.assertEqual(t.container_type, "int")

    def test_from_literal_dict(self):
        """Test type inference from dict literal."""
        t = InferredType.from_literal({"a": 1, "b": 2})
        self.assertEqual(t.type_name, "dict")
        self.assertEqual(t.key_type, "str")
        self.assertEqual(t.value_type, "int")

    def test_common_type_same(self):
        """Test finding common type when all types are same."""
        types = [
            InferredType(type_name="int", confidence=1.0),
            InferredType(type_name="int", confidence=1.0),
        ]
        common = InferredType._common_type(types)
        self.assertEqual(common.type_name, "int")

    def test_common_type_numeric_hierarchy(self):
        """Test numeric type hierarchy (int + float -> float)."""
        types = [
            InferredType(type_name="int", confidence=1.0),
            InferredType(type_name="float", confidence=1.0),
        ]
        common = InferredType._common_type(types)
        self.assertEqual(common.type_name, "float")

    def test_str_representation(self):
        """Test string representation of types."""
        t = InferredType(type_name="list", container_type="int")
        self.assertEqual(str(t), "list[int]")

        t2 = InferredType(type_name="dict", key_type="str", value_type="int")
        self.assertEqual(str(t2), "dict[str, int]")


class TestTypeInference(unittest.TestCase):
    """Test suite for TypeInference analyzer."""

    def test_simple_assignment(self):
        """Test type inference from simple assignment."""
        code = """
x = 10
y = "hello"
z = 3.14
"""
        inferencer = TypeInference()
        report = inferencer.analyze(code)

        self.assertTrue(report.parse_successful)
        self.assertGreater(report.total_variables, 0)

    def test_function_return_type(self):
        """Test function return type inference."""
        code = """
def add(a, b):
    return a + b
"""
        inferencer = TypeInference()
        report = inferencer.analyze(code)

        self.assertTrue(report.parse_successful)
        # Should have recorded return type info

    def test_builtin_call_types(self):
        """Test type inference from builtin calls."""
        code = """
length = len([1, 2, 3])
text = str(42)
number = int("100")
"""
        inferencer = TypeInference()
        report = inferencer.analyze(code)

        self.assertTrue(report.parse_successful)
        binding_names = [b.name for b in report.all_bindings]
        self.assertIn('length', binding_names)
        self.assertIn('text', binding_names)
        self.assertIn('number', binding_names)

    def test_binary_operation_types(self):
        """Test type inference from binary operations."""
        code = """
a = 10
b = 20
c = a + b  # Should be int
d = a / b  # Should be float
e = a > b  # Should be bool
"""
        inferencer = TypeInference()
        report = inferencer.analyze(code)

        self.assertTrue(report.parse_successful)

    def test_list_comprehension(self):
        """Test type inference with list comprehension."""
        code = """
numbers = [1, 2, 3, 4, 5]
squares = [x**2 for x in numbers]
"""
        inferencer = TypeInference()
        report = inferencer.analyze(code)

        self.assertTrue(report.parse_successful)

    def test_function_with_annotations(self):
        """Test type inference respects annotations."""
        code = """
def greet(name: str, count: int) -> str:
    return name * count
"""
        inferencer = TypeInference()
        report = inferencer.analyze(code)

        self.assertTrue(report.parse_successful)

    def test_syntax_error_handling(self):
        """Test handling of syntax errors."""
        code = """
x = 
"""
        inferencer = TypeInference()
        report = inferencer.analyze(code)

        self.assertFalse(report.parse_successful)
        self.assertIsNotNone(report.syntax_error)

    def test_typing_coverage(self):
        """Test typing coverage calculation."""
        code = """
x = 10
y = "hello"
z = unknown_func()
"""
        inferencer = TypeInference()
        report = inferencer.analyze(code)

        self.assertGreaterEqual(report.typing_coverage, 0.0)
        self.assertLessEqual(report.typing_coverage, 1.0)

    def test_to_dict_serialization(self):
        """Test serialization to dictionary."""
        code = """
x = 10
"""
        inferencer = TypeInference()
        report = inferencer.analyze(code)

        result_dict = report.to_dict()
        self.assertIn('parse_successful', result_dict)
        self.assertIn('total_variables', result_dict)
        self.assertIn('typing_coverage', result_dict)
        self.assertIn('bindings', result_dict)

    def test_class_method_analysis(self):
        """Test type inference in class methods."""
        code = """
class Calculator:
    def __init__(self):
        self.value = 0
    
    def add(self, x: int) -> int:
        self.value += x
        return self.value
"""
        inferencer = TypeInference()
        report = inferencer.analyze(code)

        self.assertTrue(report.parse_successful)

    def test_augmented_assignment(self):
        """Test type inference for augmented assignment."""
        code = """
x = 10
x += 5
x *= 2
"""
        inferencer = TypeInference()
        report = inferencer.analyze(code)

        self.assertTrue(report.parse_successful)


if __name__ == '__main__':
    unittest.main()
