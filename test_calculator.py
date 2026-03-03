"""Unit tests for the scientific calculator CLI module."""

import unittest

from test import Calculator, CalculatorCLI, MathOperationError, ValidationError


class CalculatorTests(unittest.TestCase):
    """Validate core calculator behaviors and error handling."""

    def setUp(self) -> None:
        self.calculator = Calculator()

    def test_basic_expression(self) -> None:
        self.assertEqual(self.calculator.evaluate_expression("5 + 3 * (2 - 1)"), 8.0)

    def test_advanced_functions(self) -> None:
        self.assertEqual(self.calculator.evaluate_expression("sqrt(16) + factorial(4)"), 28.0)

    def test_degree_mode_trig(self) -> None:
        self.calculator.set_mode("degree")
        self.assertAlmostEqual(self.calculator.evaluate_expression("sin(30)"), 0.5, places=12)
        self.assertAlmostEqual(self.calculator.evaluate_expression("asin(0.5)"), 30.0, places=12)

    def test_memory_functions(self) -> None:
        self.calculator.evaluate_expression("10")
        self.calculator.memory_add()
        self.assertEqual(self.calculator.memory_recall(), 10.0)
        self.calculator.memory_subtract(3.0)
        self.assertEqual(self.calculator.memory_recall(), 7.0)
        self.calculator.memory_clear()
        self.assertEqual(self.calculator.memory_recall(), 0.0)

    def test_invalid_expression(self) -> None:
        with self.assertRaises(ValidationError):
            self.calculator.evaluate_expression("bad(1)")

    def test_division_by_zero(self) -> None:
        with self.assertRaises(MathOperationError):
            self.calculator.evaluate_expression("5 / 0")


class CalculatorCliTests(unittest.TestCase):
    """Validate CLI easter egg trigger matching only."""

    def setUp(self) -> None:
        self.cli = CalculatorCLI()

    def test_easter_egg_triggers(self) -> None:
        self.assertTrue(self.cli._is_easter_egg_trigger("1837837"))
        self.assertTrue(self.cli._is_easter_egg_trigger("leblebi"))
        self.assertTrue(self.cli._is_easter_egg_trigger("1837837(leblebi)"))

    def test_easter_egg_non_trigger(self) -> None:
        self.assertFalse(self.cli._is_easter_egg_trigger("2+2"))


if __name__ == "__main__":
    unittest.main()
