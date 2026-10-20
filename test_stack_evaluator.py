"""
Unit tests for stack_evaluator.py

Run with:
    python -m unittest -v test_stack_evaluator.py
"""

import tempfile
import unittest
from pathlib import Path

from stack_evaluator import (
    ExpressionError,
    Stack,
    StackUnderflowError,
    evaluate_expression,
    evaluate_lines,
    format_result,
    main,
)


class StackTests(unittest.TestCase):
    def test_new_stack_is_empty(self):
        stack = Stack()
        self.assertTrue(stack.is_empty())
        self.assertEqual(stack.size(), 0)

    def test_push_and_peek(self):
        stack = Stack()
        stack.push(1)
        stack.push(2)
        self.assertEqual(stack.peek(), 2)
        self.assertEqual(stack.size(), 2)  # peek does not remove

    def test_pop_is_lifo(self):
        stack = Stack()
        for item in (1, 2, 3):
            stack.push(item)
        self.assertEqual([stack.pop(), stack.pop(), stack.pop()], [3, 2, 1])
        self.assertTrue(stack.is_empty())

    def test_pop_empty_raises(self):
        with self.assertRaises(StackUnderflowError):
            Stack().pop()

    def test_peek_empty_raises(self):
        with self.assertRaises(StackUnderflowError):
            Stack().peek()


class EvaluationTests(unittest.TestCase):
    def check(self, expression, expected):
        self.assertAlmostEqual(evaluate_expression(expression), expected, places=9)

    def test_assignment_examples(self):
        self.check("3 + 5 * 2", 13)
        self.check("(8 / 4) + 7 * 2", 16)
        self.check("10 - (2 + 3) * 4", -10)

    def test_precedence(self):
        self.check("2 + 3 * 4", 14)
        self.check("2 * 3 + 4", 10)
        self.check("10 - 6 / 3", 8)

    def test_left_to_right_associativity(self):
        self.check("20 - 5 - 3", 12)
        self.check("100 / 10 / 5", 2)

    def test_parentheses(self):
        self.check("(2 + 3) * 4", 20)
        self.check("((2 + 3) * (4 + 6)) / 5", 10)
        self.check("2 * (3 + (4 - 1)) * 5", 60)
        self.check("((((7))))", 7)

    def test_decimals(self):
        self.check("1.5 * 4 + 0.25", 6.25)
        self.check(".5 + .5", 1)

    def test_unary_operators(self):
        self.check("-5 + 12", 7)
        self.check("7 - -3", 10)
        self.check("-(2 + 3)", -5)
        self.check("2 * -3", -6)
        self.check("+4", 4)
        self.check("--5", 5)

    def test_no_spaces(self):
        self.check("3+5*2", 13)

    def test_single_number(self):
        self.check("42", 42)


class ErrorTests(unittest.TestCase):
    def assert_error(self, expression, fragment):
        with self.assertRaises(ExpressionError) as context:
            evaluate_expression(expression)
        self.assertIn(fragment, str(context.exception))

    def test_division_by_zero(self):
        self.assert_error("8 / (4 - 4)", "Division by zero")

    def test_missing_closing_parenthesis(self):
        self.assert_error("(3 + 4", "missing ')'")

    def test_unexpected_closing_parenthesis(self):
        self.assert_error("3 + 4)", "unexpected ')'")

    def test_missing_operand(self):
        self.assert_error("3 + * 4", "Missing operand")
        self.assert_error("3 +", "ends with an operator")
        self.assert_error("()", "Missing operand")

    def test_missing_operator(self):
        self.assert_error("3 4", "Missing operator")
        self.assert_error("(2)(3)", "Missing operator")

    def test_invalid_character(self):
        self.assert_error("5 $ 3", "Invalid character")
        self.assert_error("2 ^ 3", "Invalid character")

    def test_empty_expression(self):
        self.assert_error("   ", "Empty expression")


class FormattingTests(unittest.TestCase):
    def test_whole_numbers_have_no_decimal_point(self):
        self.assertEqual(format_result(13.0), "13")
        self.assertEqual(format_result(-10.0), "-10")
        self.assertEqual(format_result(-0.0), "0")

    def test_fractions(self):
        self.assertEqual(format_result(2.5), "2.5")
        self.assertEqual(format_result(10 / 3), "3.333333")
        self.assertEqual(format_result(0.1 + 0.2), "0.3")


class FileTests(unittest.TestCase):
    def test_separators_and_order_are_preserved(self):
        lines = ["3 + 5 * 2", "-----", "(8 / 4) + 7 * 2", "-----", "10 - (2 + 3) * 4"]
        output, results = evaluate_lines(lines)
        self.assertEqual(output, ["13", "-----", "16", "-----", "-10"])
        self.assertEqual(len(results), 3)

    def test_bad_line_does_not_stop_processing(self):
        output, _ = evaluate_lines(["1 / 0", "-----", "2 + 2"])
        self.assertEqual(output, ["Error: Division by zero", "-----", "4"])

    def test_blank_lines_are_kept(self):
        output, _ = evaluate_lines(["1 + 1", "", "2 + 2"])
        self.assertEqual(output, ["2", "", "4"])

    def test_end_to_end(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / "in.txt"
            target = Path(folder) / "out.txt"
            source.write_text("3 + 5 * 2\n-----\n10 - (2 + 3) * 4\n", encoding="utf-8")
            self.assertEqual(main([str(source), str(target)]), 0)
            self.assertEqual(target.read_text(encoding="utf-8"), "13\n-----\n-10\n")

    def test_missing_input_file(self):
        with tempfile.TemporaryDirectory() as folder:
            missing = Path(folder) / "nope.txt"
            self.assertEqual(main([str(missing), str(Path(folder) / "o.txt")]), 1)


if __name__ == "__main__":
    unittest.main()
