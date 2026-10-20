#!/usr/bin/env python3
"""
Stack-Based Mathematical Expression Evaluator
=============================================

Reads mathematical expressions from a text file (input.txt), evaluates each
one using a stack-based approach (a two-stack algorithm based on Dijkstra's
"shunting-yard" idea), and writes the results to another text file
(output.txt) in the same layout as the input.

Supported features
------------------
* Operators:  +  -  *  /
* Parentheses for grouping, including nested parentheses
* Correct operator precedence (* and / before + and -)
* Left-to-right evaluation of operators with equal precedence
* Integers and decimal numbers (e.g. 12, 3.5, .25)
* Unary minus / plus (e.g. -5 + 12, 7 - -3)
* Clear error messages for invalid expressions (the program never crashes
  because of a bad line; it writes "Error: <reason>" and carries on)

Usage
-----
    python stack_evaluator.py                      # input.txt -> output.txt
    python stack_evaluator.py in.txt out.txt       # custom file names
"""

import argparse
import math
import re
import sys
from pathlib import Path

# --------------------------------------------------------------------------
# Custom exceptions
# --------------------------------------------------------------------------


class StackUnderflowError(Exception):
    """Raised when pop() or peek() is called on an empty stack."""


class ExpressionError(Exception):
    """Raised when an expression is invalid or cannot be evaluated."""


# --------------------------------------------------------------------------
# 1. Stack class
# --------------------------------------------------------------------------


class Stack:
    """A simple Last-In-First-Out (LIFO) stack backed by a Python list.

    The end of the list is the *top* of the stack, so push and pop are both
    O(1) operations.
    """

    def __init__(self):
        self._items = []

    def push(self, item):
        """Place *item* on the top of the stack."""
        self._items.append(item)

    def pop(self):
        """Remove and return the item on the top of the stack."""
        if self.is_empty():
            raise StackUnderflowError("pop from an empty stack")
        return self._items.pop()

    def peek(self):
        """Return the item on the top of the stack without removing it."""
        if self.is_empty():
            raise StackUnderflowError("peek at an empty stack")
        return self._items[-1]

    def is_empty(self):
        """Return True if the stack holds no items."""
        return len(self._items) == 0

    def size(self):
        """Return the number of items in the stack."""
        return len(self._items)

    def to_list(self):
        """Return a copy of the contents, bottom first (used for tracing)."""
        return list(self._items)

    def __len__(self):
        return len(self._items)

    def __repr__(self):
        return f"Stack({self._items})"


# --------------------------------------------------------------------------
# 2. Tokenizer, precedence rules and operator application
# --------------------------------------------------------------------------

# Token kinds
NUMBER, OPERATOR, LPAREN, RPAREN = "NUMBER", "OPERATOR", "LPAREN", "RPAREN"

# Internal symbol used for unary minus (so it can be told apart from the
# binary minus on the operator stack).
UNARY_MINUS = "u-"

# Higher number = binds more tightly.
PRECEDENCE = {"+": 1, "-": 1, "*": 2, "/": 2, UNARY_MINUS: 3}

_TOKEN_PATTERN = re.compile(
    r"\s*(?:(?P<number>\d+\.?\d*|\.\d+)|(?P<symbol>[+\-*/()])|(?P<bad>\S))"
)

# A separator line contains only dashes (three or more), e.g. -----
SEPARATOR_PATTERN = re.compile(r"^\s*-{3,}\s*$")


def tokenize(expression):
    """Split *expression* into a list of (kind, value) tokens.

    Raises ExpressionError if an unsupported character is found.
    """
    tokens = []
    for match in _TOKEN_PATTERN.finditer(expression):
        if match.group("number") is not None:
            tokens.append((NUMBER, float(match.group("number"))))
        elif match.group("symbol") is not None:
            symbol = match.group("symbol")
            if symbol == "(":
                tokens.append((LPAREN, symbol))
            elif symbol == ")":
                tokens.append((RPAREN, symbol))
            else:
                tokens.append((OPERATOR, symbol))
        else:
            raise ExpressionError(f"Invalid character '{match.group('bad')}'")
    return tokens


def apply_operator(operator, operands):
    """Pop the needed operand(s) from *operands*, apply *operator*, and push
    the result back onto *operands*."""
    try:
        if operator == UNARY_MINUS:
            operands.push(-operands.pop())
            return

        right = operands.pop()  # the second operand is popped first
        left = operands.pop()

        if operator == "+":
            operands.push(left + right)
        elif operator == "-":
            operands.push(left - right)
        elif operator == "*":
            operands.push(left * right)
        elif operator == "/":
            if right == 0:
                raise ExpressionError("Division by zero")
            operands.push(left / right)
        else:  # pragma: no cover - tokenizer prevents this
            raise ExpressionError(f"Unknown operator '{operator}'")
    except StackUnderflowError:
        raise ExpressionError("Malformed expression (missing operand)")


# --------------------------------------------------------------------------
# 3. Expression evaluation (two-stack algorithm)
# --------------------------------------------------------------------------


def evaluate_expression(expression, trace=None):
    """Evaluate one mathematical expression and return its value (a float).

    Two stacks are used:
      * operands  - numbers waiting to be used
      * operators - operators and '(' waiting to be applied

    If *trace* is a list, a snapshot of both stacks is appended after every
    step as (description, operand_stack, operator_stack).  This is only used
    for demonstration/testing and does not change the result.

    Raises ExpressionError for any invalid expression.
    """
    tokens = tokenize(expression)
    if not tokens:
        raise ExpressionError("Empty expression")

    operands = Stack()
    operators = Stack()
    expect_operand = True  # True when the next token must start an operand

    def snapshot(label):
        if trace is not None:
            trace.append((label, operands.to_list(), operators.to_list()))

    for kind, value in tokens:
        if kind == NUMBER:
            if not expect_operand:
                raise ExpressionError("Missing operator before number")
            operands.push(value)
            expect_operand = False
            snapshot(_label(value))

        elif kind == LPAREN:
            if not expect_operand:
                raise ExpressionError("Missing operator before '('")
            operators.push("(")
            snapshot("(")

        elif kind == RPAREN:
            if expect_operand:
                raise ExpressionError("Missing operand before ')'")
            # Apply everything back to the matching '('
            while True:
                if operators.is_empty():
                    raise ExpressionError("Mismatched parentheses: unexpected ')'")
                if operators.peek() == "(":
                    break
                apply_operator(operators.pop(), operands)
            operators.pop()  # discard the '('
            expect_operand = False
            snapshot(")")

        else:  # OPERATOR
            if expect_operand:
                # The operator appears where an operand should be -> unary
                if value == "-":
                    operators.push(UNARY_MINUS)
                    snapshot("-  (unary)")
                elif value == "+":
                    snapshot("+  (unary, ignored)")
                else:
                    raise ExpressionError(f"Missing operand before '{value}'")
                continue

            # Binary operator: first apply any operator on the stack that has
            # higher or equal precedence (this gives left-to-right grouping).
            while (
                not operators.is_empty()
                and operators.peek() != "("
                and PRECEDENCE[operators.peek()] >= PRECEDENCE[value]
            ):
                apply_operator(operators.pop(), operands)
            operators.push(value)
            expect_operand = True
            snapshot(value)

    if expect_operand:
        raise ExpressionError("Expression ends with an operator (missing operand)")

    # Apply whatever operators are left
    while not operators.is_empty():
        operator = operators.pop()
        if operator == "(":
            raise ExpressionError("Mismatched parentheses: missing ')'")
        apply_operator(operator, operands)
        snapshot("end of input")

    if operands.size() != 1:  # pragma: no cover - guarded by the checks above
        raise ExpressionError("Malformed expression")
    return operands.pop()


def _label(number):
    """Short text for a number in a trace."""
    return format_result(number)


def format_result(value):
    """Turn a numeric result into clean text: 13.0 -> '13', 2.5 -> '2.5',
    1/3 -> '0.333333'."""
    if not math.isfinite(value):
        raise ExpressionError("Result is out of range")
    if value == int(value):
        return str(int(value))
    text = f"{value:.6f}".rstrip("0").rstrip(".")
    return "0" if text in ("", "-0") else text


# --------------------------------------------------------------------------
# 4. File handling
# --------------------------------------------------------------------------


def read_input_file(path):
    """Read *path* and return its lines (without line-break characters)."""
    with open(path, "r", encoding="utf-8-sig") as handle:
        return handle.read().splitlines()


def evaluate_lines(lines):
    """Evaluate every expression in *lines*.

    * separator lines (-----) are copied unchanged,
    * blank lines stay blank,
    * every other line is treated as an expression and replaced by its result
      (or by "Error: <reason>" if it cannot be evaluated).

    Returns (output_lines, results) where *results* is a list of
    (expression, result_text) pairs that stores every evaluated expression.
    """
    output_lines = []
    results = []
    for line in lines:
        if SEPARATOR_PATTERN.match(line):
            output_lines.append(line.strip())
        elif line.strip() == "":
            output_lines.append("")
        else:
            expression = line.strip()
            try:
                result_text = format_result(evaluate_expression(expression))
            except ExpressionError as error:
                result_text = f"Error: {error}"
            output_lines.append(result_text)
            results.append((expression, result_text))
    return output_lines, results


def write_output_file(path, output_lines):
    """Write *output_lines* to *path*, one per line."""
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(output_lines))
        handle.write("\n")


# --------------------------------------------------------------------------
# 5. Command-line entry point
# --------------------------------------------------------------------------


def main(argv=None):
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Evaluate mathematical expressions from a file using a stack."
    )
    parser.add_argument(
        "input_file", nargs="?", default=str(here / "input.txt"),
        help="file containing the expressions (default: input.txt)",
    )
    parser.add_argument(
        "output_file", nargs="?", default=str(here / "output.txt"),
        help="file the results are written to (default: output.txt)",
    )
    args = parser.parse_args(argv)

    try:
        lines = read_input_file(args.input_file)
    except FileNotFoundError:
        print(f"Error: input file '{args.input_file}' was not found.", file=sys.stderr)
        return 1
    except OSError as error:
        print(f"Error: could not read '{args.input_file}': {error}", file=sys.stderr)
        return 1

    output_lines, results = evaluate_lines(lines)

    try:
        write_output_file(args.output_file, output_lines)
    except OSError as error:
        print(f"Error: could not write '{args.output_file}': {error}", file=sys.stderr)
        return 1

    # Console summary
    print(f"{'Expression':<32} Result")
    print("-" * 52)
    for expression, result_text in results:
        print(f"{expression:<32} {result_text}")
    errors = sum(1 for _, text in results if text.startswith("Error:"))
    print("-" * 52)
    print(f"{len(results)} expression(s) processed, {errors} error(s).")
    print(f"Results written to: {args.output_file}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
