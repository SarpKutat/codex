"""Production-ready scientific calculator CLI application."""

from __future__ import annotations

import ast
import math
import os
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple


class CalculatorError(Exception):
    """Base exception for calculator-related errors."""


class ValidationError(CalculatorError):
    """Raised when user input or expression structure is invalid."""


class MathOperationError(CalculatorError):
    """Raised when a mathematical operation cannot be completed."""


@dataclass
class HistoryEntry:
    """Represents one completed calculation."""

    expression: str
    result: float


class ExpressionEvaluator:
    """Safely evaluates mathematical expressions using an AST whitelist."""

    def __init__(self, calculator: "Calculator") -> None:
        """Initialize evaluator with calculator context."""
        self.calculator = calculator

    def evaluate(self, expression: str) -> float:
        """Parse and evaluate a user expression."""
        normalized = expression.strip()
        if not normalized:
            raise ValidationError("Expression cannot be empty.")

        try:
            parsed = ast.parse(normalized, mode="eval")
        except SyntaxError as exc:
            raise ValidationError("Invalid expression syntax.") from exc

        return float(self._eval_node(parsed.body))

    def _eval_node(self, node: ast.AST) -> float:
        """Recursively evaluate supported AST nodes."""
        if isinstance(node, ast.Constant):
            if not isinstance(node.value, (int, float)):
                raise ValidationError("Only numeric constants are allowed.")
            return float(node.value)

        if isinstance(node, ast.Num):  # pragma: no cover (compatibility path)
            return float(node.n)

        if isinstance(node, ast.BinOp):
            left = self._eval_node(node.left)
            right = self._eval_node(node.right)
            return self._eval_binop(node.op, left, right)

        if isinstance(node, ast.UnaryOp):
            value = self._eval_node(node.operand)
            return self._eval_unary(node.op, value)

        if isinstance(node, ast.Call):
            return self._eval_call(node)

        if isinstance(node, ast.Name):
            return self._eval_name(node.id)

        raise ValidationError("Unsupported expression element detected.")

    def _eval_binop(self, op: ast.operator, left: float, right: float) -> float:
        """Evaluate supported binary operations."""
        try:
            if isinstance(op, ast.Add):
                return left + right
            if isinstance(op, ast.Sub):
                return left - right
            if isinstance(op, ast.Mult):
                return left * right
            if isinstance(op, ast.Div):
                if right == 0:
                    raise MathOperationError("Division by zero is not allowed.")
                return left / right
            if isinstance(op, ast.FloorDiv):
                if right == 0:
                    raise MathOperationError("Floor division by zero is not allowed.")
                return left // right
            if isinstance(op, ast.Mod):
                if right == 0:
                    raise MathOperationError("Modulus by zero is not allowed.")
                return left % right
            if isinstance(op, ast.Pow):
                return math.pow(left, right)
        except OverflowError as exc:
            raise MathOperationError("Result is too large to represent.") from exc
        except ValueError as exc:
            raise MathOperationError(str(exc)) from exc

        raise ValidationError("Unsupported binary operator.")

    def _eval_unary(self, op: ast.unaryop, value: float) -> float:
        """Evaluate unary plus/minus operations."""
        if isinstance(op, ast.UAdd):
            return +value
        if isinstance(op, ast.USub):
            return -value
        raise ValidationError("Unsupported unary operator.")

    def _eval_call(self, node: ast.Call) -> float:
        """Evaluate allowed function calls with validated arguments."""
        if not isinstance(node.func, ast.Name):
            raise ValidationError("Only direct function calls are allowed.")

        name = node.func.id
        if name not in self.calculator.functions:
            raise ValidationError(f"Unknown function: {name}")

        if node.keywords:
            raise ValidationError("Keyword arguments are not supported.")

        args = [self._eval_node(arg) for arg in node.args]
        function = self.calculator.functions[name]

        try:
            return float(function(*args))
        except TypeError as exc:
            raise ValidationError(f"Invalid arguments for function '{name}'.") from exc
        except ValueError as exc:
            raise MathOperationError(str(exc)) from exc
        except OverflowError as exc:
            raise MathOperationError("Result is too large to represent.") from exc

    def _eval_name(self, name: str) -> float:
        """Resolve supported constants and memory references."""
        constants = {
            "pi": math.pi,
            "e": math.e,
            "ans": self.calculator.last_result,
            "mr": self.calculator.memory,
        }

        if name not in constants:
            raise ValidationError(f"Unknown identifier: {name}")
        return float(constants[name])


class Calculator:
    """Core scientific calculator logic and state management."""

    def __init__(self) -> None:
        """Initialize calculator state and operation registry."""
        self.mode = "radian"
        self.memory = 0.0
        self.last_result = 0.0
        self.history: List[HistoryEntry] = []
        self.evaluator = ExpressionEvaluator(self)
        self.functions = self._build_functions()

    def _build_functions(self) -> Dict[str, Callable[..., float]]:
        """Build supported function map for expression evaluation."""
        return {
            "sqrt": self._sqrt,
            "cbrt": self._cbrt,
            "factorial": self._factorial,
            "abs": abs,
            "log": self._log10,
            "ln": self._ln,
            "exp": self._exp,
            "sin": self._sin,
            "cos": self._cos,
            "tan": self._tan,
            "asin": self._asin,
            "acos": self._acos,
            "atan": self._atan,
            "arcsin": self._asin,
            "arccos": self._acos,
            "arctan": self._atan,
        }

    def set_mode(self, mode: str) -> None:
        """Set angle mode to degree or radian."""
        cleaned = mode.strip().lower()
        if cleaned not in {"degree", "radian"}:
            raise ValidationError("Mode must be 'degree' or 'radian'.")
        self.mode = cleaned

    def evaluate_expression(self, expression: str) -> float:
        """Evaluate expression and update answer/history state."""
        result = self.evaluator.evaluate(expression)
        self.last_result = result
        self.history.append(HistoryEntry(expression=expression, result=result))
        return result

    def memory_add(self, value: float | None = None) -> float:
        """Add provided value (or last result) to memory."""
        self.memory += self.last_result if value is None else value
        return self.memory

    def memory_subtract(self, value: float | None = None) -> float:
        """Subtract provided value (or last result) from memory."""
        self.memory -= self.last_result if value is None else value
        return self.memory

    def memory_recall(self) -> float:
        """Return current memory value."""
        return self.memory

    def memory_clear(self) -> None:
        """Reset memory storage to zero."""
        self.memory = 0.0

    def clear_history(self) -> None:
        """Clear stored calculation history."""
        self.history.clear()

    def get_history(self) -> List[Tuple[str, float]]:
        """Return history as a list of expression/result tuples."""
        return [(item.expression, item.result) for item in self.history]

    def _sqrt(self, value: float) -> float:
        """Compute square root with domain validation."""
        if value < 0:
            raise MathOperationError("Square root is undefined for negative numbers.")
        return math.sqrt(value)

    def _cbrt(self, value: float) -> float:
        """Compute cube root preserving sign."""
        return math.copysign(abs(value) ** (1 / 3), value)

    def _factorial(self, value: float) -> float:
        """Compute factorial for non-negative integers only."""
        if not float(value).is_integer() or value < 0:
            raise MathOperationError("Factorial is only defined for non-negative integers.")
        return float(math.factorial(int(value)))

    def _log10(self, value: float) -> float:
        """Compute base-10 logarithm with domain validation."""
        if value <= 0:
            raise MathOperationError("Logarithm is only defined for positive numbers.")
        return math.log10(value)

    def _ln(self, value: float) -> float:
        """Compute natural logarithm with domain validation."""
        if value <= 0:
            raise MathOperationError("Natural logarithm is only defined for positive numbers.")
        return math.log(value)

    def _exp(self, value: float) -> float:
        """Compute exponential e^x."""
        return math.exp(value)

    def _to_radian(self, value: float) -> float:
        """Convert value to radians when in degree mode."""
        return math.radians(value) if self.mode == "degree" else value

    def _from_radian(self, value: float) -> float:
        """Convert radian result to configured angle mode."""
        return math.degrees(value) if self.mode == "degree" else value

    def _sin(self, value: float) -> float:
        """Compute sine in selected angle mode."""
        return math.sin(self._to_radian(value))

    def _cos(self, value: float) -> float:
        """Compute cosine in selected angle mode."""
        return math.cos(self._to_radian(value))

    def _tan(self, value: float) -> float:
        """Compute tangent in selected angle mode."""
        return math.tan(self._to_radian(value))

    def _asin(self, value: float) -> float:
        """Compute inverse sine in selected angle mode."""
        if not -1 <= value <= 1:
            raise MathOperationError("asin domain is [-1, 1].")
        return self._from_radian(math.asin(value))

    def _acos(self, value: float) -> float:
        """Compute inverse cosine in selected angle mode."""
        if not -1 <= value <= 1:
            raise MathOperationError("acos domain is [-1, 1].")
        return self._from_radian(math.acos(value))

    def _atan(self, value: float) -> float:
        """Compute inverse tangent in selected angle mode."""
        return self._from_radian(math.atan(value))


class CalculatorCLI:
    """Command-line user interface for interactive calculator usage."""

    def __init__(self) -> None:
        """Initialize CLI with calculator instance."""
        self.calculator = Calculator()

    def run(self) -> None:
        """Start continuous interactive calculation loop."""
        self._print_welcome()

        while True:
            try:
                user_input = input("calc> ").strip()
                if not user_input:
                    continue
                if self._is_easter_egg_trigger(user_input):
                    self._show_galata_tower_easter_egg()
                    continue
                if self._handle_command(user_input):
                    continue

                result = self.calculator.evaluate_expression(user_input)
                print(f"= {self._format_result(result)}")

            except (ValidationError, MathOperationError) as exc:
                print(f"Error: {exc}")
            except KeyboardInterrupt:
                print("\nUse 'exit' to quit the calculator.")
            except EOFError:
                print("\nGoodbye!")
                break

    def _is_easter_egg_trigger(self, user_input: str) -> bool:
        """Return True when the hidden Galata Tower easter egg is triggered."""
        normalized = user_input.strip().lower()
        return normalized in {"1837837", "leblebi", "1837837(leblebi)"}

    def _show_galata_tower_easter_egg(self) -> None:
        """Print hidden Galata Tower ASCII art and a fun unlock message."""
        print(
            """
               /\
              /  \\
             /____\\
               ||
             __||__
            /  __  \\
           /  /  \\  \\
          |  | () |  |
          |  |____|  |
          |  |    |  |
          |  |    |  |
          |  |____|  |
          |__________|
          /__________\\
            """.rstrip("\n")
        )
        print("Leblebi unlocked! Welcome to Galata Tower 🗼")

    def _handle_command(self, raw_input: str) -> bool:
        """Process supported non-expression commands."""
        command = raw_input.lower()

        if command == "exit":
            print("Goodbye!")
            raise SystemExit(0)

        if command == "history":
            self._print_history()
            return True

        if command == "clear":
            self._clear_screen()
            self._print_welcome()
            return True

        if command == "mc":
            self.calculator.memory_clear()
            print("Memory cleared.")
            return True

        if command == "mr":
            print(f"MR = {self._format_result(self.calculator.memory_recall())}")
            return True

        if command in {"m+", "m-"}:
            value = self.calculator.memory_add() if command == "m+" else self.calculator.memory_subtract()
            print(f"Memory = {self._format_result(value)}")
            return True

        if command.startswith("m+") or command.startswith("m-"):
            return self._handle_memory_with_value(raw_input)

        if command.startswith("mode "):
            _, mode = raw_input.split(maxsplit=1)
            self.calculator.set_mode(mode)
            print(f"Angle mode set to {self.calculator.mode}.")
            return True

        return False

    def _handle_memory_with_value(self, raw_input: str) -> bool:
        """Handle M+ value and M- value command variants."""
        op = raw_input[:2].lower()
        payload = raw_input[2:].strip()
        if not payload:
            return False

        value = self.calculator.evaluate_expression(payload)
        if op == "m+":
            new_value = self.calculator.memory_add(value)
        elif op == "m-":
            new_value = self.calculator.memory_subtract(value)
        else:
            return False

        print(f"Memory = {self._format_result(new_value)}")
        return True

    def _print_history(self) -> None:
        """Display stored calculation history."""
        history = self.calculator.get_history()
        if not history:
            print("No history available.")
            return

        for index, (expression, result) in enumerate(history, start=1):
            print(f"{index}. {expression} = {self._format_result(result)}")

    def _clear_screen(self) -> None:
        """Clear terminal screen in a cross-platform manner."""
        os.system("cls" if os.name == "nt" else "clear")

    def _print_welcome(self) -> None:
        """Print welcome banner and quick command reference."""
        print("Scientific Calculator (Python CLI)")
        print("Type expressions such as: 5 + 3 * (2 - 1)")
        print("Functions: sqrt, cbrt, factorial, abs, log, ln, exp, sin, cos, tan, asin, acos, atan")
        print("Constants: pi, e, ans, mr")
        print("Commands: history, clear, exit, mode degree, mode radian, M+, M-, MR, MC")

    def _format_result(self, value: float) -> str:
        """Format result with floating-point normalization."""
        if abs(value) < 1e-15:
            value = 0.0
        return f"{value:.15g}"


def main() -> None:
    """Application entry point."""
    CalculatorCLI().run()


if __name__ == "__main__":
    main()
