"""Bounded symbolic algebra tools for reproducible mathematical checks.

Only a small, explicit expression grammar is accepted. This is a symbolic
calculation aid, not a theorem prover or a substitute for a mathematical proof.
"""
from __future__ import annotations

import ast
import math
from typing import Callable

import sympy as sp


_FUNCTIONS: dict[str, Callable] = {
    "sin": sp.sin,
    "cos": sp.cos,
    "tan": sp.tan,
    "asin": sp.asin,
    "acos": sp.acos,
    "atan": sp.atan,
    "sinh": sp.sinh,
    "cosh": sp.cosh,
    "tanh": sp.tanh,
    "exp": sp.exp,
    "log": sp.log,
    "sqrt": sp.sqrt,
    "Abs": sp.Abs,
}
_CONSTANTS = {"pi": sp.pi, "E": sp.E}


def _parse_expression(source: str, symbols: dict[str, sp.Symbol]) -> sp.Expr:
    if not isinstance(source, str) or not source.strip() or len(source) > 2000:
        raise ValueError("Each expression must be a non-empty string of at most 2000 characters.")
    try:
        tree = ast.parse(source, mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"Invalid symbolic expression: {exc.msg}") from exc

    def visit(node):
        if isinstance(node, ast.Expression):
            return visit(node.body)
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            if isinstance(node.value, float) and not math.isfinite(node.value):
                raise ValueError("Non-finite numeric literals are not supported.")
            return sp.Integer(node.value) if isinstance(node.value, int) else sp.Float(node.value)
        if isinstance(node, ast.Name):
            if node.id in symbols:
                return symbols[node.id]
            if node.id in _CONSTANTS:
                return _CONSTANTS[node.id]
            raise ValueError(f"Unknown symbol or function name: {node.id}")
        if isinstance(node, ast.BinOp):
            left, right = visit(node.left), visit(node.right)
            operations = {
                ast.Add: lambda a, b: a + b,
                ast.Sub: lambda a, b: a - b,
                ast.Mult: lambda a, b: a * b,
                ast.Div: lambda a, b: a / b,
                ast.Pow: lambda a, b: a ** b,
            }
            operation = operations.get(type(node.op))
            if operation is None:
                raise ValueError("Only +, -, *, /, and ** are supported.")
            if isinstance(node.op, ast.Pow) and isinstance(right, sp.Integer) and abs(int(right)) > 1000:
                raise ValueError("Integer exponents must have magnitude at most 10000.")
            return operation(left, right)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            value = visit(node.operand)
            return value if isinstance(node.op, ast.UAdd) else -value
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            function = _FUNCTIONS.get(node.func.id)
            if function is None or node.keywords or len(node.args) != 1:
                raise ValueError("Only listed one-argument mathematical functions are supported.")
            return function(visit(node.args[0]))
        raise ValueError(f"Unsupported expression syntax: {type(node).__name__}")

    return visit(tree)


def symbolic_math_tool(
    operation: str,
    expressions: list[str],
    variables: list[str] | None = None,
) -> dict:
    """Simplify expressions, differentiate one expression, or form a Jacobian."""
    allowed_operations = {"simplify", "differentiate", "jacobian"}
    if operation not in allowed_operations:
        raise ValueError(f"operation must be one of {sorted(allowed_operations)}.")
    if not isinstance(expressions, list) or not expressions or len(expressions) > 20:
        raise ValueError("Provide between 1 and 20 expressions.")
    variables = variables or []
    if not isinstance(variables, list) or len(variables) > 20:
        raise ValueError("variables must be a list of at most 20 symbol names.")
    if any(not isinstance(name, str) or not name.isidentifier() or name in _FUNCTIONS or name in _CONSTANTS for name in variables):
        raise ValueError("Variables must be simple identifiers and must not shadow supported functions/constants.")
    symbols = {name: sp.Symbol(name, real=True) for name in variables}
    parsed = [_parse_expression(source, symbols) for source in expressions]

    if operation == "simplify":
        output = [sp.simplify(expr) for expr in parsed]
        result = {"expressions": [sp.sstr(expr) for expr in output]}
        latex = [sp.latex(expr) for expr in output]
    elif operation == "differentiate":
        if len(parsed) != 1 or len(variables) != 1:
            raise ValueError("differentiate requires exactly one expression and one variable.")
        derivative = sp.simplify(sp.diff(parsed[0], symbols[variables[0]]))
        result = {"expression": sp.sstr(parsed[0]), "variable": variables[0], "derivative": sp.sstr(derivative)}
        latex = [sp.latex(derivative)]
    else:
        if not variables:
            raise ValueError("jacobian requires at least one variable.")
        matrix = sp.Matrix(parsed).jacobian([symbols[name] for name in variables])
        matrix = matrix.applyfunc(sp.simplify)
        result = {
            "expressions": [sp.sstr(expr) for expr in parsed],
            "variables": variables,
            "jacobian": [[sp.sstr(matrix[i, j]) for j in range(matrix.cols)] for i in range(matrix.rows)],
        }
        latex = [sp.latex(matrix)]

    return {
        "status": "symbolic_result",
        "operation": operation,
        "result": result,
        "latex": latex,
        "assumptions": "Declared variables are real; no other domain or parameter assumptions were imposed.",
        "limitations": [
            "Symbolic simplification is not a proof of a dynamical-systems theorem.",
            "The accepted expression grammar is deliberately restricted.",
            "Check domain restrictions, singularities, branch choices, and model assumptions separately.",
        ],
    }
