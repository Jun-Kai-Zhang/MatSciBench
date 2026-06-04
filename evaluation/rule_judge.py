"""Reusable rule-judge helpers for evaluation."""

from __future__ import annotations

import re
from typing import Any

from sympy import N, sympify

from evaluation.grader import math_equal


SCIENTIFIC_NOTATION_PATTERN = re.compile(
    r"([+-]?\s*(?:\d+(?:\.\d*)?|\.\d+))\s*(?:\\times|×|x|\*)\s*10\s*\^\s*\{?\s*([+-]?\d+)\s*\}?",
    flags=re.IGNORECASE,
)

EXACT_EXPRESSION_PATTERN = re.compile(
    r"\\frac|\\sqrt|sqrt|\\pi|π|\\ln|\bln\s*\(|\\log|\blog\s*\(|\d+\s*/\s*\d+|10\s*\^",
    flags=re.IGNORECASE,
)

SUPERSCRIPT_EXPONENT_TRANS = str.maketrans({
    "⁰": "0",
    "¹": "1",
    "²": "2",
    "³": "3",
    "⁴": "4",
    "⁵": "5",
    "⁶": "6",
    "⁷": "7",
    "⁸": "8",
    "⁹": "9",
    "⁺": "+",
    "⁻": "-",
})

SUPERSCRIPT_EXPONENT_PATTERN = re.compile(r"10\s*([⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻]+)")


def as_text(value: Any) -> str:
    if value is None:
        return ""
    try:
        if value != value:
            return ""
    except Exception:
        pass
    return str(value)


def normalize_unicode_exponents(text: str) -> str:
    return SUPERSCRIPT_EXPONENT_PATTERN.sub(
        lambda match: f"10^{{{match.group(1).translate(SUPERSCRIPT_EXPONENT_TRANS)}}}",
        text,
    )


def extract_numeric_values(value: Any) -> list[float]:
    text = as_text(value)
    text = text.replace("−", "-").replace("–", "-")
    text = normalize_unicode_exponents(text)

    def scientific_repl(match: re.Match[str]) -> str:
        coeff = match.group(1).replace(" ", "")
        exponent = match.group(2).replace(" ", "")
        return f"{coeff}e{exponent}"

    text = SCIENTIFIC_NOTATION_PATTERN.sub(scientific_repl, text)
    text = re.sub(r"\^\{[^}]*\}", " ", text)
    text = re.sub(r"_\{[^}]*\}", " ", text)
    text = re.sub(r"\^[+-]?\d+[+-]?", " ", text)
    text = re.sub(r"(?<=\d),(?=\d{3}\b)", "", text)

    number_pattern = re.compile(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?")
    values: list[float] = []
    for match in number_pattern.finditer(text):
        try:
            values.append(float(match.group(0)))
        except ValueError:
            continue
    return values


def has_exact_expression(value: Any) -> bool:
    text = normalize_unicode_exponents(as_text(value))
    text = SCIENTIFIC_NOTATION_PATTERN.sub(" ", text)
    return bool(EXACT_EXPRESSION_PATTERN.search(text))


def strip_outer_container(text: str) -> str:
    text = text.strip()
    pairs = {"(": ")", "[": "]"}
    if not text or text[0] not in pairs or text[-1] != pairs[text[0]]:
        return text
    depth = 0
    for index, char in enumerate(text):
        if char == text[0]:
            depth += 1
        elif char == pairs[text[0]]:
            depth -= 1
            if depth == 0 and index != len(text) - 1:
                return text
    return text[1:-1].strip()


def split_top_level_values(value: Any) -> list[str]:
    text = strip_outer_container(as_text(value))
    parts: list[str] = []
    current: list[str] = []
    depth = 0
    for char in text:
        if char in "([{":
            depth += 1
        elif char in ")]}":
            depth = max(depth - 1, 0)
        if char == "," and depth == 0:
            part = "".join(current).strip()
            if part:
                parts.append(part)
            current = []
        else:
            current.append(char)
    tail = "".join(current).strip()
    if tail:
        parts.append(tail)
    return parts


def extract_balanced_span(text: str, start_index: int) -> str | None:
    pairs = {"(": ")", "[": "]", "{": "}"}
    opener = text[start_index]
    closer = pairs.get(opener)
    if closer is None:
        return None
    depth = 0
    for index in range(start_index, len(text)):
        char = text[index]
        if char == opener:
            depth += 1
        elif char == closer:
            depth -= 1
            if depth == 0:
                return text[start_index:index + 1]
    return None


def candidate_answer_spans(value: Any) -> list[str]:
    text = as_text(value).strip()
    spans: list[str] = []

    for boxed_match in re.finditer(r"\\boxed\s*\{", text):
        boxed_span = extract_balanced_span(text, boxed_match.end() - 1)
        if boxed_span:
            spans.append(boxed_span[1:-1].strip())

    if text.startswith(("(", "[")):
        leading_span = extract_balanced_span(text, 0)
        if leading_span:
            spans.append(leading_span)

    unique_spans: list[str] = []
    for span in spans:
        if span and span not in unique_spans:
            unique_spans.append(span)
    return unique_spans


def expression_to_float(value: Any) -> float | None:
    text = as_text(value).strip().strip("$")
    if not text:
        return None
    text = normalize_unicode_exponents(text)

    def scientific_repl(match: re.Match[str]) -> str:
        coeff = match.group(1).replace(" ", "")
        exponent = match.group(2).replace(" ", "")
        return f"{coeff}e{exponent}"

    text = SCIENTIFIC_NOTATION_PATTERN.sub(scientific_repl, text)
    if re.fullmatch(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?", text):
        return float(text)

    text = text.replace("−", "-").replace("–", "-")
    text = re.sub(r"\\text\{([^}]*)\}", r"\1", text)
    text = text.replace("\\pi", "pi").replace("π", "pi")
    text = text.replace("\\ln", "log").replace("\\log", "log")
    text = re.sub(r"\\sqrt\{([^{}]+)\}", r"sqrt(\1)", text)
    text = re.sub(r"\\sqrt\s*([A-Za-z0-9.]+)", r"sqrt(\1)", text)
    for _ in range(4):
        updated = re.sub(r"\\frac\{([^{}]+)\}\{([^{}]+)\}", r"(\1)/(\2)", text)
        if updated == text:
            break
        text = updated
    text = text.replace("^", "**")
    text = text.replace("{", "(").replace("}", ")")
    text = text.replace("\\", "")
    text = re.sub(r"/\s*(\d+(?:\.\d*)?)\s*(sqrt\([^()]+\)|pi)", r"/(\1*\2)", text)
    for _ in range(3):
        updated = re.sub(r"(\d|\)|pi)\s*(sqrt|pi|log|\()", r"\1*\2", text)
        updated = re.sub(r"(pi|\))\s*(\d)", r"\1*\2", updated)
        if updated == text:
            break
        text = updated

    try:
        return float(N(sympify(text)))
    except Exception:
        return None


def parse_expression_values(value: Any) -> list[float] | None:
    values: list[float] = []
    for part in split_top_level_values(value):
        parsed = expression_to_float(part)
        if parsed is None:
            return None
        values.append(parsed)
    return values


def parse_answer_span_values(value: Any) -> list[float] | None:
    if has_exact_expression(value):
        expression_values = parse_expression_values(value)
        if expression_values is not None:
            return expression_values
    numeric_values = extract_numeric_values(value)
    if numeric_values:
        return numeric_values
    return None


def compare_values(model_values: list[float], reference_values: list[float]) -> list[str]:
    mismatches = []
    for index, (model_value, reference_value) in enumerate(zip(model_values, reference_values), start=1):
        if reference_value == 0:
            if abs(model_value) > 1e-12:
                mismatches.append(f"#{index}: {model_value} != {reference_value}")
            continue
        if (model_value < 0 < reference_value) or (reference_value < 0 < model_value):
            mismatches.append(f"#{index}: sign mismatch {model_value} != {reference_value}")
            continue
        relative_error = abs(model_value - reference_value) / abs(reference_value)
        if relative_error > 5e-2:
            mismatches.append(f"#{index}: {model_value} != {reference_value} (relative error {relative_error:.4g})")
    return mismatches


def judge_exact_expression_answer(final_answer: Any, correct_answer: Any) -> dict[str, Any] | None:
    model_answer = as_text(final_answer).strip()
    reference_answer = as_text(correct_answer).strip()
    if not (has_exact_expression(model_answer) or has_exact_expression(reference_answer)):
        return None
    try:
        if math_equal(model_answer, reference_answer, include_percentage=False, timeout=True):
            return {
                "is_correct": True,
                "judge_reasoning": (
                    "Rule judge for NUM with exact-expression support and 5% relative tolerance: "
                    f"{model_answer!r} matches {reference_answer!r}."
                ),
            }
    except Exception as exc:
        return {
            "is_correct": False,
            "judge_reasoning": (
                "Rule judge for NUM with exact-expression support failed before numeric fallback: "
                f"{exc}"
            ),
        }

    model_values = parse_expression_values(model_answer)
    reference_values = parse_expression_values(reference_answer)
    if model_values is None or reference_values is None or len(model_values) != len(reference_values):
        return None
    mismatches = compare_values(model_values, reference_values)
    if mismatches:
        return None
    return {
        "is_correct": True,
        "judge_reasoning": (
            "Rule judge for NUM with exact-expression support and 5% relative tolerance: "
            f"parsed model={model_values}, reference={reference_values}; all values match."
        ),
    }


def judge_candidate_span_answer(final_answer: Any, correct_answer: Any) -> dict[str, Any] | None:
    reference_values = parse_answer_span_values(correct_answer)
    if reference_values is None:
        return None
    for span in candidate_answer_spans(final_answer):
        model_values = parse_answer_span_values(span)
        if model_values is None or len(model_values) != len(reference_values):
            continue
        mismatches = compare_values(model_values, reference_values)
        if not mismatches:
            return {
                "is_correct": True,
                "judge_reasoning": (
                    "Rule judge for NUM with leading/boxed answer extraction and 5% relative tolerance: "
                    f"candidate={span!r}, parsed model={model_values}, reference={reference_values}; all values match."
                ),
            }
    return None


def judge_num_answer(final_answer: Any, correct_answer: Any, multiple: bool) -> dict[str, Any]:
    _ = multiple
    model_answer = as_text(final_answer).strip()
    reference_answer = as_text(correct_answer).strip()
    if not model_answer:
        return {"is_correct": False, "judge_reasoning": "Model answer is empty."}
    if not reference_answer:
        return {"is_correct": False, "judge_reasoning": "Correct answer is empty in remote benchmark."}

    exact_judgment = judge_exact_expression_answer(model_answer, reference_answer)
    if exact_judgment is not None and exact_judgment["is_correct"]:
        return exact_judgment

    candidate_judgment = judge_candidate_span_answer(model_answer, reference_answer)
    if candidate_judgment is not None and candidate_judgment["is_correct"]:
        return candidate_judgment

    model_values = extract_numeric_values(model_answer)
    reference_values = extract_numeric_values(reference_answer)
    if not model_values:
        return {
            "is_correct": False,
            "judge_reasoning": f"Rule judge for NUM with 5% relative tolerance: no numeric model answer parsed from {model_answer!r}.",
        }
    if not reference_values:
        return {
            "is_correct": False,
            "judge_reasoning": f"Rule judge for NUM with 5% relative tolerance: no numeric reference answer parsed from {reference_answer!r}.",
        }
    if len(model_values) != len(reference_values):
        return {
            "is_correct": False,
            "judge_reasoning": (
                "Rule judge for NUM with 5% relative tolerance: "
                f"number of parsed values differs. Model={model_values}, reference={reference_values}."
            ),
        }

    mismatches = compare_values(model_values, reference_values)
    if mismatches:
        return {
            "is_correct": False,
            "judge_reasoning": (
                "Rule judge for NUM with 5% relative tolerance: "
                f"parsed model={model_values}, reference={reference_values}; mismatches: {', '.join(mismatches)}"
            ),
        }
    return {
        "is_correct": True,
        "judge_reasoning": (
            "Rule judge for NUM with 5% relative tolerance: "
            f"parsed model={model_values}, reference={reference_values}; all values match."
        ),
    }
