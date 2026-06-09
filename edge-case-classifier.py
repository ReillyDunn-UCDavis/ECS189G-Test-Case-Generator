"""
Uses Gemini to generate an edge-case taxonomy from a problem statement
and/or solution code. The returned taxonomy is a Dict[str, EdgeClassifier]
that plugs directly into eval.py's run_coverage_eval().
"""

from __future__ import annotations

import json
import os
import re
import textwrap
import time
from typing import Dict

from google import genai
from dotenv import load_dotenv

load_dotenv()
_client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

from eval import TestCase, EdgeClassifier, evaluate_model, generate_report

_SAFE_NAMESPACE = {
    "__builtins__": {},
    "len": len,
    "any": any,
    "all": all,
    "isinstance": isinstance,
    "abs": abs,
    "min": min,
    "max": max,
    "sum": sum,
    "sorted": sorted,
    "list": list,
    "set": set,
    "str": str,
    "int": int,
    "float": float,
    "bool": bool,
}


def _make_classifier(expression: str) -> EdgeClassifier:
    def classifier(tc: TestCase) -> bool:
        try:
            return bool(eval(expression, _SAFE_NAMESPACE, {"tc": tc}))
        except Exception:
            return False
    return classifier


def _parse_retry_seconds(error_str: str) -> float | None:
    m = re.search(r"retry in (\d+(?:\.\d+)?)\s*s", error_str, re.IGNORECASE)
    return float(m.group(1)) if m else None


def generate_taxonomy(
    problem_statement: str = "",
    solution_code: str = "",
    model: str = "gemini-2.0-flash",
    num_categories: int = 8,
) -> Dict[str, EdgeClassifier]:
    """
    Call Gemini to generate an edge-case taxonomy for a given problem.

    At least one of problem_statement or solution_code must be provided.
    Providing both gives the model the most context to work with.
    """
    if not problem_statement and not solution_code:
        raise ValueError("Provide at least one of problem_statement or solution_code.")

    context_parts: list[str] = []
    if problem_statement:
        context_parts.append(f"## Problem Statement\n{problem_statement}")
    if solution_code:
        context_parts.append(f"## Solution Code\n```python\n{solution_code.strip()}\n```")
    context = "\n\n".join(context_parts)

    prompt = f"""\
You are an expert software tester. Analyze the problem below and generate a comprehensive edge-case taxonomy.

{context}

Generate {num_categories} distinct edge-case categories that together expose the most important \
boundary conditions and failure modes for this problem.

For each category produce:
1. A short snake_case name (e.g. empty_input, single_element, all_negative)
2. A Python boolean expression that returns True when a test case belongs to this category

The expression is evaluated with `tc` available as a TestCase object:
  tc.inputs        — dict of keyword arguments, e.g. {{"nums": [1, 2, 3], "target": 5}}
  tc.expected_output — the expected return value

Allowed symbols in expressions: len, any, all, isinstance, abs, min, max, sum, sorted, \
list, set, str, int, float, bool

Return ONLY a JSON object with no surrounding text or markdown fences:
{{
  "categories": [
    {{"name": "empty_input", "expression": "tc.inputs.get('nums', []) == []"}},
    ...
  ]
}}

Expression rules:
- Single-line valid Python boolean expression
- Use .get(key, default) for dict access to avoid KeyError
- Reference only tc.inputs and tc.expected_output"""

    # Retry with exponential backoff on per-minute rate-limit errors
    # Daily quota exhaustion cannot be resolved by waiting — fail fast
    max_retries = 5
    default_wait = 30.0  # seconds
    wait = default_wait

    for attempt in range(max_retries):
        try:
            response = _client.models.generate_content(model=model, contents=prompt)
            raw = response.text.strip()
            break
        except Exception as e:
            err_str = str(e)
            is_rate_limit = (
                "429" in err_str
                or "quota" in err_str.lower()
                or "rate" in err_str.lower()
                or "resource exhausted" in err_str.lower()
            )

            if not is_rate_limit:
                raise

            # Daily quota cannot be resolved by retrying — tell the user clearly.
            if "perday" in err_str.lower() or "per_day" in err_str.lower():
                raise RuntimeError(
                    "Gemini daily quota exhausted. The free tier allows a limited number of "
                    "requests per day per model. Please try again tomorrow, or switch to a "
                    "paid plan / a different model."
                ) from e

            if attempt >= max_retries - 1:
                raise

            # Use the delay the API suggests, otherwise fall back to exponential backoff
            suggested = _parse_retry_seconds(err_str)
            wait = suggested if suggested is not None else wait
            print(f"Rate limit hit, retrying in {wait:.0f}s... (attempt {attempt + 1}/{max_retries})")
            time.sleep(wait)
            if suggested is None:
                wait *= 2  # exponential backoff only when API gave no hint

    fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", raw)
    if fence_match:
        raw = fence_match.group(1).strip()

    data = json.loads(raw)

    taxonomy: Dict[str, EdgeClassifier] = {}
    for cat in data["categories"]:
        taxonomy[cat["name"]] = _make_classifier(cat["expression"])

    return taxonomy


# Demo — Sum Integer Array
def main():
    solution_code = textwrap.dedent("""\
        def sum_array(nums):
            total = 0
            for n in nums:
                total += n
            return total
    """)

    problem_statement = (
        "Given a list of integers `nums`, return the sum of all elements. "
        "The list may be empty, contain negative numbers, zeros, or very large values."
    )

    print("Generating edge-case taxonomy via Gemini...")
    taxonomy = generate_taxonomy(
        problem_statement=problem_statement,
        solution_code=solution_code,
    )
    print(f"Generated {len(taxonomy)} categories: {list(taxonomy.keys())}\n")

    def sum_array(nums):
        return sum(nums)

    model_tests = [
        {"input": {"nums": []},              "expected_output": 0},
        {"input": {"nums": [5]},             "expected_output": 5},
        {"input": {"nums": [1, 2, 3]},       "expected_output": 6},
        {"input": {"nums": [-1, -2, -3]},    "expected_output": -6},
        {"input": {"nums": [0, 0, 0]},       "expected_output": 0},
        {"input": {"nums": [10**6] * 10},    "expected_output": 10**7},
        {"input": {"nums": [1, -1, 1, -1]},  "expected_output": 0},
        {"input": {"nums": list(range(10_000))}, "expected_output": sum(range(10_000))},
    ]

    result = evaluate_model(
        model_name="demo-model",
        raw_test_cases=model_tests,
        reference_solution=sum_array,
        taxonomy=taxonomy,
        order_sensitive=False,
    )

    report = generate_report([result])
    print(report)


if __name__ == "__main__":
    import os
    os.makedirs("results", exist_ok=True)
    main()
