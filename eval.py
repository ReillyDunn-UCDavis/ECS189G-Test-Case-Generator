"""
Metrics:
  1. Accuracy  — what fraction of generated test cases have correct expected_output
  2. Coverage  — what fraction of a ground-truth edge-case taxonomy is touched

Usage:
  python eval.py          # runs the built-in Two Sum demo
"""

from __future__ import annotations

import json
import traceback
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

# a single generated test case
@dataclass
class TestCase:
    inputs: Dict[str, Any] # keyword-argument dict, e.g. {"nums": [2,7], "target": 9}
    expected_output: Any
    raw: Optional[Dict] = None # original dict from the model, for debugging


@dataclass
class ModelEvalResult:
    model_name: str
    total_tests: int = 0

    # Accuracy
    passing_tests: int = 0
    failing_tests: List[Dict] = field(default_factory=list) # {tc, actual, expected}
    errored_tests: List[Dict] = field(default_factory=list) # {tc, error}
    accuracy: float = 0.0

    # Coverage
    covered_categories: List[str] = field(default_factory=list)
    missing_categories: List[str] = field(default_factory=list)
    coverage: float = 0.0

    # Composite
    composite_score: float = 0.0

# Deep-equality check that handles common LeetCode output types
def compare_outputs(
    actual: Any,
    expected: Any,
    order_sensitive: bool = True,
) -> bool:
    if type(actual) != type(expected):
        # allow int/float cross-comparison for numeric problems
        if isinstance(actual, (int, float)) and isinstance(expected, (int, float)):
            return actual == expected
        return False

    if isinstance(expected, list):
        if len(actual) != len(expected):
            return False # what about problems where the input can be any size list
        if not order_sensitive:
            # compare as sorted lists (handles problems like "return any valid answer")
            try:
                return sorted(actual) == sorted(expected)
            except TypeError:
                # elements not sortable — fall back to set-of-tuples for lists of lists
                return sorted(map(tuple, actual)) == sorted(map(tuple, expected))
        return all(compare_outputs(a, e, order_sensitive=True) for a, e in zip(actual, expected))

    if isinstance(expected, dict):
        if set(actual.keys()) != set(expected.keys()):
            return False
        return all(compare_outputs(actual[k], expected[k]) for k in expected)

    if isinstance(expected, set):
        return actual == expected

    return actual == expected

# Metric 1 — Accuracy
def run_accuracy_eval(
    reference_solution: Callable,
    test_cases: List[TestCase],
    order_sensitive: bool = True,
) -> Tuple[int, List[Dict], List[Dict]]:
    passing = 0
    failing: List[Dict] = []
    errored: List[Dict] = []

    for tc in test_cases:
        try:
            actual = reference_solution(**tc.inputs)
            if compare_outputs(actual, tc.expected_output, order_sensitive=order_sensitive):
                passing += 1
            else:
                failing.append({
                    "inputs": tc.inputs,
                    "expected": tc.expected_output,
                    "actual": actual,
                })
        except Exception:
            errored.append({
                "inputs": tc.inputs,
                "expected": tc.expected_output,
                "error": traceback.format_exc(limit=3),
            })

    return passing, failing, errored

# Metric 2 — Edge Case Coverage
EdgeClassifier = Callable[[TestCase], bool]

# Map each test case to edge-case categories using classifier functions.
def run_coverage_eval(
    test_cases: List[TestCase],
    taxonomy: Dict[str, EdgeClassifier],
) -> Tuple[List[str], List[str]]:
    covered: set[str] = set()
    for tc in test_cases:
        for category, classifier in taxonomy.items():
            if category not in covered:
                try:
                    if classifier(tc):
                        covered.add(category)
                except Exception:
                    pass  # classifier bug → skip, don't crash eval

    all_categories = set(taxonomy.keys())
    missing = sorted(all_categories - covered)
    return sorted(covered), missing


# Report generation
def _truncate(value: Any, max_len: int = 80) -> str:
    """Repr of value, truncated for readable report lines."""
    s = repr(value)
    return s if len(s) <= max_len else s[:max_len] + "…"


def generate_report(
    results: List[ModelEvalResult],
    accuracy_weight: float = 0.6,
    coverage_weight: float = 0.4,
    save_path: Optional[str] = None,
) -> str:
    assert abs(accuracy_weight + coverage_weight - 1.0) < 1e-9, \
        "Weights must sum to 1.0"

    lines: List[str] = []
    lines.append("=" * 60)
    lines.append("  MODEL EVALUATION REPORT")
    lines.append("=" * 60)

    serializable: List[Dict] = []

    for r in results:
        lines.append(f"\nModel: {r.model_name}")
        lines.append(f"  Total test cases : {r.total_tests}")
        lines.append(f"  Accuracy         : {r.accuracy:.2%}  "
                     f"({r.passing_tests}/{r.total_tests} passing, "
                     f"{len(r.failing_tests)} failing, "
                     f"{len(r.errored_tests)} errored)")

        if r.failing_tests:
            lines.append("  Failing cases:")
            for i, f in enumerate(r.failing_tests, 1):
                lines.append(f"    [{i}] inputs={_truncate(f['inputs'])}  "
                              f"expected={_truncate(f['expected'])}  "
                              f"actual={_truncate(f['actual'])}")

        if r.errored_tests:
            lines.append("  Errored cases (reference solution crashed):")
            for i, e in enumerate(r.errored_tests, 1):
                lines.append(f"    [{i}] inputs={_truncate(e['inputs'])}  "
                              f"error={e['error'].splitlines()[-1]}")

        lines.append(f"  Coverage         : {r.coverage:.2%}  "
                     f"({len(r.covered_categories)}/{len(r.covered_categories) + len(r.missing_categories)} categories)")
        lines.append(f"  Covered  : {r.covered_categories}")
        lines.append(f"  Missing  : {r.missing_categories}")
        lines.append(f"  Composite score  : {r.composite_score:.4f}  "
                     f"(accuracy×{accuracy_weight} + coverage×{coverage_weight})")
        lines.append("-" * 60)

        serializable.append({
            "model": r.model_name,
            "total_tests": r.total_tests,
            "accuracy": r.accuracy,
            "passing_tests": r.passing_tests,
            "failing_tests": r.failing_tests,
            "errored_tests": [
                {k: v for k, v in e.items() if k != "error"} for e in r.errored_tests
            ],
            "coverage": r.coverage,
            "covered_categories": r.covered_categories,
            "missing_categories": r.missing_categories,
            "composite_score": r.composite_score,
        })

    report_str = "\n".join(lines)

    if save_path:
        with open(save_path, "w", encoding="utf-8") as fh:
            json.dump(serializable, fh, indent=2)
        report_str += f"\n\nReport saved to {save_path}"

    return report_str


# Top-level evaluator
"""
    Parse raw model output dicts, run both metrics, return a ModelEvalResult.

    raw_test_cases format expected from models:
        [{"input": {...kwargs...}, "expected_output": <value>}, ...]

    "input" may also be a list of positional args — these are not supported
    here; normalise them to kwargs before calling, or override _parse_test_cases.
    """
def evaluate_model(
    model_name: str,
    raw_test_cases: List[Dict],
    reference_solution: Callable,
    taxonomy: Dict[str, EdgeClassifier],
    accuracy_weight: float = 0.6,
    coverage_weight: float = 0.4,
    order_sensitive: bool = True,
) -> ModelEvalResult:
    test_cases = _parse_test_cases(raw_test_cases)
    total = len(test_cases)

    passing, failing, errored = run_accuracy_eval(
        reference_solution, test_cases, order_sensitive=order_sensitive
    )
    accuracy = passing / total if total else 0.0

    covered, missing = run_coverage_eval(test_cases, taxonomy)
    n_all = len(taxonomy)
    coverage = len(covered) / n_all if n_all else 0.0

    composite = accuracy_weight * accuracy + coverage_weight * coverage

    return ModelEvalResult(
        model_name=model_name,
        total_tests=total,
        passing_tests=passing,
        failing_tests=failing,
        errored_tests=errored,
        accuracy=accuracy,
        covered_categories=covered,
        missing_categories=missing,
        coverage=coverage,
        composite_score=composite,
    )


def _parse_test_cases(raw: List[Dict]) -> List[TestCase]:
    cases: List[TestCase] = []
    for item in raw:
        inp = item.get("input", {})
        if not isinstance(inp, dict):
            raise ValueError(
                f"Expected 'input' to be a kwargs dict, got {type(inp).__name__}. "
                "Convert positional args to kwargs before calling evaluate_model()."
            )
        cases.append(TestCase(
            inputs=inp,
            expected_output=item.get("expected_output"),
            raw=item,
        ))
    return cases



import argparse
import json
import re

from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import torch
from peft import PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)
from finetune_qwen_asserts import (
    clean_text,
    extract_function_signature,
    extract_asserts,
    build_user_prompt,
    SYSTEM_PROMPT,
)

TEST_PATH = "data/test.json"
BASE_MODEL = "Qwen/Qwen2.5-Coder-0.5B-Instruct"
ADAPTER_PATH = "qwen_asserts_adapter"

def main():
    # Reference solution (known-correct)
    def two_sum(nums, target):
        seen = {}
        for i, n in enumerate(nums):
            if target - n in seen:
                return [seen[target - n], i]
            seen[n] = i
        return []

    # Edge-case taxonomy 
    taxonomy: Dict[str, EdgeClassifier] = {
        "empty_input":      lambda tc: tc.inputs.get("nums") == [],
        "single_element":   lambda tc: len(tc.inputs.get("nums", [])) == 1,
        "negative_numbers": lambda tc: any(x < 0 for x in tc.inputs.get("nums", [])),
        "duplicates":       lambda tc: len(tc.inputs["nums"]) != len(set(tc.inputs["nums"]))
                                       if "nums" in tc.inputs else False,
        "max_constraints":  lambda tc: len(tc.inputs.get("nums", [])) >= 10_000,
        "no_solution":      lambda tc: tc.inputs.get("expected_output") == []
                                       or tc.expected_output == [],
        "zero_values":      lambda tc: 0 in tc.inputs.get("nums", []),
        "answer_at_ends":   lambda tc: (
            len(tc.inputs.get("nums", [])) >= 2 and
            tc.inputs["nums"][0] + tc.inputs["nums"][-1] == tc.inputs.get("target", None)
        ),
    }

    # Mock model outputs
    model_a_tests = [
        {"input": {"nums": [2, 7, 11, 15], "target": 9},   "expected_output": [0, 1]},
        {"input": {"nums": [3, 2, 4],       "target": 6},   "expected_output": [1, 2]},
        {"input": {"nums": [3, 3],           "target": 6},   "expected_output": [0, 1]},
        {"input": {"nums": [-1, -2, -3, -4, -5], "target": -8}, "expected_output": [2, 4]},
        {"input": {"nums": [0, 4, 3, 0],    "target": 0},   "expected_output": [0, 3]},
        # deliberate wrong expected output — accuracy test
        {"input": {"nums": [1, 2, 3],        "target": 5},   "expected_output": [1, 2]}, # wrong: should be [1,2]
    ]

    # Run evaluation 
    results = [
        evaluate_model(
            model_name="Qwen 0.5B LoRA",
            raw_test_cases=model_a_tests,
            reference_solution=two_sum,
            taxonomy=taxonomy,
            order_sensitive=False, # Two Sum allows any valid pair order
        ),
    ]

    report = generate_report(
        results,
        accuracy_weight=0.6,
        coverage_weight=0.4,
        save_path="results/eval_report.json",
    )
    print(report)

if __name__ == "__main__":
    import os
    os.makedirs("results", exist_ok=True)
    main()
