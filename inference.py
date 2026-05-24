import json
import os
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from utils import (
    is_valid_python,
    extract_signature,
    extract_func_name,
    build_test_assertion,
    call_uses_valid_params,
    extract_param_names,
)

# -- Config --------------------------------------------------------------------

MODEL_PATH = "./testgen_model"
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -- Step 1: Generate raw input calls from the model ----------------------------

def generate_input_calls(
    model,
    tokenizer,
    device,
    signature: str,
    valid_params: set[str],
    n: int = 8,
    max_new_tokens: int = 64,
) -> list[str]:
    prompt = (
        "Generate Python test inputs.\n\n"
        f"Function:\n{signature}\n\n"
        "Inputs:\n"
    )
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=192,
    ).to(device)

    seen = set()
    calls = []

    for _ in range(n):
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.9,
            top_p=0.95,
        )
        decoded = tokenizer.decode(output[0], skip_special_tokens=True).strip()

        if not decoded.startswith("candidate("):
            continue
        if not is_valid_python(decoded):
            continue
        if decoded in seen:
            continue
        if not call_uses_valid_params(decoded, valid_params):
            print(f"  [bad params] {decoded!r} — expected: {valid_params}")
            continue

        seen.add(decoded)
        calls.append(decoded)

    return calls

# -- Step 2: Execute each input call against the real solution ------------------

def build_test_suite(
    solution_code: str,
    func_name: str,
    input_calls: list[str],
) -> list[str]:
    """
    For each input call, runs the real solution and builds a correct assert.
    Skips any call that causes the solution to throw an exception.
    
    Returns a list of assert strings with verified expected values.
    """
    tests = []
    for call in input_calls:
        assertion = build_test_assertion(solution_code, func_name, call)
        if assertion is not None:
            tests.append(assertion)
        else:
            print(f"  [skip] execution failed for: {call}")
    return tests

# -- Main -----------------------------------------------------------------------

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=False)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH).to(device)

    solution_code = """
        class Solution:
            def lengthOfLongestSubstring(self, s: str) -> int:
                d = {}
                res = 0
                left = 0
                for right, c in enumerate(s):
                    if c in d and d[c] >= left:
                        left = d[c] + 1
                    d[c] = right
                    res = max(res, right - left + 1)
                return res
        """
    
    signature = extract_signature(solution_code)
    func_name = extract_func_name(solution_code)
    valid_params = extract_param_names(solution_code, func_name)
    
    print(f"Signature: {signature}")

    print("\nGenerating input calls...")
    input_calls = generate_input_calls(model, tokenizer, device, signature, valid_params)
    print(f"Raw model output ({len(input_calls)}) calls:")
    for c in input_calls:
        print(f"    {c}")
    
    print("\nBuilding test cases...")
    tests = build_test_suite(solution_code, func_name, input_calls)

    print(f"\nFinal tests ({len(tests)}):")
    for t in tests:
        print(f"    {t}")

    result = {
        "signature":    signature,
        "input_calls":  input_calls,
        "tests:":       tests,
    }
    out_path = os.path.join(OUTPUT_DIR, "inference_result.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved to {out_path}")

if __name__ == "__main__":
    main()