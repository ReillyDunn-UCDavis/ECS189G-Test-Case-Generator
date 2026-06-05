import json
import os
import textwrap
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from utils import (
    is_valid_python,
    extract_signature,
    extract_func_name,
    extract_body,
    call_uses_valid_params,
    extract_param_names,
)

# -- Config --------------------------------------------------------------------

MODEL_PATH = "./testgen_model"
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def generate_inputs(
    model,
    tokenizer,
    device,
    signature: str,
    body: str,
    valid_params: set[str],
    n: int = 8,
    max_new_tokens: int = 64,
) -> list[str]:
    """
    Generates n unique test assertions of the form:
        assert candidate(...) == <expected>

    The model is responsible for predicting both the input arguments
    and the expected return value.
    """
    params_str = ", ".join(sorted(valid_params)) if valid_params else "(none)"

    prompt = (
        "Generate a test assertion with input and expected output.\n\n"
        f"Function:\n{signature}\n{body}\n\n"
        f"Parameters: {params_str}\n\n"
        "Test assertion:\n"
    )
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=256,
    ).to(device)

    seen = set()
    assertions = []

    for _ in range(n):
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.9,
            top_p=0.95,
        )
        decoded = tokenizer.decode(output[0], skip_special_tokens=True).strip()

        # Must be a full assertion
        if not decoded.startswith("assert candidate("):
            continue
        # Must contain == to have an expected value
        if " == " not in decoded:
            continue
        # Must be syntactically valid Python
        if not is_valid_python(decoded):
            continue
        # No duplicates
        if decoded in seen:
            continue

        seen.add(decoded)
        assertions.append(decoded)

    return assertions

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=False)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH).to(device)

    solution_code = textwrap.dedent("""
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
""").strip()

    signature = extract_signature(solution_code)
    func_name = extract_func_name(solution_code)
    body = extract_body(solution_code)
    valid_params = extract_param_names(solution_code, func_name)

    print(f"Function: {func_name}")
    print(f"Params:   {valid_params}\n")

    assertions = generate_inputs(model, tokenizer, device, signature, body, valid_params)

    print(f"Generated assertions ({len(assertions)}):")
    for a in assertions:
        print(f"  {a}")

    out_path = os.path.join(OUTPUT_DIR, "suggested_inputs.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"function": func_name, "assertions": assertions}, f, indent=2)
    print(f"\nSaved to {out_path}")

if __name__ == "__main__":
    main()