import json
import os
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from utils import (
    is_valid_python,
    extract_signature,
    extract_func_name,
    extract_body,
    build_test_assertion,
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
    prompt = (
        "Generate an interesting test input that exercises edge cases.\n\n"
        f"Function:\n{signature}\n{body}\n\n"
        "Interesting input:\n"
    )
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=256,
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
    body = extract_body(solution_code)
    valid_params = extract_param_names(solution_code, func_name)
    
    print(f"Function: {func_name}")
    print(f"Params:   {valid_params}\n")

    inputs = generate_inputs(model, tokenizer, device, signature, body, valid_params)

    print(f"Suggested inputs ({len(inputs)}):")
    for call in inputs:
        print(f"  {call}")
    
    out_path = os.path.join(OUTPUT_DIR, "suggested_inputs.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"function": func_name, "inputs": inputs}, f, indent=2)
    print(f"\nSaved to {out_path}")

if __name__ == "__main__":
    main()