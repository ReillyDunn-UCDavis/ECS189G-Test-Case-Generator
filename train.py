import json
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer, AutoModelForSeq2SeqLM,
    TrainingArguments, Trainer,
)
from utils import (
    is_valid_python,
    normalize_whitespace,
    simplify_assert,
    extract_input_args,
    extract_signature,
    extract_body,
    generate_edge_case_call,
    extract_func_name,
    build_test_assertion,
)

# -- CONFIG -----------------------------------------------

MODEL_NAME   = "Salesforce/codet5-base"
DATA_PATH    = "data/train.json"
OUTPUT_DIR   = "./testgen_model"
#MAX_SAMPLES  = 5000
MAX_TESTS    = 5
INPUT_LEN    = 192
OUTPUT_LEN   = 128

# -- Data Helpers -------------------------------------------

def extract_full_assertions(test_code: str, solution_code: str, max_tests: int = 5) -> list[str]:
    """
    Returns up to max_tests full assert statements sampled across the full
    check() function. Each assertion is the complete string:
        assert candidate(...) == <expected>

    Spread sampling ensures diversity (early tests tend to be similar;
    later ones cover edge cases).
    """
    all_assertions = []
    inside_check = False

    for line in test_code.split("\n"):
        if "def check(candidate):" in line:
            inside_check = True
            continue
        if not inside_check:
            continue

        stripped = normalize_whitespace(line.strip())
        if not stripped.startswith("assert"):
            continue
        if len(stripped) > 200:
            continue

        stripped = simplify_assert(stripped)
        if "sorted(" in stripped:
            continue

        # Keep the full assert, not just the input args
        if not is_valid_python(stripped):
            continue

        # Must match the standard pattern: assert candidate(...) == <val>
        if not stripped.startswith("assert candidate("):
            continue

        all_assertions.append(stripped)

    if not all_assertions:
        return []

    if len(all_assertions) <= max_tests:
        return all_assertions

    step = len(all_assertions) / max_tests
    indices = [int(i * step) for i in range(max_tests)]
    return [all_assertions[i] for i in indices]


def build_prompt(signature: str, body: str) -> str:
    return (
        "Generate a test assertion with input and expected output.\n\n"
        f"Function:\n{signature}\n{body}\n\n"
        "Test assertion:\n"
    )


def build_pairs(data: list) -> list:
    pairs = []
    for item in data:
        solution_code = item["solution_code"]
        signature     = extract_signature(solution_code)
        body          = extract_body(solution_code)
        func_name     = extract_func_name(solution_code)
        assertions    = extract_full_assertions(item["test_code"], solution_code)

        if not assertions:
            continue

        # # Attempt to add one edge-case assertion via execution
        # edge_call = generate_edge_case_call(solution_code, func_name)
        # if edge_call:
        #     edge_assert = build_test_assertion(solution_code, func_name, edge_call)
        #     if edge_assert and edge_assert not in assertions:
        #         assertions.insert(0, edge_assert)

        for assertion in assertions:
            pairs.append({
                "input":  build_prompt(signature, body),
                "output": assertion,          # full "assert candidate(...) == ..."
            })

    return pairs

# ── Tokenization ──────────────────────────────────────────────────────────────

def make_tokenize_fn(tokenizer):
    def tokenize(example):
        model_inputs = tokenizer(
            example["input"],
            truncation=True,
            padding="max_length",
            max_length=INPUT_LEN,
        )
        labels = tokenizer(
            example["output"],
            truncation=True,
            padding="max_length",
            max_length=OUTPUT_LEN,
        )
        # Replace padding token id with -100 so loss ignores it
        model_inputs["labels"] = [
            tok if tok != tokenizer.pad_token_id else -100
            for tok in labels["input_ids"]
        ]
        return model_inputs
    return tokenize

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Load data
    with open("data/train.json", "r", encoding="utf-8") as f:
        train_data = json.load(f)
    #train_data = train_data[:MAX_SAMPLES]

    with open("data/test.json", "r", encoding="utf-8") as f:
        val_data = json.load(f)

    # Build pairs
    train_pairs = build_pairs(train_data)
    val_pairs   = build_pairs(val_data)
    print(f"Train: {len(train_pairs)}  Val: {len(val_pairs)}")

    # Sanity-check: print a few examples so you can verify the format
    print("\nSample training targets:")
    for p in train_pairs[:3]:
        print(f"  {p['output']}")

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
    model     = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to(device)

    # Tokenize
    tokenize = make_tokenize_fn(tokenizer)
    train_ds = Dataset.from_list(train_pairs).map(tokenize, remove_columns=["input", "output"])
    val_ds   = Dataset.from_list(val_pairs).map(tokenize,   remove_columns=["input", "output"])

    # Train
    training_args = TrainingArguments(
        output_dir                  = "./results",
        per_device_train_batch_size = 4,
        num_train_epochs            = 10,
        learning_rate               = 2e-5,
        logging_steps               = 50,
        logging_dir                 = "./logs",
        report_to                   = "none",
        save_strategy               = "epoch",
        save_total_limit            = 2,
        evaluation_strategy         = "epoch",
        load_best_model_at_end      = True,
        metric_for_best_model       = "eval_loss",
    )
    trainer = Trainer(
        model         = model,
        args          = training_args,
        train_dataset = train_ds,
        eval_dataset  = val_ds,
    )
    trainer.train()

    # Save
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"\nModel saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()