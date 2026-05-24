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
)

# -- CONFIG -----------------------------------------------

MODEL_NAME   = "Salesforce/codet5-base"
DATA_PATH    = "data/train.json"
OUTPUT_DIR   = "./testgen_model"
MAX_SAMPLES  = 5000
MAX_TESTS    = 5
INPUT_LEN    = 192
OUTPUT_LEN   = 128

# -- Data Helpers -------------------------------------------

def extract_input_calls(test_code: str, max_tests: int = 5) -> list[str]:
    """
    Returns up to max_tests input calls sampled across the full check()
    function rather than taken from the top.  Spread ensures diversity
    (early tests tend to be similar; later ones cover edge cases).
    """
    all_calls = []
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

        call = extract_input_args(stripped)
        if call is None:
            continue
        if not is_valid_python(call):
            continue

        all_calls.append(call)

    if not all_calls:
        return []

    if len(all_calls) <= max_tests:
        return all_calls

    step = len(all_calls) / max_tests
    indices = [int(i * step) for i in range(max_tests)]
    return [all_calls[i] for i in indices]


def build_prompt(signature: str) -> str:
    return (
        "Generate Python test inputs.\n\n"
        f"Function:\n{signature}\n\n"
        "Inputs:\n"
    )


def build_pairs(data: list) -> list:
    pairs = []
    for item in data:
        signature = extract_signature(item["solution_code"])
        calls = extract_input_calls(item["test_code"])

        if not calls:
            continue
        
        for call in calls:
            pairs.append({
                "input":  build_prompt(signature),
                "output": call,
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

    with open("data/train.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    data = data[:MAX_SAMPLES]

    pairs = build_pairs(data)
    print(f"Training pairs built: {len(pairs)}")

    for i, p in enumerate(pairs[:3]):
        print(f"\n--- Example {i} ---")
        print("INPUT:\n", p["input"])
        print("OUTPUT:\n", p["output"])
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to(device)

    dataset = Dataset.from_list(pairs)
    tokenize = make_tokenize_fn(tokenizer)
    tokenized = dataset.map(tokenize, remove_columns=["input", "output"])

    training_args = TrainingArguments(
        output_dir            = "./results",
        per_device_train_batch_size = 4,
        num_train_epochs      = 5,
        learning_rate         = 2e-5,
        logging_steps         = 50,
        logging_dir           = "./logs",
        report_to             = "none",
        save_steps            = 500,
        save_total_limit      = 2,
    )
    trainer = Trainer(
        model        = model,
        args         = training_args,
        train_dataset = tokenized,
    )
    trainer.train()

    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"\nModel saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()