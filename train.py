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


def build_prompt(signature: str, body: str) -> str:
    return (
        "Generate an interesting test input that exercises edge cases.\n\n"
        f"Function:\n{signature}\n{body}\n\n"
        "Interesting input:\n"
    )


def build_pairs(data: list) -> list:
    pairs = []
    for item in data:
        signature = extract_signature(item["solution_code"])
        body      = extract_body(item["solution_code"])
        calls     = extract_input_calls(item["test_code"])

        if not calls:
            continue

        edge = generate_edge_case_call(item["solution_code"], extract_func_name(item["solution_code"]))
        if edge and edge not in calls:
            calls.insert(0, edge)

        for call in calls:
            pairs.append({
                "input":  build_prompt(signature, body),
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

    # Load data
    with open("data/train.json", "r", encoding="utf-8") as f:
        train_data = json.load(f)
    train_data = train_data[:MAX_SAMPLES]

    with open("data/test.json", "r", encoding="utf-8") as f:
        val_data = json.load(f)

    # Build pairs
    train_pairs = build_pairs(train_data)
    val_pairs   = build_pairs(val_data)
    print(f"Train: {len(train_pairs)}  Val: {len(val_pairs)}")

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