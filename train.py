import ast
import json
import re
import sys

import torch
from datasets import Dataset
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)

sys.stdout.reconfigure(line_buffering=True)

MODEL_NAME = "Salesforce/codet5-base"
TRAIN_PATH = "data/train.json"
VAL_PATH = "data/test.json"
OUTPUT_DIR = "./testgen_model"
RESULTS_DIR = "./results"
LOG_DIR = "./logs"

MAX_TRAIN_PROBLEMS = 1000
MAX_VAL_PROBLEMS = 200
MAX_EXAMPLES_PER_PROBLEM = 3
MAX_INPUT_LENGTH = 384
MAX_OUTPUT_LENGTH = 96
PER_DEVICE_BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = 4
NUM_EPOCHS = 2
LEARNING_RATE = 3e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.05


def clean_text(text):
    text = text.replace("\u00a0", " ")
    text = text.replace("Â", " ")
    text = re.sub(r"\s+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_problem_context(problem_description):
    description = clean_text(problem_description)
    constraints = ""

    if "Constraints:" in description:
        constraints = description.split("Constraints:", 1)[1]
        if "Follow-up:" in constraints:
            constraints = constraints.split("Follow-up:", 1)[0]
        constraints = constraints.strip()

    for marker in ("Example 1:", "Examples:"):
        if marker in description:
            description = description.split(marker, 1)[0].strip()

    if constraints:
        description = f"{description}\n\nConstraints:\n{constraints}"

    return description


def extract_signature(solution_code):
    tree = ast.parse(solution_code)

    for node in tree.body:
        if not isinstance(node, ast.ClassDef) or node.name != "Solution":
            continue

        for item in node.body:
            if not isinstance(item, ast.FunctionDef) or item.name == "__init__":
                continue

            args = []
            positional = list(item.args.posonlyargs) + list(item.args.args)
            if positional and positional[0].arg == "self":
                positional = positional[1:]

            defaults = list(item.args.defaults)
            default_offset = len(positional) - len(defaults)

            for index, arg in enumerate(positional):
                piece = arg.arg
                if arg.annotation is not None:
                    piece += f": {ast.unparse(arg.annotation)}"
                if index >= default_offset:
                    piece += f" = {ast.unparse(defaults[index - default_offset])}"
                args.append(piece)

            if item.returns is not None:
                returns = f" -> {ast.unparse(item.returns)}"
            else:
                returns = ""

            return f"def {item.name}({', '.join(args)}){returns}"

    return None


def normalize_example_text(text):
    text = clean_text(text)
    text = re.sub(r"\s*=\s*", "=", text)
    text = re.sub(r",\s*", ", ", text)
    return text


def format_example(example):
    example_input = normalize_example_text(example["input"])
    example_output = normalize_example_text(example["output"])
    return f"Input: {example_input}\nOutput: {example_output}"


def build_prompt(signature, description):
    return (
        "Read the function signature and problem description.\n"
        "Generate one valid example test case that matches the task.\n"
        "Use exactly this format:\n"
        "Input: <arguments>\n"
        "Output: <expected result>\n\n"
        f"Signature:\n{signature}\n\n"
        f"Description:\n{description}\n\n"
        "Example test case:\n"
    )


def build_pairs(items):
    pairs = []

    for item in items:
        signature = extract_signature(item["solution_code"])
        description = extract_problem_context(item["problem_description"])
        examples = item.get("examples", [])[:MAX_EXAMPLES_PER_PROBLEM]

        if not signature or not description or not examples:
            continue

        prompt = build_prompt(signature, description)
        for example in examples:
            pairs.append({
                "input": prompt,
                "output": format_example(example),
            })

    return pairs


def make_tokenize_fn(tokenizer):
    def tokenize(example):
        model_inputs = tokenizer(
            example["input"],
            truncation=True,
            max_length=MAX_INPUT_LENGTH,
        )
        labels = tokenizer(
            example["output"],
            truncation=True,
            max_length=MAX_OUTPUT_LENGTH,
        )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    return tokenize


def load_json(path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def main():
    print("CUDA available:", torch.cuda.is_available())
    print("Using device:", "cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    train_items = load_json(TRAIN_PATH)[:MAX_TRAIN_PROBLEMS]
    val_items = load_json(VAL_PATH)[:MAX_VAL_PROBLEMS]

    train_pairs = build_pairs(train_items)
    val_pairs = build_pairs(val_items)

    print(f"Train problems: {len(train_items)}")
    print(f"Validation problems: {len(val_items)}")
    print(f"Train pairs: {len(train_pairs)}")
    print(f"Validation pairs: {len(val_pairs)}")

    if not train_pairs or not val_pairs:
        raise ValueError("No training pairs were created from the dataset.")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    tokenize = make_tokenize_fn(tokenizer)
    train_dataset = Dataset.from_list(train_pairs).map(
        tokenize,
        remove_columns=["input", "output"],
    )
    val_dataset = Dataset.from_list(val_pairs).map(
        tokenize,
        remove_columns=["input", "output"],
    )

    training_args = TrainingArguments(
        output_dir=RESULTS_DIR,
        per_device_train_batch_size=PER_DEVICE_BATCH_SIZE,
        per_device_eval_batch_size=PER_DEVICE_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        num_train_epochs=NUM_EPOCHS,
        dataloader_num_workers=0,
        logging_steps=50,
        logging_dir=LOG_DIR,
        report_to="none",
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        warmup_ratio=WARMUP_RATIO,
        save_strategy="epoch",
        evaluation_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=torch.cuda.is_available(),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    )

    trainer.train()

    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"Saved model to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
