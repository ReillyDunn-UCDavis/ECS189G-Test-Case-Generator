import argparse
import json
import re

from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

# This is the system prompt for the model to avoid writing anything but asserts
#SYSTEM_PROMPT = (
#    "You write Python assert statements only."
#    "Every line must start with 'assert'. "
#    "Every assert must call candidate(...), not the original function name. "
#    "Do not explain anything. "
#    "Do not write markdown. "
#    "Do not write a solution."
#)

SYSTEM_PROMPT = (
    "You write Python assert statements only."
    "Every line must start with 'assert'. "
    "Every assert must call candidate(...), not the original function name. "
    "Every assert have the correct output given the input."
    "Write exactly the requested number of assertions."
    "Use the same style as the training data."
    "Do not write anything else."
    "Do not explain anything."
    "Do not write markdown."
)


# Data processing cleaning
def clean_text(text):
    text = text.replace("\u00a0", " ").replace("Â", " ")
    text = re.sub(r"\s+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# Return function signature which should be the first function of Solution class
def extract_function_signature(solution_code):
    inside_solution = False
    for line in solution_code.splitlines():
        stripped = line.strip()
        if stripped.startswith("class Solution"):
            inside_solution = True
            continue
        if inside_solution and stripped.startswith("def "):
            return stripped
    return None

# Get all assert statements from test_code
def extract_asserts(test_code, max_asserts):
    return [
        line.strip()
        for line in test_code.splitlines()
        if line.strip().startswith("assert")
    ][:max_asserts]


# Create user prompt that has problem description and function signature from example
def build_user_prompt(example):
    signature = extract_function_signature(example["solution_code"])
    if not signature:
        return None

    return (
        "Problem description:\n"
        f"{clean_text(example['problem_description'])}\n\n"
        "Function signature:\n"
        f"{signature}\n"
    )


# Create prompt for model to continue generating asserts
def build_text(example, tokenizer, max_asserts):
    prompt = build_user_prompt(example)
    asserts = extract_asserts(example["test_code"], max_asserts)
    if not prompt or not asserts:
        return None

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt
          + f"\nWrite exactly {len(asserts)} assert statements"
         },
        {"role": "assistant", "content": "\n".join(asserts)},
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )

# Create training dataset from JSON file and convert into list of prompts
def build_dataset(path, tokenizer, max_items, max_asserts):
    with open(path, "r", encoding="utf-8") as file:
        items = json.load(file)

    rows = []
    for example in items[:max_items]:
        text = build_text(example, tokenizer, max_asserts)
        if text:
            rows.append({"text": text})

    if not rows:
        raise ValueError(f"No rows were built from {path}.")

    return Dataset.from_list(rows)

# Convert dataset into token IDs via tokenizer
def tokenize_dataset(dataset, tokenizer, max_length):
    def tokenize(example):
        return tokenizer(
            example["text"],
            truncation=True,
            max_length=max_length,
        )

    return dataset.map(tokenize, remove_columns=["text"])


def main():

    # Command Options
    parser = argparse.ArgumentParser()

    # Model we are using is Qwen2.5-Coder-0.5B-Instruct, a very lightweight LLM
    parser.add_argument("--model", default="Qwen/Qwen2.5-Coder-0.5B-Instruct")
    parser.add_argument("--train-path", default="data/train.json")
    parser.add_argument("--val-path", default="data/test.json")
    parser.add_argument("--output-dir", default="qwen-assert-lora")
    parser.add_argument("--max-train-items", type=int, default=1000)
    parser.add_argument("--max-val-items", type=int, default=200)
    parser.add_argument("--max-asserts", type=int, default=5)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token


    # Create training dataset from train path
    train_dataset = build_dataset(
        args.train_path,
        tokenizer,
        args.max_train_items,
        args.max_asserts,
    )

    # Create validation dataset from validation path
    val_dataset = build_dataset(
        args.val_path,
        tokenizer,
        args.max_val_items,
        args.max_asserts,
    )

    # Tokenize datasets
    train_dataset = tokenize_dataset(train_dataset, tokenizer, args.max_length)
    val_dataset = tokenize_dataset(val_dataset, tokenizer, args.max_length)


    # Use quantization to improve memory usage
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        device_map="auto",
        quantization_config=quantization_config,
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)

    # Set up LoRA(Low rank adapation) config used for fine-tuning
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )

    # Confidugre LoRA with model
    model = get_peft_model(model, lora_config)


    # Set training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        logging_steps=20,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none",
        bf16=False,
        fp16=True,
    )

    # Train Model
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )

    trainer.train()

    # Save model and tokenizer
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
