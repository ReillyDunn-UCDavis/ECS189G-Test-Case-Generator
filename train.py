from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import TrainingArguments, Trainer
from datasets import Dataset
import json
import sys
sys.stdout.reconfigure(line_buffering=True)

MODEL_NAME = "Salesforce/codet5-small"

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    use_fast=False
)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

with open("data/train.json", "r", encoding="utf-8") as f:
    data = json.load(f)

data = data[:10]

def extract_asserts(code):
    lines = []

    for line in code.split("\n"):
        line = line.strip()

        if line.startswith("assert candidate("):
            lines.append(line)

    return "\n".join(lines)

pairs = []

for item in data:
    pairs.append({
        "input": "Generate unit tests for this Python function:\n\n" + item["solution_code"],
        "output": extract_asserts(item["test_code"])
    })

dataset = Dataset.from_list(pairs)

def tokenize(example):
    model_inputs = tokenizer(
        example["input"],
        truncation=True,
        padding="max_length",
        max_length=512
    )

    labels = tokenizer(
        example["output"],
        truncation=True,
        padding="max_length",
        max_length=512
    )

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized = dataset.map(tokenize)

training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=2,
    num_train_epochs=1,
    logging_steps=1,
    logging_dir="./logs",
    report_to="none",
    disable_tqdm=False,
    log_level="info",
    save_steps=100,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized
)

trainer.train()

model.save_pretrained("./testgen_model")
tokenizer.save_pretrained("./testgen_model")