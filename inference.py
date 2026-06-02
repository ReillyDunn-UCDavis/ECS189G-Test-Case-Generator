import json
import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from finetune_qwen_asserts import (
    SYSTEM_PROMPT,
    clean_text,
    extract_function_signature)

    

MODEL_NAME = "Qwen/Qwen2.5-Coder-0.5B-Instruct"
TEST_PATH = "data/test.json"
MAX_ASSERTS = 10




def generate_qwen_prompt(example):
    example_test_cases = '\n'.join([
        line.strip()
        for line in example["test_code"].splitlines()
        if line.strip().startswith("assert")
    ][:MAX_ASSERTS])
    signature = extract_function_signature(example["solution_code"])
    return (
        "Read the problem description and generate one valid Python assert statement "
        "for testing the candidate function.\n\n"
        "Problem description:\n"
        f"{clean_text(example['problem_description'])}\n\n"
        "Function signature:\n"
        f"{signature}\n\n"
    )


def extract_assert_line(text):
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("assert"):
            return line
    return "assert candidate(" + text.strip()


with open(TEST_PATH, "r", encoding="utf-8") as file:
    test_items = json.load(file)

target_item = test_items[0]
prompt = generate_qwen_prompt(target_item)

print(prompt)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16,
)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    trust_remote_code=True,
    quantization_config=quantization_config,
)

messages = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": prompt},
]
chat_text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)
inputs = tokenizer(chat_text, return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=160,
        temperature = 0.3,
        top_p = 0.9,
        do_sample=True,
    )

response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
print("===========Model Response==========")
print(response)
