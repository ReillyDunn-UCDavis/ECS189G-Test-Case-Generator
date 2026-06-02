import json
import re

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from finetune_qwen_asserts import (
    SYSTEM_PROMPT,
    build_user_prompt,
)
BASE_MODEL = "Qwen/Qwen2.5-Coder-0.5B-Instruct"
ADAPTER_PATH = "qwen-assert-lora"
TEST_PATH = "data/test.json"






with open(TEST_PATH, "r", encoding="utf-8") as file:
    test_items = json.load(file)


tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    trust_remote_code=True,
    device_map="auto",
)
model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
model.eval()

def get_response(description, signature,n):

    
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": 
        "Problem description:\n"
        f"{description}\n\n"
        "Function signature:\n"
        f"{signature}\n\n"
        f"Write exactly {n} assert statements"
        },
        {"role": "assistant", "content": "assert "},
    ]
    print(messages)
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False,
        )
 
    arr= tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    )

    return arr

