from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import json
import sys
sys.stdout.reconfigure(line_buffering=True)

model_path = "./testgen_model"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

code = """
Generate unit tests for this Python function:

def add(a,b):
    return a+b
"""

inputs = tokenizer(code, return_tensors="pt")

outputs = model.generate(
    **inputs,
    max_length=128,
    num_beams=4,
    early_stopping=True,
    no_repeat_ngram_size=2
)

decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)

result = {
    "input": code,
    "output": decoded
}

with open("outputs/inference_result.json", "w", encoding="utf-8") as f:
    json.dump(result, f, indent=2)

print(decoded, flush=True)