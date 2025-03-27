import torch
from transformers import pipeline

pipe = pipeline(
    "text-generation",
    model="google/gemma-2-2b-IT",
    device="mps",  # replace with "mps" to run on a Mac device
)

text = "Tell me a joke"
outputs = pipe(text, max_new_tokens=80)
response = outputs[0]["generated_text"]
print(response)
