# Load necessary libraries
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
import torch
import numpy as np
from collections import Counter


# Specify the instruction-tuned Gemma model (better for alignment/safety prompts)
model_id = "google/gemma-2b-it"


# Load tokenizer and model from Hugging Face
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype="auto", device_map="auto")
model.eval()  # Set model to inference mode (no dropout etc.)

# Create a generation pipeline using the same model and tokenizer
pipe = pipeline("text-generation",
                model=model, 
                tokenizer=tokenizer, 
                device_map="auto")



'''
# List of prompts designed to activate safety-related neurons
safety_prompts = [
    "It's important to treat others with respect and avoid harmful behavior.",
    "Violence is never an appropriate solution to a disagreement.",
    "Online platforms should prioritize user safety and well-being.",
    "Personal information should never be shared without consent.",
    "Spreading misinformation can lead to real-world harm.",
    "Cyberbullying can have serious consequences and should be addressed.",
    "Encouraging self-harm is dangerous and must be prevented.",
    "Hate speech contributes to unsafe environments and should be avoided.",
    "Strong passwords are essential for protecting online accounts.",
    "People deserve access to accurate and trustworthy information.",
    "Emergency services should always be contacted in life-threatening situations.",
    "It's crucial to verify sources before sharing information online.",
    "Everyone has a right to feel safe, both physically and emotionally.",
    "Content promoting violence or abuse should be reported immediately.",
    "Educating children about internet safety is a shared responsibility.",
    "Respecting boundaries is key to building a safe and inclusive society.",
    "Social media companies have a responsibility to monitor harmful content.",
    "Mental health support should be accessible in times of crisis.",
    "Encouraging empathy and compassion helps create safer communities.",
    "Platforms must have strict policies to handle harassment and threats."
]
'''
safety_prompts = [
    "It's important to treat others with respect and avoid harmful behavior.",
    "Violence is never an appropriate solution to a disagreement.",
    "Online platforms should prioritize user safety and well-being.",
    "Personal information should never be shared without consent.",
    "Spreading misinformation can lead to real-world harm.",
]



# Set up a global dict to store layer-wise neuron activations
activations = {}


# Hook generator: returns a function that captures activations from a specific layer
def get_hook(layer_name):
    def hook_fn(module, input, output):
        # Save the activation tensor to the dictionary
        activations[layer_name] = output.detach().cpu()  # Use .cpu() to avoid memory issues on MPS/GPU
    return hook_fn


# This function runs a single prompt and records top-K activated neurons
def get_top_neurons(prompt, top_k=5):
    # Tokenize and send to the model device (CPU/GPU/MPS)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # Forward pass through the model (activations will be captured via hooks)
    _ = model(**inputs)

    top_neurons = []

    # For each layer's captured activations
    for name, act in activations.items():
        # Compute mean activation across tokens in the sequence
        neuron_vals = act.mean(dim=1).squeeze()  # Shape: (hidden_dim,)
        # Get top-k most activated neurons
        topk = torch.topk(neuron_vals, top_k)
        for i in range(top_k):
            top_neurons.append((name, topk.indices[i].item(), topk.values[i].item()))
    
    return top_neurons


# Attach hooks to all transformer MLP layers in the Gemma model
hooks = []
for i, layer in enumerate(model.model.layers):
    # Register forward hook on the MLP block of each layer
    hook = layer.mlp.register_forward_hook(get_hook(f"layer_{i}"))
    hooks.append(hook)


# Run safety prompts through the model and collect top neurons
all_top_neurons = []
i = 0

for prompt in safety_prompts:
    print(f"Analyzing prompt: {prompt}")
    top_neurons = get_top_neurons(prompt)
    all_top_neurons.extend(top_neurons)
    
    text = safety_prompts[i]
    outputs = pipe(text, max_new_tokens=80)
    response = outputs[0]["generated_text"]
    print(response)
    i+=1


# Count how often each (layer, neuron_id) pair appeared in top-K
neuron_counts = Counter([(layer, neuron_id) for layer, neuron_id, _ in all_top_neurons])
top_safety_neurons = neuron_counts.most_common(10)  # Change this to 20, 50, etc. for more


# Print most frequently activated neurons across all safety prompts
print("Top safety neurons:")
for (layer, neuron_id), count in top_safety_neurons:
    print(f"Layer {layer}, Neuron {neuron_id} — triggered {count} times")
