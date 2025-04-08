import os
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# CONFIGURAZIONE
model_dir = "/mnt/storage/optsmall/facebook_opt-1.3b"  # Cartella del modello salvato
output_file = "/home/ubuntu/projects/LLMRSResearch/llm_usage/llm_responses/opt1.3_eval.json"  # File di output
num_generations = 5  # Numero di risposte generate per ogni prompt

# Caricamento del modello e del tokenizer
device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForCausalLM.from_pretrained(model_dir).to(device)

# Prompt di input
prompts = [
    "Describe Sustainability Champion profile and his values",
    "Describe a Financially-minded enterpreneur and his values",
    "Describe a Socially-conscious advocate consumer and his values",
    "Describe a Skeptical and conservative energy consumer and his values",
    "What are the differences between Sustainability Champion and Skeptical and conservative energy consumer?",
    "What are the differences between Skeptical and conservative energy consumer and Financially-minded enterpreneur?",
    "What are the differences between Sustainability Champion and Socially-conscious advocate consumer?",
]

# Generazione delle risposte
response_data = []
for prompt in prompts:
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(
        **inputs,
        max_length=200,
        temperature=0.7,
        top_k=50,
        top_p=0.9,
        num_return_sequences=num_generations,
        do_sample=True      
    )
    
    for i, output in enumerate(outputs):
        response_text = tokenizer.decode(output, skip_special_tokens=True)
        response_data.append({"prompt": prompt, "response": response_text})
        print(f"Prompt: {prompt}\nRisposta {i+1}: {response_text}\n")

# Salvataggio delle risposte
with open(output_file, "w") as f:
    json.dump(response_data, f, indent=4)

print(f"Risposte salvate in {output_file}")
