import os
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# CONFIGURAZIONE
base_model_dir = "/mnt/storage/huggingface/hub/models--meta-llama--Meta-Llama-3-8B"  # Cartella del modello base LLaMA 8B
lora_dir = "/mnt/storage/optsmall/meta-llama_Meta-Llama-3-8B"  # Cartella con gli adapter LoRA
output_file = "/home/ubuntu/projects/LLMRSResearch/llm_usage/llm_responses/llama8b_responses.json"
num_generations = 5  # Numero di risposte generate per prompt

# Verifica se è disponibile la GPU


# Caricamento del tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model_dir, padding_side="left")
tokenizer.pad_token = tokenizer.eos_token  # Imposta il token di padding

# Caricamento del modello base e degli adapter LoRA
base_model = AutoModelForCausalLM.from_pretrained(base_model_dir, torch_dtype=torch.bfloat16, device_map="auto")
model = PeftModel.from_pretrained(base_model, lora_dir)  # Applica LoRA

model.eval()  # Metti il modello in modalità valutazione

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
    inputs = tokenizer(prompt, return_tensors="pt", padding=True)
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
