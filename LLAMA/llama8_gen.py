import os
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# CONFIGURAZIONE
model_dir = "/mnt/storage/optsmall/meta-llama_Meta-Llama-3-8B"  # Cartella del modello LLaMA 8B finetunato con LoRA
output_file = "/home/ubuntu/projects/LLMRSResearch/llm_usage/llm_responses/llama8b_responses.json"  # File di output
num_generations = 5  # Numero di risposte generate per ogni prompt

# Caricamento del tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_dir)

# Caricamento del modello base LLaMA e successiva applicazione del fine-tuning LoRA
model = AutoModelForCausalLM.from_pretrained(model_dir)
model = PeftModel.from_pretrained(model, model_dir)  # Carica LoRA

# Impostazione del dispositivo
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Definire il token di padding
pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

# Prompt di input con il ruolo dell'assistente
prompts = [
    "You are a helpful assistant. Answer the following question concisely and do not repeat the question: What is an energy community?",
    "You are a knowledgeable assistant. Describe a Sustainability Champion profile, focusing on key qualities such as vision, commitment, and ethics.",
    "You are an expert on sustainability. Describe a Sustainability Champion profile based on the following user values: committed, visionary, educated, proactive, ethical, influential.",
    "You are an expert on sustainability. Describe a Sustainability Champion profile based on the following user values: committed, visionary, educated, proactive, ethical.",
    "You are a sustainability advisor. Describe a Sustainability Champion profile based on the following user values: committed, visionary, educated, proactive.",
    "You are an expert in sustainability. Describe a Sustainability Champion profile based on the following user values: committed, visionary, educated.",
    "You are a sustainability advisor. Describe a Sustainability Champion profile based on the following user values: committed, visionary.",
    "You are an expert in sustainability. Describe a Sustainability Champion profile based on the following user values: committed.",
]

# Generazione delle risposte
response_data = []
for prompt in prompts:
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    # Generazione delle risposte
    outputs = model.generate(
        **inputs,
        max_length=256,
        temperature=0.7,
        top_k=50,
        top_p=0.8,
        num_return_sequences=num_generations,
        do_sample=True,
        pad_token_id=pad_token_id  # Impedisce che il modello ripeta il prompt
    )

    # Salvataggio delle risposte
    for i, output in enumerate(outputs):
        response_text = tokenizer.decode(output, skip_special_tokens=True)
        
        # Evitare che la risposta contenga il prompt
        response_text = response_text[len(prompt):].strip()
        
        response_data.append({"prompt": prompt, "response": response_text})
        print(f"Prompt: {prompt}\nRisposta {i+1}: {response_text}\n")

# Salvataggio delle risposte in un file JSON
with open(output_file, "w") as f:
    json.dump(response_data, f, indent=4)

print(f"Risposte salvate in {output_file}")
