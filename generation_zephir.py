from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch
import os
from transformers.trainer_utils import get_last_checkpoint

output_dir = "/mnt/storage/checkpoints"
save_directory = "/mnt/storage/huggingface/hub"

# Verifica se esiste un checkpoint
if os.path.isdir(output_dir):
    last_checkpoint = get_last_checkpoint(output_dir)
    if last_checkpoint is None:
        print("Nessun checkpoint trovato. Carico il modello da zero.")
        model_name = "HuggingFaceH4/zephyr-7b-beta"
    else:
        print(f"Carico il modello dal checkpoint: {last_checkpoint}")
        model_name = last_checkpoint
else:
    print("La directory di checkpoint non esiste. Carico il modello da zero.")
    model_name = "HuggingFaceH4/zephyr-7b-beta"


model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=save_directory, torch_dtype=torch.bfloat16, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=save_directory)


pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, torch_dtype=torch.bfloat16, device_map="auto")


messages = [
    {
        "role": "system",
        "content": "You are an expert recommendation system in the field of sustainability",
    },
    {"role": "user", "content": "Create a profile for a Sustainability Champions"}
]


prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


outputs = pipe(prompt, max_new_tokens=512, do_sample=True, temperature=1.2, top_k=100, top_p=0.50)


print(outputs[0]["generated_text"])
