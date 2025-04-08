import transformers
import torch

save_directory = "/mnt/storage/huggingface/hub"
model_id = "meta-llama/Meta-Llama-3-8B"

pipeline = transformers.pipeline(
    "text-generation", 
    model=model_id, 
    model_kwargs={"torch_dtype": torch.bfloat16}, 
    device_map="auto"
)

# Generazione con max_new_tokens
output = pipeline(
    "Who is a Sustainability Champion?", 
    max_new_tokens=150  # Imposta il numero massimo di token generati
)

print(output)
