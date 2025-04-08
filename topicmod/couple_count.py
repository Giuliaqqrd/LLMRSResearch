import json

# Percorso del file
file_path = "/home/ubuntu/projects/LLMRSResearch/data/tune2/unique_train_data.json"

# Carica il file JSON
with open(file_path, "r") as f:
    data = json.load(f)

# Conta le coppie prompt-completion
count = len(data)

print(f"Numero di coppie prompt-completion: {count}")
