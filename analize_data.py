import json

data_file = "data/tune2/unique_train_data.json"

with open(data_file, 'r', encoding='utf-8') as f:
    data = json.load(f)

unique_prompts = set()
unique_entries = []  # Lista per salvare i dizionari unici

prompt_count = 0

for entry in data:
    if "prompt" in entry:
        prompt_count += 1

print(f"Il numero di occorrenze di 'prompt' è: {prompt_count}")

# Iteriamo su ogni elemento della lista
# for entry in data:
#     if "prompt" in entry:
#         prompt_cleaned = entry["prompt"].strip()  # Rimuoviamo spazi superflui
#         if prompt_cleaned not in unique_prompts:
#             unique_prompts.add(prompt_cleaned)
#             unique_entries.append(entry)  # Aggiungiamo il dizionario originale senza duplicati

# Salviamo il risultato in un nuovo file JSON
# output_filename = "unique_train_data.json"
# with open(output_filename, "w", encoding="utf-8") as outfile:
#     json.dump(unique_entries, outfile, indent=4, ensure_ascii=False)

# print(f"File salvato come '{output_filename}' con {len(unique_entries)} prompt unici.")