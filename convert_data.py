import json

data_file = "data/tune2/unique_train_data.json"
def convert_to_qwen_format(input_file, output_file):
    """
    Converte un file JSON con prompt e completion nel formato JSONL compatibile con Qwen.
    
    Args:
        input_file (str): Nome del file di input (JSON).
        output_file (str): Nome del file di output (JSONL).
    """
    # Leggi i dati di input
    with open(data_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Lista per salvare i dati convertiti
    formatted_data = []
    
    # Conversione
    for item in data:
        messages = [
            {
                "role": "user",
                "content": item["prompt"].strip()
            },
            {
                "role": "assistant",
                "content": item["completion"].strip()
            }
        ]
        formatted_data.append({"messages": messages})
    
    # Scrivi i dati in formato JSONL
    with open(output_file, 'w', encoding='utf-8') as f:
        for entry in formatted_data:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')

# Esempio di utilizzo
input_file = 'data/unique_train_data.json'  # Nome del file con i dati originali
output_file = 'data/tune2/tuning_dataset_final.jsonl'  # Nome del file di output
convert_to_qwen_format(input_file, output_file)

print(f"Conversione completata! File salvato come {output_file}.")

