import nlpaug.augmenter.word as naw
import nltk
import json
from tqdm import tqdm  # Importa tqdm per la barra di avanzamento


input_file = "/home/ubuntu/projects/LLMRSResearch/data/tune2/unique_train_data.json"   
output_file = "/home/ubuntu/projects/LLMRSResearch/nlpaug/dataset_augmented_2.json"


aug = naw.SynonymAug(aug_src='wordnet', model_path=None, name='Synonym_Aug', aug_min=6, aug_max=15, aug_p=0.3, lang='eng', 
                     stopwords=None, tokenizer=None, reverse_tokenizer=None, stopwords_regex=None, force_reload=False, 
                     verbose=1)

# Funzione per leggere il file JSON in modo sicuro
def load_json_file(file_path):
    try:
        with open(file_path, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Errore: Il file {file_path} non esiste.")
        return []
    except json.JSONDecodeError:
        print(f"Errore: Il file {file_path} non è un JSON valido.")
        return []

# Funzione per salvare il dataset aumentato in formato JSON
def save_json_file(data, file_path):
    try:
        with open(file_path, "w") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)  # Usa indent per una formattazione più leggibile
        print(f"Dataset aumentato salvato in {file_path}")
    except Exception as e:
        print(f"Errore durante il salvataggio del file: {e}")

# Leggi il file JSON
data = load_json_file(input_file)

if not data:
    print("Nessun dato trovato nel file JSON.")
else:

    # Creazione del dataset aumentato
    augmented_dataset = []
    for item in tqdm(data, desc="Elaborazione dataset", unit="elemento"):
        couple = {}
        couple["prompt"] = " ".join(aug.augment(nltk.sent_tokenize(item["prompt"])))
        couple["completion"] = " ".join(aug.augment(nltk.sent_tokenize(item["completion"])))
        augmented_dataset.append(couple)

    # Stampa il dataset aumentato (se non è troppo grande)
    print(f"\nDataset aumentato creato")  # Mostra solo il primo elemento per esempio

    # Salva il dataset aumentato
    save_json_file(augmented_dataset, output_file)