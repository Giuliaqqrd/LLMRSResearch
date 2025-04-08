import json
import os
import pandas as pd

# Cartella principale che contiene le cartelle delle run (modifica con il percorso corretto)
root_dir = "/home/ubuntu/projects/LLMRSResearch/LLAMA/llamainstr"

# Lista per salvare i dati delle run
train_runtimes = []

# Scansiona tutte le cartelle dentro root_dir
for run_folder in os.listdir(root_dir):
    run_path = os.path.join(root_dir, run_folder)
    if os.path.isdir(run_path):  # Controlla che sia una cartella
        print(f"Trovata cartella: {run_folder}")  # Debug
        log_file = os.path.join(run_path, "training_log.json")  # Supponiamo si chiami così
        if os.path.exists(log_file):  # Verifica che il file esista
            print(f"Trovato file: {log_file}")  # Debug
            with open(log_file, "r") as f:
                data = json.load(f)
                if "train_log" in data and isinstance(data["train_log"], list):
                    print(f"Lunghezza train_log: {len(data['train_log'])}")  # Debug
                    for entry in data["train_log"]:
                        if "train_runtime" in entry:
                            train_runtimes.append({
                                "run_name": data.get("run_name", run_folder),
                                "train_runtime": entry["train_runtime"]
                            })
                            break  # Prende il primo che trova

# Converti i risultati in DataFrame
df = pd.DataFrame(train_runtimes)

# Stampa i primi risultati
print(df.head(10))

# Calcola la media del tempo di addestramento
if not df.empty:
    mean_train_runtime = df["train_runtime"].mean()
    print(f"\nTempo medio di addestramento: {mean_train_runtime:.2f} secondi")
else:
    print("\nNessun dato trovato. Controlla la struttura dei file JSON e le cartelle.")

# Salva i risultati su CSV per analisi più comoda
df.to_csv("/home/ubuntu/projects/LLMRSResearch/OPT/results/train_runtime_resultsllama8instr.csv", index=False)
