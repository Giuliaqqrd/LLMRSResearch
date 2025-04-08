import json
import os
import pandas as pd

# Cartella principale che contiene le cartelle delle run (modifica con il percorso corretto)
root_dir = "/home/ubuntu/projects/LLMRSResearch/LLAMA/llamainstr"

# Lista per salvare i dati delle run
results = []

# Scansiona tutte le cartelle dentro root_dir
for run_folder in os.listdir(root_dir):
    run_path = os.path.join(root_dir, run_folder)
    if os.path.isdir(run_path):  # Controlla che sia una cartella
        log_file = os.path.join(run_path, "training_log.json")  # Supponiamo si chiami così
        if os.path.exists(log_file):  # Verifica che il file esista
            with open(log_file, "r") as f:
                data = json.load(f)
                # Estrai metriche principali
                results.append({
                    "run_name": data["run_name"],
                    "learning_rate": data["learning_rate"],
                    "scheduler": data["lr_scheduler"],
                    #"weight_decay": data["weight_decay"],
                    #"epochs": data["num_train_epochs"],
                    "train_loss": data["train_log"][-2]["train_loss"],  # Ultima train loss registrata
                    "eval_loss": data["train_log"][-1]["eval_loss"],  # Ultima eval loss registrata
                    "perplexity": data["perplexity"]
                })

# Converti i risultati in DataFrame
df = pd.DataFrame(results)

print(df.head(10))

# Ordina per eval_loss crescente (e poi perplexity per disambiguare)
best_run = df.sort_values(["eval_loss", "perplexity"]).iloc[0]
worst_run = df.sort_values(["eval_loss", "perplexity"], ascending=False).iloc[0]

# Stampa i risultati
print("\n📈 Migliore run:")
print(best_run)

print("\n📉 Peggiore run:")
print(worst_run)

# Salva i risultati su CSV per analisi più comoda
df.to_csv("/home/ubuntu/projects/LLMRSResearch/OPT/results/finetuningllama8inst_results.csv", index=False)
