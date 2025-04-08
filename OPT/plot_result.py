import json
import os
import matplotlib.pyplot as plt
import seaborn as sns

# Dati della migliore run
best_run = {
    "name": "Best Run (run_7)",
    "train_loss": 1.956402,
    "eval_loss": 1.712478,
    "perplexity": 5.542681
}

# Dati della peggiore run
worst_run = {
    "name": "Worst Run (run_18)",
    "train_loss": 3.021542,
    "eval_loss": 2.560054,
    "perplexity": 12.936517
}

# 📊 1. Bar chart per train_loss, eval_loss e perplexity
fig, ax = plt.subplots(figsize=(8, 5))
metrics = ["Train Loss", "Eval Loss", "Perplexity"]
best_values = [best_run["train_loss"], best_run["eval_loss"], best_run["perplexity"]]
worst_values = [worst_run["train_loss"], worst_run["eval_loss"], worst_run["perplexity"]]

x = range(len(metrics))
width = 0.4

ax.bar([p - width/2 for p in x], best_values, width, label=best_run["name"], color="royalblue")
ax.bar([p + width/2 for p in x], worst_values, width, label=worst_run["name"], color="red")

ax.set_xticks(x)
ax.set_xticklabels(metrics)
ax.set_ylabel("Value")
ax.set_title("LLaMA3 8B-Instruct - PET - Comparison of Train Loss, Eval Loss, and Perplexity")

# Creazione della legenda con i valori numerici
legend_labels = [
    f"{best_run['name']} (Train Loss: {best_run['train_loss']:.3f}, Eval Loss: {best_run['eval_loss']:.3f}, Perplexity: {best_run['perplexity']:.2f})",
    f"{worst_run['name']} (Train Loss: {worst_run['train_loss']:.3f}, Eval Loss: {worst_run['eval_loss']:.3f}, Perplexity: {worst_run['perplexity']:.2f})"
]
ax.legend(legend_labels)

# Salvataggio del grafico nella cartella specificata
output_dir = "/home/ubuntu/projects/LLMRSResearch/OPT/plot"
os.makedirs(output_dir, exist_ok=True)  # Crea la cartella se non esiste
file_path = os.path.join(output_dir, "comparison_plotllama8instr.png")
plt.savefig(file_path)

# Visualizzazione del grafico
plt.show()

