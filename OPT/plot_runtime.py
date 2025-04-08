import matplotlib.pyplot as plt
import os
import matplotlib.pyplot as plt

# Modelli e tempi medi di addestramento
models = ["OPT 125M", "OPT 350M", "OPT 1.3B", "LLaMA3 8B"]
training_times = [122.37, 238.14, 1048.20, 3833.65]

# Creazione del grafico
plt.figure(figsize=(8, 5))
plt.plot(models, training_times, marker='o', linestyle='-', color='b', label="Training Time (s)")

# Aggiunta di etichette
plt.xlabel("Model")
plt.ylabel("Training Time (seconds)")
plt.title("Average Training Time per Run")
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend()

# Mostra il grafico
output_dir = "/home/ubuntu/projects/LLMRSResearch/OPT/plot"
os.makedirs(output_dir, exist_ok=True)  # Crea la cartella se non esiste
file_path = os.path.join(output_dir, "comparison_trainruntime.png")
plt.savefig(file_path)

# Visualizzazione del grafico
plt.show()
