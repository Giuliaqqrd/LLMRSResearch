import json
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Funzione per Mean Pooling (considerando l'attenzione)
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]  # Primo elemento contiene gli embeddings dei token
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

# Carica il file JSON
data_file = "/home/ubuntu/projects/LLMRSResearch/llm_usage/llm_responses/opt350_onlyprofile.json"
with open(data_file, "r", encoding="utf-8") as f:
    data = json.load(f)

# Estrai tutte le risposte
all_responses = [entry["response"] for entry in data]

# Carica il modello e il tokenizer da Hugging Face
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

# Tokenizza tutte le risposte
encoded_input = tokenizer(all_responses, padding=True, truncation=True, return_tensors="pt")

# Disabilita il calcolo dei gradienti (per efficienza)
with torch.no_grad():
    model_output = model(**encoded_input)

# Ottieni gli embedding medi delle frasi
sentence_embeddings = mean_pooling(model_output, encoded_input["attention_mask"])

# Normalizza gli embedding per avere vettori unitari
sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)

# Converti i tensori PyTorch in array NumPy
embeddings_np = sentence_embeddings.cpu().numpy()

# Calcola la matrice di similarità coseno tra tutti gli embedding
similarity_matrix = cosine_similarity(embeddings_np, embeddings_np)

# Prendi solo i valori della parte superiore della matrice (senza diagonale)
num_responses = len(all_responses)
similarity_values = []
for i in range(num_responses):
    for j in range(i + 1, num_responses):
        similarity_values.append(similarity_matrix[i, j])

# Calcola la variabilità globale (1 - similarità media)
global_variability = 1 - np.mean(similarity_values)

# Stampa il risultato
print(f"Variabilità globale tra tutte le risposte: {global_variability:.2f}")

# **Parte 2: Visualizza un grafico della distribuzione della similarità**

# Crea un istogramma della distribuzione delle similarità
plt.figure(figsize=(10, 6))
sns.histplot(similarity_values, kde=True, color="blue", bins=20)
plt.title("Distribuzione della similarità coseno tra le risposte")
plt.xlabel("Similarità coseno")
plt.ylabel("Frequenza")
plt.grid(True)

# Salva il grafico nel percorso specificato
plot_path = "/home/ubuntu/projects/LLMRSResearch/OPT/plot/similarity_distribution.png"
plt.savefig(plot_path)

# Mostra il grafico (opzionale)
plt.show()

print(f"Grafico salvato come {plot_path}")
