import json
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np

# Funzione per Mean Pooling (considerando l'attenzione)
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]  # Primo elemento contiene gli embeddings dei token
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

data_file = "/home/ubuntu/projects/LLMRSResearch/llm_usage/llm_responses/opt350_onlyprofile.json"
# Carica il file JSON
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

# Determina il valore massimo possibile per n_components in PCA
max_components = min(embeddings_np.shape[0], embeddings_np.shape[1])  # min(n_samples, n_features)

# PCA con valore massimo possibile
pca = PCA(n_components=max_components)
embeddings_pca = pca.fit_transform(embeddings_np)

# Ora applichiamo t-SNE per ridurre la dimensionalità a 2D per il grafico
tsne = TSNE(n_components=2, random_state=42)
embeddings_tsne = tsne.fit_transform(embeddings_pca)

# Eseguiamo KMeans per il clustering
num_clusters = 7  # Puoi scegliere il numero di cluster che ti aspetti (6 in questo caso)
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
clusters = kmeans.fit_predict(embeddings_tsne)

# Creazione del grafico
plt.figure(figsize=(10, 8))

# Assegna colori diversi per ogni cluster
for cluster in range(num_clusters):
    cluster_indices = np.where(clusters == cluster)[0]
    plt.scatter(embeddings_tsne[cluster_indices, 0], embeddings_tsne[cluster_indices, 1], label=f'Cluster {cluster+1}')

# Aggiungi etichette per le risposte
for i, response in enumerate(all_responses):
    plt.annotate(f'{i+1}', (embeddings_tsne[i, 0], embeddings_tsne[i, 1]), fontsize=8)

plt.title("Distribuzione delle risposte nel piano 2D con Clustering (t-SNE + PCA + KMeans)")
plt.xlabel("Dimensione 1")
plt.ylabel("Dimensione 2")

# Aggiungi legenda
plt.legend()

# Salva il grafico
plot_path = "/home/ubuntu/projects/LLMRSResearch/OPT/plot/responses_tsne_cluster_plot.png"
plt.savefig(plot_path)

# Mostra il grafico
plt.show()

print(f"Grafico salvato in: {plot_path}")
