import json
from collections import defaultdict
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

data_file = "/home/ubuntu/projects/LLMRSResearch/llm_usage/llm_responses/opt125_responses.json"
# Carica il file JSON
with open(data_file, "r", encoding="utf-8") as f:
    data = json.load(f)

# Raggruppa le risposte per domanda
responses_by_prompt = defaultdict(list)
for entry in data:
    responses_by_prompt[entry["prompt"]].append(entry["response"])

# Modello per embedding
model = SentenceTransformer("all-MiniLM-L6-v2")

# Analizza la variabilità per ogni domanda
variability_scores = {}

for prompt, responses in responses_by_prompt.items():
    if len(responses) < 2:
        variability_scores[prompt] = None  # Non ha senso calcolare la variabilità con una sola risposta
        continue

    # Calcolo embedding
    embeddings = model.encode(responses)

    # Calcolo similarità coseno
    sim_matrix = cosine_similarity(embeddings)

    # Media delle similarità escluse le diagonali (auto-similarità)
    mean_similarity = np.mean(sim_matrix[np.triu_indices(len(responses), k=1)])

    # Variabilità = 1 - similarità media (più è alto, più sono variabili le risposte)
    variability_scores[prompt] = 1 - mean_similarity

# Stampa i risultati
for prompt, score in variability_scores.items():
    print(f"Domanda: {prompt}\nVariabilità: {score:.2f}\n")
