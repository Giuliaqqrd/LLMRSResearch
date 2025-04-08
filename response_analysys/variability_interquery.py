import json
from collections import defaultdict
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

data_file = "/home/ubuntu/projects/LLMRSResearch/llm_usage/llm_responses/opt125_responses.json"
# Carica il file JSON
with open(data_file, "r", encoding="utf-8") as f:
    data = json.load(f)

# Domande di interesse (in ordine crescente di dettaglio)
ordered_prompts = [
    "Describe a Sustainability Champion profile based on the following user values: committed",
    "Describe a Sustainability Champion profile based on the following user values: committed, visionary",
    "Describe a Sustainability Champion profile based on the following user values: committed, visionary, educated",
    "Describe a Sustainability Champion profile based on the following user values: committed, visionary, educated, proactive",
    "Describe a Sustainability Champion profile based on the following user values: committed, visionary, educated, proactive, ethical",
    "Describe a Sustainability Champion profile based on the following user values: committed, visionary, educated, proactive, ethical, influential"
]

# Raggruppa le risposte per ciascuna domanda
responses_by_prompt = defaultdict(list)
for entry in data:
    if entry["prompt"] in ordered_prompts:
        responses_by_prompt[entry["prompt"]].append(entry["response"])

# Modello per embedding
model = SentenceTransformer("all-MiniLM-L6-v2")

# Funzione per calcolare la similarità tra due gruppi di risposte
def group_similarity(responses1, responses2):
    if not responses1 or not responses2:
        return None  # Evita errori su gruppi vuoti

    # Ottieni embedding delle risposte
    embeddings1 = model.encode(responses1)
    embeddings2 = model.encode(responses2)

    # Calcola la similarità tra ogni coppia (risposta del gruppo 1 vs risposta del gruppo 2)
    sim_matrix = cosine_similarity(embeddings1, embeddings2)

    # Restituisci la media delle similarità tra gruppi
    return np.mean(sim_matrix)

# Calcola la variabilità tra gruppi successivi di risposte
variability_between_groups = {}
for i in range(len(ordered_prompts) - 1):
    prompt1 = ordered_prompts[i]
    prompt2 = ordered_prompts[i + 1]
    
    sim = group_similarity(responses_by_prompt[prompt1], responses_by_prompt[prompt2])
    if sim is not None:
        variability_between_groups[(prompt1, prompt2)] = 1 - sim

# Stampa i risultati
for (p1, p2), var in variability_between_groups.items():
    print(f"Variabilità tra risposte di:\n  '{p1}'\n  '{p2}'\n  → {var:.2f}\n")
