import pandas as pd
from scipy.stats import kendalltau
import numpy as np

# Caricare il CSV con il separatore corretto
file_path = "/home/ubuntu/projects/LLMRSResearch/data/tune2/HALL_eval.csv"  # Cambia il percorso se necessario
df = pd.read_csv(file_path, delimiter=";")

# Convertire colonne numeriche da stringhe con virgola a float
cols_to_fix = ["STDDEV", "MEAN", "TOT_MEAN_OPT125", "TOT_MEAN_OPT350", "TOT_MEAN_OPT1.3", "TOT_MEAN_LL8", "TOT_MEAN_LL8INSTR"]
for col in cols_to_fix:
    if col in df.columns:
        df[col] = df[col].astype(str).str.replace(",", ".").astype(float, errors='ignore')

# Identificare le colonne delle valutazioni
val_cols = [col for col in df.columns if "ID" in col]

# Calcolare la coerenza inter-valutatori usando Kendall's W (approssimato come media di Kendall tau tra valutatori)
kendall_w_values = []
for i in range(len(val_cols)):
    for j in range(i + 1, len(val_cols)):
        tau, _ = kendalltau(df[val_cols[i]], df[val_cols[j]])
        kendall_w_values.append(tau)

# Media di Kendall's tau come proxy per Kendall's W
kendall_w = np.nanmean(kendall_w_values)

# Identificare le categorie con maggiore deviazione dalla media
df_sorted_by_std = df.sort_values(by="STDDEV", ascending=False)

# Stampare i risultati
print(f"Coerenza inter-valutatori (Kendall's W approssimato): {kendall_w:.2f}")
print("\nCategorie con maggiore deviazione dalla media:")
print(df_sorted_by_std[['MODEL', 'OUTPUT', 'STDDEV']].head(5))
