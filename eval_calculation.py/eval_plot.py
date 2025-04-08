import pandas as pd
import matplotlib.pyplot as plt
import os

# Caricare il CSV con il separatore corretto
file_path = "/home/ubuntu/projects/LLMRSResearch/data/tune2/LIK_eval.csv"  # Cambia il percorso se necessario
df = pd.read_csv(file_path, delimiter=";")

# Identify evaluator columns
val_cols = [col for col in df.columns if "ID" in col]


# Identify and merge Q1 and Q2 into a single category per evaluator
category_cols = {}
for col in val_cols:
    if "Q1" in col or "Q2" in col:
        base_col = col.replace("Q1", "Category_1").replace("Q2", "Category_1")
        if base_col not in category_cols:
            category_cols[base_col] = []
        category_cols[base_col].append(col)

# Compute the mean for each category
for category, cols in category_cols.items():
    df[category] = df[cols].mean(axis=1)


# Use only category-based columns for plotting
category_val_cols = list(category_cols.keys())

# Convert responses to numeric values
for col in category_val_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Create the destination folder if it does not exist
save_path = "/home/ubuntu/projects/LLMRSResearch/OPT/plot"
os.makedirs(save_path, exist_ok=True)

# Identify models and sort them correctly
models = sorted(df['MODEL'].unique(), reverse=True)


# Plot the average rating trend for each evaluator per model
plt.figure(figsize=(12, 6))
for index, col in enumerate(category_val_cols):
    means = df.groupby('MODEL')[col].mean().reindex(models)
    col = "Evaluator_"+str((index+1))
    plt.plot(models, means, marker='o', linestyle='-', label=col)

plt.title("Human-Likeness: Average Ratings per Model")
plt.xlabel("Model Name")
plt.ylabel("Average Rating")
plt.legend()
plt.grid(True)
plt.xticks(rotation=0)

# Save the plot
plot_file = os.path.join(save_path, "LIKresponse_trend_average_grouped.png")
plt.savefig(plot_file)
plt.close()

print(f"Plot saved in: {plot_file}")
