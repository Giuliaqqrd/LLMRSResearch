import matplotlib.pyplot as plt
import numpy as np
import os

# Data for the 3 best runs
runs = [
    {"run_name": "OPT125M", "train_loss": 1.431285, "eval_loss": 2.086062},
    {"run_name": "OPT350M", "train_loss": 0.982018, "eval_loss": 1.76262},
    {"run_name": "OPT1.3B", "train_loss": 0.784829, "eval_loss": 1.643905}
]

# Extract run names, train_loss, and eval_loss values
run_names = [run["run_name"] for run in runs]
train_losses = [run["train_loss"] for run in runs]
eval_losses = [run["eval_loss"] for run in runs]

# Positions for the bars
x = np.arange(len(run_names))

# Bar width
width = 0.35

# Create the plot
fig, ax = plt.subplots(figsize=(10, 6))

# Add the bars
bar1 = ax.bar(x - width/2, train_losses, width, label='Train Loss', color='skyblue')
bar2 = ax.bar(x + width/2, eval_losses, width, label='Eval Loss', color='salmon')

# Labels and title
ax.set_xlabel('Run Name')
ax.set_ylabel('Loss')
ax.set_title('Comparison of the Top 3 Runs')
ax.set_xticks(x)
ax.set_xticklabels(run_names, rotation=45, ha="right")
ax.legend()

# Save the plot in the specified directory
output_dir = "/home/ubuntu/projects/LLMRSResearch/OPT/plot"
os.makedirs(output_dir, exist_ok=True)  # Create the folder if it doesn't exist
file_path = os.path.join(output_dir, "comparison_plot_best_runs.png")
plt.savefig(file_path)

# Show the plot
plt.show()
