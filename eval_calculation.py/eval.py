import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the CSV into a DataFrame
df = pd.read_csv('/home/ubuntu/projects/LLMRSResearch/data/tune2/eval_red.csv')

# Assuming each column (Hallucinations, Domain, HL) has summed scores from 4 experts
# Calculate the mean for each model and category by dividing the sum by the number of experts (4) -> 8 because we have 2 questions
mean_per_model = df.groupby('MODEL')[['Hallucinations', 'Domain', 'HL']].sum() / 5  # 

# Trova il min e max per ciascuna colonna
min_vals = mean_per_model.min()
max_vals = mean_per_model.max()

# Normalizza i valori tra 1 e 5
mean_per_model = 1 + (mean_per_model - min_vals) * 4 / (max_vals - min_vals)


# Calculate the overall mean for each model (average across categories)
overall_mean_per_model = mean_per_model.mean(axis=1)

# Add the overall mean to the DataFrame
mean_per_model['Overall Mean'] = overall_mean_per_model

# Set the plot size
plt.figure(figsize=(10, 6))

# Plot the means by category (Hallucinations, Domain, HL)
mean_per_model[['Hallucinations', 'Domain', 'HL']].plot(kind='bar', figsize=(10, 6))


# Add titles and labels
plt.title('Means by Category for Each Model')
plt.xlabel('Model')
plt.ylabel('Mean')

# Adjust the legend title
plt.legend(title="Categories", labels=['Hallucinations', 'Domain', 'HL'], bbox_to_anchor=(1.05, 1), loc='upper left')

# Save the plot to the specified directory
plt.tight_layout()
plt.savefig('/home/ubuntu/projects/LLMRSResearch/OPT/plot/means_by_category.png')

# Display the plot
plt.show()

# Plot the overall mean for each model (averaged over 4 experts)
plt.figure(figsize=(8, 5))
overall_mean_per_model.plot(kind='bar', color='skyblue')

# Add custom labels to the x and y axes
plt.title('Overall Mean for Each Model')
plt.xlabel('Model')
plt.ylabel('Overall Mean')

# Save the plot to the specified directory
plt.tight_layout()
plt.savefig('/home/ubuntu/projects/LLMRSResearch/OPT/plot/overall_mean.png')

# Display the plot
plt.show()
