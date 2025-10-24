import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
try:
    df = pd.read_csv('trails/numerical_analysis_results.csv')
except FileNotFoundError:
    print("Error: 'numerical_analysis_results.csv' not found. Make sure the file is in the same directory as the script.")
    exit()

# Data Cleaning
df.columns = df.columns.str.strip()
df = df.rename(columns={'Trail #1': 'Trail_Num'})

# Clean and convert 'L-Shifted Delta' to numeric
df['L-Shifted Delta'] = df['L-Shifted Delta'].str.rstrip('%').astype(float)

# Clean and convert 'Balance' and 'Limit' to numeric
for col in ['Balance', 'Limit']:
    if df[col].dtype == 'object':
        df[col] = df[col].str.replace('"', '').str.replace(',', '').astype(float)

# Separate analyzers into two groups
all_analyzers = df['Analyzer Name'].unique()
account_n_analyzers = sorted([name for name in all_analyzers if name.startswith('Account n=')])
other_analyzers = sorted([name for name in all_analyzers if not name.startswith('Account n=')])

# Create two subplots, one for each group
fig, axes = plt.subplots(2, 1, figsize=(12, 14), sharex=True)

# Plot for non-'Account n=' analyzers
if other_analyzers:
    ax1 = axes[0]
    other_df = df[df['Analyzer Name'].isin(other_analyzers)]
    
    # Reverse order for plot layering and legend
    other_analyzer_order = sorted(other_df['Analyzer Name'].unique())

    sns.scatterplot(
        data=other_df,
        x='Trail_Num',
        y='L-Shifted Delta',
        hue='Analyzer Name',
        hue_order=other_analyzer_order,
        palette='viridis',
        ax=ax1,
        s=100
    )
    ax1.set_title('L-Shifted Delta for Other Analyzers', fontsize=14)
    ax1.set_ylabel('L-Shifted Delta (%)', fontsize=10)
    ax1.grid(True)
    ax1.legend(title='Analyzer Name', bbox_to_anchor=(1.05, 1), loc='upper left')

# Plot for 'Account n=' analyzers
if account_n_analyzers:
    ax2 = axes[1]
    account_n_df = df[df['Analyzer Name'].isin(account_n_analyzers)]

    # Reverse order for plot layering and legend
    account_n_analyzer_order = sorted(account_n_df['Analyzer Name'].unique())

    sns.scatterplot(
        data=account_n_df,
        x='Trail_Num',
        y='L-Shifted Delta',
        hue='Analyzer Name',
        hue_order=account_n_analyzer_order,
        palette='viridis',
        ax=ax2,
        s=100
    )
    ax2.set_title('L-Shifted Delta for "Account n=" 0-balance Analyzers', fontsize=14)
    ax2.set_xlabel('Trail Number', fontsize=12)
    ax2.set_ylabel('L-Shifted Delta (%)', fontsize=10)
    ax2.grid(True)
    ax2.legend(title='Analyzer Name', bbox_to_anchor=(1.05, 1), loc='upper left')

# Final layout adjustments
plt.tight_layout(rect=[0, 0, 0.9, 1]) # Adjust layout to make space for legend

# Save the plot
plt.savefig('l_shifted_delta_distribution_grouped.png')

# Show the plot
plt.show()

print("Chart saved as 'l_shifted_delta_distribution_grouped.png'")