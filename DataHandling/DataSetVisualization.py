# Personal Finance Dataset Visualization
# This script visualizes the personal finance dataset using various plots.
# It includes univariate, bivariate, multivariate and time series analyses.


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv("DataHandling/cleaned_full_personal_finance.csv")

# Parse datetime
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
df['Time'] = pd.to_datetime(df['Time'], format='%H:%M:%S', errors='coerce').dt.time

# Seaborn style
sns.set(style="whitegrid")

# Set up 2x2 subplot grid
fig, axs = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle("Personal Finance Visualizations", fontsize=18)

# --- Plot 1: Top 10 Merchants by Total Spend (Bar Chart)
top_merchants = df.groupby('Merchant')['Amount'].sum().sort_values(ascending=False).head(10)
sns.barplot(
    x=top_merchants.values,
    y=top_merchants.index,
    palette=sns.color_palette("tab10"),
    ax=axs[0, 0]
)
axs[0, 0].set_title("Top 10 Merchants by Total Spend")
axs[0, 0].set_xlabel("Total Amount")
axs[0, 0].set_ylabel("Merchant")

# --- Plot 2: Amount by Category and Transaction Type (Box Plot)
sns.boxplot(
    data=df,
    x="Category",
    y="Amount",
    hue="Transaction Type",
    ax=axs[0, 1]
)
axs[0, 1].set_title("Amount by Category and Transaction Type")
axs[0, 1].tick_params(axis='x', rotation=45)
axs[0, 1].legend(title="Transaction Type", loc="upper right")

# --- Plot 3: Transaction Type by Account Type (Heatmap)
heatmap_data = pd.crosstab(df['Account Type'], df['Transaction Type'])
sns.heatmap(
    heatmap_data,
    annot=True,
    fmt='d',
    cmap="YlGnBu",
    ax=axs[1, 0]
)
axs[1, 0].set_title("Transaction Type Frequency by Account Type")
axs[1, 0].set_xlabel("Transaction Type")
axs[1, 0].set_ylabel("Account Type")

# --- Plot 4: Violin Plot of Amount by Payment Method and Transaction Type
sns.violinplot(
    data=df,
    x="Payment Method",
    y="Amount",
    hue="Transaction Type",
    split=True,
    palette="Set2",
    ax=axs[1, 1]
)
axs[1, 1].set_title("Amount by Payment Method and Transaction Type")
axs[1, 1].tick_params(axis='x', rotation=45)
axs[1, 1].legend(title="Transaction Type", loc="upper right")
axs[1, 1].set_ylabel("Amount")


# Final layout
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

