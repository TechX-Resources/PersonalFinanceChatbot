# Anomaly Detection using Isolation Forest with Time and Context
# This script detects anomalies in financial transactions using Isolation Forest,
# incorporating time and contextual features.


import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv("DataHandling/cleaned_full_personal_finance.csv")

# Drop unneeded columns
df = df.drop(columns=['User ID', 'Transaction Rating'], errors='ignore')

# Convert numeric columns
df['Amount'] = pd.to_numeric(df['Amount'], errors='coerce')
df['Balance After Transaction'] = pd.to_numeric(df['Balance After Transaction'], errors='coerce')

# Drop missing numeric values
df = df.dropna(subset=['Amount', 'Balance After Transaction'])

# Convert Date and Time
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
df['Time'] = pd.to_datetime(df['Time'], format='%H:%M:%S', errors='coerce')

# Extract features
df['DayOfWeek'] = df['Date'].dt.day_name()
df['Hour'] = df['Time'].dt.hour

# Fill missing categorical data
categorical_features = [
    'Type', 'Category', 'Merchant', 'Transaction Type',
    'Account Type', 'Location', 'DayOfWeek'
]
numeric_features = ['Amount', 'Balance After Transaction', 'Hour']

df[categorical_features] = df[categorical_features].fillna('Unknown')
df[numeric_features] = df[numeric_features].fillna(0)

# Preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
    ],
    remainder='passthrough'  # Leave numeric columns as is
)

X_processed = preprocessor.fit_transform(df[categorical_features + numeric_features])

# Isolation Forest
iso_forest = IsolationForest(n_estimators=100, contamination=0.01, random_state=42)
df['anomaly'] = iso_forest.fit_predict(X_processed)
df['is_anomaly'] = df['anomaly'] == -1

# Save anomalies
# df[df['is_anomaly']].to_csv("DataHandling/anomalies_with_context.csv", index=False)

# Visualize Using a small sample of the data set
sample_df = df.sample(750, random_state=42)  
plt.figure(figsize=(10, 6))
plt.scatter(sample_df['Amount'], sample_df['Balance After Transaction'],
            c=sample_df['is_anomaly'], cmap='coolwarm', edgecolor='k', alpha=0.6)
plt.xlabel("Transaction Amount")
plt.ylabel("Balance After Transaction")
plt.title("Anomalies with Time and Context")
plt.grid(True)
plt.show()
