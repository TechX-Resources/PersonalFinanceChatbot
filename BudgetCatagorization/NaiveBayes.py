# Naive Bayes Classifier for Budget Categorization
# This script uses a Naive Bayes classifier to categorize budget transactions based on their descriptions.

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix

# --- Step 1: Load and prepare the dataset
df = pd.read_csv("DataHandling/cleaned_full_personal_finance.csv")

# Only keep necessary columns
df = df[['Description', 'Category']].dropna()

# Rename for clarity
df = df.rename(columns={'Description': 'TransactionDescription', 'Category': 'FinalCategory'})

# Optional: lowercase and strip whitespace
df['TransactionDescription'] = df['TransactionDescription'].str.lower().str.strip()

# --- Step 2: Train/test split
X = df['TransactionDescription']
y = df['FinalCategory']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Step 3: Vectorize text using TF-IDF
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# --- Step 4: Train the Naive Bayes model
nb_model = MultinomialNB()
nb_model.fit(X_train_vec, y_train)

# --- Step 5: Evaluate the model
y_pred = nb_model.predict(X_test_vec)

print("Classification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
