# Naive Bayes Classifier for Budget Categorization
# This script uses a Naive Bayes classifier to categorize budget transactions based on their descriptions.

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix

# Load and prepare the dataset
df = pd.read_csv("DataHandling/cleaned_full_personal_finance.csv")

# Only keep necessary columns
df = df[['Description', 'Category']].dropna()

# Rename for clarity
df = df.rename(columns={'Description': 'TransactionDescription', 'Category': 'FinalCategory'})

# lowercase and strip whitespace
df['TransactionDescription'] = df['TransactionDescription'].str.lower().str.strip()

# Train/test split
X = df['TransactionDescription']
y = df['FinalCategory']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorize text using TF-IDF
vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words='english',             # removes common English words like "the", "a", "and"
    min_df=2,                         # remove words that appear in only 1 document
    max_df=0.95,                      # remove words that appear in 95%+ of documents
    ngram_range=(1, 2)                # includes unigrams and bigrams (e.g. 'uber', 'uber ride')
)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train the Naive Bayes model
nb_model = MultinomialNB()
nb_model.fit(X_train_vec, y_train)

# Evaluate the model
y_pred = nb_model.predict(X_test_vec)

print("Classification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
