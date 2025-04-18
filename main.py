import pandas as pd
import joblib
import os
import nltk
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Ensure nltk data is downloaded
nltk.download('stopwords')

# Define clean_text function locally
def clean_text(text):
    """Clean and preprocess text data"""
    import re
    import string
    from nltk.corpus import stopwords
    
    # Convert to lowercase
    text = str(text).lower()
    
    # Remove punctuation
    text = re.sub(f'[{string.punctuation}]', '', text)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    text = ' '.join([word for word in text.split() if word not in stop_words])
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

# Create a simple test dataset
print("Creating a test dataset...")
test_data = [
    {"headline": "This is clearly sarcastic", "is_sarcastic": 1},
    {"headline": "This is not sarcastic at all", "is_sarcastic": 0},
    {"headline": "Another sarcastic headline", "is_sarcastic": 1},
    {"headline": "Regular news without sarcasm", "is_sarcastic": 0},
    {"headline": "Sarcastic content here", "is_sarcastic": 1},
    {"headline": "Factual news headline", "is_sarcastic": 0},
    {"headline": "More sarcasm for testing", "is_sarcastic": 1},
    {"headline": "Standard news report", "is_sarcastic": 0},
    {"headline": "Highly sarcastic statement", "is_sarcastic": 1},
    {"headline": "Plain non-sarcastic headline", "is_sarcastic": 0},
]

# Create dataframe from test data
df = pd.DataFrame(test_data)

print(f"Test dataset shape: {df.shape}")
print(f"Value counts of is_sarcastic: {df['is_sarcastic'].value_counts()}")

# Clean text
df['cleaned'] = df['headline'].apply(clean_text)

# Vectorize
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['cleaned'])
y = df['is_sarcastic']

# Force manual split to ensure both classes are in training set
# First, split the data by class
df_class_0 = df[df['is_sarcastic'] == 0]
df_class_1 = df[df['is_sarcastic'] == 1]

print(f"Class 0 samples: {len(df_class_0)}")
print(f"Class 1 samples: {len(df_class_1)}")

# Now manually create train/test sets with both classes
train_ratio = 0.8
train_size_0 = int(len(df_class_0) * train_ratio)
train_size_1 = int(len(df_class_1) * train_ratio)

train_indices = list(df_class_0.index[:train_size_0]) + list(df_class_1.index[:train_size_1])
test_indices = list(df_class_0.index[train_size_0:]) + list(df_class_1.index[train_size_1:])

# Get training and test sets
df_train = df.loc[train_indices]
df_test = df.loc[test_indices]

print(f"Training set class distribution: {df_train['is_sarcastic'].value_counts()}")
print(f"Test set class distribution: {df_test['is_sarcastic'].value_counts()}")

# Create feature matrices and target vectors
X_train = vectorizer.transform(df_train['cleaned'])
y_train = df_train['is_sarcastic']
X_test = vectorizer.transform(df_test['cleaned'])
y_test = df_test['is_sarcastic']

# Verify we have both classes in the training set
print(f"Classes in y_train: {pd.Series(y_train).unique()}")

# Model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Save model and vectorizer
os.makedirs('models', exist_ok=True)
joblib.dump(model, 'models/sarcasm_model.pkl')
joblib.dump(vectorizer, 'models/tfidf_vectorizer.pkl')

# Evaluation
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))