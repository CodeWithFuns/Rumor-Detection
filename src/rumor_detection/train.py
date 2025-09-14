import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from models import preprocess_text, train_model, load_model
from sklearn.feature_extraction.text import TfidfVectorizer

# Load and preprocess the data
data = pd.read_csv("your_data.csv")  # Replace with your actual data file
X = data['text']  # Column containing tweet text
y = data['label']  # Column containing labels ('Rumor' or 'Real')

# Apply TF-IDF vectorization
vectorizer = TfidfVectorizer(lowercase=True, preprocessor=preprocess_text, stop_words='english')
X_vectorized = vectorizer.fit_transform(X)

# Train-test split (ensure you're using stratification)
X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.2, stratify=y, random_state=42)

# Train model
model = train_model(X_train, y_train, model_type='logistic')  # You can switch to 'svm' for SVM

# Save the trained model
joblib.dump(model, 'rumor_detection_model.joblib')

# Optionally, save the vectorizer as well
joblib.dump(vectorizer, 'tfidf_vectorizer.joblib')
