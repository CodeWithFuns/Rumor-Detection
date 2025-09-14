import joblib
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
import numpy as np

# Load the spaCy model for lemmatization
nlp = spacy.load('en_core_web_sm')

def preprocess_text(text):
    """
    Preprocess the text (lowercase, lemmatize, remove stop words).
    Args:
        text (str): Input text.
    Returns:
        str: Preprocessed text.
    """
    # Apply lemmatization, lowercase conversion, stopword removal, etc.
    return ' '.join([word.lemma_ for word in nlp(text) if word.is_stop == False])

def load_model(model_path):
    """
    Load a trained model from a file.
    Args:
        model_path (str): Path to the saved model.
    Returns:
        The loaded model.
    """
    return joblib.load(model_path)

def train_model(X_train, y_train, model_type='logistic'):
    """
    Train the model using Logistic Regression or SVM.
    Args:
        X_train (array): The training feature data.
        y_train (array): The training labels.
        model_type (str): Model type ('logistic' or 'svm').
    Returns:
        The trained model.
    """
    if model_type == 'logistic':
        model = LogisticRegression(class_weight='balanced')
    elif model_type == 'svm':
        model = LinearSVC(class_weight='balanced')
    else:
        raise ValueError("Invalid model type")

    model.fit(X_train, y_train)
    return model

def predict_with_threshold(model, X, threshold=0.7):
    """
    Make predictions with a custom threshold.
    Args:
        model: The trained model.
        X: Feature data (e.g., TF-IDF features).
        threshold: Probability threshold for classifying as 'Rumor'.
    Returns:
        Predicted labels based on the custom threshold.
    """
    probs = model.predict_proba(X)  # Get probabilities for each class
    predictions = (probs[:, 1] >= threshold).astype(int)  # Classify as 'Rumor' (1) if prob > threshold
    return predictions
