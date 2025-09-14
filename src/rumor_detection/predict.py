import joblib
from models import preprocess_text, predict_with_threshold

# Load the trained model and vectorizer
model = joblib.load('rumor_detection_model.joblib')
vectorizer = joblib.load('tfidf_vectorizer.joblib')

def predict_new_data(text, threshold=0.7):
    """
    Predict whether the input text is a rumor or not.
    Args:
        text (str): The input text to classify.
        threshold (float): The probability threshold for labeling a text as 'Rumor'.
    """
    # Preprocess and vectorize the input text
    X_input = vectorizer.transform([text])
    
    # Predict with the custom threshold
    prediction = predict_with_threshold(model, X_input, threshold)
    
    return "Rumor" if prediction == 1 else "Real"

# Example usage
input_text = "Sample text to classify as rumor or real"
prediction = predict_new_data(input_text, threshold=0.7)
print(f"The text is classified as: {prediction}")
