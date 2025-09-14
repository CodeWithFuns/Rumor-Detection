import joblib
import pandas as pd
from models import predict_with_threshold

# Load the trained model and vectorizer
model = joblib.load('rumor_detection_model.joblib')
vectorizer = joblib.load('tfidf_vectorizer.joblib')

# Load and preprocess the data for evaluation
data = pd.read_csv("your_test_data.csv")  # Replace with your actual test data file
X_test = data['text']
y_test = data['label']

# Vectorize the test data
X_test_vectorized = vectorizer.transform(X_test)

# Predict with the custom threshold
y_pred = predict_with_threshold(model, X_test_vectorized, threshold=0.7)

# Evaluate the performance
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))
