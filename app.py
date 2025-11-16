# app.py
from flask import Flask, render_template, request
import joblib
import train_model  # contains extract_features_improved

app = Flask(__name__)

# Load the trained model
MODEL_FILE = "model.pkl"
try:
    model = joblib.load(MODEL_FILE)
except FileNotFoundError:
    raise RuntimeError(f"Model file {MODEL_FILE} not found. Run train_model.py first.")

# Use improved feature extractor
def extract_features(url: str):
    """Return feature vector using the improved extractor from train_model.py"""
    return train_model.extract_features_improved(url)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    url = request.form['url'].strip().lower()

    # Extract features
    features = extract_features(url)

    # Predict using the trained model
    pred = model.predict(features)[0]
    prob = model.predict_proba(features)[0]

    # Interpret prediction
    if pred == 1:
        result = f"Phishing Website 🚨 (Confidence: {prob[1]:.2f})"
    else:
        result = f"Legitimate Website ✅ (Confidence: {prob[0]:.2f})"

    return render_template('index.html', result=result, url=url)

if __name__ == '__main__':
    app.run(debug=True)
