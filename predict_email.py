import joblib
import numpy as np
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

# ===== Load Model and Vectorizer =====
MODEL_PATH = "phishing_model.pkl"   # Your trained model
VECTORIZER_PATH = "tfidf_vectorizer.pkl"

print("\n🔐 Loading model and tokenizer...")
model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)

# ===== Adjustable Threshold =====
THRESHOLD = 0.80  # Change this to 0.7–0.85 depending on performance

# ===== Define Prediction Function =====
def predict_email(subject, body):
    text = f"{subject} {body}"
    X = vectorizer.transform([text])
    prob = model.predict_proba(X)[0]

    phishing_prob = prob[1]
    legit_prob = prob[0]

    if phishing_prob >= THRESHOLD:
        label = "Phishing"
    else:
        label = "Legitimate"

    # Determine risk level
    if phishing_prob >= 0.9:
        risk = "⚠️ High Risk"
    elif phishing_prob >= 0.7:
        risk = "🟠 Medium Risk"
    else:
        risk = "🟢 Low Risk"

    print("\n--- Email Prediction ---")
    print(f"Subject: {subject}")
    print(f"Prediction: {label}")
    print(f"Phishing Probability: {phishing_prob:.4f}")
    print(f"Legit Probability: {legit_prob:.4f}")
    print(f"Risk Level: {risk}")
    print("----------------------------------------")


# ===== Example Test Emails =====
emails = [
    {
        "subject": "Your Netflix account has been suspended",
        "body": "We noticed a problem with your billing. Please update your payment information to restore access."
    },
    {
        "subject": "Google security alert: new sign-in from Lagos",
        "body": "A new device just signed into your account. If this wasn’t you, please secure your account immediately."
    },
    {
        "subject": "Amazon order confirmation",
        "body": "Thank you for your purchase. Your order #123456 will be delivered soon."
    },
    {
        "subject": "Welcome to PayPal!",
        "body": "Your account has been successfully created. You can start sending and receiving payments safely."
    },
    {
        "subject": "Microsoft Account Password Changed",
        "body": "Your Microsoft account password was recently changed. If this wasn’t you, reset your password immediately."
    }
]

# ===== Run Predictions =====
for email in emails:
    predict_email(email["subject"], email["body"])
