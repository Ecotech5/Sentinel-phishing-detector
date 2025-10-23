import re
import joblib

def clean_text(text: str) -> str:
    """Clean email text by removing URLs, symbols, and extra spaces."""
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.lower().strip()

def save_model(vectorizer, model, vec_path="tfidf_vectorizer.pkl", model_path="phishing_model.pkl"):
    """Save trained model and vectorizer."""
    joblib.dump(vectorizer, vec_path)
    joblib.dump(model, model_path)
    print(f"💾 Model saved to {model_path}, Vectorizer saved to {vec_path}")

def load_model(vec_path="tfidf_vectorizer.pkl", model_path="phishing_model.pkl"):
    """Load saved model and vectorizer."""
    vectorizer = joblib.load(vec_path)
    model = joblib.load(model_path)
    return vectorizer, model
