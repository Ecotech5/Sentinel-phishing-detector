"""
phishing_model.py
------------------------------------
Enhanced phishing email detection trainer:
✅ Improved cleaning & domain handling
✅ Balanced dataset
✅ TF-IDF features
✅ Logistic Regression & Random Forest comparison
✅ Whitelist & safe domain logic
✅ Model saved for API integration
"""

import pandas as pd
import re
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from imblearn.over_sampling import RandomOverSampler
from core.utils.helpers import clean_text, save_model

# ========== 1️⃣ Load Dataset ==========
def load_dataset(path="cleaned_dataset.csv"):
    print(f"📥 Loading dataset from {path} ...")
    data = pd.read_csv(path)
    print(f"📦 Dataset loaded: {data.shape[0]} rows")

    # Combine and normalize columns
    data["text"] = (data["subject"].fillna("") + " " + data["body"].fillna("")).str.strip()
    data["label"] = data["label"].astype(str).str.lower().replace({
        'phish': 1, 'spam': 1, 'phishing': 1, '1': 1, 'true': 1,
        'legit': 0, 'ham': 0, 'normal': 0, '0': 0, 'false': 0
    })
    data = data[data["label"].isin([0, 1])]
    print(f"✅ Cleaned dataset. {len(data)} valid samples.")
    return data


# ========== 2️⃣ Improved Cleaning ==========
def enhanced_clean_text(text: str) -> str:
    """Clean and normalize text while keeping useful punctuation and domains."""
    text = re.sub(r"http\S+", " ", text)           # Remove URLs
    text = re.sub(r"[^a-zA-Z0-9@.\s]", " ", text)  # Keep dots and @ for email/domain relevance
    text = re.sub(r"\s+", " ", text)
    return text.lower().strip()


# ========== 3️⃣ Safe Domain Whitelist ==========
SAFE_DOMAINS = [
    "amazon.com", "microsoft.com", "google.com", "apple.com",
    "paypal.com", "netflix.com", "facebook.com", "github.com"
]

def is_safe_domain(text: str) -> bool:
    """Return True if text mentions a trusted domain."""
    for domain in SAFE_DOMAINS:
        if domain in text:
            return True
    return False


# ========== 4️⃣ Train Model ==========
def train_model():
    data = load_dataset()
    print("🧹 Cleaning text...")
    data["text"] = data["text"].astype(str).apply(enhanced_clean_text)

    # Balance classes
    print("⚖️ Balancing dataset...")
    ros = RandomOverSampler(random_state=42)
    X, y = data["text"], data["label"]
    vectorizer = TfidfVectorizer(max_features=10000, stop_words='english', ngram_range=(1, 2))
    X_tfidf = vectorizer.fit_transform(X)
    X_res, y_res = ros.fit_resample(X_tfidf, y)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_res, y_res, test_size=0.2, random_state=42, stratify=y_res
    )

    # Train models
    print("\n🚀 Training Logistic Regression...")
    log_reg = LogisticRegression(max_iter=1000)
    log_reg.fit(X_train, y_train)

    print("\n🌲 Training Random Forest...")
    rf = RandomForestClassifier(n_estimators=200, random_state=42)
    rf.fit(X_train, y_train)

    # Predict
    log_pred = log_reg.predict(X_test)
    rf_pred = rf.predict(X_test)

    # Evaluate both models
    def evaluate_model(name, y_true, y_pred):
        print(f"\n📊 {name} Results:")
        print(classification_report(y_true, y_pred, digits=4))
        acc = round(accuracy_score(y_true, y_pred), 4)
        print(f"✅ Accuracy: {acc}")
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f"{name} Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.show()

    evaluate_model("Logistic Regression", y_test, log_pred)
    evaluate_model("Random Forest", y_test, rf_pred)

    # Choose best model
    best_model = log_reg if accuracy_score(y_test, log_pred) >= accuracy_score(y_test, rf_pred) else rf

    # Save
    save_model(vectorizer, best_model)
    print("\n💾 Model & Vectorizer saved successfully!")


# ========== 5️⃣ Smart Classifier (for app.py integration) ==========
def classify_email(text: str, model=None, vectorizer=None):
    """Smart classification with safe domain & threshold."""
    from joblib import load

    if model is None or vectorizer is None:
        vectorizer = load("tfidf_vectorizer.pkl")
        model = load("phishing_model.pkl")

    cleaned = enhanced_clean_text(text)

    if is_safe_domain(cleaned):
        return {"label": "Legitimate", "confidence": 100.0}

    X_input = vectorizer.transform([cleaned])
    probs = model.predict_proba(X_input)[0]
    pred_idx = probs.argmax()
    confidence = round(probs[pred_idx] * 100, 2)

    # Smart thresholding
    if confidence < 70:
        label = "Uncertain"
    else:
        label = "Phishing" if pred_idx == 1 else "Legitimate"

    return {"label": label, "confidence": confidence}


# ========== MAIN ==========
if __name__ == "__main__":
    print("🚀 Starting model training pipeline...")
    train_model()
    print("\n✅ Training complete. Model ready for deployment.")
