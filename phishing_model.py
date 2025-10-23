import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from core.utils.helpers import clean_text, save_model

# 1️⃣ Load dataset
data = pd.read_csv("cleaned_dataset.csv")
print(f"📦 Dataset loaded: {data.shape[0]} rows")

# 2️⃣ Combine and clean text
data["text"] = (data["subject"].fillna("") + " " + data["body"].fillna("")).str.strip()
data["label"] = data["label"].astype(str).str.lower().replace({
    'phish': 1, 'spam': 1, 'phishing': 1, '1': 1, 'true': 1,
    'legit': 0, 'ham': 0, 'normal': 0, '0': 0, 'false': 0
})
data = data[data["label"].isin([0, 1])]
data["text"] = data["text"].apply(clean_text)

print(f"✅ Cleaned labels. Phishing: {data['label'].sum()}, Legitimate: {len(data) - data['label'].sum()}")

# 3️⃣ Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    data["text"], data["label"], test_size=0.2, random_state=42, stratify=data["label"]
)

# 4️⃣ TF-IDF Vectorization
vectorizer = TfidfVectorizer(max_features=10000, stop_words='english', ngram_range=(1,2))
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)
print(f"🧠 TF-IDF features: {X_train_tfidf.shape[1]}")

# 5️⃣ Train models
log_reg = LogisticRegression(max_iter=1000)
rf = RandomForestClassifier(n_estimators=200, random_state=42)

print("\n🚀 Training Logistic Regression...")
log_reg.fit(X_train_tfidf, y_train)
log_pred = log_reg.predict(X_test_tfidf)

print("\n🌲 Training Random Forest...")
rf.fit(X_train_tfidf, y_train)
rf_pred = rf.predict(X_test_tfidf)

# 6️⃣ Evaluate models
def evaluate_model(name, y_true, y_pred):
    print(f"\n📊 Results for {name}:")
    print(classification_report(y_true, y_pred))
    print("Accuracy:", round(accuracy_score(y_true, y_pred), 4))
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f"{name} Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

evaluate_model("Logistic Regression", y_test, log_pred)
evaluate_model("Random Forest", y_test, rf_pred)

# 7️⃣ Save the better model
save_model(vectorizer, log_reg)
print("\n✅ Training complete.")
