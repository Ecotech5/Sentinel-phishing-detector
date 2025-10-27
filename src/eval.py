import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load the saved model and vectorizer
model = joblib.load("phishing_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Load your dataset for evaluation
df = pd.read_csv(r"C:\Users\User\Desktop\phishing detection\data\final_merged_dataset.csv")

# Combine subject and body into text
df['text'] = df['subject'].fillna('') + ' ' + df['body'].fillna('')

# Prepare test data (you can use part of your dataset or a dedicated test file)
X = df['text']
y = df['label']

# Vectorize
X_tfidf = vectorizer.transform(X)

# Predict
y_pred = model.predict(X_tfidf)

# Evaluate
print("\n--- Model Evaluation ---")
print(f"Accuracy : {accuracy_score(y, y_pred):.4f}")
print(f"Precision: {precision_score(y, y_pred):.4f}")
print(f"Recall   : {recall_score(y, y_pred):.4f}")
print(f"F1 Score : {f1_score(y, y_pred):.4f}")
