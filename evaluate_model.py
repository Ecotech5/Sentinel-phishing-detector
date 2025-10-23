import torch
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import seaborn as sns
import os
import random

# ============== CONFIG ==============
MODEL_PATH = "fine_tuned_model"
DATA_PATH = "cleaned_dataset.csv"
BATCH_SIZE = 16
MAX_SAMPLES = 5000  # evaluate on smaller sample for speed
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f" Using device: {DEVICE}")

# ============== LOAD MODEL ==============
print(" Loading fine-tuned model...")
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f" Model not found at: {MODEL_PATH}")

tokenizer = DistilBertTokenizer.from_pretrained(MODEL_PATH)
model = DistilBertForSequenceClassification.from_pretrained(MODEL_PATH)
model.to(DEVICE)
model.eval()

# ============== LOAD DATASET ==============
print(" Reading dataset...")
data = pd.read_csv(DATA_PATH)

data["text"] = (data["subject"].astype(str) + " " + data["body"].astype(str)).fillna("")
if len(data) > MAX_SAMPLES:
    data = data.sample(n=MAX_SAMPLES, random_state=42).reset_index(drop=True)
    print(f" Using a random subset of {MAX_SAMPLES} samples for faster evaluation")

texts = data["text"].tolist()
labels = data["label"].tolist()

# ============== TOKENIZATION ==============
print(" Tokenizing text...")
inputs = tokenizer(
    texts,
    padding=True,
    truncation=True,
    return_tensors="pt",
    max_length=256
)

dataset = TensorDataset(inputs["input_ids"], inputs["attention_mask"], torch.tensor(labels))
loader = DataLoader(dataset, batch_size=BATCH_SIZE)

# ============== EVALUATION ==============
print(" Evaluating model...")

all_preds, all_labels = [], []
total_batches = len(loader)

with torch.no_grad():
    for i, batch in enumerate(loader, start=1):
        input_ids, attention_mask, batch_labels = [b.to(DEVICE) for b in batch]
        outputs = model(input_ids, attention_mask=attention_mask)
        preds = torch.argmax(outputs.logits, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(batch_labels.cpu().numpy())

        # Progress update
        if i % 20 == 0 or i == total_batches:
            print(f" Processed {i}/{total_batches} batches...")

# ============== METRICS ==============
accuracy = accuracy_score(all_labels, all_preds)
precision = precision_score(all_labels, all_preds)
recall = recall_score(all_labels, all_preds)
f1 = f1_score(all_labels, all_preds)

print("\n Classification Report:")
print(classification_report(all_labels, all_preds, target_names=["Legitimate", "Phishing"]))

print(" Summary:")
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-Score:  {f1:.4f}")

# ============== CONFUSION MATRIX ==============
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Legit", "Phish"], yticklabels=["Legit", "Phish"])
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix - Phishing Email Detection")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.show()

# ============== SAVE METRICS ==============
metrics_summary = {
    "accuracy": accuracy,
    "precision": precision,
    "recall": recall,
    "f1_score": f1
}
pd.DataFrame([metrics_summary]).to_csv("evaluation_results.csv", index=False)
print("\n Metrics saved to evaluation_results.csv")
print("️ Confusion matrix saved as confusion_matrix.png")
