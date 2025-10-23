import os
import pandas as pd
import torch
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

# =============================
# 🧩 Configuration
# =============================
DATA_PATH = "C:/Users/User/Desktop/phishing detection/cleaned_dataset.csv"
MODEL_NAME = "distilbert-base-uncased"
OUTPUT_DIR = "./phishing_bert_model"

# =============================
# 🧠 Detect Hardware
# =============================
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🚀 Device detected: {device.upper()}")

# =============================
# 📦 Load Dataset
# =============================
print("📦 Loading dataset...")
data = pd.read_csv(DATA_PATH)
print(f"✅ Dataset loaded successfully: {len(data)} rows")

# Combine subject and body into one text column
data["text"] = data["subject"].fillna("") + " " + data["body"].fillna("")
data = data[["text", "label"]]

# ✅ Sample smaller subset if on CPU
if device == "cpu":
    data = data.sample(2000, random_state=42)
    print("⚡ Using 2,000 samples for quick CPU training.")
else:
    print("💪 Full dataset will be used for GPU training.")

# =============================
# 🔀 Split Dataset
# =============================
train_texts, val_texts, train_labels, val_labels = train_test_split(
    data["text"].tolist(), data["label"].tolist(), test_size=0.2, random_state=42
)

# =============================
# 🔠 Tokenization
# =============================
print("🔠 Tokenizing text with DistilBERT...")
tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)

train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=256)
val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=256)

# =============================
# 🧾 Dataset Class
# =============================
class PhishingDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


train_dataset = PhishingDataset(train_encodings, train_labels)
val_dataset = PhishingDataset(val_encodings, val_labels)

# =============================
# 🧠 Load Model
# =============================
print("🧠 Loading DistilBERT model...")
model = DistilBertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
model.to(device)

# =============================
# ⚙️ Training Arguments
# =============================
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=1 if device == "cpu" else 3,
    per_device_train_batch_size=4 if device == "cpu" else 16,
    per_device_eval_batch_size=4 if device == "cpu" else 16,
    learning_rate=2e-5 if device == "cpu" else 3e-5,
    warmup_steps=100,
    weight_decay=0.01,
    logging_dir="./logs",
    eval_strategy="epoch",  # Compatible with newer Transformers versions
    save_strategy="epoch",
    logging_steps=100,
)

# =============================
# 🚀 Trainer
# =============================
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

# =============================
# 🏋️ Fine-tuning
# =============================
print("🚀 Starting fine-tuning...")
trainer.train()

# =============================
# 💾 Save Model
# =============================
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"✅ Model saved to {OUTPUT_DIR}")

