from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

MODEL_PATH = "./phishing_bert_model"

# Load fine-tuned model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

def classify_email(subject, body):
    text = f"{subject} {body}"
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=1)
        pred = torch.argmax(probs, dim=1).item()
    label = "Phishing" if pred == 1 else "Legitimate"
    confidence = probs[0][pred].item()
    return label, confidence

# 🔍 Test Example
subject = "Important: Verify your account immediately"
body = "Your bank account will be suspended unless you confirm your login details here."
label, conf = classify_email(subject, body)
print(f"Prediction: {label} ({conf:.2%} confidence)")
