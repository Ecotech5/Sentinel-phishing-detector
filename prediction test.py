# diagnose_probs.py
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np

model_path = r"C:\Users\User\Desktop\phishing detection\fine_tuned_model"  # change if needed
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

emails = [
    {"id":1, "subject":"Welcome to Amazon Prime — Your benefits are active",
     "body":"Hi Emmanuel,\n\nThanks for joining Amazon Prime. Your membership is now active. You can manage your account at https://www.amazon.com/your-account. If you have questions reply to this email or visit our Help Center.\n\nBest,\nAmazon Prime Team"},
    {"id":2, "subject":"Your Netflix receipt for March 2025",
     "body":"Hello,\n\nThanks for your payment. A receipt for your Netflix Standard Plan is attached. If you did not authorize this payment, please visit https://www.netflix.com/account to review billing details.\n\nRegards,\nNetflix Billing"},
    {"id":3, "subject":"Your PayPal weekly summary is available",
     "body":"Hi,\n\nYour PayPal activity summary is ready. Log in at https://www.paypal.com to view transactions. We never ask for your password via email.\n\nThanks,\nPayPal Security Team"},
    {"id":5, "subject":"Your UPS tracking update — Package delivered",
     "body":"Hello,\n\nYour UPS package (Tracking #1Z999AA10123456784) was delivered today at 2:14 PM. Track details: https://www.ups.com/track.\n\nUPS Customer Service"},
    {"id":6, "subject":"Google Workspace: New device signed in",
     "body":"Hi,\n\nA new device signed into your Google account from Lagos, NG. If this wasn't you, go to https://myaccount.google.com/security to review. Otherwise no action is necessary.\n\nGoogle Security Team"},
    {"id":10, "subject":"Receipt — Apple Store purchase confirmation",
     "body":"Hello,\n\nThank you for your purchase at the Apple Store. Order #M123456789. Review your order at https://www.apple.com/your-orders.\n\nApple Store Team"},
]

for e in emails:
    text = (e["subject"] or "") + " " + (e["body"] or "")
    enc = tokenizer(text, truncation=True, padding=True, return_tensors="pt", max_length=512).to(device)
    with torch.no_grad():
        out = model(**enc)
        logits = out.logits[0].cpu().numpy()
        probs = torch.softmax(out.logits, dim=1)[0].cpu().numpy()

    tokens = tokenizer.convert_ids_to_tokens(enc["input_ids"][0])
    print(f"\n--- EMAIL {e['id']} ---")
    print("Phishing prob:", probs[1].round(4), " Legit prob:", probs[0].round(4))
    print("Logits:", logits.round(4))
    print("First 40 tokens:", tokens[:40])
