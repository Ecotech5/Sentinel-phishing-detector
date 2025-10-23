from flask import Flask, request, jsonify
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch
from core.analyzer import SentinelAnalyzer
import re

# ================== CONFIG ==================
MODEL_PATH = "fine_tuned_model"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ================== INIT APP ==================
app = Flask(__name__)

# Load DistilBERT model
print("🚀 Loading fine-tuned phishing detection model...")
tokenizer = DistilBertTokenizer.from_pretrained(MODEL_PATH)
model = DistilBertForSequenceClassification.from_pretrained(MODEL_PATH)
model.to(DEVICE)
model.eval()

# Initialize Sentinel Analyzer
analyzer = SentinelAnalyzer()

# ================== HELPERS ==================
def clean_text(text):
    """Remove URLs and special characters from text"""
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.lower().strip()

def predict_phishing(text):
    """Run phishing prediction using fine-tuned DistilBERT"""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=256)
    inputs = {key: val.to(DEVICE) for key, val in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=1)
        pred_class = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred_class].item()
    return {
        "label": "Phishing" if pred_class == 1 else "Legitimate",
        "confidence": round(confidence * 100, 2)
    }

# ================== ROUTES ==================

@app.route("/")
def index():
    return jsonify({
        "message": "SentinelURL - Real-Time Phishing Detection API",
        "endpoints": ["/api/detect", "/api/analyze"]
    })

@app.route("/api/detect", methods=["POST"])
def detect_phishing():
    data = request.get_json()
    text = data.get("text")
    if not text:
        return jsonify({"error": "Missing 'text' field"}), 400

    text = clean_text(text)
    prediction = predict_phishing(text)

    return jsonify({
        "input_text": text[:200] + "...",
        "phishing_result": prediction
    })

@app.route("/api/analyze", methods=["POST"])
def analyze_url():
    data = request.get_json()
    url = data.get("url")
    if not url:
        return jsonify({"error": "Missing 'url' field"}), 400

    # Use Sentinel Threat Analyzer (VirusTotal, AbuseIPDB, URLScan)
    threat_report = analyzer.full_analysis(url)

    # Optional: also run ML-based prediction on domain name
    ml_result = predict_phishing(url)

    return jsonify({
        "url": url,
        "threat_intelligence": threat_report,
        "ml_detection": ml_result
    })

# ================== RUN SERVER ==================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
