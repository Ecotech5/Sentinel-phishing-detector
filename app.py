from flask import Flask, jsonify, request, render_template
from core.analyzer import SentinelAnalyzer
import os
import re
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

# ================== CONFIG ==================
MODEL_PATH = "fine_tuned_model"
Drive_URL = "https://drive.google.com/drive/folders/1I6ZPoaSvt7SgsBmqw8Fn303VYQ3o21AY?usp=drive_link"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ================== INIT APP ==================
app = Flask(__name__, template_folder="templates", static_folder="static")

# ================== LOAD MODEL ==================
print("🚀 Loading fine-tuned phishing detection model...")
try:
    tokenizer = DistilBertTokenizer.from_pretrained(MODEL_PATH)
    model = DistilBertForSequenceClassification.from_pretrained(MODEL_PATH)
    model.to(DEVICE)
    model.eval()
    print("✅ Model loaded successfully.")
except Exception as e:
    print(f"⚠️ Model loading failed: {e}")
    tokenizer, model = None, None

# Initialize Analyzer
analyzer = SentinelAnalyzer()

# ================== HELPERS ==================
def clean_text(text: str) -> str:
    """Clean text by removing URLs, special chars, and extra spaces."""
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.lower().strip()


def predict_phishing(text: str):
    """Run phishing detection using the fine-tuned DistilBERT model."""
    if not model or not tokenizer:
        return {"error": "Model not loaded"}

    inputs = tokenizer(
        text, return_tensors="pt", truncation=True, padding=True, max_length=256
    )
    inputs = {key: val.to(DEVICE) for key, val in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=1)
        pred_class = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred_class].item()

    return {
        "label": "Phishing" if pred_class == 1 else "Legitimate",
        "confidence": round(confidence * 100, 2),
    }

# ================== FRONTEND ==================
@app.route("/")
def index():
    """Render the main Sentinel Phishing Detection UI."""
    return render_template("index.html")

# ================== API ROUTES ==================
@app.route("/api/detect", methods=["POST"])
def detect_phishing():
    """Detect phishing in email or message text."""
    data = request.get_json(force=True)
    text = data.get("text")

    if not text:
        return jsonify({"error": "Missing 'text' field"}), 400

    cleaned = clean_text(text)
    result = predict_phishing(cleaned)

    # Add contextual recommendations
    recommendation = (
        "⚠️ Be cautious — this email may be a phishing attempt. Avoid clicking links or sharing personal info."
        if result["label"] == "Phishing"
        else "✅ This email appears legitimate, but always verify sender identity before acting."
    )

    return jsonify({
        "input_text": cleaned[:250] + "...",
        "phishing_result": result,
        "recommendation": recommendation
    })

@app.route("/api/analyze", methods=["POST"])
def analyze_url():
    """Perform full URL analysis using threat intelligence + ML model."""
    data = request.get_json(force=True)
    url = data.get("url")

    if not url:
        return jsonify({"error": "Missing 'url' field"}), 400

    # Threat intelligence check
    threat_data = analyzer.full_analysis(url)
    # ML detection
    ml_result = predict_phishing(url)

    recommendation = (
        "⚠️ Potentially unsafe URL detected. Avoid visiting this site."
        if ml_result["label"] == "Phishing"
        else "✅ URL appears safe, but verify with caution."
    )

    return jsonify({
        "url": url,
        "threat_intelligence": threat_data,
        "ml_detection": ml_result,
        "recommendation": recommendation
    })

@app.route("/api")
def api_info():
    """API Information endpoint."""
    return jsonify({
        "message": "🛡️ SentinelURL - Real-Time Phishing Detection API",
        "endpoints": ["/api/detect", "/api/analyze"]
    })

# ================== SERVER RUN ==================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
