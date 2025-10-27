import re
import requests
from urllib.parse import urlparse
from bs4 import BeautifulSoup
from core.utils.helpers import clean_text, load_model



def extract_features(url: str) -> dict:
    """Extracts key URL and webpage features for phishing analysis."""
    features = {
        "url_length": len(url),
        "has_https": int(url.startswith("https")),
        "num_digits": len(re.findall(r"\d", url)),
        "num_special_chars": len(re.findall(r"[@_!#$%^&*()<>?/|}{~:]", url)),
        "domain": urlparse(url).netloc
    }

    try:
        response = requests.get(url, timeout=5)
        soup = BeautifulSoup(response.text, "html.parser")
        features["num_links"] = len(soup.find_all("a"))
        features["num_forms"] = len(soup.find_all("form"))
        features["num_images"] = len(soup.find_all("img"))
        features["page_title_length"] = len(soup.title.string if soup.title else "")
    except Exception:
        features.update({
            "num_links": 0,
            "num_forms": 0,
            "num_images": 0,
            "page_title_length": 0,
        })

    return features


def analyze_url(url: str) -> dict:
    """Perform ML-based phishing classification using saved model."""
    try:
        vectorizer, model = load_model()
    except Exception as e:
        return {"error": "Model/vectorizer not found", "details": str(e), "url": url}

    cleaned_text = clean_text(url)
    text_vector = vectorizer.transform([cleaned_text])
    prediction = model.predict(text_vector)[0]
    confidence = getattr(model, "predict_proba", lambda x: [[0.5, 0.5]])(text_vector)[0]

    return {
        "url": url,
        "prediction": "phishing" if prediction == 1 else "legitimate",
        "confidence": round(float(max(confidence)), 4),
        "features": extract_features(url)
    }


class SentinelAnalyzer:
    """Core threat analyzer used by app.py"""

    @staticmethod
    def analyze(url: str):
        """Simple analysis using the saved ML model."""
        return analyze_url(url)

    @staticmethod
    def full_analysis(url: str):
        """
        Simulated threat intelligence aggregation.
        (Placeholder for VirusTotal, URLScan, etc.)
        """
        try:
            features = extract_features(url)
            return {
                "url": url,
                "domain": features.get("domain"),
                "has_https": features.get("has_https"),
                "url_length": features.get("url_length"),
                "num_links": features.get("num_links"),
                "num_forms": features.get("num_forms"),
                "num_images": features.get("num_images"),
                "risk_score": round((features["num_forms"] + features["num_links"]) / 10, 2),
                "status": "Potentially suspicious" if features["num_forms"] > 2 else "Likely safe"
            }
        except Exception as e:
            return {"error": "Analysis failed", "details": str(e)}

    @staticmethod
    def result_to_json(result: dict, indent: int = 2):
        """Convert result dict to formatted JSON."""
        import json
        return json.dumps(result, indent=indent)
