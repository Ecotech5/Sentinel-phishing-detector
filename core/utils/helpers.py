import re
import joblib
from urllib.parse import urlparse
import ipaddress


def clean_text(text: str) -> str:
    """Clean email text by removing URLs, symbols, and extra spaces."""
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.lower().strip()


def normalize_url(url: str) -> str:
    """Normalize the URL by ensuring consistent scheme and formatting."""
    if not url.startswith(("http://", "https://")):
        url = "http://" + url
    parsed = urlparse(url)
    normalized = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
    return normalized.strip("/")


def extract_domain(url: str) -> str:
    """Extract domain name from URL."""
    parsed = urlparse(url)
    return parsed.netloc.lower()


def extract_ip(url: str) -> str:
    """Extract IP address from URL if available."""
    parsed = urlparse(url)
    try:
        ip = ipaddress.ip_address(parsed.hostname)
        return str(ip)
    except Exception:
        return None


def clamp_int(value, min_value=0, max_value=100):
    """Clamp integer between min and max values."""
    try:
        return max(min_value, min(int(value), max_value))
    except Exception:
        return min_value


def save_model(vectorizer, model, vec_path="tfidf_vectorizer.pkl", model_path="phishing_model.pkl"):
    """Save trained model and vectorizer."""
    joblib.dump(vectorizer, vec_path)
    joblib.dump(model, model_path)
    print(f"💾 Model saved to {model_path}, Vectorizer saved to {vec_path}")


def load_model(vec_path="tfidf_vectorizer.pkl", model_path="phishing_model.pkl"):
    """Load saved model and vectorizer."""
    vectorizer = joblib.load(vec_path)
    model = joblib.load(model_path)
    return vectorizer, model
