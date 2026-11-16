"""
predict.py
----------
High-level API: given a URL, domain, or QR image, produce a verdict:
  - "phishing" / "suspicious" / "safe"
  - confidence score
  - human-readable reasons (hybrid of ML score + rule-based flags)
"""

import os
import sys
import pickle
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.features.url_features import extract_url_features, explain_url_flags
from src.features.domain_features import analyze_domain, explain_domain_flags
from src.qr.qr_decoder import decode_qr_from_path, decode_qr_from_bytes, is_url

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "models", "phishing_model.pkl")

_model_cache = {}


def _load_model():
    if "model" not in _model_cache:
        with open(MODEL_PATH, "rb") as f:
            bundle = pickle.load(f)
        _model_cache["model"] = bundle["model"]
        _model_cache["feature_columns"] = bundle["feature_columns"]
    return _model_cache["model"], _model_cache["feature_columns"]


def _verdict_from_score(score: float) -> str:
    if score >= 0.7:
        return "phishing"
    if score >= 0.35:
        return "suspicious"
    return "safe"


def analyze_url(url: str, include_domain_check: bool = True) -> dict:
    """
    Full analysis of a single URL: ML score + rule-based reasons +
    (optionally) domain/WHOIS checks.
    """
    model, feature_columns = _load_model()

    url_feats = extract_url_features(url)
    X = pd.DataFrame([url_feats])[feature_columns]
    ml_score = float(model.predict_proba(X)[0][1])

    reasons = explain_url_flags(url_feats)

    domain_info = None
    if include_domain_check:
        try:
            from urllib.parse import urlparse
            hostname = urlparse(url if "://" in url else "http://" + url).hostname
            if hostname:
                domain_info = analyze_domain(hostname)
                reasons.extend(explain_domain_flags(domain_info))
        except Exception:
            pass

    # Hybrid rule override: certain hard signals bump the score up
    # regardless of what the ML model says (defense in depth).
    hard_flags = 0
    if url_feats.get("has_ip_address"):
        hard_flags += 1
    if domain_info and domain_info.get("is_typosquat_suspect"):
        hard_flags += 1
    if domain_info and domain_info.get("resolves") is False:
        hard_flags += 1

    final_score = min(1.0, ml_score + 0.15 * hard_flags)
    verdict = _verdict_from_score(final_score)

    return {
        "input": url,
        "verdict": verdict,
        "confidence": round(final_score, 4),
        "ml_score": round(ml_score, 4),
        "reasons": reasons or ["No suspicious indicators found"],
        "url_features": url_feats,
        "domain_info": domain_info,
    }


def analyze_domain_only(domain: str) -> dict:
    """Standalone domain analysis (no full URL available)."""
    info = analyze_domain(domain)
    reasons = explain_domain_flags(info)
    risk_points = sum([
        1 for k in ("is_typosquat_suspect",) if info.get(k)
    ]) + (1 if info.get("resolves") is False else 0)
    verdict = "suspicious" if risk_points > 0 else "safe"
    return {
        "input": domain,
        "verdict": verdict,
        "reasons": reasons or ["No suspicious indicators found"],
        "domain_info": info,
    }


def analyze_qr_image(image_path: str = None, image_bytes: bytes = None) -> dict:
    """Decode a QR image and run the extracted URL through analyze_url()."""
    if image_path:
        decode_result = decode_qr_from_path(image_path)
    elif image_bytes:
        decode_result = decode_qr_from_bytes(image_bytes)
    else:
        return {"success": False, "error": "No image provided"}

    if not decode_result.get("success"):
        return {"success": False, "error": decode_result.get("error", "Could not decode QR")}

    data = decode_result["data"]
    result = {"success": True, "decoded_content": data}

    if is_url(data):
        result["type"] = "url"
        result["analysis"] = analyze_url(data)
    else:
        result["type"] = "text"
        result["analysis"] = {
            "verdict": "unknown",
            "reasons": ["QR content is not a URL; no phishing analysis applicable"],
        }

    return result


if __name__ == "__main__":
    test_urls = [
        "https://paypa1-secure-login.tk/verify/account",
        "https://www.google.com",
        "http://192.168.1.5/login.php?verify=1",
        "https://bit.ly/3xyzAbc",
    ]
    for u in test_urls:
        r = analyze_url(u, include_domain_check=True)
        print(f"\nURL: {u}")
        print(f"  Verdict: {r['verdict']}  (confidence={r['confidence']}, ml_score={r['ml_score']})")
        print(f"  Reasons: {r['reasons']}")
