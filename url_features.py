"""
url_features.py
----------------
Extracts lexical & structural features from a raw URL string.
These features are cheap to compute (no network calls) and form the
first line of phishing detection.
"""

import re
import math
from urllib.parse import urlparse
from collections import Counter

SHORTENER_DOMAINS = {
    "bit.ly", "tinyurl.com", "goo.gl", "t.co", "ow.ly", "is.gd",
    "buff.ly", "adf.ly", "shorte.st", "cutt.ly", "rebrand.ly", "tiny.cc",
}

SUSPICIOUS_KEYWORDS = [
    "login", "verify", "secure", "account", "update", "confirm",
    "bank", "signin", "sign-in", "password", "pay", "billing",
    "suspend", "unlock", "invoice", "refund", "webscr", "wallet",
    "support", "security", "alert",
]

IP_REGEX = re.compile(
    r"^(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}"
    r"(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)$"
)


def _shannon_entropy(s: str) -> float:
    """Higher entropy => more 'random looking' string (common in DGA / phishing domains)."""
    if not s:
        return 0.0
    counts = Counter(s)
    length = len(s)
    return -sum((c / length) * math.log2(c / length) for c in counts.values())


def _get_hostname(url: str) -> str:
    parsed = urlparse(url if "://" in url else "http://" + url)
    return parsed.hostname or ""


def extract_url_features(url: str) -> dict:
    """
    Extract a dictionary of numeric/boolean features from a URL.
    Works even if the URL is missing a scheme (adds http:// as default).
    """
    raw_url = url.strip()
    if "://" not in raw_url:
        raw_url = "http://" + raw_url

    parsed = urlparse(raw_url)
    hostname = parsed.hostname or ""
    path = parsed.path or ""
    query = parsed.query or ""

    features = {
        "url_length": len(raw_url),
        "hostname_length": len(hostname),
        "path_length": len(path),
        "num_dots": raw_url.count("."),
        "num_hyphens": raw_url.count("-"),
        "num_underscores": raw_url.count("_"),
        "num_at_symbols": raw_url.count("@"),
        "num_digits": sum(c.isdigit() for c in raw_url),
        "num_slashes": raw_url.count("/"),
        "num_query_params": query.count("=") if query else 0,
        "num_subdomains": max(hostname.count(".") - 1, 0) if hostname else 0,
        "has_ip_address": 1 if IP_REGEX.match(hostname) else 0,
        "has_https": 1 if parsed.scheme == "https" else 0,
        "has_port": 1 if parsed.port else 0,
        "is_shortener": 1 if hostname in SHORTENER_DOMAINS else 0,
        "double_slash_in_path": 1 if "//" in path else 0,
        "suspicious_keyword_count": sum(
            1 for kw in SUSPICIOUS_KEYWORDS if kw in raw_url.lower()
        ),
        "hostname_entropy": round(_shannon_entropy(hostname), 3),
        "digit_ratio": round(
            sum(c.isdigit() for c in raw_url) / len(raw_url), 3
        ) if raw_url else 0,
        "has_non_standard_tld": 0,  # set below
    }

    # Flag uncommon / high-abuse TLDs (heuristic list, extend as needed)
    high_risk_tlds = {"zip", "mov", "xyz", "top", "gq", "tk", "ml", "cf", "click", "work"}
    tld = hostname.split(".")[-1].lower() if "." in hostname else ""
    features["has_non_standard_tld"] = 1 if tld in high_risk_tlds else 0

    return features


def explain_url_flags(features: dict) -> list:
    """Turn extracted features into short human-readable reasons (for UI display)."""
    reasons = []
    if features.get("has_ip_address"):
        reasons.append("URL uses a raw IP address instead of a domain name")
    if not features.get("has_https"):
        reasons.append("Connection is not secured with HTTPS")
    if features.get("is_shortener"):
        reasons.append("URL uses a link-shortening service (destination is hidden)")
    if features.get("num_at_symbols", 0) > 0:
        reasons.append("URL contains '@' which can be used to obscure the real destination")
    if features.get("suspicious_keyword_count", 0) >= 2:
        reasons.append("URL contains multiple credential/urgency-related keywords")
    if features.get("num_subdomains", 0) >= 3:
        reasons.append("Unusually high number of subdomains")
    if features.get("hostname_entropy", 0) > 4.0:
        reasons.append("Domain name looks randomly generated (high entropy)")
    if features.get("has_non_standard_tld"):
        reasons.append("Domain uses a top-level domain often associated with abuse")
    if features.get("double_slash_in_path"):
        reasons.append("Suspicious redirect pattern ('//') found in URL path")
    if features.get("url_length", 0) > 100:
        reasons.append("URL is unusually long")
    return reasons


if __name__ == "__main__":
    test_urls = [
        "http://paypa1-secure-login.tk/verify/account?ref=123",
        "https://www.google.com",
        "http://192.168.1.5/login.php",
        "https://bit.ly/3xyzAbc",
    ]
    for u in test_urls:
        f = extract_url_features(u)
        print(u)
        print(" features:", f)
        print(" reasons :", explain_url_flags(f))
        print()
