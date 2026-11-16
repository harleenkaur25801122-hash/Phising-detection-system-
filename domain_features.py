"""
domain_features.py
-------------------
Domain-level checks that require network access (WHOIS / DNS) plus
offline heuristics (typosquatting against a popular-domains list).

NOTE: WHOIS/DNS lookups require internet access and the optional
packages `python-whois` and `dnspython`. If they are not installed or
the network is unavailable, these functions fail gracefully and
return None / neutral values instead of crashing, so the rest of the
pipeline (URL features + ML model) can still produce a verdict.
"""

import socket
from datetime import datetime

try:
    import whois as whois_lib  # python-whois
    HAS_WHOIS = True
except ImportError:
    HAS_WHOIS = False

try:
    import dns.resolver  # dnspython
    HAS_DNS = True
except ImportError:
    HAS_DNS = False


# A small sample of popular/high-value brand domains to check typosquatting
# against. Extend this list (or load from Tranco top-1M) for production use.
POPULAR_DOMAINS = [
    "google.com", "paypal.com", "apple.com", "microsoft.com", "amazon.com",
    "facebook.com", "netflix.com", "instagram.com", "linkedin.com",
    "chase.com", "bankofamerica.com", "wellsfargo.com", "hdfcbank.com",
    "icicibank.com", "sbi.co.in", "irs.gov", "whatsapp.com", "outlook.com",
]


def _levenshtein(a: str, b: str) -> int:
    """Plain-python edit distance (no external dependency needed)."""
    if a == b:
        return 0
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        curr = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            cost = 0 if ca == cb else 1
            curr[j] = min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost)
        prev = curr
    return prev[-1]


def check_typosquatting(domain: str, threshold: int = 2) -> dict:
    """
    Compare domain against a list of popular domains.
    Returns the closest match and whether it looks like typosquatting
    (small edit distance but NOT an exact match).
    """
    domain = domain.lower().strip()
    best_match, best_dist = None, None
    for popular in POPULAR_DOMAINS:
        dist = _levenshtein(domain, popular)
        if best_dist is None or dist < best_dist:
            best_dist, best_match = dist, popular

    is_suspicious = bool(best_dist is not None and 0 < best_dist <= threshold and domain != best_match)
    return {
        "closest_popular_domain": best_match,
        "edit_distance": best_dist,
        "is_typosquat_suspect": is_suspicious,
    }


def get_domain_age_days(domain: str):
    """Returns domain age in days via WHOIS, or None if unavailable."""
    if not HAS_WHOIS:
        return None
    try:
        w = whois_lib.whois(domain)
        creation_date = w.creation_date
        if isinstance(creation_date, list):
            creation_date = creation_date[0]
        if creation_date is None:
            return None
        return (datetime.now() - creation_date).days
    except Exception:
        return None


def has_dns_record(domain: str) -> bool:
    """Check if the domain resolves at all (dead domains are often unregistered/parked)."""
    try:
        socket.gethostbyname(domain)
        return True
    except Exception:
        return False


def get_mx_records_exist(domain: str) -> bool:
    """Legit business domains almost always have MX records; some throwaway phishing domains don't."""
    if not HAS_DNS:
        return None
    try:
        answers = dns.resolver.resolve(domain, "MX")
        return len(answers) > 0
    except Exception:
        return False


def analyze_domain(domain: str) -> dict:
    """Bundle all domain-level checks into one dict."""
    domain = domain.lower().strip()
    result = {
        "domain": domain,
        "resolves": has_dns_record(domain),
        "age_days": get_domain_age_days(domain),
        "has_mx_records": get_mx_records_exist(domain),
    }
    result.update(check_typosquatting(domain))
    return result


def explain_domain_flags(result: dict) -> list:
    reasons = []
    if result.get("resolves") is False:
        reasons.append("Domain does not currently resolve (may be dead or parked)")
    age = result.get("age_days")
    if age is not None and age < 30:
        reasons.append(f"Domain was registered very recently ({age} days ago)")
    if result.get("has_mx_records") is False:
        reasons.append("Domain has no mail (MX) records, unusual for a legitimate business")
    if result.get("is_typosquat_suspect"):
        reasons.append(
            f"Domain closely resembles '{result.get('closest_popular_domain')}' "
            f"(edit distance {result.get('edit_distance')}) — possible typosquatting"
        )
    return reasons


if __name__ == "__main__":
    for d in ["paypa1-secure.com", "google.com", "totally-random-xyz123.tk"]:
        r = analyze_domain(d)
        print(d, "->", r)
        print(" reasons:", explain_domain_flags(r))
        print()
