# train_model.py
"""
Train a RandomForest phishing URL classifier and save model.pkl.

Usage:
    python train_model.py

Make sure you have a dataset CSV in data/ with columns:
    - url
    - label   (0 = legitimate, 1 = phishing)

This script will:
 - load dataset (tries several filenames)
 - extract 18 features per URL (function extract_features_improved)
 - train/test split, train RandomForest
 - print metrics and feature importances
 - save model.pkl for use in the Flask app
"""

import os
import re
import math
from typing import List
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import tldextract

# ---------- Feature extractor (returns 2D list: [[f1, f2, ...]]) ----------
def shannon_entropy(s: str) -> float:
    if not s:
        return 0.0
    probs = [s.count(c) / len(s) for c in set(s)]
    return -sum(p * math.log2(p) for p in probs)

def extract_features_improved(url: str) -> List[List[float]]:
    """
    Extract the 18 features used for training.
    Returns a 2D list shape (1, 18) suitable for sklearn predict.
    Feature order MUST match the order used during training and in the Flask app.
    """
    u = (url or "").strip().lower()
    ext = tldextract.extract(u)
    domain = ext.domain or ""
    subdomain = ext.subdomain or ""
    host = (subdomain + "." + domain).strip(".") if subdomain else domain

    # 1) uses_https
    uses_https = 1 if u.startswith("https://") else 0

    # 2) count_special (non-alphanumeric and non-dot)
    count_special = len(re.findall(r'[^a-zA-Z0-9\.]', u))

    # 3) count_hyphens
    count_hyphens = u.count('-')

    # 4) url_length
    url_length = len(u)

    # 5) domain_length
    domain_length = len(domain)

    # 6) count_dots
    count_dots = u.count('.')

    # 7) has_susp_keyword
    suspicious_keywords = [
        'login', 'verify', 'update', 'secure', 'account', 'bank', 'confirm',
        'free', 'paypal', 'signin', 'ebanking', 'webscr', 'authenticate'
    ]
    has_susp_keyword = 1 if any(w in u for w in suspicious_keywords) else 0

    # 8) subdomains_count
    subdomains_count = len([s for s in subdomain.split('.') if s]) if subdomain else 0

    # 9) host_entropy
    host_entropy = shannon_entropy(host)

    # 10) host_length
    host_length = len(host)

    # 11) ratio_letters_digits
    letters = sum(c.isalpha() for c in u)
    digits = sum(c.isdigit() for c in u)
    ratio_letters_digits = (letters / digits) if digits > 0 else float(letters)

    # 12) count_digits
    count_digits = digits

    # 13) has_ip
    has_ip = 1 if re.search(r"(\d{1,3}\.){3}\d{1,3}", u) else 0

    # 14) prefix_suffix_flag (domain contains hyphen)
    prefix_suffix_flag = 1 if '-' in domain else 0

    # 15) uncommon_tld
    common_tlds = {'com', 'org', 'net', 'edu', 'gov', 'mil', 'info', 'io', 'co'}
    tld = (ext.suffix or "").lower()
    uncommon_tld = 0 if tld in common_tlds else 1

    # 16) punycode_flag
    punycode_flag = 1 if 'xn--' in u else 0

    # 17) count_at
    count_at = u.count('@')

    # 18) has_port (like :8080)
    has_port = 1 if re.search(r":\d{1,5}", u) else 0

    features = [
        uses_https, count_special, count_hyphens, url_length, domain_length,
        count_dots, has_susp_keyword, subdomains_count, host_entropy, host_length,
        ratio_letters_digits, count_digits, has_ip, prefix_suffix_flag,
        uncommon_tld, punycode_flag, count_at, has_port
    ]

    return [features]

# ---------- Helper: find dataset ----------
def find_dataset_file():
    candidates = [
        "data/dataset_final.csv",
        "data/combined_urls.csv",
        "data/sample-urls.csv",
        "data/dataset.csv",
        "data/urlset.csv"
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None

# ---------- Main training pipeline ----------
def main():
    dataset_path = find_dataset_file()
    if dataset_path is None:
        raise FileNotFoundError(
            "No dataset found. Put a CSV with 'url' and 'label' columns into data/ "
            "(e.g., data/dataset_final.csv or data/combined_urls.csv)."
        )

    print("Using dataset:", dataset_path)
    # read csv robustly
    try:
        df = pd.read_csv(dataset_path, on_bad_lines='skip', encoding='utf-8')
    except Exception:
        df = pd.read_csv(dataset_path, on_bad_lines='skip', encoding='ISO-8859-1')

    # Normalize column names
    cols = [c.lower().strip() for c in df.columns]
    df.columns = cols

    if 'url' not in df.columns:
        # try domain -> url
        if 'domain' in df.columns:
            df = df.rename(columns={'domain': 'url'})
        else:
            print("Columns found:", df.columns.tolist())
            raise RuntimeError("Dataset must contain a column named 'url' (or 'domain').")

    if 'label' not in df.columns:
        # try to guess a label column
        possible = [c for c in df.columns if 'label' in c or 'class' in c]
        if possible:
            df = df.rename(columns={possible[0]: 'label'})
        else:
            raise RuntimeError("Dataset must contain a 'label' column (0=legit, 1=phish).")

    df = df[['url', 'label']].dropna()
    df['label'] = df['label'].astype(int)

    # Optionally limit size for quick training (comment out for full)
    # df = df.sample(n=min(10000, len(df)), random_state=42)

    print("Total rows:", len(df))
    # Extract features for all rows (this may take a while for very large dataset)
    X = []
    for u in df['url'].astype(str).tolist():
        X.append(extract_features_improved(u)[0])
    X = np.array(X)
    y = df['label'].values

    print("Feature matrix shape:", X.shape)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y if len(np.unique(y))>1 else None
    )

    # Train model
    clf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)

    # Evaluate
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print("\n=== Metrics ===")
    print("Accuracy: {:.4f}\n".format(acc))
    print("Classification report:")
    print(classification_report(y_test, y_pred, zero_division=0))
    print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))

    # Feature importances (print names + importance)
    feature_names = [
        'uses_https', 'count_special', 'count_hyphens', 'url_length', 'domain_length',
        'count_dots', 'has_susp_keyword', 'subdomains_count', 'host_entropy', 'host_length',
        'ratio_letters_digits', 'count_digits', 'has_ip', 'prefix_suffix_flag',
        'uncommon_tld', 'punycode_flag', 'count_at', 'has_port'
    ]
    importances = clf.feature_importances_
    ranked = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)
    print("\nTop feature importances:")
    for name, val in ranked:
        print(f"{name:20s} -> {val:.4f}")

    # Save model
    joblib.dump(clf, "model.pkl")
    print("\n[SAVED] Model written to model.pkl")

# Run when executed
if __name__ == "__main__":
    main()
