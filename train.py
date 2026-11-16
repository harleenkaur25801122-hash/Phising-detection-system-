"""
train.py
--------
Trains a RandomForest classifier on URL features extracted from the
dataset CSV, evaluates it, and saves the trained model + feature list
to models/phishing_model.pkl
"""

import sys
import os
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, accuracy_score
)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.features.url_features import extract_url_features


def build_feature_dataframe(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    feature_rows = [extract_url_features(u) for u in df["url"]]
    features_df = pd.DataFrame(feature_rows)
    features_df["label"] = df["label"].values
    return features_df


def train_and_save(
    csv_path="data/urls_dataset.csv",
    model_out="models/phishing_model.pkl",
):
    print("Extracting features...")
    df = build_feature_dataframe(csv_path)
    feature_cols = [c for c in df.columns if c != "label"]

    X = df[feature_cols]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Training on {len(X_train)} samples, testing on {len(X_test)}...")
    clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced",
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]

    print("\n=== Evaluation ===")
    print("Accuracy:", round(accuracy_score(y_test, y_pred), 4))
    print("ROC-AUC :", round(roc_auc_score(y_test, y_proba), 4))
    print("\nClassification report:")
    print(classification_report(y_test, y_pred, target_names=["legitimate", "phishing"]))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\nTop 10 most important features:")
    importances = pd.Series(clf.feature_importances_, index=feature_cols)
    print(importances.sort_values(ascending=False).head(10))

    os.makedirs(os.path.dirname(model_out), exist_ok=True)
    with open(model_out, "wb") as f:
        pickle.dump({"model": clf, "feature_columns": feature_cols}, f)
    print(f"\nModel saved to {model_out}")

    return clf, feature_cols


if __name__ == "__main__":
    train_and_save()
