"""
Evaluate both Demographic Joblib Classifiers and Remote 17-Party Transformer on the
Real-World Gold Standard Benchmark Dataset.
"""

import json
from pathlib import Path
from sklearn.metrics import accuracy_score, classification_report
import joblib

GOLD_PATH = Path("artifacts/demographics_models/gold_standard_benchmark.jsonl")
MODELS_DIR = Path("artifacts/demographics_models")

def evaluate_joblib_models():
    print("=" * 70)
    print("1. EVALUATING SKLEARN DEMOGRAPHIC CLASSIFIERS ON REAL-WORLD GOLD SET")
    print("=" * 70)

    gold_samples = []
    with open(GOLD_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                gold_samples.append(json.loads(line))

    texts = [s["text"] for s in gold_samples]

    tasks = [
        ("mental_state", "mental_state_classifier.joblib"),
        ("age_bracket", "age_classifier.joblib"),
        ("gender", "gender_classifier.joblib"),
        ("impatience", "impatience_classifier.joblib"),
        ("irrationality", "irrationality_classifier.joblib"),
        ("party_support", "party_support_classifier.joblib")
    ]

    for attr, model_file in tasks:
        p = MODELS_DIR / model_file
        if not p.exists():
            print(f"Skipping {attr}: {model_file} not found.")
            continue
        clf = joblib.load(p)
        y_true = [s.get(attr) for s in gold_samples if s.get(attr)]
        matched_texts = [s["text"] for s in gold_samples if s.get(attr)]
        
        y_pred = clf.predict(matched_texts)
        acc = accuracy_score(y_true, y_pred)
        print(f"\nTask: {attr:15s} | Accuracy: {acc*100:.2f}% ({sum(1 for t, p in zip(y_true, y_pred) if t == p)}/{len(y_true)})")
        print(classification_report(y_true, y_pred, zero_division=0, digits=2))

if __name__ == "__main__":
    evaluate_joblib_models()
