import json
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from sklearn.metrics import accuracy_score, classification_report
from pathlib import Path

gold_file = Path("artifacts/demographics_models/gold_standard_benchmark.jsonl")
samples = [json.loads(l) for l in open(gold_file)]

mod_dir = "/home/s0r0qw6/party_bert_training/output_model"
tokenizer = AutoTokenizer.from_pretrained(mod_dir)
model = AutoModelForSequenceClassification.from_pretrained(mod_dir).cuda().eval()

PARTIES_17 = [
    "BJP", "Congress", "AAP", "TMC", "SP", "DMK",
    "Shiv Sena", "NCP", "RJD", "JDU", "AIADMK",
    "TDP", "YSRCP", "BRS", "BSP", "CPIM", "CJP", "None"
]

y_true = []
y_pred = []
for s in samples:
    text = s["text"]
    expected = s["party"]
    enc = tokenizer(text, return_tensors="pt").to("cuda")
    with torch.no_grad():
        probs = torch.softmax(model(**enc).logits, dim=-1)[0]
        idx = torch.argmax(probs).item()
        pred = PARTIES_17[idx]
    y_true.append(expected)
    y_pred.append(pred)

acc = accuracy_score(y_true, y_pred)
print("\n=== 17-PARTY BERT EVALUATION ON REAL-WORLD GOLD STANDARD ===")
print(f"Accuracy: {acc*100:.2f}% ({sum(1 for t, p in zip(y_true, y_pred) if t == p)}/{len(y_true)})")
print(classification_report(y_true, y_pred, digits=3, zero_division=0))

print("=== MISCLASSIFIED SAMPLES ===")
for s, yt, yp in zip(samples, y_true, y_pred):
    if yt != yp:
        t_preview = s['text'][:70]
        print(f"Expected: {yt:10s} | Pred: {yp:10s} | Text: {t_preview}...")
