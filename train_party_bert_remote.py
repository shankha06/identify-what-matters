import json, time, random, os
from pathlib import Path
from collections import Counter
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score, accuracy_score

SEED = 42
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)
random.seed(SEED)

BASE_MODEL = "microsoft/mdeberta-v3-base"
OUTPUT_DIR = Path("/home/s0r0qw6/party_bert_training/output_model")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PARTIES = ["BJP", "Congress", "AAP", "TMC", "SP", "DMK", "None"]
label_of = {p: i for i, p in enumerate(PARTIES)}

# 1. Load data
samples = []
with open("/home/s0r0qw6/party_bert_training/data/party_train_all.jsonl") as f:
    for line in f:
        d = json.loads(line)
        t = d.get("text", "").strip()
        p = d.get("party")
        w = float(d.get("weight", 1.0))
        if t and p in label_of:
            samples.append({"text": t, "party": p, "weight": w})

print(f"Loaded {len(samples)} total samples.")

train_items, val_items = train_test_split(
    samples, test_size=0.15, stratify=[s["party"] for s in samples], random_state=SEED
)
print(f"Train split: {len(train_items)} | Val split: {len(val_items)}")

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
model = AutoModelForSequenceClassification.from_pretrained(
    BASE_MODEL,
    num_labels=len(PARTIES),
    dtype=torch.float32
)

class TextDataset(Dataset):
    def __init__(self, items):
        self.items = items
    def __len__(self):
        return len(self.items)
    def __getitem__(self, idx):
        return self.items[idx]

def collate_fn(batch):
    texts = [item["text"] for item in batch]
    labels = torch.tensor([label_of[item["party"]] for item in batch], dtype=torch.long)
    weights = torch.tensor([item["weight"] for item in batch], dtype=torch.float)
    enc = tokenizer(texts, padding=True, truncation=True, max_length=128, return_tensors="pt")
    enc["labels"] = labels
    enc["weights"] = weights
    return enc

train_loader = DataLoader(TextDataset(train_items), batch_size=32, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(TextDataset(val_items), batch_size=64, shuffle=False, collate_fn=collate_fn)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using training device:", device)
model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
epochs = 4

print("Starting Fine-Tuning mDeBERTa-v3 on GPU VM...")
best_f1 = 0.0

for epoch in range(epochs):
    model.train()
    total_loss = 0.0
    start_ep = time.time()
    
    for step, batch in enumerate(train_loader):
        optimizer.zero_grad()
        labels = batch["labels"].to(device)
        weights = batch["weights"].to(device)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits.float()
        
        # Weighted Cross-Entropy Loss
        ce_loss_raw = F.cross_entropy(logits, labels, reduction="none")
        loss = (ce_loss_raw * weights).mean()
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
        
    # Validation evaluation
    model.eval()
    val_preds = []
    val_targets = []
    with torch.no_grad():
        for batch in val_loader:
            labels = batch["labels"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            preds = torch.argmax(logits, dim=-1)
            val_preds.extend(preds.cpu().tolist())
            val_targets.extend(labels.cpu().tolist())
            
    val_acc = accuracy_score(val_targets, val_preds)
    val_f1 = f1_score(val_targets, val_preds, average="macro")
    print(f"Epoch {epoch+1}/{epochs} ({time.time()-start_ep:.1f}s) - Loss: {total_loss/len(train_loader):.4f} | Val Acc: {val_acc*100:.2f}% | Val Macro F1: {val_f1:.4f}")
    
    if val_f1 > best_f1:
        best_f1 = val_f1
        model.save_pretrained(str(OUTPUT_DIR))
        tokenizer.save_pretrained(str(OUTPUT_DIR))
        print("  -> New best model checkpoint saved!")

print("\n=== FINAL HELD-OUT VALIDATION REPORT ===")
print(classification_report(val_targets, val_preds, target_names=PARTIES))

(OUTPUT_DIR / "model_config.json").write_text(json.dumps({
    "base_model": BASE_MODEL,
    "architecture": "mDeBERTa-v3 Multi-Party Sequence Classifier",
    "classes": PARTIES,
    "val_accuracy": float(val_acc),
    "val_macro_f1": float(best_f1),
    "num_samples": len(samples)
}, indent=2))
print("Finished!")

# Continue training for 3 more epochs with lower LR
print("\n--- Phase 2: Iterative Fine-Tuning with lr=1e-5 ---")
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=0.01)
for epoch in range(4, 7):
    model.train()
    total_loss = 0.0
    start_ep = time.time()
    for step, batch in enumerate(train_loader):
        optimizer.zero_grad()
        labels = batch["labels"].to(device)
        weights = batch["weights"].to(device)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits.float()
        ce_loss_raw = F.cross_entropy(logits, labels, reduction="none")
        loss = (ce_loss_raw * weights).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
        
    model.eval()
    val_preds = []
    val_targets = []
    with torch.no_grad():
        for batch in val_loader:
            labels = batch["labels"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            preds = torch.argmax(logits, dim=-1)
            val_preds.extend(preds.cpu().tolist())
            val_targets.extend(labels.cpu().tolist())
    val_acc = accuracy_score(val_targets, val_preds)
    val_f1 = f1_score(val_targets, val_preds, average="macro")
    print(f"Epoch {epoch+1}/7 ({time.time()-start_ep:.1f}s) - Loss: {total_loss/len(train_loader):.4f} | Val Acc: {val_acc*100:.2f}% | Val Macro F1: {val_f1:.4f}")
    if val_f1 > best_f1:
        best_f1 = val_f1
        model.save_pretrained(str(OUTPUT_DIR))
        tokenizer.save_pretrained(str(OUTPUT_DIR))
        print("  -> New best model checkpoint saved!")

print("\n=== FINAL REFINED REPORT ===")
print(classification_report(val_targets, val_preds, target_names=PARTIES))
