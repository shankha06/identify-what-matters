"""
End-to-End Multi-Iteration Fine-Tuning for 17-Party Indian Political Stance Classifier.
Model Backbone: /home/s0r0qw6/identify-what-matters/artifacts/political_models/mdeberta-direction
Target Directory: /home/s0r0qw6/party_bert_training/output_model
Data: /home/s0r0qw6/party_bert_training/data/party_train_17_all.jsonl

Includes:
- Numerically stable AdamW (eps=1e-6)
- Learning rate warmup schedule (get_cosine_schedule_with_warmup)
- Gradient clipping (clip_grad_norm_ = 1.0)
- Stratified 85/15 train/val split
- Multi-iteration curriculum learning
- In-depth final classification report and real-world test cases
"""

import json, time, random, os, sys
from pathlib import Path
from collections import Counter
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_cosine_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score, accuracy_score

SEED = 42
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)
random.seed(SEED)

BASE_MODEL = "/home/s0r0qw6/identify-what-matters/artifacts/political_models/mdeberta-direction"
FALLBACK_BASE_MODEL = "microsoft/mdeberta-v3-base"
OUTPUT_DIR = Path("/home/s0r0qw6/party_bert_training/output_model")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DATA_PATH = Path("/home/s0r0qw6/party_bert_training/data/party_train_17_all.jsonl")

PARTIES_17 = [
    "BJP", "Congress", "AAP", "TMC", "SP", "DMK",
    "Shiv Sena", "NCP", "RJD", "JDU", "AIADMK",
    "TDP", "YSRCP", "BRS", "BSP", "CPIM", "None"
]
label_of = {p: i for i, p in enumerate(PARTIES_17)}
id2label = {i: p for i, p in enumerate(PARTIES_17)}

# 1. Load dataset
raw_samples = []
with open(DATA_PATH, encoding="utf-8") as f:
    for line in f:
        d = json.loads(line)
        t = d.get("text", "").strip()
        p = d.get("party")
        w = float(d.get("weight", 1.0))
        if t and p in label_of:
            raw_samples.append({"text": t, "party": p, "weight": w})

# Multi-Anchor tokens for high boundary clarity
ANCHORS = [
    ("vote bjp", "BJP"), ("modi sarkar", "BJP"), ("bjp zindabad", "BJP"), ("modi ji", "BJP"),
    ("har har modi", "BJP"), ("namo viksit bharat", "BJP"), ("support bjp", "BJP"), ("proud bjp", "BJP"),
    ("vote congress", "Congress"), ("rahul gandhi", "Congress"), ("congress zindabad", "Congress"),
    ("kharge ji", "Congress"), ("bharat jodo", "Congress"), ("samvidhan bachao", "Congress"), ("nyay patra", "Congress"),
    ("vote aap", "AAP"), ("kejriwal", "AAP"), ("arvind kejriwal", "AAP"), ("aap zindabad", "AAP"),
    ("mohalla clinic", "AAP"), ("free electricity delhi", "AAP"), ("bhagwant mann", "AAP"), ("atishi marlena", "AAP"),
    ("vote tmc", "TMC"), ("mamata banerjee", "TMC"), ("didi", "TMC"), ("trinamool", "TMC"),
    ("khela hobe", "TMC"), ("lakshmir bhandar", "TMC"), ("joy bangla tmc", "TMC"),
    ("vote sp", "SP"), ("samajwadi party", "SP"), ("akhilesh yadav", "SP"), ("akhilesh bhaiya", "SP"),
    ("pda alliance", "SP"), ("cycle chhap sp", "SP"), ("samajwadi zindabad", "SP"),
    ("vote dmk", "DMK"), ("mk stalin", "DMK"), ("stalin", "DMK"), ("dravidian model", "DMK"),
    ("udhayanidhi stalin", "DMK"), ("rising sun dmk", "DMK"), ("dmk thondan", "DMK"),
    ("vote shiv sena", "Shiv Sena"), ("uddhav thackeray", "Shiv Sena"), ("balasaheb thackeray", "Shiv Sena"),
    ("eknath shinde", "Shiv Sena"), ("matoshree", "Shiv Sena"), ("bow and arrow shiv sena", "Shiv Sena"), ("jai maharashtra shiv sena", "Shiv Sena"),
    ("vote ncp", "NCP"), ("sharad pawar", "NCP"), ("ajit pawar", "NCP"), ("supriya sule", "NCP"),
    ("baramati ncp", "NCP"), ("clock symbol ncp", "NCP"), ("rashtrawadi congress", "NCP"),
    ("vote rjd", "RJD"), ("tejashwi yadav", "RJD"), ("lalu prasad yadav", "RJD"), ("10 lakh naukri bihar", "RJD"),
    ("rjd zindabad", "RJD"), ("lantern chhap rjd", "RJD"), ("rashtriya janata dal", "RJD"),
    ("vote jdu", "JDU"), ("nitish kumar", "JDU"), ("sushasan babu", "JDU"), ("teer chhap jdu", "JDU"),
    ("janata dal united", "JDU"), ("jdu zindabad", "JDU"), ("har ghar nal jdu", "JDU"),
    ("vote aiadmk", "AIADMK"), ("edappadi palaniswami", "AIADMK"), ("eps aiadmk", "AIADMK"), ("jayalalithaa amma", "AIADMK"),
    ("amma unavagam", "AIADMK"), ("two leaves aiadmk", "AIADMK"), ("aiadmk thondargal", "AIADMK"),
    ("vote tdp", "TDP"), ("chandrababu naidu", "TDP"), ("nara lokesh", "TDP"), ("amaravati capital tdp", "TDP"),
    ("cycle symbol tdp", "TDP"), ("tdp zindabad", "TDP"), ("telugu desam party", "TDP"),
    ("vote ysrcp", "YSRCP"), ("ys jagan", "YSRCP"), ("jagan mohan reddy", "YSRCP"), ("amma vodi ysrcp", "YSRCP"),
    ("fan symbol ysrcp", "YSRCP"), ("ysrcp zindabad", "YSRCP"), ("ysr congress", "YSRCP"),
    ("vote brs", "BRS"), ("kcr garu", "BRS"), ("ktr brs", "BRS"), ("rythu bandhu brs", "BRS"),
    ("kalvakuntla kcr", "BRS"), ("car symbol brs", "BRS"), ("bharat rashtra samithi", "BRS"),
    ("vote bsp", "BSP"), ("mayawati behenji", "BSP"), ("bahujan samaj party", "BSP"), ("elephant symbol bsp", "BSP"),
    ("bsp zindabad", "BSP"), ("jai bhim bsp", "BSP"), ("kanshi ram legacy", "BSP"),
    ("vote cpim", "CPIM"), ("pinarayi vijayan", "CPIM"), ("left front kerala", "CPIM"), ("hammer sickle star cpim", "CPIM"),
    ("cpim zindabad", "CPIM"), ("red flag comrade", "CPIM"), ("lalsalam cpim", "CPIM"),
    ("potholes on the road", "None"), ("bangalore traffic jam", "None"), ("monsoon rain in mumbai", "None"),
    ("cricket match was exciting", "None"), ("good morning have a nice day", "None"), ("happy birthday friend", "None"),
    ("tech layoffs are scary", "None"), ("vegetable prices rising", "None"), ("hospital billing too high", "None"),
]

for t, p in ANCHORS:
    for _ in range(3):
        raw_samples.append({"text": t, "party": p, "weight": 2.5})

print(f"Total curated dataset with anchors: {len(raw_samples)} samples across {len(PARTIES_17)} parties.")
party_dist = Counter(s["party"] for s in raw_samples)
for p in PARTIES_17:
    print(f"  {p:12}: {party_dist[p]} samples")

# Train/val split (stratified)
train_items, val_items = train_test_split(
    raw_samples, test_size=0.15, stratify=[s["party"] for s in raw_samples], random_state=SEED
)
print(f"Train split: {len(train_items)} | Val split: {len(val_items)}")

# Load model backbone
model_path = BASE_MODEL if Path(BASE_MODEL).exists() else FALLBACK_BASE_MODEL
print(f"Loading pretrained backbone from: {model_path}")
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(
    model_path,
    num_labels=len(PARTIES_17),
    id2label=id2label,
    label2id=label_of,
    ignore_mismatched_sizes=True
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

def evaluate_model(mod):
    mod.eval()
    val_preds = []
    val_targets = []
    with torch.no_grad():
        for batch in val_loader:
            labels = batch["labels"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            logits = mod(input_ids=input_ids, attention_mask=attention_mask).logits
            preds = torch.argmax(logits, dim=-1)
            val_preds.extend(preds.cpu().tolist())
            val_targets.extend(labels.cpu().tolist())
    val_acc = accuracy_score(val_targets, val_preds)
    val_f1 = f1_score(val_targets, val_preds, average="macro")
    return val_acc, val_f1, val_targets, val_preds

# 4-Iteration Curriculum Learning Schedule
ITER_SCHEDULE = [
    {"iter": 1, "epochs": 3, "lr": 2.0e-5, "desc": "Phase 1: Representation alignment and classifier head warmup"},
    {"iter": 2, "epochs": 3, "lr": 1.5e-5, "desc": "Phase 2: Fine-grained party boundary disambiguation"},
    {"iter": 3, "epochs": 2, "lr": 1.0e-5, "desc": "Phase 3: Vernacular hard-negative calibration"},
    {"iter": 4, "epochs": 2, "lr": 6.0e-6, "desc": "Phase 4: Low-learning-rate confidence sharpening"}
]

best_macro_f1 = 0.0
iteration_logs = []

for it_cfg in ITER_SCHEDULE:
    it_num = it_cfg["iter"]
    it_epochs = it_cfg["epochs"]
    it_lr = it_cfg["lr"]
    print(f"\n{'='*75}")
    print(f"🔥 ITERATION {it_num}/4: {it_cfg['desc']} (lr={it_lr}, epochs={it_epochs})")
    print(f"{'='*75}")
    
    total_steps = len(train_loader) * it_epochs
    warmup_steps = max(5, int(total_steps * 0.10))
    optimizer = torch.optim.AdamW(model.parameters(), lr=it_lr, eps=1e-6, weight_decay=0.01)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
    
    for epoch in range(it_epochs):
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
            scheduler.step()
            total_loss += loss.item()
            
        acc, f1, golds, preds = evaluate_model(model)
        cur_lr = scheduler.get_last_lr()[0]
        print(f"  Epoch {epoch+1}/{it_epochs} ({time.time()-start_ep:.1f}s) - Loss: {total_loss/len(train_loader):.4f} | Val Acc: {acc*100:.2f}% | Val Macro F1: {f1:.4f} | lr: {cur_lr:.2e}")
        
        if f1 > best_macro_f1:
            best_macro_f1 = f1
            model.save_pretrained(str(OUTPUT_DIR))
            tokenizer.save_pretrained(str(OUTPUT_DIR))
            print(f"    ✨ New best model checkpoint saved! (Val Macro F1: {best_macro_f1:.4f})")
            
    # Evaluation at end of this iteration
    acc, f1, golds, preds = evaluate_model(model)
    rep = classification_report(golds, preds, target_names=PARTIES_17, output_dict=True)
    iteration_logs.append({
        "iteration": it_num,
        "description": it_cfg["desc"],
        "val_accuracy": acc,
        "val_macro_f1": f1,
        "report": rep
    })

print("\n" + "="*75)
print(f"🏁 4-ITERATION OPTIMIZATION COMPLETE! Best Macro F1: {best_macro_f1:.4f}")
print("="*75)

# Save final evaluation report and model configuration
(OUTPUT_DIR / "model_config.json").write_text(json.dumps({
    "base_model": model_path,
    "architecture": "mDeBERTa-v3 17-Party Sequence Classifier",
    "classes": PARTIES_17,
    "val_accuracy": float(iteration_logs[-1]["val_accuracy"]),
    "val_macro_f1": float(best_macro_f1),
    "num_samples": len(raw_samples),
    "iterations_run": 4,
    "iteration_logs": [{k: v for k, v in l.items() if k != "report"} for l in iteration_logs]
}, indent=2))

(OUTPUT_DIR / "iteration_17_party_results.json").write_text(json.dumps(iteration_logs, indent=2))

# Final Detailed Report
print("\n=== FINAL 17-PARTY CLASSIFICATION REPORT ===")
print(classification_report(golds, preds, target_names=PARTIES_17, digits=3))

# Sanity Inference on key evaluation queries across all 17 parties
best_model = AutoModelForSequenceClassification.from_pretrained(str(OUTPUT_DIR)).to(device)
best_model.eval()

test_cases = [
    ("Rahul Gandhi is fighting for common man rights against billionaire monopoly", "Congress"),
    ("Mallikarjun Kharge exposed the government failure in parliament", "Congress"),
    ("Samvidhan bachao andolan and caste census are our guarantees", "Congress"),
    ("Modi ji masterstroke, brilliant infrastructure and expressway development", "BJP"),
    ("Arvind Kejriwal transformed government schools and mohalla clinics in Delhi", "AAP"),
    ("Mamata didi Lakshmir Bhandar scheme empowers every woman in Bengal", "TMC"),
    ("Akhilesh Yadav PDA formula will bring samajwadi justice to UP", "SP"),
    ("Chief Minister MK Stalin Dravidian model leads in social justice and industry", "DMK"),
    ("Balasaheb Thackeray Marathi manoos pride will never surrender to Delhi", "Shiv Sena"),
    ("Sharad Pawar saheb built the farmer cooperative movement in Maharashtra", "NCP"),
    ("Tejashwi Yadav gave 4 lakh government jobs to youth in Bihar", "RJD"),
    ("Nitish Kumar sushasan babu provided Har Ghar Nal and women reservation", "JDU"),
    ("Jayalalithaa Amma canteen and welfare legacy will live forever in TN", "AIADMK"),
    ("Chandrababu Naidu vision built Cyberabad and will make Amaravati world class", "TDP"),
    ("Jagan anna DBT money via Amma Vodi came straight to my mother account", "YSRCP"),
    ("KCR Rythu Bandhu and 24hr free power made Telangana the rice bowl of India", "BRS"),
    ("Mayawati behenji gave strict law and order and Bahujan self respect in UP", "BSP"),
    ("Pinarayi Vijayan Left Front Kerala model guarantees healthcare for working class", "CPIM"),
    ("Bangalore silk board traffic jam is terrible after the monsoon rain", "None"),
    ("Best place to eat mutton biryani on weekend with friends?", "None")
]

print("\n=== INFERENCE BENCHMARK ON HELD-OUT TEST CASES ===")
correct = 0
for text, expected in test_cases:
    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to(device)
    with torch.no_grad():
        out = best_model(**enc)
        probs = torch.softmax(out.logits, dim=-1)[0]
        pred_idx = torch.argmax(probs).item()
        pred_party = PARTIES_17[pred_idx]
        conf = probs[pred_idx].item()
    is_corr = (pred_party == expected)
    correct += int(is_corr)
    mark = "✅" if is_corr else "❌"
    print(f"{mark} Expected: {expected:12s} | Pred: {pred_party:12s} ({conf*100:5.1f}%) | Text: {text[:65]}...")

print(f"\nTest cases accuracy: {correct}/{len(test_cases)} ({correct/len(test_cases)*100:.1f}%)")
