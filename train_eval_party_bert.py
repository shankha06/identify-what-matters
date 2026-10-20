import json, time, random, re
from pathlib import Path
from collections import Counter
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sklearn.metrics import classification_report, f1_score, accuracy_score
from sklearn.model_selection import train_test_split

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

BASE_MODEL = "artifacts/political_models/mdeberta-direction"
OUTPUT_DIR = Path("artifacts/demographics_models/bert-party-support")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PARTIES = ["BJP", "Congress", "AAP", "TMC", "SP", "DMK", "None"]
label_of = {p: i for i, p in enumerate(PARTIES)}

# 1. Load Gold Datasets
samples = []
# (a) New gold samples
with open("artifacts/demographics_models/party_gold_corpus/gold_samples.jsonl") as f:
    for line in f:
        d = json.loads(line)
        t = d.get("text", "").strip()
        p = d.get("party_support")
        if t and p in label_of:
            samples.append({"text": t, "party": p, "weight": 3.0})

# (b) Party lean corpus
import glob
for p_file in glob.glob("artifacts/demographics_models/party_lean_corpus/*.jsonl"):
    with open(p_file) as f:
        for line in f:
            d = json.loads(line)
            t = d.get("text", "").strip()
            p = d.get("party_support")
            if t and p in label_of:
                samples.append({"text": t, "party": p, "weight": 2.5})

# (c) Clean synthetic samples (balanced max 350 per party)
syn_by_party = {p: [] for p in PARTIES}
with open("artifacts/demographics_models/synthetic_dataset.jsonl") as f:
    for line in f:
        d = json.loads(line)
        t = d.get("text", "").strip()
        p = d.get("party_support")
        if t and p in label_of and len(t) >= 15:
            syn_by_party[p].append(t)

for p in PARTIES:
    picked = random.sample(syn_by_party[p], min(len(syn_by_party[p]), 350))
    for t in picked:
        samples.append({"text": t, "party": p, "weight": 1.0})

# (d) Critical anchor samples for unambiguous party tokens
ANCHORS = [
    ("bjp", "BJP"), ("vote bjp", "BJP"), ("modi sarkar", "BJP"), ("bjp zindabad", "BJP"), ("modi ji", "BJP"),
    ("har har modi", "BJP"), ("namo", "BJP"), ("bhagwa", "BJP"), ("support bjp", "BJP"), ("proud bjp", "BJP"),
    ("congress", "Congress"), ("vote congress", "Congress"), ("rahul gandhi", "Congress"), ("congress zindabad", "Congress"),
    ("kharge ji", "Congress"), ("inc india", "Congress"), ("bharat jodo", "Congress"), ("samvidhan bachao", "Congress"),
    ("aap", "AAP"), ("vote aap", "AAP"), ("kejriwal", "AAP"), ("arvind kejriwal", "AAP"), ("aap zindabad", "AAP"),
    ("mohalla clinic", "AAP"), ("free electricity delhi", "AAP"), ("bhagwant mann", "AAP"),
    ("tmc", "TMC"), ("vote tmc", "TMC"), ("mamata banerjee", "TMC"), ("didi", "TMC"), ("trinamool", "TMC"),
    ("khela hobe", "TMC"), ("lakshmir bhandar", "TMC"), ("bengal development", "TMC"),
    ("sp", "SP"), ("vote sp", "SP"), ("samajwadi party", "SP"), ("akhilesh yadav", "SP"), ("akhilesh bhaiya", "SP"),
    ("pda alliance", "SP"), ("cycle chhap", "SP"),
    ("dmk", "DMK"), ("vote dmk", "DMK"), ("mk stalin", "DMK"), ("stalin", "DMK"), ("dravidian model", "DMK"),
    ("udhayanidhi", "DMK"), ("rising sun dmk", "DMK"),
    ("potholes on the road", "None"), ("good morning have a great day", "None"), ("happy birthday", "None"),
    ("cricket match was amazing", "None"), ("bangalore weather is nice", "None"), ("monsoon rain started", "None"),
    ("movie was entertaining", "None"), ("delicious biryani recipe", "None"), ("tech industry recession", "None"),
]
for t, p in ANCHORS:
    for _ in range(5):
        samples.append({"text": t, "party": p, "weight": 4.0})

print(f"Total curated training samples: {len(samples)}")
counts = Counter(s["party"] for s in samples)
for p in PARTIES:
    print(f"  {p:10}: {counts[p]} samples")

# Split train / test stratified
train_items, test_items = train_test_split(samples, test_size=0.15, stratify=[s["party"] for s in samples], random_state=SEED)
print(f"Train split: {len(train_items)} | Test split: {len(test_items)}")

# Tokenizer & Model
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
model = AutoModelForSequenceClassification.from_pretrained(
    BASE_MODEL,
    num_labels=len(PARTIES),
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
    enc = tokenizer(texts, padding=True, truncation=True, max_length=128, return_tensors="pt")
    enc["labels"] = labels
    return enc

train_loader = DataLoader(TextDataset(train_items), batch_size=16, shuffle=True, collate_fn=collate_fn)
test_loader = DataLoader(TextDataset(test_items), batch_size=32, shuffle=False, collate_fn=collate_fn)

device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
model.to(device)

class_counts = [sum(s["party"] == p for s in train_items) for p in PARTIES]
total_tr = len(train_items)
class_weights = torch.tensor([total_tr / (len(PARTIES) * max(c, 1)) for c in class_counts], device=device).float()

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5, weight_decay=0.01)
epochs = 4

print("Starting mDeBERTa fine-tuning for Party Support...")
start_t = time.time()
for epoch in range(epochs):
    model.train()
    total_loss = 0.0
    for step, batch in enumerate(train_loader):
        optimizer.zero_grad()
        labels = batch.pop("labels").to(device)
        inputs = {k: v.to(device) for k, v in batch.items()}
        logits = model(**inputs).logits
        loss = torch.nn.functional.cross_entropy(logits.float(), labels, weight=class_weights)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")

print(f"Training completed in {time.time() - start_t:.1f}s")

# Save final model
model.save_pretrained(str(OUTPUT_DIR))
tokenizer.save_pretrained(str(OUTPUT_DIR))

# Save metadata
(OUTPUT_DIR / "training_run.json").write_text(json.dumps({
    "task": "party_support",
    "base_model": "mdeberta-v3-base-xnli",
    "classes": PARTIES,
    "train_rows": len(train_items),
    "test_rows": len(test_items),
    "epochs": epochs,
    "device": device,
    "classes_distribution": dict(counts)
}, indent=2))

# Evaluation on held-out test split
model.eval()
all_preds = []
all_golds = []

with torch.no_grad():
    for batch in test_loader:
        labels = batch.pop("labels")
        inputs = {k: v.to(device) for k, v in batch.items()}
        logits = model(**inputs).logits
        preds = logits.argmax(dim=-1).cpu().numpy()
        all_preds.extend(preds)
        all_golds.extend(labels.numpy())

acc = accuracy_score(all_golds, all_preds)
macro_f1_score = f1_score(all_golds, all_preds, average="macro")

print("\n" + "="*50)
print(f"EVALUATION RESULTS (Held-out Test Set n={len(all_golds)}):")
print(f"Accuracy: {acc*100:.2f}% | Macro F1: {macro_f1_score:.4f}")
print("="*50)
print(classification_report(all_golds, all_preds, target_names=PARTIES, digits=3))

# Sanity Check Wild Inputs
test_queries = [
    "vote bjp",
    "modi sarkar zindabad",
    "rahul gandhi bharat jodo yatra zindabad",
    "kejriwal ke mohalla clinic best hai",
    "didi rockz in bengal, joy bangla",
    "akhilesh bhaiya will bring socialist change in up",
    "stalin dravidian model is super in tn",
    "monsoon weather in pune is really rainy today"
]

print("\n--- SANITY CHECK PREDICTIONS ---")
for q in test_queries:
    inputs = tokenizer([q], return_tensors="pt", truncation=True, max_length=128).to(device)
    with torch.no_grad():
        probs = torch.softmax(model(**inputs).logits, dim=-1).cpu().numpy()[0]
    best_idx = probs.argmax()
    print(f"'{q}' -> {PARTIES[best_idx]} (conf: {probs[best_idx]:.2f})")
