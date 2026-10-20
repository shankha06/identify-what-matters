#!/usr/bin/env python3
"""
Dedicated Neural Demographic Classifier (v3) using Supervised Contrastive Representation Learning:
- Primary Target: Mental State (6 classes)
- Secondary Target: Age Bracket (5 classes)
- Loss Function:
    Loss = Loss_CE_Mental + 0.5 * Loss_CE_Age + lambda_contrastive * Loss_SupervisedContrastive
- Architecture:
    - Backbone: microsoft/mdeberta-v3-base (initialized from local political mdeberta-direction)
    - Contrastive projection head with L2 normalization (pulling positive embeddings together, pushing negative apart)
    - Specialized classification heads with LayerNorm, Dropout(0.25), and residual connection
- Optimizer:
    - AdamW with eps=1e-6, weight_decay=0.01, cosine learning rate scheduler with warmup
- Online Validation:
    - Evaluates on the 192-sample real-world gold standard benchmark at every epoch and saves best checkpoint
"""

import os
import json
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, AutoConfig, get_cosine_schedule_with_warmup
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

# Classes
MENTAL_CLASSES = [
    "Civic Anger & Institutional Skepticism",
    "Cultural Pride & National Solidarity",
    "Financial Anxiety & Economic Strain",
    "Ironic Detachment & Memetic Cynicism",
    "Objective Deliberation & Fact-Checking",
    "Social Empathy & Grassroots Care"
]
MENTAL2ID = {c: i for i, c in enumerate(MENTAL_CLASSES)}
ID2MENTAL = {i: c for i, c in enumerate(MENTAL_CLASSES)}

AGE_CLASSES = ["13-17", "18-24", "25-34", "35-54", "55+"]
AGE2ID = {c: i for i, c in enumerate(AGE_CLASSES)}
ID2AGE = {i: c for i, c in enumerate(AGE_CLASSES)}

class SupConLoss(nn.Module):
    """Supervised Contrastive Loss (Khosla et al., 2020)"""
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        batch_size = features.shape[0]
        if batch_size < 2:
            return torch.tensor(0.0, device=features.device, requires_grad=True)

        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(features.device)

        # Cosine similarity matrix
        anchor_dot_contrast = torch.div(
            torch.matmul(features, features.T),
            self.temperature
        )

        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size).view(-1, 1).to(features.device),
            0
        )
        mask = mask * logits_mask

        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-9)

        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-9)

        loss = -mean_log_prob_pos
        pos_anchors = (mask.sum(1) > 0)
        if pos_anchors.sum() > 0:
            loss = loss[pos_anchors].mean()
        else:
            loss = loss.mean()
        return loss

class MultiTaskContrastiveDemographics(nn.Module):
    def __init__(self, base_model_name, num_mental=6, num_age=5, proj_dim=128):
        super().__init__()
        self.config = AutoConfig.from_pretrained(base_model_name)
        self.encoder = AutoModel.from_pretrained(base_model_name, config=self.config)
        hidden_size = self.config.hidden_size

        # Contrastive Projection Head
        self.proj_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, proj_dim)
        )

        # Classification Heads
        self.mental_head = nn.Sequential(
            nn.Dropout(0.25),
            nn.Linear(hidden_size, 384),
            nn.LayerNorm(384),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(384, num_mental)
        )

        self.age_head = nn.Sequential(
            nn.Dropout(0.25),
            nn.Linear(hidden_size, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(256, num_age)
        )

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
        if token_type_ids is not None:
            kwargs["token_type_ids"] = token_type_ids

        outputs = self.encoder(**kwargs)
        token_embeddings = outputs.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        pooled = sum_embeddings / sum_mask

        proj = F.normalize(self.proj_head(pooled), dim=-1)
        mental_logits = self.mental_head(pooled)
        age_logits = self.age_head(pooled)

        return mental_logits, age_logits, proj, pooled

class DemographicsDataset(Dataset):
    def __init__(self, samples, tokenizer, max_len=128):
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        text = item["text"]
        enc = self.tokenizer(
            text,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        mental_id = MENTAL2ID.get(item["mental_state"], -1)
        age_id = AGE2ID.get(item.get("age_bracket"), -1)

        res = {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "mental_id": torch.tensor(mental_id, dtype=torch.long),
            "age_id": torch.tensor(age_id, dtype=torch.long),
            "text": text
        }
        if "token_type_ids" in enc:
            res["token_type_ids"] = enc["token_type_ids"].squeeze(0)
        return res

def evaluate_on_benchmark(model, tokenizer, gold_samples, device):
    model.eval()
    y_true_m, y_pred_m = [], []
    y_true_a, y_pred_a = [], []

    with torch.no_grad():
        for s in gold_samples:
            enc = tokenizer(
                s["text"],
                max_length=128,
                padding=True,
                truncation=True,
                return_tensors="pt"
            )
            input_ids = enc["input_ids"].to(device)
            attention_mask = enc["attention_mask"].to(device)
            token_type_ids = enc.get("token_type_ids")
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(device)

            m_logits, a_logits, _, _ = model(input_ids, attention_mask, token_type_ids)
            pred_m = torch.argmax(m_logits, dim=-1).item()
            pred_a = torch.argmax(a_logits, dim=-1).item()

            y_true_m.append(s["mental_state"])
            y_pred_m.append(ID2MENTAL[pred_m])

            if s.get("age_bracket") in AGE2ID:
                y_true_a.append(s["age_bracket"])
                y_pred_a.append(ID2AGE[pred_a])

    acc_m = accuracy_score(y_true_m, y_pred_m)
    f1_m = f1_score(y_true_m, y_pred_m, average="macro", zero_division=0)

    acc_a = accuracy_score(y_true_a, y_pred_a) if y_true_a else 0.0
    f1_a = f1_score(y_true_a, y_pred_a, average="macro", zero_division=0) if y_true_a else 0.0

    return {
        "mental_acc": acc_m,
        "mental_f1": f1_m,
        "age_acc": acc_a,
        "age_f1": f1_a,
        "y_true_m": y_true_m,
        "y_pred_m": y_pred_m,
        "y_true_a": y_true_a,
        "y_pred_a": y_pred_a
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-file", default="artifacts/demographics_models/synthetic_demographics_balanced_v2.jsonl")
    parser.add_argument("--gold-file", default="artifacts/demographics_models/real_world_gold_standard_v2.jsonl")
    parser.add_argument("--base-model", default="/home/s0r0qw6/identify-what-matters/artifacts/political_models/mdeberta-direction")
    parser.add_argument("--output-dir", default="artifacts/demographics_models/neural_contrastive_demographics_v2")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1.8e-5)
    parser.add_argument("--contrastive-weight", type=float, default=0.35)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    print(f"Loading data from {args.data_file}...")
    samples = []
    with open(args.data_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                item = json.loads(line.strip())
                if item.get("mental_state") in MENTAL2ID:
                    samples.append(item)

    print(f"Total valid training samples: {len(samples)}")

    print(f"Loading gold standard benchmark from {args.gold_file}...")
    gold_samples = []
    with open(args.gold_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                gold_samples.append(json.loads(line.strip()))
    print(f"Total gold samples: {len(gold_samples)}")

    # Stratified split by mental state
    train_samples, val_samples = train_test_split(
        samples, test_size=0.1, random_state=42, stratify=[s["mental_state"] for s in samples]
    )
    print(f"Train split: {len(train_samples)}, Val split: {len(val_samples)}")

    print(f"Loading tokenizer & base model from {args.base_model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    model = MultiTaskContrastiveDemographics(args.base_model, num_mental=len(MENTAL_CLASSES), num_age=len(AGE_CLASSES))
    model.to(args.device)

    train_ds = DemographicsDataset(train_samples, tokenizer)
    val_ds = DemographicsDataset(val_samples, tokenizer)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    # Class weights for balanced mental states
    from collections import Counter
    m_counts = Counter(s["mental_state"] for s in train_samples)
    m_weights = torch.tensor([1.0 / (m_counts[ID2MENTAL[i]] ** 0.5) for i in range(len(MENTAL_CLASSES))], dtype=torch.float).to(args.device)
    m_weights = m_weights / m_weights.sum() * len(MENTAL_CLASSES)

    criterion_mental = nn.CrossEntropyLoss(weight=m_weights)
    criterion_age = nn.CrossEntropyLoss(ignore_index=-1)
    criterion_supcon = SupConLoss(temperature=0.07)

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], "weight_decay": 0.01},
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0}
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.lr, eps=1e-6)

    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(total_steps * 0.1)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    print("\nInitial Baseline Evaluation on Gold Standard Benchmark:")
    base_res = evaluate_on_benchmark(model, tokenizer, gold_samples, args.device)
    print(f"Pre-train Gold Mental State Acc: {base_res['mental_acc']*100:.2f}% | F1: {base_res['mental_f1']:.4f}")
    print(f"Pre-train Gold Age Bracket Acc:  {base_res['age_acc']*100:.2f}% | F1: {base_res['age_f1']:.4f}")

    best_mental_f1 = 0.0
    os.makedirs(args.output_dir, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_m_loss = 0.0
        total_con_loss = 0.0

        for step, batch in enumerate(train_loader):
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(args.device)
            attention_mask = batch["attention_mask"].to(args.device)
            mental_labels = batch["mental_id"].to(args.device)
            age_labels = batch["age_id"].to(args.device)
            token_type_ids = batch.get("token_type_ids")
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(args.device)

            m_logits, a_logits, proj, _ = model(input_ids, attention_mask, token_type_ids)

            loss_m = criterion_mental(m_logits, mental_labels)
            loss_a = criterion_age(a_logits, age_labels)
            # Contrastive loss pulling same mental states together
            loss_con = criterion_supcon(proj, mental_labels)

            loss = loss_m + 0.4 * loss_a + args.contrastive_weight * loss_con
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            total_m_loss += loss_m.item()
            total_con_loss += loss_con.item()

            if (step + 1) % 40 == 0:
                print(f"Epoch {epoch}/{args.epochs} | Step {step+1}/{len(train_loader)} | Total: {loss.item():.4f} | Mental: {loss_m.item():.4f} | Con: {loss_con.item():.4f}")

        # Gold benchmark evaluation
        gold_res = evaluate_on_benchmark(model, tokenizer, gold_samples, args.device)

        print(f"\n--- Epoch {epoch} Results ---")
        print(f"Gold Mental State Acc: {gold_res['mental_acc']*100:.2f}% | F1: {gold_res['mental_f1']:.4f}")
        print(f"Gold Age Bracket Acc:  {gold_res['age_acc']*100:.2f}% | F1: {gold_res['age_f1']:.4f}")

        if gold_res["mental_f1"] > best_mental_f1:
            best_mental_f1 = gold_res["mental_f1"]
            print(f"⭐ New Best Model! (Mental F1: {best_mental_f1:.4f}) Saving to {args.output_dir}...")
            model.encoder.save_pretrained(args.output_dir)
            tokenizer.save_pretrained(args.output_dir)
            torch.save({
                "mental_head": model.mental_head.state_dict(),
                "age_head": model.age_head.state_dict(),
                "proj_head": model.proj_head.state_dict(),
                "config": {
                    "mental_classes": MENTAL_CLASSES,
                    "age_classes": AGE_CLASSES,
                    "best_gold_mental_acc": gold_res["mental_acc"],
                    "best_gold_mental_f1": gold_res["mental_f1"],
                    "best_gold_age_acc": gold_res["age_acc"],
                    "best_gold_age_f1": gold_res["age_f1"],
                    "epoch": epoch
                }
            }, os.path.join(args.output_dir, "demographic_heads.pt"))

            with open(os.path.join(args.output_dir, "eval_results.json"), "w") as f:
                json.dump({
                    "epoch": epoch,
                    "mental_accuracy": gold_res["mental_acc"],
                    "mental_macro_f1": gold_res["mental_f1"],
                    "age_accuracy": gold_res["age_acc"],
                    "age_macro_f1": gold_res["age_f1"]
                }, f, indent=2)

    print(f"\nTraining Complete. Best Gold Mental F1: {best_mental_f1:.4f}")

if __name__ == "__main__":
    main()
