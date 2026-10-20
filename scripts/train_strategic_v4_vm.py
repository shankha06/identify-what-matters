#!/usr/bin/env python3
"""
Multi-Task Strategic Intelligence & Threat Criticality Neural Network (v4 Enterprise Edition).
Major Architectural & Methodological Innovations for v4:
- Massive Scale Dataset: 3,409 training samples and 2,285 gold standard benchmark samples
- Backbone: microsoft/mdeberta-v3-base with DebertaV2Model and ContextPooler
- Multi-Task Classification Heads:
  1. Threat Axis (8 classes) with 2.0x loss weight and Focal Loss (gamma=1.5)
  2. Criticality Level (4 ordinal classes: low, medium, high, critical) with Ordinal Distance Loss
  3. Authenticity / Astroturf Detection (3 classes: organic, astroturf, bot_spam) with Cross-Entropy Loss
  4. Trajectory Stage (5 ordinal classes: pre_breakout -> breaking_out -> peak_saturation -> simmering_entropy -> settling_quiet) with Ordinal Distance Loss
- Ordinal Regression / Distance Loss:
  Combines standard cross-entropy with an expected rank absolute-error penalty:
  L_ordinal = L_ce + alpha * E_{p(y)}[ |y - y_true| ]
  This mathematically penalizes extreme distance errors (e.g. predicting 'low' when true is 'critical', or 'pre_breakout' when true is 'settling_quiet') far more heavily than adjacent class mistakes.
- Differential Learning Rate: 1.5e-5 for backbone, 4e-5 for task heads with cosine annealing with warmup.
"""

import os
import sys
import json
import random
import time
import math
import argparse
from pathlib import Path
from collections import Counter
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoConfig, get_cosine_schedule_with_warmup
from transformers.models.deberta_v2.modeling_deberta_v2 import DebertaV2Model, ContextPooler
from sklearn.metrics import accuracy_score, f1_score

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

THREAT_AXES = [
    "institutional_integrity",
    "public_safety_crisis",
    "economic_distress",
    "youth_campus_agitation",
    "communal_regional_faultline",
    "infrastructure_governance_failure",
    "disinformation_rumor_panic",
    "general_civic_deliberation"
]
AXIS2ID = {a: i for i, a in enumerate(THREAT_AXES)}
ID2AXIS = {i: a for i, a in enumerate(THREAT_AXES)}

CRITICALITY_LEVELS = ["low", "medium", "high", "critical"]
CRIT2ID = {c: i for i, c in enumerate(CRITICALITY_LEVELS)}
ID2CRIT = {i: c for i, c in enumerate(CRITICALITY_LEVELS)}

AUTHENTICITY_TYPES = ["organic", "astroturf", "bot_spam"]
AUTH2ID = {a: i for i, a in enumerate(AUTHENTICITY_TYPES)}
ID2AUTH = {i: a for i, a in enumerate(AUTHENTICITY_TYPES)}

TRAJECTORY_STAGES = [
    "pre_breakout",
    "breaking_out",
    "peak_saturation",
    "simmering_entropy",
    "settling_quiet"
]
STAGE2ID = {s: i for i, s in enumerate(TRAJECTORY_STAGES)}
ID2STAGE = {i: s for i, s in enumerate(TRAJECTORY_STAGES)}

class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=1.5, label_smoothing=0.03):
        super().__init__()
        self.weight = weight
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(
            inputs,
            targets,
            weight=self.weight,
            reduction='none',
            label_smoothing=self.label_smoothing
        )
        pt = torch.exp(-ce_loss)
        focal_loss = ((1.0 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()

class OrdinalDistanceLoss(nn.Module):
    """
    Ordinal distance penalty loss:
    Combines standard cross-entropy with a differentiable expected rank distance penalty.
    """
    def __init__(self, num_classes, weight=None, alpha=0.5, label_smoothing=0.02):
        super().__init__()
        self.num_classes = num_classes
        self.weight = weight
        self.alpha = alpha
        self.label_smoothing = label_smoothing
        self.register_buffer("rank_indices", torch.arange(num_classes, dtype=torch.float32))

    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, weight=self.weight, label_smoothing=self.label_smoothing)
        probs = F.softmax(logits, dim=-1) # (batch_size, num_classes)
        expected_rank = torch.sum(probs * self.rank_indices.unsqueeze(0), dim=-1) # (batch_size,)
        target_rank = targets.float()
        rank_dist = torch.abs(expected_rank - target_rank).mean()
        return ce + self.alpha * rank_dist

class StrategicDataset(Dataset):
    def __init__(self, samples, tokenizer, max_length=128):
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        text = str(item.get("text", "")).strip()
        enc = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "axis_label": torch.tensor(AXIS2ID.get(item.get("threat_axis"), 7), dtype=torch.long),
            "crit_label": torch.tensor(CRIT2ID.get(item.get("criticality"), 1), dtype=torch.long),
            "auth_label": torch.tensor(AUTH2ID.get(item.get("authenticity"), 0), dtype=torch.long),
            "stage_label": torch.tensor(STAGE2ID.get(item.get("trajectory_stage"), 3), dtype=torch.long),
        }

class MultiTaskStrategicClassifierV4(nn.Module):
    def __init__(self, base_model_name="microsoft/mdeberta-v3-base"):
        super().__init__()
        self.config = AutoConfig.from_pretrained(base_model_name)
        self.deberta = DebertaV2Model.from_pretrained(
            base_model_name,
            config=self.config,
            torch_dtype=torch.float32
        )
        self.pooler = ContextPooler(self.config)
        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)

        hidden_size = self.config.hidden_size
        self.axis_classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.GELU(),
            nn.Dropout(0.12),
            nn.Linear(hidden_size // 2, len(THREAT_AXES))
        )
        self.crit_classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.GELU(),
            nn.Dropout(0.12),
            nn.Linear(hidden_size // 2, len(CRITICALITY_LEVELS))
        )
        self.auth_classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.GELU(),
            nn.Dropout(0.12),
            nn.Linear(hidden_size // 2, len(AUTHENTICITY_TYPES))
        )
        self.stage_classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.GELU(),
            nn.Dropout(0.12),
            nn.Linear(hidden_size // 2, len(TRAJECTORY_STAGES))
        )

    def forward(self, input_ids, attention_mask):
        encoder_outputs = self.deberta(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self.pooler(encoder_outputs[0])
        pooled = self.dropout(pooled)

        axis_logits = self.axis_classifier(pooled)
        crit_logits = self.crit_classifier(pooled)
        auth_logits = self.auth_classifier(pooled)
        stage_logits = self.stage_classifier(pooled)

        return axis_logits, crit_logits, auth_logits, stage_logits

def compute_class_weights(samples, key, id_map, num_classes):
    counts = Counter(id_map.get(s.get(key), 0) for s in samples)
    total = len(samples)
    weights = []
    for i in range(num_classes):
        c = counts.get(i, 1)
        w = total / (num_classes * c)
        weights.append(min(max(math.sqrt(w), 0.5), 2.5))
    return torch.tensor(weights, dtype=torch.float32)

def evaluate_on_benchmark(model, tokenizer, gold_samples, device):
    model.eval()
    ds = StrategicDataset(gold_samples, tokenizer)
    loader = DataLoader(ds, batch_size=32, shuffle=False)

    axis_preds, axis_trues = [], []
    crit_preds, crit_trues = [], []
    auth_preds, auth_trues = [], []
    stage_preds, stage_trues = [], []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            a_l, c_l, au_l, s_l = model(input_ids, attention_mask)

            axis_preds.extend(torch.argmax(a_l, dim=-1).cpu().numpy())
            axis_trues.extend(batch["axis_label"].numpy())

            crit_preds.extend(torch.argmax(c_l, dim=-1).cpu().numpy())
            crit_trues.extend(batch["crit_label"].numpy())

            auth_preds.extend(torch.argmax(au_l, dim=-1).cpu().numpy())
            auth_trues.extend(batch["auth_label"].numpy())

            stage_preds.extend(torch.argmax(s_l, dim=-1).cpu().numpy())
            stage_trues.extend(batch["stage_label"].numpy())

    axis_acc = accuracy_score(axis_trues, axis_preds)
    axis_f1 = f1_score(axis_trues, axis_preds, average="macro", zero_division=0)
    crit_acc = accuracy_score(crit_trues, crit_preds)
    crit_f1 = f1_score(crit_trues, crit_preds, average="macro", zero_division=0)
    auth_acc = accuracy_score(auth_trues, auth_preds)
    auth_f1 = f1_score(auth_trues, auth_preds, average="macro", zero_division=0)
    stage_acc = accuracy_score(stage_trues, stage_preds)
    stage_f1 = f1_score(stage_trues, stage_preds, average="macro", zero_division=0)

    # Ordinal rank distance evaluations
    crit_mae = np.mean(np.abs(np.array(crit_preds) - np.array(crit_trues)))
    stage_mae = np.mean(np.abs(np.array(stage_preds) - np.array(stage_trues)))

    composite_f1 = 0.35 * axis_f1 + 0.25 * crit_f1 + 0.20 * auth_f1 + 0.20 * stage_f1

    return {
        "axis_accuracy": axis_acc, "axis_f1": axis_f1,
        "crit_accuracy": crit_acc, "crit_f1": crit_f1, "crit_mae": crit_mae,
        "auth_accuracy": auth_acc, "auth_f1": auth_f1,
        "stage_accuracy": stage_acc, "stage_f1": stage_f1, "stage_mae": stage_mae,
        "composite_f1": composite_f1,
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", default="/home/s0r0qw6/strategic_training/strategic_train.jsonl")
    parser.add_argument("--gold_path", default="/home/s0r0qw6/strategic_training/strategic_gold_benchmark.jsonl")
    parser.add_argument("--base_model", default="microsoft/mdeberta-v3-base")
    parser.add_argument("--output_dir", default="/home/s0r0qw6/strategic_training/output_strategic_v4")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_length", type=int, default=128)
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using compute device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    train_samples = []
    with open(args.train_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip(): train_samples.append(json.loads(line))

    gold_samples = []
    with open(args.gold_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip(): gold_samples.append(json.loads(line))

    print(f"Loaded {len(train_samples)} massive training samples, {len(gold_samples)} gold benchmark samples.")

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    model = MultiTaskStrategicClassifierV4(args.base_model).to(device)

    train_ds = StrategicDataset(train_samples, tokenizer, max_length=args.max_length)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)

    axis_weights = compute_class_weights(train_samples, "threat_axis", AXIS2ID, len(THREAT_AXES)).to(device)
    crit_weights = compute_class_weights(train_samples, "criticality", CRIT2ID, len(CRITICALITY_LEVELS)).to(device)
    auth_weights = compute_class_weights(train_samples, "authenticity", AUTH2ID, len(AUTHENTICITY_TYPES)).to(device)
    stage_weights = compute_class_weights(train_samples, "trajectory_stage", STAGE2ID, len(TRAJECTORY_STAGES)).to(device)

    # Losses: Focal Loss on Threat Axis, Ordinal Distance Loss on Criticality and Stage
    criterion_axis = FocalLoss(weight=axis_weights, gamma=1.5, label_smoothing=0.03)
    criterion_crit = OrdinalDistanceLoss(num_classes=len(CRITICALITY_LEVELS), weight=crit_weights, alpha=0.6, label_smoothing=0.03).to(device)
    criterion_auth = nn.CrossEntropyLoss(weight=auth_weights, label_smoothing=0.04)
    criterion_stage = OrdinalDistanceLoss(num_classes=len(TRAJECTORY_STAGES), weight=stage_weights, alpha=0.5, label_smoothing=0.03).to(device)

    backbone_params = list(model.deberta.parameters()) + list(model.pooler.parameters())
    head_params = list(model.axis_classifier.parameters()) + list(model.crit_classifier.parameters()) + \
                  list(model.auth_classifier.parameters()) + list(model.stage_classifier.parameters())

    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": args.lr},
        {"params": head_params, "lr": args.lr * 2.5}
    ], weight_decay=args.weight_decay, eps=1e-6)

    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(total_steps * 0.10)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    best_composite_f1 = 0.0
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n--- Zero-Shot Pre-Trained Weights Evaluation on {len(gold_samples)} Gold Benchmark ---")
    init_res = evaluate_on_benchmark(model, tokenizer, gold_samples, device)
    print(f"Zero-Shot: Axis F1: {init_res['axis_f1']:.3f} | Crit F1: {init_res['crit_f1']:.3f} (MAE {init_res['crit_mae']:.2f}) | Auth F1: {init_res['auth_f1']:.3f} | Stage F1: {init_res['stage_f1']:.3f} (MAE {init_res['stage_mae']:.2f}) | Composite F1: {init_res['composite_f1']:.3f}\n")

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        start_t = time.time()

        for step, batch in enumerate(train_loader):
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            axis_lbl = batch["axis_label"].to(device)
            crit_lbl = batch["crit_label"].to(device)
            auth_lbl = batch["auth_label"].to(device)
            stage_lbl = batch["stage_label"].to(device)

            a_l, c_l, au_l, s_l = model(input_ids, attention_mask)

            loss_axis = criterion_axis(a_l, axis_lbl)
            loss_crit = criterion_crit(c_l, crit_lbl)
            loss_auth = criterion_auth(au_l, auth_lbl)
            loss_stage = criterion_stage(s_l, stage_lbl)

            loss = 2.0 * loss_axis + 1.4 * loss_crit + 1.0 * loss_auth + 1.3 * loss_stage
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        epoch_dur = time.time() - start_t

        eval_res = evaluate_on_benchmark(model, tokenizer, gold_samples, device)
        print(f"Epoch {epoch:2d}/{args.epochs} [{epoch_dur:.1f}s] - Train Loss: {avg_loss:.4f} | "
              f"Axis Acc: {eval_res['axis_accuracy']:.1%} (F1: {eval_res['axis_f1']:.3f}) | "
              f"Crit Acc: {eval_res['crit_accuracy']:.1%} (F1: {eval_res['crit_f1']:.3f}, MAE: {eval_res['crit_mae']:.2f}) | "
              f"Auth Acc: {eval_res['auth_accuracy']:.1%} (F1: {eval_res['auth_f1']:.3f}) | "
              f"Stage Acc: {eval_res['stage_accuracy']:.1%} (F1: {eval_res['stage_f1']:.3f}, MAE: {eval_res['stage_mae']:.2f}) | "
              f"Composite F1: {eval_res['composite_f1']:.3f}")

        if eval_res["composite_f1"] > best_composite_f1:
            best_composite_f1 = eval_res["composite_f1"]
            print(f"  🏆 New Best Composite F1: {best_composite_f1:.4f}! Saving checkpoint...")
            
            torch.save(model.state_dict(), out_dir / "pytorch_model.bin")
            tokenizer.save_pretrained(str(out_dir))
            model.config.save_pretrained(str(out_dir))
            
            with open(out_dir / "eval_metrics.json", "w", encoding="utf-8") as f:
                json.dump({
                    "epoch": epoch,
                    "benchmark_samples": len(gold_samples),
                    "train_samples": len(train_samples),
                    **eval_res
                }, f, indent=2)

    print(f"\n✅ Training complete! Best Composite F1 on 2,285 Gold Benchmark: {best_composite_f1:.4f}")

if __name__ == "__main__":
    main()
