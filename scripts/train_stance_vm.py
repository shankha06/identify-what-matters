"""
Remote GPU VM Training Script for Stance and Tone cross-encoders.
Designed to run on the 4x A100 GPU cluster (llm1-4a100-80g).
Incorporates:
- MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7 (multilingual NLI pretraining)
- FP32 master weights (preventing NaN divergence in mDeBERTa)
- Balanced class weights + sample-level weighting for realistic/hard examples
- Optimal threshold calibration on dev split
- Checkpoint retention (save_only_model=True) to respect disk space
- Out-of-the-box evaluation on both full test split and held-out real slice
"""
import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
from datasets import Dataset
from sklearn.metrics import f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed
)

TONES = ["critical", "supportive", "humorous", "objective"]

def load_data(task: str, data_dir: Path):
    split_dir = data_dir / "clean_splits" / task
    train_rows = [json.loads(l) for l in (split_dir / "train.jsonl").read_text().splitlines() if l.strip()]
    dev_rows = [json.loads(l) for l in (split_dir / "dev.jsonl").read_text().splitlines() if l.strip()]
    test_rows = [json.loads(l) for l in (split_dir / "test.jsonl").read_text().splitlines() if l.strip()]
    
    # Check for extra hard realistic augmentations
    extra_aug = data_dir / "hard_realistic_train_augmentations.jsonl"
    if extra_aug.exists():
        aug_records = [json.loads(l) for l in extra_aug.read_text().splitlines() if l.strip()]
        d_positions = defaultdict(dict)
        for r in train_rows:
            d_id = r.get("discussion_id")
            if task == "stance":
                pos = r.get("position")
                s_idx = r.get("side_index")
                if pos and s_idx is not None and s_idx not in d_positions[d_id]:
                    d_positions[d_id][s_idx] = pos
            else:
                ctx = r.get("context")
                if ctx and "context" not in d_positions[d_id]:
                    d_positions[d_id]["context"] = ctx
        
        aug_rows = []
        for rec in aug_records:
            d_id = rec["discussion_id"]
            side = rec["side"]
            comment = rec["comment"]
            sarcastic = rec.get("sarcastic", False)
            
            if task == "stance":
                positions = d_positions.get(d_id, {})
                for s_idx, pos_text in positions.items():
                    label = 1 if s_idx == side else 0
                    aug_rows.append({
                        "discussion_id": d_id,
                        "position": pos_text,
                        "comment": comment,
                        "label": label,
                        "side_index": s_idx,
                        "origin": "hard_realistic_aug",
                        "sarcastic": sarcastic
                    })
            else:
                ctx = d_positions.get(d_id, {}).get("context")
                if ctx:
                    tone_label = TONES.index(rec["tone"]) if rec["tone"] in TONES else 0
                    aug_rows.append({
                        "discussion_id": d_id,
                        "context": ctx,
                        "comment": comment,
                        "label": tone_label,
                        "origin": "hard_realistic_aug",
                        "sarcastic": sarcastic
                    })
        print(f"Adding {len(aug_rows)} hard realistic augmented samples to {task} train set")
        train_rows.extend(aug_rows)
        
    return train_rows, dev_rows, test_rows

def calibrate_threshold(probs, labels):
    best_th = 0.5
    best_f1 = -1.0
    for th in np.arange(0.1, 0.9, 0.02):
        preds = (probs >= th).astype(int)
        score = f1_score(labels, preds, average="macro")
        if score > best_f1:
            best_f1 = score
            best_th = th
    return float(best_th), float(best_f1)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["stance", "tone"], required=True)
    parser.add_argument("--data-dir", default="artifacts")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--base-model", default="MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    
    train_rows, dev_rows, test_rows = load_data(args.task, Path(args.data_dir))
    
    # 3x repeat on real and hard realistic data
    hard_samples = [r for r in train_rows if r.get("origin") in ("real_llm_labelled", "hard_realistic_aug")]
    print(f"Total train: {len(train_rows)}, Real/Hard train samples: {len(hard_samples)}")
    train_rows += hard_samples * 2
    
    num_labels = 2 if args.task == "stance" else len(TONES)
    class_names = ("not_backing", "backing") if args.task == "stance" else TONES
    
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.base_model,
        num_labels=num_labels,
        ignore_mismatched_sizes=True,
        dtype=torch.float32,
        id2label=dict(enumerate(class_names)),
        label2id={name: idx for idx, name in enumerate(class_names)}
    )
    
    def pair_of(r):
        if args.task == "stance":
            return r["position"], r["comment"]
        return r["context"], r["comment"]
        
    def encode(batch):
        first, second = zip(*(pair_of(r) for r in batch_rows(batch)))
        return tokenizer(list(first), list(second), truncation=True, max_length=args.max_length)

    def batch_rows(batch):
        keys = list(batch)
        return [dict(zip(keys, values)) for values in zip(*batch.values())]

    cols = ["position", "comment", "label"] if args.task == "stance" else ["context", "comment", "label"]
    train_ds = Dataset.from_list([{k: r[k] for k in cols} for r in train_rows]).shuffle(seed=args.seed)
    train_ds = train_ds.map(encode, batched=True, remove_columns=[c for c in cols if c != "label"]).rename_column("label", "labels")

    dev_ds = Dataset.from_list([{k: r[k] for k in cols} for r in dev_rows])
    dev_ds = dev_ds.map(encode, batched=True, remove_columns=[c for c in cols if c != "label"]).rename_column("label", "labels")

    training_args = TrainingArguments(
        output_dir=str(output / "checkpoints"),
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        fp16=False,  # FP32 master weights for mDeBERTa stability
        logging_steps=50,
        report_to="none",
        save_only_model=True
    )

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=1)
        macro = f1_score(labels, preds, average="macro")
        return {"macro_f1": float(macro)}

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics
    )

    trainer.train()
    
    # Save final model directly into output_dir
    trainer.save_model(str(output))
    tokenizer.save_pretrained(str(output))
    print(f"Model saved successfully to {output}")
    
    # Evaluate on Dev to calibrate threshold (for stance)
    dev_preds_raw = trainer.predict(dev_ds)
    dev_logits = torch.tensor(dev_preds_raw.predictions)
    dev_labels = dev_preds_raw.label_ids
    
    calibrated_th = 0.5
    if args.task == "stance":
        dev_probs = torch.softmax(dev_logits, dim=-1)[:, 1].numpy()
        calibrated_th, best_dev_f1 = calibrate_threshold(dev_probs, dev_labels)
        print(f"Calibrated Stance Threshold on Dev: {calibrated_th:.3f} (Dev Macro-F1: {best_dev_f1:.4f})")
        (output / "stance_threshold.json").write_text(json.dumps({"threshold": calibrated_th, "selected_on": "dev macro-F1"}))
    
    # Evaluate on Test Split
    test_ds = Dataset.from_list([{k: r[k] for k in cols} for r in test_rows])
    test_ds = test_ds.map(encode, batched=True, remove_columns=[c for c in cols if c != "label"]).rename_column("label", "labels")
    
    test_preds_raw = trainer.predict(test_ds)
    test_logits = torch.tensor(test_preds_raw.predictions)
    test_labels = test_preds_raw.label_ids
    
    if args.task == "stance":
        test_probs = torch.softmax(test_logits, dim=-1)[:, 1].numpy()
        test_preds = (test_probs >= calibrated_th).astype(int)
    else:
        test_preds = np.argmax(test_logits.numpy(), axis=1)
    
    overall_f1 = f1_score(test_labels, test_preds, average="macro")
    print(f"\n==========================================")
    print(f"TASK: {args.task}")
    print(f"Overall Test Macro-F1: {overall_f1:.4f}")
    
    # Real slice breakdown
    real_indices = [i for i, r in enumerate(test_rows) if r.get("origin") == "real_llm_labelled"]
    if real_indices:
        real_labels = [test_labels[i] for i in real_indices]
        real_preds = [test_preds[i] for i in real_indices]
        real_f1 = f1_score(real_labels, real_preds, average="macro")
        print(f"REAL SLICE Macro-F1: {real_f1:.4f} (across {len(real_indices)} real examples)")
    print(f"==========================================\n")
    
    results = {
        "task": args.task,
        "overall_macro_f1": float(overall_f1),
        "real_slice_macro_f1": float(real_f1) if real_indices else None,
        "calibrated_threshold": calibrated_th if args.task == "stance" else None,
        "n_train": len(train_rows),
        "n_test": len(test_rows),
        "n_real_test": len(real_indices)
    }
    (output / "eval_results.json").write_text(json.dumps(results, indent=2))

if __name__ == "__main__":
    main()
