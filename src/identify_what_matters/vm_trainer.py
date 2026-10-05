"""
GPU VM Fine-Tuning Pipeline for Domain-Specific Trend Ranking
Leverages:
- NVIDIA A100 GPU VM (llm1-4a100-80g) via GCP IAP
- Base Model: all-mpnet-base-v2
- MultipleNegativesRankingLoss / CosineSimilarityLoss for specialized Indian Trend & Fashion retrieval
"""

import os
import json
import argparse
from typing import List, Dict

try:
    import torch
    from sentence_transformers import SentenceTransformer, InputExample, losses
    from torch.utils.data import DataLoader
except ImportError:
    pass


def prepare_trend_training_pairs(scraped_data_file: str, output_pairs_file: str):
    """
    Construct positive pairs from trend clusters:
    - (Post Caption, Relevant Trend Query / Co-clustered Post)
    - (Buyer Comment Question, Product / Post Caption)
    """
    with open(scraped_data_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    posts = data.get("posts", [])
    pairs = []

    # 1. Tag - Caption pairs
    for p in posts:
        caption = p.get("caption", "").strip()
        tag = p.get("tag", "").strip()
        if len(caption) > 20 and tag:
            pairs.append({"query": f"Trending in India #{tag}", "positive": caption[:200]})

    # 2. Comment intent pairs
    for p in posts:
        caption = p.get("caption", "").strip()
        for c in p.get("comments", []):
            comment_text = c.get("text", "").strip()
            if len(comment_text) > 5 and len(caption) > 20:
                pairs.append({"query": comment_text, "positive": caption[:200]})

    os.makedirs(os.path.dirname(output_pairs_file), exist_ok=True)
    with open(output_pairs_file, "w", encoding="utf-8") as f:
        for item in pairs:
            f.write(json.dumps(item) + "\n")

    print(f"Prepared {len(pairs)} contrastive training pairs saved to {output_pairs_file}")


def train_vm_model(
    base_model_path: str,
    train_data_path: str,
    output_dir: str,
    epochs: int = 3,
    batch_size: int = 64,
    lr: float = 2e-5,
):
    """
    Runs fine-tuning with MultipleNegativesRankingLoss on GPU VM.
    """
    print(f"Loading base model: {base_model_path}")
    model = SentenceTransformer(base_model_path)

    examples = []
    with open(train_data_path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            examples.append(InputExample(texts=[item["query"], item["positive"]]))

    train_dataloader = DataLoader(examples, shuffle=True, batch_size=batch_size)
    train_loss = losses.MultipleNegativesRankingLoss(model=model)

    print(f"Starting training on device: {model.device} for {epochs} epochs...")
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=epochs,
        warmup_steps=100,
        optimizer_params={"lr": lr},
        output_path=output_dir,
        show_progress_bar=True,
    )
    print(f"Training completed successfully! Saved tuned model to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Train Trend MPNet Model")
    parser.add_argument("--prepare-pairs", action="store_true")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--data-file", default="output_sample.json")
    parser.add_argument("--pairs-file", default="data/trend_train_pairs.jsonl")
    parser.add_argument("--base-model", default="/Users/s0r0qw6/model-py-java/artifacts/all-mpnet-base-v2")
    parser.add_argument("--output-dir", default="artifacts/tuned-trend-mpnet")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    if args.prepare_pairs:
        prepare_trend_training_pairs(args.data_file, args.pairs_file)

    if args.train:
        train_vm_model(
            base_model_path=args.base_model,
            train_data_path=args.pairs_file,
            output_dir=args.output_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
        )


if __name__ == "__main__":
    main()
