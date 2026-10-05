"""Isolated, synthetic same-event embedding pilot; never production gold."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
from pathlib import Path
import random
import re
import sys
from datetime import datetime, timezone

from identify_what_matters.openrouter import DEFAULT_MODEL, OpenRouterClient

MODEL = DEFAULT_MODEL
DOMAINS = (
    "public_transport", "agriculture", "education", "public_health",
    "cricket", "cinema", "technology", "environment",
    "food", "music", "space_science", "civic_services",
)
REGIONAL = ("bn", "ta", "te", "mr", "gu", "kn")
IDENTIFIERS = (
    r"https?://\S+|www\.\S+", r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}",
    r"(?<!\w)[@#][\w.]+", r"(?<!\w)\+?\d[\d\s().-]{6,}\d(?!\w)",
    r"\b[A-Z]{5}\d{4}[A-Z]\b", r"\b[A-Z]{2}\d{2}[A-Z]{1,3}\d{4}\b",
    r"\b[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}\b",
    r"\b(?:user|account|customer|order|ticket|device|session|aadhaar|pan|id)"
    r"\s*[:=#-]\s*[\w.-]+",
)


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def sanitize(text):
    for pattern in IDENTIFIERS:
        text = re.sub(pattern, " ", text, flags=re.IGNORECASE)
    return " ".join(text.split())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def write_rows(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
    temporary.replace(path)


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def failure_category(error):
    message = str(error)
    http_status = re.search(r"\b([45]\d{2}) (?:Client|Server) Error", message)
    if http_status:
        return "http_" + http_status.group(1)
    for marker in ("No module named", "NameResolutionError", "ConnectTimeout", "ReadTimeout",
                   "Connection refused", "SSLError", "InvalidSignature", "ProxyError"):
        if marker in message:
            return marker.replace(" ", "_")
    if isinstance(error, json.JSONDecodeError):
        return "invalid_json"
    if isinstance(error, ValueError):
        return "schema_rejection"
    return type(error).__name__


def validate_batch(payload, domains, batch_index, metadata):
    if not isinstance(payload, dict) or set(payload) != {"events", "negatives"}:
        raise ValueError("schema_root")
    events, negatives = payload["events"], payload["negatives"]
    if not isinstance(events, list) or len(events) != 8:
        raise ValueError("event_count")
    indexed, seen_texts = {}, set()
    for event in events:
        if not isinstance(event, dict) or set(event) != {"event_id", "domain", "paraphrases"}:
            raise ValueError("event_schema")
        domain, event_id = event["domain"], event["event_id"]
        if domain not in domains or event_id not in {f"{domain}_{number}" for number in range(4)}:
            raise ValueError("event_identity")
        if event_id in indexed or len(event["paraphrases"]) != 4:
            raise ValueError("event_duplicate_or_size")
        required = {"en", "hi", "hinglish", REGIONAL[DOMAINS.index(domain) % 6]}
        languages = set()
        for text in event["paraphrases"]:
            if not isinstance(text, dict) or set(text) != {"text", "language"}:
                raise ValueError("text_schema")
            content, language = text["text"], text["language"]
            if not isinstance(content, str) or not 15 <= len(content) <= 300:
                raise ValueError("text_length")
            if sanitize(content) != content or content.casefold() in seen_texts:
                raise ValueError("unsafe_or_duplicate_text")
            if language not in required or language in languages:
                raise ValueError("language_coverage")
            if language == "hi" and not re.search(r"[\u0900-\u097f]", content):
                raise ValueError("hindi_script")
            script_ranges = {"bn": "\u0980-\u09ff", "ta": "\u0b80-\u0bff",
                             "te": "\u0c00-\u0c7f", "mr": "\u0900-\u097f",
                             "gu": "\u0a80-\u0aff", "kn": "\u0c80-\u0cff"}
            if language in script_ranges and not re.search(f"[{script_ranges[language]}]", content):
                raise ValueError("regional_script")
            languages.add(language)
            seen_texts.add(content.casefold())
        indexed[event_id] = {**event, **metadata, "batch_index": batch_index,
                     "language": sorted(languages)}
    if not isinstance(negatives, list) or len(negatives) != 24:
        raise ValueError("negative_count")
    edge_counts, seen_pairs = {}, set()
    labeled = []
    for pair in negatives:
        if not isinstance(pair, dict) or set(pair) != {
            "event_a", "event_b", "variant_a", "variant_b", "same_event", "reason"
        }:
            raise ValueError("negative_schema")
        first, second = pair["event_a"], pair["event_b"]
        if first not in indexed or second not in indexed or first == second:
            raise ValueError("negative_event")
        if indexed[first]["domain"] != indexed[second]["domain"] or pair["same_event"] is not False:
            raise ValueError("negative_label")
        for key in ("variant_a", "variant_b"):
            if type(pair[key]) is not int or not 0 <= pair[key] < 4:
                raise ValueError("negative_variant")
        if not isinstance(pair["reason"], str) or not 10 <= len(pair["reason"]) <= 300:
            raise ValueError("negative_reason")
        if sanitize(pair["reason"]) != pair["reason"]:
            raise ValueError("unsafe_reason")
        edge = tuple(sorted((first, second)))
        identity = tuple(sorted(((first, pair["variant_a"]), (second, pair["variant_b"]))))
        if identity in seen_pairs:
            raise ValueError("duplicate_negative")
        seen_pairs.add(identity)
        edge_counts[edge] = edge_counts.get(edge, 0) + 1
        labeled.append({**pair, **metadata, "domain": indexed[first]["domain"],
                        "language": [indexed[first]["paraphrases"][pair["variant_a"]]["language"],
                                     indexed[second]["paraphrases"][pair["variant_b"]]["language"]],
                        "event_id": [first, second]})
    expected = {tuple(sorted(edge)) for domain in domains for edge in
                itertools.combinations([f"{domain}_{number}" for number in range(4)], 2)}
    if set(edge_counts) != expected or any(count != 2 for count in edge_counts.values()):
        raise ValueError("negative_edge_coverage")
    return list(indexed.values()), labeled


def prompt_for(domains):
    specifications = [{"domain": domain, "regional": REGIONAL[DOMAINS.index(domain) % 6]}
                      for domain in domains]
    return sanitize(
        "Create fictional Indian public-interest social posts, not facts or real people. "
        "Return strict JSON only: begin with { and end with }, no Markdown fences or prose, "
        "with exactly keys events and negatives. Domains: "
        + json.dumps(specifications) + ". For each domain create FOUR distinct specific events "
        "numbered 0 through 3. Each event_id is domain followed by underscore and number. "
        "Events must be confusable within the same broad category but differ in action, public "
        "location, timing or outcome. No personal names, usernames, hashtags, URLs, emails, "
        "phones, addresses, account numbers or personal identifiers. Use fictional public "
        "venues rather than individuals. Each event has exactly keys event_id, domain, "
        "paraphrases. Each paraphrases array has FOUR objects with exactly text and language: "
        "en, hi in Devanagari, hinglish in romanized Hindi mixed with English, and the specified "
        "regional language in its native script. Each text 35-180 characters; preserve SAME "
        "specific event across translations and informal paraphrases, not just broad topic. "
        "No generic slogans. Explicitly label ALL SIX unordered different-event combinations "
        "per domain as hard negatives, with TWO different paraphrase pairs per combination, "
        "so 24 negatives total. Each negative has exactly event_a, event_b, variant_a, variant_b "
        "(zero-based paraphrase indices), same_event false, reason (brief distinguishing "
        "event fact, 15-160 characters). 8 events and 24 negatives total."
    )


def build_dataset(events, negatives, out, seed, metadata):
    rng = random.Random(seed)
    assignment = {}
    for index, domain in enumerate(DOMAINS):
        identities = sorted(event["event_id"] for event in events if event["domain"] == domain)
        if len(identities) != 4:
            raise ValueError("incomplete_domain")
        rng.shuffle(identities)
        for identity in identities[:2]:
            assignment[identity] = "train"
        for identity in identities[2:]:
            assignment[identity] = "dev" if index % 2 == 0 else "test"
    write_json(out / "split_manifest.json", {**metadata, "seed": seed,
               "strategy": "2 train events per domain; 2 dev/test events in alternating domains",
               "assignment": assignment, "human_gold": False})
    rows = {split: [] for split in ("train", "dev", "test")}
    by_id = {event["event_id"]: event for event in events}
    for event in events:
        for first, second in itertools.combinations(event["paraphrases"], 2):
            rows[assignment[event["event_id"]]].append({**metadata, "event_id": [event["event_id"]],
                "domain": event["domain"], "language": [first["language"], second["language"]],
                "text_a": first["text"], "text_b": second["text"], "label": 1.0,
                "label_basis": "LLM generated same-event group"})
    edges = {tuple(sorted((pair["event_a"], pair["event_b"]))) for pair in negatives}
    discarded = 0
    for first_id, second_id in sorted(edges):
        if assignment[first_id] != assignment[second_id]:
            discarded += 1
            continue
        first_event, second_event = by_id[first_id], by_id[second_id]
        for first, second in itertools.product(first_event["paraphrases"], second_event["paraphrases"]):
            rows[assignment[first_id]].append({**metadata, "event_id": [first_id, second_id],
                "domain": first_event["domain"], "language": [first["language"], second["language"]],
                "text_a": first["text"], "text_b": second["text"], "label": 0.0,
                "label_basis": "LLM negative event relation expanded to paraphrases"})
    for split, pairs in rows.items():
        rng.shuffle(pairs)
        write_rows(out / f"{split}.jsonl", pairs)
    summary = {**metadata, "events": len(events), "texts": sum(len(event["paraphrases"]) for event in events),
               "domains": len(DOMAINS), "llm_labeled_hard_negatives": len(negatives),
               "discarded_cross_split_event_edges": discarded,
               "pairs": {split: {"total": len(pairs), "positive": sum(row["label"] == 1 for row in pairs),
                                  "negative": sum(row["label"] == 0 for row in pairs)}
                         for split, pairs in rows.items()}, "adequate_real_data": False}
    write_json(out / "dataset_summary.json", summary)
    print(json.dumps(summary))


def generate(args):
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    state_path = out / "generation_state.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {
        "run_id": "topic-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "timestamp": timestamp(), "origin": "synthetic", "model": MODEL,
        "seed": args.seed, "calls": 0, "reserved_tokens": 0, "usage": [], "failures": []}
    if state["seed"] != args.seed or state["model"] != MODEL:
        raise ValueError("resume_configuration_mismatch")
    metadata = {key: state[key] for key in ("run_id", "timestamp", "origin", "model")}
    client = None
    all_events, all_negatives = [], []
    for batch_index in range(6):
        domains = DOMAINS[batch_index * 2:batch_index * 2 + 2]
        checkpoint = out / f"batch_{batch_index}.json"
        if checkpoint.exists():
            saved = json.loads(checkpoint.read_text())
            if saved["metadata"] != metadata:
                raise ValueError("checkpoint_run_mismatch")
            events, negatives = validate_batch(saved["payload"], domains, batch_index, metadata)
        else:
            prompt = prompt_for(domains)
            reservation = 2 * (len(prompt.encode("utf-8")) + 6500 + 512)
            if state["calls"] + 2 > min(args.max_calls, 8) or state["reserved_tokens"] + reservation > args.token_cap:
                write_json(state_path, state)
                continue
            if client is None:
                try:
                    client = OpenRouterClient(model=MODEL, max_tokens=6500, temperature=0.3)
                except Exception as error:
                    raise ValueError(f"openrouter_initialization_failed:{type(error).__name__}") from None
            state["calls"] += 2
            state["reserved_tokens"] += reservation
            write_json(state_path, state)
            try:
                response = client.generate(user_prompt=prompt,
                    system_instruction="Return strict JSON only. Synthetic non-personal text only.",
                    request_timeout=150)
                state["usage"].append({"batch": batch_index, "input_tokens": response.input_tokens,
                                       "output_tokens": response.output_tokens})
                stripped = response.text.strip()
                state.setdefault("response_format", []).append({"batch": batch_index,
                    "characters": len(stripped), "starts_object": stripped.startswith("{"),
                    "ends_object": stripped.endswith("}"), "markdown_fence": stripped.startswith("```")})
                payload = json.loads(response.text)
                events, negatives = validate_batch(payload, domains, batch_index, metadata)
            except Exception as error:
                state["failures"].append({"batch": batch_index, "error_type": type(error).__name__,
                                          "category": failure_category(error),
                                          "json_error_position": error.pos if isinstance(error, json.JSONDecodeError) else None,
                                          "timestamp": timestamp()})
                write_json(state_path, state)
                print(json.dumps({"batch": batch_index, "status": "rejected_or_unavailable",
                                  "error_type": type(error).__name__,
                                  "category": failure_category(error)}), flush=True)
                continue
            write_json(checkpoint, {"metadata": metadata, "payload": payload})
            write_json(state_path, state)
            print(json.dumps({"batch": batch_index, "status": "validated", "calls": state["calls"]}), flush=True)
        all_events.extend(events)
        all_negatives.extend(negatives)
    write_rows(Path(args.labels_prefix + ".jsonl"), all_events)
    write_rows(Path(args.labels_prefix + "_negatives.jsonl"), all_negatives)
    if len(all_events) != 48:
        write_json(out / "incomplete_summary.json", {**metadata, "events": len(all_events),
            "texts": sum(len(event["paraphrases"]) for event in all_events),
            "domains": len({event["domain"] for event in all_events}),
            "llm_labeled_hard_negatives": len(all_negatives), "calls": state["calls"],
            "reserved_tokens": state["reserved_tokens"], "required_events": 48,
            "training_ready": False, "adequate_real_data": False})
        raise ValueError("incomplete_generation_or_budget_exhausted")
    if len({item["text"].casefold() for event in all_events for item in event["paraphrases"]}) != 192:
        raise ValueError("cross_batch_duplicate_text")
    build_dataset(all_events, all_negatives, out, args.seed, metadata)


def load_splits(dataset):
    dataset = Path(dataset)
    manifest = json.loads((dataset / "split_manifest.json").read_text())
    splits = {split: read_rows(dataset / f"{split}.jsonl") for split in ("train", "dev", "test")}
    text_splits = {}
    for split, rows in splits.items():
        if {row["label"] for row in rows} != {0.0, 1.0}:
            raise ValueError("missing_label_class")
        for row in rows:
            if any(manifest["assignment"][identity] != split for identity in row["event_id"]):
                raise ValueError("event_leakage")
            for key in ("text_a", "text_b"):
                digest = hashlib.sha256(row[key].encode()).hexdigest()
                if digest in text_splits and text_splits[digest] != split:
                    raise ValueError("text_leakage")
                text_splits[digest] = split
    return splits


def train(args):
    import torch
    from sentence_transformers import InputExample, SentenceTransformer, losses
    from torch.utils.data import DataLoader

    if args.batch_size not in (8, 16) or args.epochs != 1:
        raise ValueError("pilot_requires_batch_8_or_16_and_one_epoch")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError("pilot_requires_exactly_one_visible_gpu")
    output = Path(args.output)
    if output.exists():
        raise ValueError("checkpoint_output_already_exists")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    splits = load_splits(args.dataset)
    model = SentenceTransformer(args.base_model, device="cuda")
    model.max_seq_length = 128
    samples = [InputExample(texts=[row["text_a"], row["text_b"]], label=float(row["label"]))
               for row in splits["train"]]
    loader = DataLoader(samples, batch_size=args.batch_size, shuffle=True,
                        generator=torch.Generator().manual_seed(args.seed))
    started = timestamp()
    model.fit(train_objectives=[(loader, losses.CosineSimilarityLoss(model))], epochs=1,
              warmup_steps=0, optimizer_params={"lr": 2e-5}, output_path=str(output),
              show_progress_bar=False, use_amp=False)
    write_json(output / "training_run.json", {"started": started, "completed": timestamp(),
        "base_model": args.base_model, "loss": "CosineSimilarityLoss", "seed": args.seed,
        "batch_size": args.batch_size, "epochs": 1, "pairs": len(samples),
        "gpu": torch.cuda.get_device_name(0), "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "torch": torch.__version__, "origin": "synthetic", "human_gold": False})
    print(json.dumps({"status": "trained", "checkpoint": str(output), "pairs": len(samples)}))


def evaluate(args):
    import numpy as np
    from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
    from sentence_transformers import SentenceTransformer

    splits = load_splits(args.dataset)
    results = {}
    for name, model_path in (("base", args.base_model), ("tuned", args.tuned_model)):
        model = SentenceTransformer(model_path, device=args.device)
        model.max_seq_length = 128
        texts = sorted({row[key] for split in ("dev", "test") for row in splits[split]
                        for key in ("text_a", "text_b")})
        embeddings = model.encode(texts, batch_size=16, normalize_embeddings=True, show_progress_bar=False)
        indexed = dict(zip(texts, embeddings))
        scores = {split: np.array([float(np.dot(indexed[row["text_a"]], indexed[row["text_b"]]))
                                  for row in splits[split]]) for split in ("dev", "test")}
        labels = {split: np.array([int(row["label"]) for row in splits[split]]) for split in scores}
        candidates = np.r_[np.nextafter(scores["dev"].max(), np.inf), np.unique(scores["dev"])]
        threshold = float(max(candidates, key=lambda value: (f1_score(labels["dev"], scores["dev"] >= value), value)))

        def metrics(truth, similarity):
            prediction = similarity >= threshold
            positive, negative = similarity[truth == 1], similarity[truth == 0]
            return {"count": int(len(truth)), "positive_count": int(len(positive)),
                    "negative_count": int(len(negative)), "roc_auc": float(roc_auc_score(truth, similarity)),
                    "f1": float(f1_score(truth, prediction, zero_division=0)),
                    "precision": float(precision_score(truth, prediction, zero_division=0)),
                    "recall": float(recall_score(truth, prediction, zero_division=0)),
                    "positive_cosine_mean": float(positive.mean()), "negative_cosine_mean": float(negative.mean()),
                    "cosine_separation": float(positive.mean() - negative.mean())}

        result = {"model_path": model_path, "dev_threshold": threshold,
                  "dev": metrics(labels["dev"], scores["dev"]),
                  "test": metrics(labels["test"], scores["test"]), "per_language": {}}
        for split in ("dev", "test"):
            language_results = {}
            for language in sorted({language for row in splits[split] for language in row["language"]}):
                mask = np.array([language in row["language"] for row in splits[split]])
                truth = labels[split][mask]
                if min(int(sum(truth == 0)), int(sum(truth == 1))) >= 5:
                    language_results[language] = metrics(truth, scores[split][mask])
                else:
                    language_results[language] = {"count": int(mask.sum()), "status": "too_few_per_class"}
            result["per_language"][split] = language_results
        results[name] = result
        del model
    tuned, base = results["tuned"], results["base"]
    generalizes = (tuned["test"]["f1"] >= tuned["dev"]["f1"] - 0.05
                   and tuned["test"]["f1"] >= base["test"]["f1"]
                   and tuned["test"]["roc_auc"] >= base["test"]["roc_auc"])
    report = {"timestamp": timestamp(), "evaluation": "event-disjoint synthetic holdout; NOT human gold",
              "models": results, "dev_threshold_generalizes": generalizes,
              "human_real_evaluation_available": False, "promotion_allowed": False,
              "promotion_reason": "Human-labeled representative real evaluation is required; synthetic scores cannot authorize promotion.",
              "uncertainty": "Pair observations are correlated within events; 48 event groups are a small pilot, not sufficient real data.",
              "language_protocol": "Pairs containing language, overlapping slices, minimum 5 positives and 5 negatives."}
    write_json(args.output, report)
    print(json.dumps(report))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    generation = commands.add_parser("generate")
    generation.add_argument("--output", default="artifacts/topic_training/pilot-20261003")
    generation.add_argument("--labels-prefix", default="data/topic_labels_pilot_20261003")
    generation.add_argument("--max-calls", type=int, default=8)
    generation.add_argument("--token-cap", type=int, default=60000)
    generation.add_argument("--seed", type=int, default=20261003)
    generation.set_defaults(handler=generate)
    training = commands.add_parser("train")
    training.add_argument("--dataset", required=True)
    training.add_argument("--base-model", required=True)
    training.add_argument("--output", required=True)
    training.add_argument("--batch-size", type=int, default=8)
    training.add_argument("--epochs", type=int, default=1)
    training.add_argument("--seed", type=int, default=20261003)
    training.set_defaults(handler=train)
    evaluation = commands.add_parser("evaluate")
    evaluation.add_argument("--dataset", required=True)
    evaluation.add_argument("--base-model", required=True)
    evaluation.add_argument("--tuned-model", required=True)
    evaluation.add_argument("--output", required=True)
    evaluation.add_argument("--device", default="cpu")
    evaluation.set_defaults(handler=evaluate)
    args = parser.parse_args()
    try:
        args.handler(args)
    except Exception as error:
        print(json.dumps({"status": "failed", "error_type": type(error).__name__,
                          "detail": str(error) if isinstance(error, ValueError) else "Details suppressed to protect credentials/content."}), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())