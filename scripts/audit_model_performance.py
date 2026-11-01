"""Measure deployed model families without changing artifacts or selecting on reporting labels."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, f1_score, roc_auc_score


def rows(path):
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def metrics(gold, predicted):
    return {
        "rows": len(gold), "accuracy": float(accuracy_score(gold, predicted)),
        "macro_f1": float(f1_score(gold, predicted, average="macro", zero_division=0)),
        "per_class": classification_report(gold, predicted, output_dict=True, zero_division=0),
    }


def evaluate(family, args):
    if family == "summary":
        return summarize(args.output, args.clean_splits)
    artifacts = Path("artifacts")
    clean = args.clean_splits
    if family == "sarcasm":
        from collections import defaultdict
        from identify_what_matters.political_corpus_trainer import train_and_evaluate
        from identify_what_matters.stance import label_comments
        dataset = args.output / "sarcasm_dataset"
        dataset.mkdir(parents=True, exist_ok=True)
        for split in ("train", "dev", "test"):
            converted = [{"text": row["context"] + " || Comment: " + row["comment"],
                          "sarcastic": bool(row["sarcastic"]), "origin": row["origin"]}
                         for row in rows(clean / "tone" / f"{split}.jsonl")]
            (dataset / f"{split}.jsonl").write_text(
                "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in converted), encoding="utf-8")
        candidate_dir = args.output / "sarcasm_candidate"
        trained = json.loads((candidate_dir / "metadata.json").read_text()) if candidate_dir.exists() else \
            train_and_evaluate(dataset, candidate_dir, search=True, tasks=("sarcastic",))
        reporting = rows(clean / "tone/test.jsonl")
        groups = defaultdict(list)
        for index, row in enumerate(reporting):
            groups[row["context"]].append((index, row))
        predicted = [False] * len(reporting)
        for context, group in groups.items():
            output = label_comments(context, [], [row["comment"] for _, row in group])
            for (index, _), label in zip(group, output):
                predicted[index] = bool(label["sarcastic"])
        gold = [str(bool(row["sarcastic"])) for row in reporting]
        candidate = joblib.load(candidate_dir / "sarcastic_classifier.joblib")
        real_dev = [row for row in rows(clean / "tone/dev.jsonl") if row["origin"] == "real_llm_labelled"]
        true_index = list(map(str, candidate.classes_)).index("True")
        dev_probabilities = candidate.predict_proba(
            [row["context"] + " || Comment: " + row["comment"] for row in real_dev])[:, true_index]
        dev_gold = [bool(row["sarcastic"]) for row in real_dev]
        threshold = float(max(np.linspace(0.05, 0.95, 37), key=lambda value:
                              f1_score(dev_gold, dev_probabilities >= value, average="macro", zero_division=0)))
        calibration = {"threshold": threshold, "selection_split": "real_dev",
                       "dev_rows": len(real_dev), "dev_positives": sum(dev_gold),
                       "caveat": "Exploratory calibration on only three real dev positives; not release evidence"}
        (candidate_dir / "sarcasm_threshold.json").write_text(json.dumps(calibration, indent=2), encoding="utf-8")
        candidate_probabilities = candidate.predict_proba(
            [row["context"] + " || Comment: " + row["comment"] for row in reporting])[:, true_index]
        candidate_predictions = [str(bool(value >= threshold)) for value in candidate_probabilities]
        heuristic_predictions = [str(value) for value in predicted]
        slices = {}
        for origin in sorted({row["origin"] for row in reporting}):
            indices = [index for index, row in enumerate(reporting) if row["origin"] == origin]
            slices[origin] = {
                "heuristic": metrics([gold[index] for index in indices],
                                     [heuristic_predictions[index] for index in indices]),
                "candidate": metrics([gold[index] for index in indices],
                                     [candidate_predictions[index] for index in indices]),
            }
        return {"current_heuristic": metrics(gold, heuristic_predictions),
                "candidate": metrics(gold, candidate_predictions), "slices": slices,
                "uncalibrated_candidate": metrics(gold, [str(bool(value >= 0.5)) for value in candidate_probabilities]),
                "real_dev_calibration": calibration,
                "candidate_training": trained, "caveat": "Sarcasm gold is LLM-labelled, not human verified"}
    if family == "civic":
        from identify_what_matters.civic_classifier import evaluate as civic_evaluate
        return civic_evaluate(SimpleNamespace(
            dataset=clean / "civic", models=[str(artifacts / "civic-models/civic-me5-base")],
            batch_size=16, device=args.device, write_threshold=False,
        ))
    if family in ("stance", "tone"):
        from identify_what_matters.stance_finetune import best_threshold, scores_for, stance_metrics, tone_metrics
        model = artifacts / "stance-models" / family
        dev = rows(clean / family / "dev.jsonl")
        reporting = rows(clean / family / "test.jsonl")
        dev_probs = scores_for(str(model), family, dev, args.device)
        reporting_probs = scores_for(str(model), family, reporting, args.device)
        if family == "tone":
            return {"dev": tone_metrics(dev, dev_probs), "reporting": tone_metrics(reporting, reporting_probs),
                    "gold_provenance": "LLM-labelled, not independently human verified"}
        threshold = best_threshold(dev, dev_probs)
        deployed_threshold = json.loads((model / "stance_threshold.json").read_text())["threshold"]
        return {"threshold_from_dev": threshold, "deployed_threshold": deployed_threshold,
                "dev": stance_metrics(dev, dev_probs, threshold),
                "reporting": stance_metrics(reporting, reporting_probs, threshold),
                "deployed_reporting": stance_metrics(reporting, reporting_probs, deployed_threshold),
                "gold_provenance": "LLM-labelled, not independently human verified"}
    if family == "relevance":
        from sentence_transformers import SentenceTransformer
        result = {}
        dev = rows(clean / "relevance_v4/dev.jsonl")
        reporting = rows(clean / "relevance_v4/test.jsonl")
        texts = list(dict.fromkeys(row[key] for row in dev + reporting for key in ("anchor", "caption")))
        for version in ("v4", "v5"):
            directory = artifacts / f"topic-models/multilingual-e5-base-events-real-{version}"
            settings = json.loads((directory / "topic_threshold.json").read_text())
            model = SentenceTransformer(str(directory), device=args.device)
            vectors = model.encode([settings.get("prefix", "") + text for text in texts],
                                   batch_size=16, normalize_embeddings=True, show_progress_bar=False)
            lookup = dict(zip(texts, vectors))
            probabilities = [np.array([np.dot(lookup[row["anchor"]], lookup[row["caption"]])
                                       for row in split]) for split in (dev, reporting)]
            gold = [np.array([int(row["label"]) for row in split]) for split in (dev, reporting)]
            thresholds = np.linspace(-1.0, 1.0, 401)
            threshold = float(max(thresholds, key=lambda value: f1_score(
                gold[0], probabilities[0] >= value, zero_division=0)))
            result[version] = {
                "threshold_from_dev": threshold,
                "dev_f1": float(f1_score(gold[0], probabilities[0] >= threshold, zero_division=0)),
                "reporting_auc": float(roc_auc_score(gold[1], probabilities[1])),
                "reporting": metrics(gold[1], probabilities[1] >= threshold),
                "runtime_fixed_045": metrics(gold[1], probabilities[1] >= 0.45),
                "saved_threshold": settings,
                "independence_caveat": "Legacy splits may have informed previous training; v5 threshold used test labels",
            }
            del model
        return result
    if family == "political":
        reporting = rows(clean / "political/test.jsonl")
        texts = [row["text"] for row in reporting]
        result = {}
        for task in ("is_political", "issue_cluster", "sentiment", "target", "direction", "narrative_type"):
            gold = [str(row[task]) for row in reporting]
            result[task] = {}
            paths = {"deployed_linear": artifacts / "political_models" / f"{task}_classifier.joblib",
                     "candidate": args.output / "political_candidates" / f"{task}_classifier.joblib"}
            for name, path in paths.items():
                if path.exists():
                    result[task][name] = metrics(gold, joblib.load(path).predict(texts))
            if task in ("is_political", "direction"):
                import torch
                from transformers import AutoModelForSequenceClassification, AutoTokenizer
                path = artifacts / f"political_models/mdeberta-{task}"
                tokenizer = AutoTokenizer.from_pretrained(path)
                model = AutoModelForSequenceClassification.from_pretrained(path).to(args.device).eval()
                classes = json.loads((path / "training_run.json").read_text())["classes"]
                predictions = []
                with torch.inference_mode():
                    for start in range(0, len(texts), 16):
                        inputs = tokenizer(texts[start:start + 16], truncation=True, max_length=160,
                                           padding=True, return_tensors="pt").to(args.device)
                        predictions.extend(str(classes[index]) for index in model(**inputs).logits.argmax(-1).tolist())
                result[task]["deployed_transformer"] = metrics(gold, predictions)
                del model
        result["caveat"] = "Linear scores exclude runtime keyword/entity overrides"
        return result
    if family == "demographics":
        reporting = rows(artifacts / "demographics_models/real_world_gold_standard_v2.jsonl")
        texts = [row["text"] for row in reporting]
        result = {"caveat": "Legacy gold used for neural checkpoint selection; text-derived age/gender are not verified demographics"}
        fields = {"age": "age_bracket", "gender": "gender", "mental_state": "mental_state",
                  "impatience": "impatience", "irrationality": "irrationality", "party_support": "party"}
        for task, field in fields.items():
            model = joblib.load(artifacts / "demographics_models" / f"{task}_classifier.joblib")
            result[task] = metrics([str(row[field]) for row in reporting], model.predict(texts))
        from identify_what_matters.demographics import _get_neural_demographics_model
        import torch
        tokenizer, model, config = _get_neural_demographics_model()
        if model is None:
            result["neural"] = {"status": "unavailable", "reason": "Runtime loader could not load checkpoint"}
        else:
            from train_contrastive_demographics_vm import ID2AGE, ID2MENTAL
            mental, ages = [], []
            model.to(args.device)
            with torch.inference_mode():
                for start in range(0, len(texts), 16):
                    inputs = tokenizer(texts[start:start + 16], padding=True, truncation=True,
                                       max_length=128, return_tensors="pt").to(args.device)
                    mental_logits, age_logits, _, _ = model(inputs["input_ids"], inputs["attention_mask"])
                    mental.extend(ID2MENTAL[index] for index in mental_logits.argmax(-1).tolist())
                    ages.extend(ID2AGE[index] for index in age_logits.argmax(-1).tolist())
            result["neural"] = {"mental_state": metrics([row["mental_state"] for row in reporting], mental),
                                "age": metrics([row["age_bracket"] for row in reporting], ages)}
        return result
    if family == "strategic":
        from identify_what_matters.strategic_classifier import load_strategic_model
        from train_strategic_v4_vm import evaluate_on_benchmark
        model, tokenizer, _ = load_strategic_model()
        if model is None:
            raise RuntimeError("Deployed strategic checkpoint could not load")
        model.to(args.device)
        result = evaluate_on_benchmark(model, tokenizer, rows("data/strategic_intelligence/strategic_gold_benchmark.jsonl"), args.device)
        return {"legacy_reporting": result, "caveat": "Benchmark was used for checkpoint selection; not independent gold"}
    raise ValueError(family)


def summarize(root: Path, clean: Path):
    def report(name):
        return json.loads((root / f"{name}_evaluation.json").read_text())

    def comparison(baseline, candidate, metric="macro_f1"):
        return {"metric": metric, "baseline": baseline, "candidate": candidate,
                "delta": candidate - baseline, "production_weights_replaced": False}

    political = report("political")
    comparisons = {}
    for task, values in political.items():
        if not isinstance(values, dict):
            continue
        source = "deployed_transformer" if "deployed_transformer" in values else "deployed_linear"
        comparisons[f"political:{task}"] = {
            **comparison(values[source]["macro_f1"], values["candidate"]["macro_f1"]),
            "baseline_source": source, "excludes_runtime_keyword_entity_overrides": True,
        }
    for family, candidate in (("stance", "stance_deberta_candidate"), ("tone", "tone_deberta_candidate")):
        comparisons[family] = comparison(report(family)["reporting"]["macro_f1"],
                                          report(candidate)["results"][0]["macro_f1"])
    sarcasm = report("sarcasm")
    comparisons["sarcasm"] = {**comparison(sarcasm["current_heuristic"]["macro_f1"],
                                          sarcasm["candidate"]["macro_f1"]),
                              "opt_in_env": "TRENDS_SARCASM_MODEL",
                              "opt_in_model": str(root / "sarcasm_candidate/sarcastic_classifier.joblib"),
                              "calibration": sarcasm["real_dev_calibration"],
                              "release_recommendation": "Keep experimental; real-dev sample is too small",
                              "slices": sarcasm["slices"]}
    demographic = report("demographics")
    candidate = report("demographics_candidate")
    for task, key in (("mental_state", "mental_f1"), ("age", "age_f1")):
        comparisons[f"demographics:{task}"] = comparison(demographic["neural"][task]["macro_f1"], candidate[key])
    comparisons["strategic"] = comparison(report("strategic")["legacy_reporting"]["composite_f1"],
                                           json.loads((root / "eval_metrics.json").read_text())["composite_f1"],
                                           "composite_f1")
    augmentation = {"comparisons": {}, "datasets": [],
                    "vm_checkpoint_root": "/home/s0r0qw6/model_audits/20261031/synthetic_augmentation",
                    "method": "Frozen deployed base copies; train-only augmentation; unchanged dev/reporting; dev-selected checkpoints",
                    "release_recommendation": "No promotion: real-comment macro-F1 regressed for both candidates"}
    for family in ("stance", "tone"):
        path = root / f"{family}_sonnet_evaluation.json"
        if not path.exists():
            continue
        baseline = report(family)["reporting"]
        augmented = json.loads(path.read_text())["results"][0]
        augmentation["comparisons"][family] = {
            **comparison(baseline["macro_f1"], augmented["macro_f1"]),
            "real_comments": comparison(baseline["slices"]["origin:real_llm_labelled"]["macro_f1"],
                                        augmented["slices"]["origin:real_llm_labelled"]["macro_f1"]),
            "report_file": str(path),
        }
    for name in ("stance_corpus_short_sonnet_20261031", "stance_corpus_micro_sonnet_20261031"):
        directory = root.parent / name
        path = directory / "dataset/summary.json"
        if path.exists():
            state = json.loads((directory / "generation_state.json").read_text())
            augmentation["datasets"].append({"directory": str(directory),
                                              "summary": json.loads(path.read_text()),
                                              "model": state["model"], "profile": state["profile"],
                                              "response_calls": len(state["usage"]),
                                              "output_tokens": sum(row["output_tokens"] for row in state["usage"])})
    return {"comparisons": comparisons, "civic_baseline": report("civic"),
            "sonnet_synthetic_augmentation": augmentation,
            "relevance_calibration": report("relevance"),
            "real_comment_slices": {family: report(family)["reporting"]["slices"].get("origin:real_llm_labelled")
                                    for family in ("stance", "tone")},
            "data_quality": json.loads((clean / "split_audit.json").read_text()),
            "report_files": sorted(str(file) for file in root.glob("*_evaluation.json")
                                   if file.name != "summary_evaluation.json"),
            "vm_candidate_archive": "/home/s0r0qw6/model_audits/20261031",
            "limitations": ["Cleaned legacy reporting sets are not new independently labelled gold",
                            "Conflict quarantine consulted labels across splits; scores are conditional on that filtering",
                            "Legacy strategic/demographic benchmarks and the v5 event threshold informed old selection",
                            "Sarcasm, stance and tone labels are LLM judgements; gains need human verification",
                            "Text-derived age/gender labels are not verified personal demographics",
                            "Joblib artifacts were evaluated across sklearn 1.7.2, 1.9.0 and 1.9.1 versions",
                            "No production classifier checkpoint was replaced, including bert-party-support-17"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", choices=("civic", "stance", "tone", "relevance", "political", "demographics", "strategic", "sarcasm", "summary"), required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--clean-splits", type=Path, default=Path("artifacts/model_audit_20261031/clean_splits"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/model_audit_20261031"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    result = evaluate(args.family, args)
    (args.output / f"{args.family}_evaluation.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Completed {args.family}: {args.output / (args.family + '_evaluation.json')}", flush=True)


if __name__ == "__main__":
    main()