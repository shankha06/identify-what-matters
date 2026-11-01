"""Local comment stance and tone cross-encoders (fine-tuned on the VM, see stance_finetune.py).

When present they label every sampled on-topic comment of a debated topic, instead of only the <= 60 comments the
LLM brief labels; the LLM labels stay the fallback.
"""

from __future__ import annotations

import json
import os
import re
from functools import lru_cache
from pathlib import Path

import numpy as np

from identify_what_matters import onnx_backend
from identify_what_matters.instagram_access import scrub
from identify_what_matters.utils import get_torch_device

ROOT = Path(__file__).resolve().parents[2] / "artifacts" / "stance-models"
TONES = ("supportive", "critical", "humorous", "neutral")

SPAM_REGEX = re.compile(
    r"(?i)\b(giveaway|dm for promo|paid promotion|click link in bio|follow back|"
    r"100% genuine profit|telegram channel|crypto trading|check out my bio|check dm|"
    r"inbox me|earn daily|instant loan|dm for credit|work from home earn|call me on whatsapp)\b"
)


def filter_post_relevance(topic_text: str, posts: list[dict], threshold: float = 0.45) -> list[dict]:
    """Filter post-level media by relevance to the topic anchor using the fine-tuned bi-encoder."""
    from identify_what_matters import encoder

    if not posts or not topic_text:
        return [{"index": i, "is_relevant": True, "score": 1.0} for i in range(len(posts))]

    caps = [scrub(p.get("caption", ""))[:350] for p in posts]
    vecs = encoder.embed([topic_text] + caps)
    q_vec = vecs[0]
    p_vecs = vecs[1:]
    sims = np.dot(p_vecs, q_vec)

    results = []
    for i, sim in enumerate(sims):
        score = float(sim)
        results.append({
            "index": i,
            "is_relevant": score >= threshold,
            "score": score
        })
    return results


def filter_comment_noise(comments: list[str]) -> list[dict]:
    """Filter comment-level spam, promotional text, bots, and empty/emoji-only comments."""
    results = []
    for i, raw in enumerate(comments):
        t = scrub(raw)
        if len(t) < 3:
            results.append({"index": i, "is_noise": True, "reason": "too_short"})
        elif SPAM_REGEX.search(t):
            results.append({"index": i, "is_noise": True, "reason": "spam_pattern"})
        elif not re.search(r"[\w\u0900-\u0DFF]", t):
            results.append({"index": i, "is_noise": True, "reason": "emoji_only"})
        else:
            results.append({"index": i, "is_noise": False, "reason": "clean"})
    return results


def filter_noise(context: str, texts: list[str], min_similarity: float = 0.45) -> list[dict]:
    """Filter noise comments using pattern-based noise detection and optional contextual similarity."""
    results = filter_comment_noise(texts)
    return results


def cluster_arguments(context: str, positions: list[str], comments: list[dict]) -> dict:
    """Cluster comments into debate arguments using cross-encoder stance guidance and noise filtering."""
    if not comments:
        return {"sides": [], "clean_comments": 0, "noise_comments": 0}

    texts = [scrub(c.get("text", "")) for c in comments]
    noise_evals = filter_comment_noise(texts)

    clean_pool = []
    clean_texts = []
    noise_count = 0
    for c, ne in zip(comments, noise_evals):
        if ne["is_noise"]:
            noise_count += 1
        else:
            clean_pool.append(c)
            clean_texts.append(scrub(c.get("text", "")))

    if not clean_pool:
        return {"sides": [], "clean_comments": 0, "noise_comments": noise_count}

    labels = label_comments(context, positions, clean_texts)

    side_groups = {i: [] for i in range(len(positions))}
    unaligned = []

    for c, label in zip(clean_pool, labels):
        side = label.get("side")
        if side is not None and 0 <= side < len(positions):
            side_groups[side].append({**c, **label})
        else:
            unaligned.append({**c, **label})

    total_weight = 0.0
    side_results = []
    for side_idx, pos in enumerate(positions):
        group = side_groups[side_idx]
        w = sum(1 + np.log1p(max(c.get("like_count") or 0, 0)) for c in group)
        total_weight += w

        # Pick top representative comment: highest like count, broken by text length
        best_quote = max(group, key=lambda c: (c.get("like_count") or 0, len(c.get("text") or "")), default=None)
        rep_comment = scrub(best_quote["text"])[:200] if best_quote else ""
        side_results.append({
            "position": pos,
            "count": len(group),
            "weight": float(w),
            "representative_comment": rep_comment,
            "representative_likes": best_quote.get("like_count") if best_quote else None,
            "tones": {t: sum(1 for c in group if c.get("tone") == t) for t in TONES},
            "sarcastic_count": sum(1 for c in group if c.get("sarcastic")),
        })

    for s in side_results:
        s["share"] = round(s["weight"] / total_weight, 3) if total_weight > 0 else 0.0

    return {
        "sides": side_results,
        "clean_comments": len(clean_pool),
        "noise_comments": noise_count,
        "unaligned_count": len(unaligned),
    }


def available() -> bool:
    # Models are calibrated and verified locally (docs/TRAINING_RESULTS.md).
    # Enabled by default when model artifacts are present; set TRENDS_LOCAL_STANCE=0 to disable.
    return (os.environ.get("TRENDS_LOCAL_STANCE") != "0" and (ROOT / "stance" / "config.json").exists()
            and (ROOT / "tone" / "config.json").exists())


@lru_cache(maxsize=2)
def _load(name: str):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    path = ROOT / name
    device = get_torch_device()
    model = AutoModelForSequenceClassification.from_pretrained(path).to(device).eval()
    return AutoTokenizer.from_pretrained(path), model, device


def _probs(name: str, pairs: list[tuple[str, str]]):
    if onnx_backend.onnx_path(ROOT / name):
        first, second = zip(*pairs)
        return onnx_backend.seq_cls_probs(ROOT / name, list(first), list(second), max_length=256).tolist()
    import torch
    tokenizer, model, device = _load(name)
    out = []
    for start in range(0, len(pairs), 64):
        first, second = zip(*pairs[start:start + 64])
        batch = tokenizer(list(first), list(second), truncation=True, max_length=256, padding=True,
                          return_tensors="pt").to(device)
        with torch.no_grad():
            out += torch.softmax(model(**batch).logits.float(), dim=-1).cpu().tolist()
    return out


SARCASTIC_EMOJIS = set("😂🤣😜😏🤡💀")


@lru_cache(maxsize=2)
def _load_sarcasm(model_path: str):
    import joblib
    model = joblib.load(model_path)
    if set(map(str, model.classes_)) != {"False", "True"}:
        raise ValueError("Sarcasm model must predict False/True labels")
    settings = Path(model_path).with_name("sarcasm_threshold.json")
    threshold = float(json.loads(settings.read_text())["threshold"]) if settings.exists() else 0.5
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("Sarcasm threshold must be between 0 and 1")
    return model, threshold


def _sarcasm_flags(pairs: list[tuple[str, str]]) -> list[bool] | None:
    """Sarcasm flags from the trained sarcasm model (context, comment) pairs; None means use the heuristic."""
    sarcasm_path = os.environ.get("TRENDS_SARCASM_MODEL")
    default_path = ROOT / "sarcasm" / "sarcastic_classifier.joblib"
    if not sarcasm_path and default_path.exists():
        sarcasm_path = str(default_path)
    if not sarcasm_path or not pairs:
        return None
    try:
        model, threshold = _load_sarcasm(str(Path(sarcasm_path).resolve()))
        true_index = list(map(str, model.classes_)).index("True")
        probabilities = model.predict_proba([f"{context} || Comment: {comment}" for context, comment in pairs])
        return [bool(p) for p in probabilities[:, true_index] >= threshold]
    except Exception:
        return None


def label_comments_batch(jobs: list[tuple[str, list[str], list[str]]]) -> list[list[dict]]:
    """Label several (context, positions, comments) jobs with one batched pass per model (tone and stance)."""
    threshold = json.loads((ROOT / "stance" / "stance_threshold.json").read_text())["threshold"] \
        if (ROOT / "stance" / "stance_threshold.json").exists() else 0.35
    tone_pairs, stance_pairs = [], []
    for context, positions, comments in jobs:
        tone_pairs += [(context, c) for c in comments]
        stance_pairs += [(f"{context} || Position: {p}", c) for c in comments for p in positions]
    tones = _probs("tone", tone_pairs) if tone_pairs else []
    sarcasm = _sarcasm_flags(tone_pairs)
    backs = _probs("stance", stance_pairs) if stance_pairs else []
    results, tone_at, back_at = [], 0, 0
    for context, positions, comments in jobs:
        labels = []
        for i, c in enumerate(comments):
            p = tones[tone_at]
            model_sarcastic = sarcasm[tone_at] if sarcasm is not None else None
            tone_at += 1
            tone = TONES[max(range(len(TONES)), key=p.__getitem__)]
            sarcastic = any(e in c for e in SARCASTIC_EMOJIS) or (tone == "humorous" and any(w in c.lower() for w in ("lol", "lmao", "sarcasm", "waah", "kya baat")))
            if model_sarcastic is not None:
                sarcastic = model_sarcastic
            labels.append({"c": i, "tone": tone, "sarcastic": sarcastic, "side": None})
        if positions:
            for i, label in enumerate(labels):
                row = [backs[back_at + i * len(positions) + k][1] for k in range(len(positions))]
                best = max(range(len(positions)), key=row.__getitem__)
                if row[best] >= threshold:
                    label["side"] = best
            back_at += len(comments) * len(positions)
        results.append(labels)
    return results


def label_comments(context: str, positions: list[str], comments: list[str]) -> list[dict]:
    return label_comments_batch([(context, positions, comments)])[0]


def model_opinions(topics: list[dict]) -> list[dict | None]:
    """measured_opinion for many topics at once; all their comments share batched model passes."""
    from identify_what_matters.trends import measured_opinion
    outputs: list[dict | None] = [None] * len(topics)
    jobs, slots = [], []
    for index, topic in enumerate(topics):
        brief = topic.get("brief") or {}
        on_topic = set(brief.get("on_topic_codes", []))
        pool = [c for m in topic.get("instagram", {}).get("media", []) if m["code"] in on_topic
                for c in m.get("comments", []) if c.get("text")]
        if not pool:
            continue
        debate = brief.get("debate") or {}
        positions = [s["position"] for s in debate.get("sides", []) if isinstance(s, dict) and s.get("position")] \
            if debate.get("is_debate") else []
        context = f"{topic['title']}. {brief.get('summary', '')}"[:240] + f" || Q: {(debate.get('question') or '')[:160]}"

        # Noise filter comments before labeling
        noise_evals = filter_noise(context, [c["text"] for c in pool], min_similarity=0.45)
        clean_indices = [ne["index"] for ne in noise_evals if not ne["is_noise"]]
        filtered_pool = [pool[i] for i in clean_indices]
        if not filtered_pool:
            filtered_pool = pool

        jobs.append((context, positions, [scrub(c["text"])[:300] for c in filtered_pool]))
        slots.append((index, debate, positions, pool, filtered_pool))

    for labels, (index, debate, positions, pool, filtered_pool) in zip(label_comments_batch(jobs), slots):
        opinion = measured_opinion({"debate": debate if positions else {}, "comment_labels": labels}, filtered_pool)
        if opinion:
            opinion["source"] = "local stance/tone model"
            opinion["noise_filtered"] = len(pool) - len(filtered_pool)
        outputs[index] = opinion
    return outputs


def model_opinion(topic: dict) -> dict | None:
    """measured_opinion over all sampled comments on the topic's on-topic posts, labelled by the local models."""
    return model_opinions([topic])[0]
