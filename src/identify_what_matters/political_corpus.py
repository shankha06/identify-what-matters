"""Political/civic undercurrent corpus for Indian Instagram posts and comments (synthetic OpenRouter data
+ real pipeline captions/comments re-labelled by the same LLM pipeline).

Builds the labelled dataset behind the pipeline the data-requirements doc asks for instead of hard-coded
"Pro-BJP"/"Anti-Congress" account labels:

    account -> post/comment -> political/non-political -> issue cluster -> sentiment -> target/direction
    -> narrative type -> (engagement is already measured elsewhere) -> undercurrent score

Each row is one post OR one comment (both share the same label schema, since a comment can disagree with
its own post's framing). Labels are per TEXT, never per account/page/subreddit - see `unsafe()` and the
prompt below, which forbid the model from using @handles as a cue.

Commands:
  generate    -- OpenRouter writes realistic Indian political/civic posts with labelled replies
  label-real  -- re-labels real captions/comments already captured in output/runs/*/report.json
  build       -- issue-cluster-disjoint-by-round train/dev/test rows for the five classifiers

Labels are LLM judgements, not human gold - see docs/TRAINING_RESULTS.md for the promotion gate before any
of these classifiers becomes the default (replacing political_pulse.py's keyword/regex scoring).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from identify_what_matters.llm import make_backend_client
from identify_what_matters.topic_corpus import NOISY_STYLE, REGIONAL_NAMES, SPLIT_OF_ROUND, unsafe
from identify_what_matters.utils import Budget, now_iso as now, parse_json_payload as parse_payload, write_json, write_rows

MODEL = "claude-sonnet-4-6"

# The data-requirements doc's "issue cluster" taxonomy, plus two buckets for negative/contrast examples.
ISSUE_CLUSTERS = (
    "employment", "inflation", "economy", "farmers", "education", "defence", "corruption",
    "communal", "women_safety", "caste", "infrastructure", "welfare", "governance", "elections",
)
NON_ISSUE_DOMAINS = ("non_political_everyday", "non_political_entertainment")
DOMAINS = ISSUE_CLUSTERS + NON_ISSUE_DOMAINS

# Real party/leader names are used as plain text (never as @handles or #hashtags - see `unsafe()`),
# matching the existing precedent in demographics_trainer.py's seed samples.
TARGETS = (
    "BJP", "Congress", "AAP", "TMC", "DMK", "AIADMK", "Shiv Sena", "NCP", "CPIM", "RJD", "JDU",
    "BSP", "SP", "Other party", "Government", "Opposition", "Election Commission", "Judiciary",
    "Police", "None",
)
SENTIMENTS = ("positive", "negative", "neutral", "mixed")
DIRECTIONS = ("pro_incumbent", "anti_incumbent", "pro_opposition", "anti_opposition", "neutral")
NARRATIVES = (
    "government_achievement", "governance_criticism", "corruption_allegation", "mobilization_call",
    "satire_meme", "human_interest", "fact_check", "communal_framing", "other",
)
_lock = threading.Lock()


def regional_for(domain: str, round_index: int) -> str:
    codes = list(REGIONAL_NAMES)
    return codes[(DOMAINS.index(domain) * 5 + round_index) % len(codes)]


def prompt_for(domain: str, round_index: int, used: list[str]) -> str:
    regional = regional_for(domain, round_index)
    is_non_political = domain.startswith("non_political")
    topic_line = (
        f"Issue cluster: {domain.replace('_', ' ')} in Indian politics/civic life."
        if not is_non_political else
        f"Non-political Indian Instagram content: {domain.replace('non_political_', '').replace('_', ' ')} "
        "(no party, politician, election, or government criticism/praise at all)."
    )
    return (
        "Generate training data for a model that reads a single Indian Instagram post or comment and labels: "
        "whether it is political, which civic issue it discusses, its sentiment, which party/institution it targets, "
        "its political direction, and its narrative framing. " + NOISY_STYLE +
        f"{topic_line}\n"
        "Create exactly 3 DIFFERENT fictional-but-realistic Instagram posts for this cluster. Real Indian party "
        "names and real politicians' names (plain text only, e.g. 'Modi', 'Rahul Gandhi', 'BJP', 'Congress') may be "
        "used naturally, the way real Instagram comments do - but NEVER as @handles or #hashtags, and never invent "
        "a scandal/claim about a named real person (keep claims generic/policy-level, e.g. 'the government's new "
        "policy' rather than a specific fabricated act). No URLs, emails, or phone numbers.\n"
        f"Avoid repeating these already-used captions: {'; '.join(used) or 'none yet'}.\n"
        "For each post give: caption (60-250 chars, English/Hindi/Hinglish), and label it with:\n"
        "  is_political (true/false), issue_cluster (one of: " + ", ".join(ISSUE_CLUSTERS) + ", or 'none' if not "
        "political), sentiment (positive/negative/neutral/mixed - the post's own emotional valence, independent of "
        "target), target (who/what the post is about: one of " + ", ".join(TARGETS) + "), direction (one of "
        "pro_incumbent/anti_incumbent/pro_opposition/anti_opposition/neutral - neutral if purely informational or "
        "non-political), narrative_type (one of " + ", ".join(NARRATIVES) + ").\n"
        "Then write exactly 6 realistic REPLY COMMENTS to that post (mix of agreeing, disagreeing, sarcastic, and "
        "off-topic/joke replies - a comment's own stance can differ from the post's framing). Label each comment "
        f"with the same 5 fields. Comments 6-200 chars, mixed {REGIONAL_NAMES[regional]} ({regional})/Hindi/"
        "Hinglish/English.\n"
        'Return ONLY JSON: {"posts": [{"key": "p0", "caption": "...", "is_political": true, "issue_cluster": "...", '
        '"sentiment": "...", "target": "...", "direction": "...", "narrative_type": "...", '
        '"comments": [{"text": "...", "is_political": true, "issue_cluster": "...", "sentiment": "...", '
        '"target": "...", "direction": "...", "narrative_type": "..."}]}]} with keys p0, p1, p2.'
    )


def _valid_item(item: dict, is_comment: bool) -> dict | None:
    text = str(item.get("text" if is_comment else "caption", "")).strip()
    if not (6 if is_comment else 40) <= len(text) <= (200 if is_comment else 300) or unsafe(text):
        return None
    sentiment, target = str(item.get("sentiment")), str(item.get("target"))
    direction, narrative = str(item.get("direction")), str(item.get("narrative_type"))
    issue = str(item.get("issue_cluster"))
    if sentiment not in SENTIMENTS or target not in TARGETS or direction not in DIRECTIONS or narrative not in NARRATIVES:
        return None
    if issue not in ISSUE_CLUSTERS and issue != "none":
        return None
    return {
        "text": text, "is_political": bool(item.get("is_political")), "issue_cluster": issue,
        "sentiment": sentiment, "target": target, "direction": direction, "narrative_type": narrative,
    }


def validate(payload: dict, domain: str, round_index: int, split: str | None = None) -> list[dict]:
    posts = payload.get("posts") if isinstance(payload, dict) else None
    if not isinstance(posts, list) or not posts:
        raise ValueError("no posts")
    split = split or SPLIT_OF_ROUND[round_index]
    out = []
    for index, raw in enumerate(posts):
        if not isinstance(raw, dict):
            continue
        post_label = _valid_item(raw, is_comment=False)
        if post_label is None:
            continue
        comments = []
        for comment in raw.get("comments") or []:
            if not isinstance(comment, dict):
                continue
            label = _valid_item(comment, is_comment=True)
            if label is not None:
                comments.append(label)
        if len(comments) < 3:
            continue
        key = hashlib.sha256(post_label["text"].casefold().encode()).hexdigest()[:16]
        out.append({"discussion_id": f"{domain}.r{round_index}.{index}", "domain": domain, "split": split,
                    "origin": "synthetic_llm", "post": post_label, "comments": comments, "hash": key})
    if not out:
        raise ValueError("no valid posts after filtering")
    return out


def generate(args) -> None:
    out = Path(args.output)
    (out / "calls").mkdir(parents=True, exist_ok=True)
    (out / "rejected").mkdir(parents=True, exist_ok=True)
    client = make_backend_client(model=args.model, max_tokens=8192)
    state_path = out / "generation_state.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {"calls": 0, "usage": [], "failures": []}
    state_lock = threading.Lock()
    budget = Budget(state, args.max_calls, args.max_tokens * 10_000, state_lock)

    def run_domain(domain: str) -> None:
        used: list[str] = []
        for round_index in range(args.rounds):
            checkpoint = out / "calls" / f"{domain}.r{round_index}.json"
            if checkpoint.exists():
                continue
            for attempt in range(args.attempts):
                if not budget.reserve(args.max_tokens * 2, calls=2):
                    return
                try:
                    response = client.generate(
                        user_prompt=prompt_for(domain, round_index, used), request_timeout=240,
                        system_instruction="You produce strict JSON datasets. Output only the JSON object.")
                except Exception as error:
                    with state_lock:
                        state["failures"].append({"call": f"{domain}.r{round_index}", "category": type(error).__name__})
                    continue
                with state_lock:
                    state["usage"].append({"call": f"{domain}.r{round_index}", "input_tokens": response.input_tokens,
                                           "output_tokens": response.output_tokens})
                try:
                    discussions = validate(parse_payload(response.text), domain, round_index, args.split)
                except Exception as error:
                    (out / "rejected" / f"{domain}.r{round_index}.a{attempt}.txt").write_text(response.text)
                    with state_lock:
                        state["failures"].append({"call": f"{domain}.r{round_index}", "category": str(error)[:60]})
                    print(json.dumps({"call": f"{domain}.r{round_index}", "status": "rejected", "why": str(error)[:60]}), flush=True)
                    continue
                write_json(checkpoint, {"model": args.model, "generated": now(), "discussions": discussions})
                used.extend(d["post"]["text"][:80] for d in discussions)
                print(json.dumps({"call": f"{domain}.r{round_index}", "status": "validated", "posts": len(discussions)}), flush=True)
                break
            with state_lock:
                write_json(state_path, state)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(run_domain, args.only or DOMAINS))
    write_json(state_path, state)


LABEL_REAL_PROMPT = (
    "Label each of these real Indian Instagram posts/comments for a political/civic-issue undercurrent dashboard. "
    "Do NOT infer the label from who posted it (you are not told the account) - judge the text alone.\n"
    "For each item return: is_political (true/false), issue_cluster (one of: " + ", ".join(ISSUE_CLUSTERS) +
    ", or 'none'), sentiment (one of " + ", ".join(SENTIMENTS) + "), target (one of " + ", ".join(TARGETS) +
    "), direction (one of " + ", ".join(DIRECTIONS) + "), narrative_type (one of " + ", ".join(NARRATIVES) + ").\n"
    'Return ONLY JSON: {"labels": [{"i": 0, "is_political": true, "issue_cluster": "...", "sentiment": "...", '
    '"target": "...", "direction": "...", "narrative_type": "..."}]} for every index given, in order.'
)


def label_real(args) -> None:
    """Pulls real captions + sampled comments already collected by trend runs and asks the same LLM to apply
    the richer political/issue/direction/narrative schema - the same "LLM judgement, not human gold" approach
    docs/TRAINING_RESULTS.md already uses for stance/tone/relevance. No report.json field currently stores
    issue_cluster/target/direction/narrative, so this is the only way to get real-text examples for now."""
    client = make_backend_client(model=args.model, max_tokens=8192)
    items: list[dict] = []
    reports = sorted(report for root in args.runs for report in Path(root).glob("*/report.json"))
    seen_text = set()
    for report in reports:
        for topic in json.loads(report.read_text()).get("topics", []):
            brief = topic.get("brief") or {}
            caption_source = topic.get("title", "")
            if caption_source and caption_source not in seen_text and len(caption_source) >= 40:
                seen_text.add(caption_source)
                items.append({"text": caption_source[:300], "kind": "post", "topic": topic.get("topic_key")})
            for comment in (brief.get("comment_pool") or [])[:8]:
                text = (comment or {}).get("text", "").strip()
                if text and 6 <= len(text) <= 200 and text not in seen_text:
                    seen_text.add(text)
                    items.append({"text": text, "kind": "comment", "topic": topic.get("topic_key")})
    if not items:
        print(json.dumps({"real_items": 0, "note": "no report.json found under the given --runs roots"}))
        return
    rng = random.Random(args.seed)
    rng.shuffle(items)
    items = items[: args.limit]
    labelled = []
    for batch_start in range(0, len(items), args.batch_size):
        batch = items[batch_start: batch_start + args.batch_size]
        prompt = LABEL_REAL_PROMPT + "\n\n" + json.dumps(
            [{"i": i, "text": item["text"]} for i, item in enumerate(batch)], ensure_ascii=False)
        try:
            response = client.generate(user_prompt=prompt, request_timeout=180,
                                       system_instruction="You produce strict JSON. Output only the JSON object.")
            result = parse_payload(response.text)
            by_index = {row["i"]: row for row in result.get("labels", []) if isinstance(row, dict) and "i" in row}
        except Exception as error:
            print(json.dumps({"batch": batch_start, "status": "failed", "error": type(error).__name__}), flush=True)
            continue
        for i, item in enumerate(batch):
            row = by_index.get(i)
            if row is None:
                continue
            label = _valid_item({**row, "text": item["text"], "caption": item["text"]}, is_comment=(item["kind"] == "comment"))
            if label is None:
                continue
            key = hashlib.sha256(item["text"].casefold().encode()).hexdigest()[:16]
            bucket = int(key[:8], 16) % 10
            labelled.append({"discussion_id": f"real.{item['kind']}.{key}", "domain": label["issue_cluster"] or "none",
                             "split": "test" if bucket < 2 else "dev" if bucket < 3 else "train",
                             "origin": "real_llm_labelled", "kind": item["kind"], "topic": item["topic"], **label})
        print(json.dumps({"batch": batch_start, "status": "labelled", "count": len(batch)}), flush=True)
    write_rows(Path(args.output) / "real_labelled.jsonl", labelled)
    print(json.dumps({"real_items": len(labelled), "by_kind": {k: sum(r["kind"] == k for r in labelled) for k in ("post", "comment")},
                      "splits": {s: sum(r["split"] == s for r in labelled) for s in ("train", "dev", "test")}}, indent=2))


def build(args) -> None:
    out = Path(args.output)
    rows = {"train": [], "dev": [], "test": []}

    def add_row(label: dict, split: str, origin: str, domain: str) -> None:
        rows[split].append({**label, "origin": origin, "domain": domain})

    for checkpoint in sorted((out / "calls").glob("*.json")):
        for discussion in json.loads(checkpoint.read_text())["discussions"]:
            add_row(discussion["post"], discussion["split"], discussion["origin"], discussion["domain"])
            for comment in discussion["comments"]:
                add_row(comment, discussion["split"], discussion["origin"], discussion["domain"])
    real_path = out / "real_labelled.jsonl"
    if real_path.exists():
        for line in real_path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            add_row({k: row[k] for k in ("text", "is_political", "issue_cluster", "sentiment", "target",
                                         "direction", "narrative_type")}, row["split"], row["origin"], row["domain"])

    rng = random.Random(args.seed)
    for split in rows:
        rng.shuffle(rows[split])
        write_rows(Path(args.output) / "dataset" / f"{split}.jsonl", rows[split])

    summary = {
        "built": now(), "human_gold": False,
        "rows": {s: len(v) for s, v in rows.items()},
        "by_origin": {o: sum(r["origin"] == o for v in rows.values() for r in v)
                     for o in {r["origin"] for v in rows.values() for r in v}},
        "by_issue_cluster": {s: {c: sum(r["issue_cluster"] == c for r in v) for c in ISSUE_CLUSTERS} for s, v in rows.items()},
        "political_share": {s: round(sum(r["is_political"] for r in v) / len(v), 3) if v else None for s, v in rows.items()},
    }
    write_json(Path(args.output) / "dataset" / "summary.json", summary)
    print(json.dumps(summary, indent=2, default=str))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)

    generation = commands.add_parser("generate")
    generation.add_argument("--output", default="artifacts/political_corpus")
    generation.add_argument("--rounds", type=int, default=8, choices=range(1, 9))
    generation.add_argument("--attempts", type=int, default=2)
    generation.add_argument("--workers", type=int, default=8)
    generation.add_argument("--max-calls", type=int, default=200)
    generation.add_argument("--max-tokens", type=int, default=4000)
    generation.add_argument("--model", default=MODEL)
    generation.add_argument("--only", nargs="*", choices=DOMAINS)
    generation.add_argument("--split", choices=("train", "dev", "test"))
    generation.set_defaults(func=generate)

    labeller = commands.add_parser("label-real")
    labeller.add_argument("--runs", nargs="+", default=["output/runs"])
    labeller.add_argument("--output", default="artifacts/political_corpus")
    labeller.add_argument("--model", default=MODEL)
    labeller.add_argument("--batch-size", type=int, default=15)
    labeller.add_argument("--limit", type=int, default=600)
    labeller.add_argument("--seed", type=int, default=20261007)
    labeller.set_defaults(func=label_real)

    builder = commands.add_parser("build")
    builder.add_argument("--output", default="artifacts/political_corpus")
    builder.add_argument("--seed", type=int, default=20261007)
    builder.set_defaults(func=build)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
