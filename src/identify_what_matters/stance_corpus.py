"""Comment stance and tone corpus for Indian Instagram debates (synthetic OpenRouter data + real pipeline labels).

Commands:
  generate     -- OpenRouter writes fictional contested discussions with labelled comments (noisy, code-mixed)
  export-real  -- per-comment labels that trend briefs already produced on real comments (report.json)
  build        -- event-disjoint train/dev/test rows for the stance (pair) and tone models

A stance row pairs a debate position with one comment: label 1 if the comment backs that position. Positions
are free text, so one model serves any debate with any number of sides. Labels are LLM judgements, not human gold.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from identify_what_matters.openrouter import DEFAULT_MODEL, OpenRouterClient
from identify_what_matters.topic_corpus import (NOISY_STYLE, REGIONAL, REGIONAL_NAMES, SPLIT_OF_ROUND, Budget, now,
                                       parse_payload, unsafe, write_json, write_rows)

MODEL = DEFAULT_MODEL
DOMAINS = (
    "elections_and_voter_rolls", "parliament_and_new_laws", "state_politics_and_defections",
    "religion_and_places_of_worship", "caste_and_reservation", "police_action_and_protests",
    "prices_tax_and_budget", "exams_jobs_and_paper_leaks", "city_infrastructure_and_civic_failures",
    "women_safety_and_courts", "language_and_regional_identity", "celebrity_controversy",
    "cricket_selection_and_results", "film_release_and_boycotts", "india_and_neighbours", "farmers_and_rural_policy",
)
TONES = ("supportive", "critical", "humorous", "neutral")
LANGUAGE_PLAN = "5 hinglish, 4 en, 3 hi (Devanagari), 3 {regional} (native script), 1 any"


def regional_for(domain: str, round_index: int) -> str:
    codes = list(REGIONAL)
    return codes[(DOMAINS.index(domain) * 5 + round_index) % len(codes)]


def prompt_for(domain: str, round_index: int, used: list[str]) -> str:
    regional = regional_for(domain, round_index)
    return (
        "Generate training data for a model that reads Indian Instagram comments under a contested post and decides "
        "which side of the debate each comment backs, and its tone.\n" + NOISY_STYLE +
        f"Domain: {domain.replace('_', ' ')} in India.\n"
        "Create exactly 2 DIFFERENT fictional but realistic contested discussions in this domain. Refer to people by "
        "role only (the minister, the chief minister, an opposition MP, the actor, the captain); generic real "
        "institutions are fine. No real personal names, @handles, #hashtags, URLs, emails or phone numbers. Heated "
        "but no slurs, threats or hate against any group. Never use the double-quote character inside text values.\n"
        f"Avoid these already used discussions: {'; '.join(used) or 'none yet'}.\n"
        "For each discussion give: caption (the Instagram post, 60-260 chars, English or Hinglish), question (the "
        "contested question, one neutral line), positions (exactly 3: two opposing sides and one third view such as "
        "'both sides are playing politics' or 'wait for the inquiry'; each one sentence), and exactly 16 comments.\n"
        f"Comment languages: {LANGUAGE_PLAN.format(regional=REGIONAL_NAMES[regional] + ' (' + regional + ')')}.\n"
        "Comment mix: at least 4 back position 0, at least 4 back position 1, at least 2 back position 2; at least 3 "
        "sarcastic comments whose literal words point the other way (e.g. praise that is really mockery); at least 2 "
        "that take no side (pure jokes, questions, tagging friends style chatter); exactly 1 off-topic spam or promo. "
        "Some comments must back a side only implicitly (no keyword overlap with the position). Comments 6-200 chars.\n"
        "Labels per comment: side (0, 1, 2 or null), tone toward the post's subject (supportive, critical, humorous, "
        "neutral; label the intended meaning of sarcasm), sarcastic (true/false), off_topic (true/false), language "
        f"(en, hi, hinglish or {regional}).\n"
        'Return ONLY JSON: {"discussions": [{"key": "d0", "caption": "...", "question": "...", "positions": ["...", '
        '"...", "..."], "comments": [{"text": "...", "language": "hinglish", "side": 0, "tone": "critical", '
        '"sarcastic": false, "off_topic": false}]}]} with keys d0, d1.'
    )


def validate(payload: dict, domain: str, round_index: int, split: str | None = None) -> list[dict]:
    regional = regional_for(domain, round_index)
    discussions = payload.get("discussions") if isinstance(payload, dict) else None
    if not isinstance(discussions, list) or len(discussions) != 2:
        raise ValueError("discussion_count")
    accepted, seen = [], set()
    for position, item in enumerate(discussions):
        caption, question, positions, comments = (item.get("caption"), item.get("question"), item.get("positions"),
                                                  item.get("comments"))
        if not (isinstance(caption, str) and 30 <= len(caption) <= 400 and isinstance(question, str)):
            raise ValueError("caption_or_question")
        if not (isinstance(positions, list) and len(positions) == 3 and all(isinstance(p, str) and p for p in positions)):
            raise ValueError("positions")
        if not (isinstance(comments, list) and 14 <= len(comments) <= 18):
            raise ValueError("comment_count")
        cleaned = []
        for comment in comments:
            text = " ".join(str(comment.get("text", "")).split())
            language, side, tone = comment.get("language"), comment.get("side"), comment.get("tone")
            if not 4 <= len(text) <= 260 or unsafe(text) or text.casefold() in seen:
                raise ValueError("comment_text")
            if side not in (0, 1, 2, None) or tone not in TONES or language not in ("en", "hi", "hinglish", regional):
                raise ValueError("comment_labels")
            if language == "hi" and not re.search("[\u0900-\u097f]", text):
                raise ValueError("hindi_script")
            if language == regional and not re.search(f"[{REGIONAL[regional]}]", text):
                language = f"{regional}_latn"
            seen.add(text.casefold())
            cleaned.append({"text": text, "language": language, "side": side, "tone": tone,
                            "sarcastic": bool(comment.get("sarcastic")), "off_topic": bool(comment.get("off_topic"))})
        sides = [c["side"] for c in cleaned if not c["off_topic"]]
        if sides.count(0) < 3 or sides.count(1) < 3:
            raise ValueError("side_balance")
        if any(unsafe(text) for text in [caption, question, *positions]):
            raise ValueError("identifier")
        accepted.append({"discussion_id": f"{domain}.r{round_index}.d{position}", "domain": domain,
                         "round": round_index, "split": split or SPLIT_OF_ROUND[round_index], "origin": "synthetic",
                         "caption": " ".join(caption.split()), "question": " ".join(question.split()),
                         "positions": [" ".join(p.split()) for p in positions], "comments": cleaned})
    return accepted


def generate(args):
    out = Path(args.output)
    (out / "calls").mkdir(parents=True, exist_ok=True)
    (out / "rejected").mkdir(parents=True, exist_ok=True)
    state_path = out / "generation_state.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {
        "created": now(), "origin": "synthetic", "model": args.model, "calls": 0, "usage": [], "failures": []}
    if state.get("model") != args.model:
        raise ValueError("resume_model_mismatch")
    budget = Budget(state, args.max_calls, args.max_output_tokens)
    state_lock = threading.Lock()
    client = OpenRouterClient(model=args.model, max_tokens=args.max_tokens, temperature=0.8)

    def run_domain(domain):
        for round_index in range(args.rounds):
            checkpoint = out / "calls" / f"{domain}.r{round_index}.json"
            if checkpoint.exists():
                continue
            used = [d["question"] for r in range(round_index) if (out / "calls" / f"{domain}.r{r}.json").exists()
                    for d in json.loads((out / "calls" / f"{domain}.r{r}.json").read_text())["discussions"]]
            for attempt in range(args.attempts):
                if not budget.reserve(args.max_tokens * 2, calls=2):
                    return
                try:
                    response = client.generate(user_prompt=prompt_for(domain, round_index, used), request_timeout=240,
                                               system_instruction="You produce strict JSON datasets. Output only the JSON object.")
                except Exception as error:
                    state["failures"].append({"call": f"{domain}.r{round_index}", "category": type(error).__name__})
                    continue
                state["usage"].append({"call": f"{domain}.r{round_index}", "input_tokens": response.input_tokens,
                                       "output_tokens": response.output_tokens})
                try:
                    discussions = validate(parse_payload(response.text), domain, round_index, args.split)
                except Exception as error:
                    (out / "rejected" / f"{domain}.r{round_index}.a{attempt}.txt").write_text(response.text)
                    state["failures"].append({"call": f"{domain}.r{round_index}", "category": str(error)[:60]})
                    print(json.dumps({"call": f"{domain}.r{round_index}", "status": "rejected", "why": str(error)[:60]}), flush=True)
                    continue
                write_json(checkpoint, {"model": args.model, "generated": now(), "discussions": discussions})
                print(json.dumps({"call": f"{domain}.r{round_index}", "status": "validated"}), flush=True)
                break
            with state_lock:
                write_json(state_path, state)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(run_domain, args.only or DOMAINS))
    write_json(state_path, state)


def export_real(args):
    """Real comments already labelled by trend briefs; split by topic hash so a story never crosses splits."""
    rows = []
    reports = sorted(report for root in args.runs for report in Path(root).glob("*/report.json"))
    for report in reports:
        for topic in json.loads(report.read_text())["topics"]:
            brief = topic.get("brief") or {}
            sides = [s.get("position") for s in (brief.get("debate") or {}).get("sides", []) if isinstance(s, dict)]
            pool, labels = brief.get("comment_pool") or [], brief.get("comment_labels") or []
            if not pool or not labels:
                continue
            comments = []
            for label in labels:
                index = label.get("c") if isinstance(label, dict) else None
                if not isinstance(index, int) or not 0 <= index < len(pool) or label.get("tone") not in TONES:
                    continue
                side = label.get("side") if isinstance(label.get("side"), int) and 0 <= label["side"] < len(sides) else None
                comments.append({"text": pool[index]["text"], "language": None, "side": side, "tone": label["tone"],
                                 "sarcastic": bool(label.get("sarcastic")), "off_topic": bool(label.get("off_topic")),
                                 "likes": pool[index]["likes"]})
            if len(comments) < 2:
                continue
            key = hashlib.sha256(topic["title"].casefold().encode()).hexdigest()
            bucket = int(key[:8], 16) % 10
            rows.append({"discussion_id": f"real.{key[:12]}", "domain": topic.get("category", "general"), "origin": "real_llm_labelled",
                         "split": "test" if bucket < 3 else "dev" if bucket < 4 else "train",
                         "caption": f"{topic['title']}. {brief.get('summary', '')}"[:300],
                         "question": (brief.get("debate") or {}).get("question", ""), "positions": sides, "comments": comments})
    unique = list({row["discussion_id"]: row for row in rows}.values())
    write_rows(Path(args.output) / "real_discussions.jsonl", unique)
    print(json.dumps({"real_discussions": len(unique), "comments": sum(len(r["comments"]) for r in unique),
                      "splits": {s: sum(r["split"] == s for r in unique) for s in ("train", "dev", "test")}}))


def context_text(discussion: dict) -> str:
    return f"{discussion['caption'][:240]} || Q: {discussion['question'][:160]}"


def build(args):
    out = Path(args.output)
    discussions = []
    for checkpoint in sorted((out / "calls").glob("*.json")):
        discussions += json.loads(checkpoint.read_text())["discussions"]
    aug_dir = out / "regional_aug"
    if aug_dir.exists():
        for checkpoint in sorted(aug_dir.glob("*.json")):
            try:
                data = json.loads(checkpoint.read_text(encoding="utf-8"))
                discussions.append(data)
            except Exception:
                pass
    real_path = out / "real_discussions.jsonl"
    if real_path.exists():
        discussions += [json.loads(line) for line in real_path.read_text().splitlines() if line.strip()]
    rng = random.Random(args.seed)
    stance = {s: [] for s in ("train", "dev", "test")}
    tone = {s: [] for s in stance}
    for d in discussions:
        context = context_text(d)
        for comment in d["comments"]:
            meta = {"discussion_id": d["discussion_id"], "origin": d["origin"], "language": comment.get("language"),
                    "sarcastic": comment["sarcastic"]}
            tone[d["split"]].append({"context": context, "comment": comment["text"], "label": TONES.index(comment["tone"]),
                                     **meta})
            if comment["off_topic"]:
                continue
            for index, position in enumerate(d["positions"]):
                stance[d["split"]].append({"position": f"{context} || Position: {position}", "comment": comment["text"],
                                           "label": int(comment["side"] == index), "side_index": index,
                                           "gold_side": comment["side"], "n_sides": len(d["positions"]), **meta})
    dataset = out / "dataset"
    for split in stance:
        rng.shuffle(stance[split])
        rng.shuffle(tone[split])
        write_rows(dataset / f"stance_{split}.jsonl", stance[split])
        write_rows(dataset / f"tone_{split}.jsonl", tone[split])
    summary = {"built": now(), "human_gold": False, "discussions": len(discussions),
               "by_origin": {o: sum(d["origin"] == o for d in discussions) for o in {d["origin"] for d in discussions}},
               "splits": {s: {"stance_pairs": len(stance[s]), "positives": sum(r["label"] for r in stance[s]),
                              "tone_rows": len(tone[s])} for s in stance}}
    write_json(dataset / "summary.json", summary)
    print(json.dumps(summary, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    generation = commands.add_parser("generate")
    generation.add_argument("--output", default="artifacts/stance_corpus")
    generation.add_argument("--rounds", type=int, default=8, choices=range(1, 9))
    generation.add_argument("--attempts", type=int, default=2)
    generation.add_argument("--workers", type=int, default=8)
    generation.add_argument("--max-tokens", type=int, default=7000)
    generation.add_argument("--max-calls", type=int, default=170)
    generation.add_argument("--max-output-tokens", type=int, default=900000)
    generation.add_argument("--model", default=MODEL)
    generation.add_argument("--split", choices=("train", "dev", "test"))
    generation.add_argument("--only", nargs="+", choices=DOMAINS)
    generation.set_defaults(handler=generate)
    real = commands.add_parser("export-real")
    real.add_argument("--runs", nargs="+", default=["output/runs"])
    real.add_argument("--output", default="artifacts/stance_corpus")
    real.set_defaults(handler=export_real)
    building = commands.add_parser("build")
    building.add_argument("--output", default="artifacts/stance_corpus")
    building.add_argument("--seed", type=int, default=20261003)
    building.set_defaults(handler=build)
    args = parser.parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
