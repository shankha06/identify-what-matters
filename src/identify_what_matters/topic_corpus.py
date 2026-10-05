"""OpenRouter-generated multilingual same-event corpus (v2) for topic-embedding fine-tuning.

Synthetic, fictional events only; never production gold. One domain per call keeps
responses small enough to parse reliably, and rounds are split train/dev/test so
no event or generation call leaks across splits.
"""

from __future__ import annotations

import argparse
import itertools
import json
import random
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from identify_what_matters.openrouter import DEFAULT_MODEL, OpenRouterClient

MODEL = DEFAULT_MODEL
DOMAINS = (
    "news_and_civic_discussion", "economy_and_cost_of_living", "jobs_and_work",
    "education_and_exams", "technology_and_ai", "sport", "cinema_and_streaming",
    "music_and_live_events", "fashion_and_beauty", "food_and_prices",
    "health_and_wellbeing", "environment_and_weather", "transport_and_city_life",
    "travel_and_local_culture", "gaming_and_esports", "agriculture_and_rural_life",
    "books_and_language", "science_and_space",
)
REGIONAL = {"bn": "\u0980-\u09ff", "ta": "\u0b80-\u0bff", "te": "\u0c00-\u0c7f",
            "mr": "\u0900-\u097f", "gu": "\u0a80-\u0aff", "kn": "\u0c80-\u0cff",
            "ml": "\u0d00-\u0d7f", "pa": "\u0a00-\u0a7f"}
REGIONAL_NAMES = {"bn": "Bengali", "ta": "Tamil", "te": "Telugu", "mr": "Marathi",
                  "gu": "Gujarati", "kn": "Kannada", "ml": "Malayalam", "pa": "Punjabi"}
SLOTS = (("caption", "en"), ("caption", "hi"), ("caption", "hinglish"),
         ("caption", "regional"), ("comment", "hinglish"), ("comment", "en"))
SPLIT_OF_ROUND = {0: "train", 1: "train", 2: "train", 3: "train", 4: "train", 5: "train",
                  6: "dev", 7: "test"}
IDENTIFIERS = (
    r"https?://\S+|www\.\S+", r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}", r"(?<!\w)[@#][\w.]+",
    r"(?<!\w)\+?\d[\d\s().-]{8,}\d(?!\w)", r"\b[A-Z]{5}\d{4}[A-Z]\b",
)
_lock = threading.Lock()


def now():
    return datetime.now(timezone.utc).isoformat()


def unsafe(text):
    return any(re.search(pattern, text) for pattern in IDENTIFIERS)


def write_json(target, value):
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(target)


def write_rows(target, rows):
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))


def regional_for(domain, round_index):
    codes = list(REGIONAL)
    return codes[(DOMAINS.index(domain) + round_index * 3) % len(codes)]


NOISY_STYLE = (
    "Write like real Indian Instagram users, not like a news desk: uneven grammar, slang, abbreviations "
    "(pls, bro, ngl, fr, kya scene), emoji, sarcasm, rhetorical questions, mid-sentence code switching, "
    "occasional typos. Captions can be short. Comments can be very short reactions that still reference "
    "one concrete detail. "
)


def prompt_for(domain, round_index, used_titles, style="clean"):
    regional = regional_for(domain, round_index)
    avoid = "; ".join(used_titles) if used_titles else "none yet"
    return (
        "Generate synthetic training data for a model that groups Indian Instagram posts and comments "
        "by the specific event or discussion they are about.\n"
        + (NOISY_STYLE if style == "noisy" else "") +
        f"Domain: {domain.replace('_', ' ')} in India.\n"
        "Create exactly 4 DIFFERENT fictional but realistic events or discussion topics in this domain. "
        "The four must be confusable: same domain, overlapping vocabulary, but each differs in the concrete "
        "happening (what changed, which kind of place, when, outcome). Prefer things people would actually "
        "argue about or react to on Instagram (announcements, controversies, launches, price changes, "
        "results, viral moments, local problems).\n"
        f"Do not reuse these already generated topics: {avoid}.\n"
        "Rules: fictional events, not claims about real ones; public institutions or generic places are fine; "
        "no personal names, @handles, #hashtags, URLs, emails or phone numbers. Inside text values never use "
        "the double-quote character; use single quotes if needed.\n"
        "For each event output exactly 6 texts in this order, all about the SAME specific event "
        "(not just the domain):\n"
        "1 caption en: English post caption, 40-220 chars.\n"
        "2 caption hi: Hindi caption in Devanagari script, 40-220 chars.\n"
        "3 caption hinglish: romanized Hindi mixed with English, Instagram style, 40-220 chars.\n"
        f"4 caption {regional}: {REGIONAL_NAMES[regional]} caption in native script, 40-220 chars.\n"
        "5 comment hinglish: a casual viewer comment reacting to this event (opinion, question, joke or "
        "complaint), 12-140 chars, emoji allowed.\n"
        "6 comment en: a different casual English comment about this event, 12-140 chars.\n"
        "Texts should paraphrase freely, not translate word for word. Comments must mention a detail of "
        "the event, not generic praise.\n"
        'Return ONLY a JSON object, no markdown: {"events": [{"event_key": "e0", "title": "<=8 English '
        'words", "distinguishing_fact": "one sentence", "texts": [{"style": "caption", "language": "en", '
        '"text": "..."}, ...]}, ...]} with event_key e0, e1, e2, e3.'
    )


def parse_payload(raw):
    text = raw.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("no_json_object")
    return json.loads(text[start:end + 1], strict=False)


def validate(payload, domain, round_index, split=None):
    regional = regional_for(domain, round_index)
    events = payload.get("events") if isinstance(payload, dict) else None
    if not isinstance(events, list) or len(events) != 4:
        raise ValueError("event_count")
    seen, accepted = set(), []
    for position, event in enumerate(events):
        if not isinstance(event, dict) or event.get("event_key") != f"e{position}":
            raise ValueError("event_key")
        title, fact, texts = event.get("title"), event.get("distinguishing_fact"), event.get("texts")
        if not isinstance(title, str) or not 3 <= len(title) <= 90 or not isinstance(fact, str):
            raise ValueError("event_title")
        if not isinstance(texts, list) or len(texts) != len(SLOTS):
            raise ValueError("text_count")
        cleaned = []
        for (style, language), item in zip(SLOTS, texts):
            language = regional if language == "regional" else language
            if not isinstance(item, dict) or item.get("style") != style or item.get("language") != language:
                raise ValueError("slot_order")
            content = " ".join(str(item.get("text", "")).split())
            low, high = (25, 300) if style == "caption" else (8, 220)
            if not low <= len(content) <= high:
                raise ValueError(f"length_{style}_{language}")
            if unsafe(content) or content.casefold() in seen:
                raise ValueError("unsafe_or_duplicate")
            if language == "hi" and not re.search("[\u0900-\u097f]", content):
                raise ValueError("hindi_script")
            if language in REGIONAL and not re.search(f"[{REGIONAL[language]}]", content):
                raise ValueError("regional_script")
            if language in ("en", "hinglish") and re.search("[\u0900-\u0d7f]", content):
                raise ValueError("latin_script")
            seen.add(content.casefold())
            cleaned.append({"style": style, "language": language, "text": content})
        accepted.append({"event_id": f"{domain}.r{round_index}.e{position}", "domain": domain,
                         "round": round_index, "split": split or SPLIT_OF_ROUND[round_index],
                         "title": title.strip(), "distinguishing_fact": fact.strip(), "texts": cleaned})
    return accepted


class Budget:
    def __init__(self, state, max_calls, max_output_tokens):
        self.state, self.max_calls, self.max_output_tokens = state, max_calls, max_output_tokens

    def reserve(self, expected_output, calls=1):
        with _lock:
            used = sum(item.get("output_tokens") or 0 for item in self.state["usage"])
            if self.state["calls"] + calls > self.max_calls or used + expected_output > self.max_output_tokens:
                return False
            self.state["calls"] += calls
            return True


def generate(args):
    out = Path(args.output)
    (out / "calls").mkdir(parents=True, exist_ok=True)
    (out / "rejected").mkdir(parents=True, exist_ok=True)
    state_path = out / "generation_state.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {
        "run_id": "topicv2-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "created": now(), "origin": "synthetic", "model": args.model, "calls": 0, "usage": [], "failures": []}
    if state.get("model") != args.model:
        raise ValueError("resume_model_mismatch")
    budget = Budget(state, args.max_calls, args.max_output_tokens)
    prior_titles = {}
    for prior in args.avoid_from or []:
        for checkpoint in Path(prior, "calls").glob("*.json"):
            for event in json.loads(checkpoint.read_text())["events"]:
                prior_titles.setdefault(event["domain"], []).append(event["title"])
    client = OpenRouterClient(model=args.model, max_tokens=args.max_tokens, temperature=0.8)

    def save_state():
        with _lock:
            write_json(state_path, state)

    def run_domain(domain):
        for round_index in range(args.rounds):
            checkpoint = out / "calls" / f"{domain}.r{round_index}.json"
            if checkpoint.exists():
                continue
            for rejected in sorted((out / "rejected").glob(f"{domain}.r{round_index}.a*.txt")):
                try:
                    events = validate(parse_payload(rejected.read_text()), domain, round_index, args.split)
                except Exception:
                    continue
                write_json(checkpoint, {"run_id": state["run_id"], "model": args.model, "generated": now(),
                                        "salvaged_from": rejected.name, "events": events})
                print(json.dumps({"call": f"{domain}.r{round_index}", "status": "salvaged"}), flush=True)
                break
            if checkpoint.exists():
                continue
            used = list(prior_titles.get(domain, []))
            for previous in range(round_index):
                earlier = out / "calls" / f"{domain}.r{previous}.json"
                if earlier.exists():
                    used += [event["title"] for event in json.loads(earlier.read_text())["events"]]
            for attempt in range(args.attempts):
                if not budget.reserve(args.max_tokens * 2, calls=2):
                    return
                prompt = prompt_for(domain, round_index, used, args.style)
                try:
                    response = client.generate(user_prompt=prompt, request_timeout=240,
                        system_instruction="You produce strict JSON datasets. Output only the JSON object.")
                except Exception as error:
                    with _lock:
                        state["failures"].append({"call": f"{domain}.r{round_index}", "attempt": attempt,
                                                  "category": type(error).__name__, "at": now()})
                    save_state()
                    continue
                with _lock:
                    state["usage"].append({"call": f"{domain}.r{round_index}", "attempt": attempt,
                                           "input_tokens": response.input_tokens,
                                           "output_tokens": response.output_tokens})
                try:
                    events = validate(parse_payload(response.text), domain, round_index, args.split)
                except Exception as error:
                    category = str(error) if isinstance(error, ValueError) and "Expecting" not in str(error) \
                        else type(error).__name__
                    (out / "rejected" / f"{domain}.r{round_index}.a{attempt}.txt").write_text(response.text)
                    with _lock:
                        state["failures"].append({"call": f"{domain}.r{round_index}", "attempt": attempt,
                                                  "category": category, "output_tokens": response.output_tokens,
                                                  "at": now()})
                    save_state()
                    print(json.dumps({"call": f"{domain}.r{round_index}", "status": "rejected",
                                      "category": category}), flush=True)
                    continue
                write_json(checkpoint, {"run_id": state["run_id"], "model": args.model, "generated": now(),
                                        "events": events})
                save_state()
                print(json.dumps({"call": f"{domain}.r{round_index}", "status": "validated",
                                  "calls": state["calls"]}), flush=True)
                break

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(run_domain, args.only or DOMAINS))
    save_state()
    build(out, args.seed)


def build(out, seed):
    out = Path(out)
    rng = random.Random(seed)
    events = []
    for checkpoint in sorted((out / "calls").glob("*.json")):
        events += json.loads(checkpoint.read_text())["events"]
    seen, duplicates = {}, 0
    for event in events:
        for item in event["texts"]:
            key = item["text"].casefold()
            if key in seen and seen[key] != event["event_id"]:
                duplicates += 1
            seen[key] = event["event_id"]
    if duplicates:
        raise ValueError(f"cross_event_duplicate_texts:{duplicates}")
    by_call = {}
    for event in events:
        by_call.setdefault(event["event_id"].rsplit(".", 1)[0], []).append(event)
    texts = {split: [] for split in ("train", "dev", "test")}
    pairs = {split: [] for split in texts}
    for event in events:
        for index, item in enumerate(event["texts"]):
            texts[event["split"]].append({**item, "text_id": f"{event['event_id']}.t{index}",
                                          "event_id": event["event_id"], "domain": event["domain"],
                                          "title": event["title"]})

    def pair(first_event, first, second_event, second, label, kind):
        return {"text_a": first["text"], "text_b": second["text"], "label": label, "kind": kind,
                "event_id": [first_event["event_id"], second_event["event_id"]],
                "domain": first_event["domain"], "language": [first["language"], second["language"]],
                "style": [first["style"], second["style"]]}

    for event in events:
        for first, second in itertools.combinations(event["texts"], 2):
            pairs[event["split"]].append(pair(event, first, event, second, 1.0, "positive"))
    for group in by_call.values():
        for first_event, second_event in itertools.combinations(group, 2):
            product = list(itertools.product(first_event["texts"], second_event["texts"]))
            for first, second in rng.sample(product, 8):
                pairs[first_event["split"]].append(pair(first_event, first, second_event, second, 0.0,
                                                        "hard_negative_same_domain"))
    for split in texts:
        split_events = [event for event in events if event["split"] == split]
        for event in split_events:
            others = [other for other in split_events if other["domain"] != event["domain"]]
            for other in rng.sample(others, min(3, len(others))):
                pairs[split].append(pair(event, rng.choice(event["texts"]), other, rng.choice(other["texts"]),
                                         0.0, "easy_negative_other_domain"))
    dataset = out / "dataset"
    for split in texts:
        rng.shuffle(pairs[split])
        write_rows(dataset / f"{split}_pairs.jsonl", pairs[split])
        write_rows(dataset / f"{split}_texts.jsonl", texts[split])
    write_rows(dataset / "events.jsonl", events)
    summary = {
        "built": now(), "seed": seed, "origin": "synthetic_openrouter", "human_gold": False,
        "events": len(events), "texts": sum(len(rows) for rows in texts.values()),
        "domains_covered": sorted({event["domain"] for event in events}),
        "split_policy": "rounds 0-5 train, round 6 dev, round 7 test (event- and call-disjoint)",
        "splits": {split: {"events": len({row["event_id"] for row in texts[split]}),
                           "texts": len(texts[split]),
                           "pairs": {kind: sum(row["kind"] == kind for row in pairs[split])
                                     for kind in ("positive", "hard_negative_same_domain",
                                                  "easy_negative_other_domain")}}
                   for split in texts},
        "languages": sorted({row["language"] for rows in texts.values() for row in rows}),
    }
    write_json(dataset / "summary.json", summary)
    print(json.dumps(summary, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    generation = commands.add_parser("generate")
    generation.add_argument("--output", default="artifacts/topic_corpus_v2")
    generation.add_argument("--rounds", type=int, default=8, choices=range(1, 9))
    generation.add_argument("--attempts", type=int, default=2)
    generation.add_argument("--workers", type=int, default=6)
    generation.add_argument("--max-tokens", type=int, default=6000)
    generation.add_argument("--max-calls", type=int, default=180)
    generation.add_argument("--max-output-tokens", type=int, default=500000)
    generation.add_argument("--seed", type=int, default=20261003)
    generation.add_argument("--only", nargs="+", choices=DOMAINS)
    generation.add_argument("--model", default=MODEL)
    generation.add_argument("--style", choices=("clean", "noisy"), default="clean")
    generation.add_argument("--split", choices=("train", "dev", "test"), help="Force one split (OOD sets)")
    generation.add_argument("--avoid-from", nargs="*", help="Corpus dirs whose event titles must not be reused")
    generation.set_defaults(handler=generate)
    building = commands.add_parser("build")
    building.add_argument("--output", default="artifacts/topic_corpus_v2")
    building.add_argument("--seed", type=int, default=20261003)
    building.set_defaults(handler=lambda args: build(args.output, args.seed))
    args = parser.parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
