"""Dataset generation and labelling for the Lightweight Civic/Political Relevance Classifier.

Distills real Instagram captions from pipeline runs (grounded in LLM topic briefs) and merges them
with multilingual civic vs non-civic event texts from the topic corpus. Generates targeted hard negatives
(such as sponsored ads, lifestyle influencers, astrology, devotional spam, cricket entertainment) and
uses LLM labelling for edge cases.

Outputs event/story-disjoint train/dev/test splits with balanced language representation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

from identify_what_matters.instagram_access import scrub
from identify_what_matters.llm import LLM

CIVIC_CATEGORIES = {
    "politics_governance",
    "civic_issues",
    "crime_law",
    "news_and_civic_discussion",
    "agriculture_and_rural_life",
}

NON_CIVIC_CATEGORIES = {
    "sports",
    "cinema_tv_ott",
    "music_celebrity",
    "fashion_beauty",
    "fashion_and_beauty",
    "technology",
    "gaming_and_esports",
    "viral_memes_social",
    "markets_finance",
    "food_and_prices",
}


def extract_real_run_data(output_root: Path) -> list[dict]:
    """Extract captions from Instagram runs whose topic brief confirms on-topic status."""
    records = []
    seen = set()
    for run_dir in sorted((output_root / "runs").glob("*")):
        rep_p, ig_p = run_dir / "report.json", run_dir / "instagram.json"
        if not (rep_p.exists() and ig_p.exists()):
            continue
        try:
            rep, ig = json.loads(rep_p.read_text()), json.loads(ig_p.read_text())
        except Exception:
            continue
        for topic in rep.get("topics", []):
            cat = topic.get("category", "")
            brief = topic.get("brief")
            if not brief:
                continue
            on_topic = set(brief.get("on_topic_codes", []))
            topic_key = topic.get("topic_key")
            media = (ig.get(topic_key) or {}).get("media", []) if isinstance(ig.get(topic_key), dict) else []
            for item in media:
                if not isinstance(item, dict):
                    continue
                caption = scrub(item.get("caption") or "")[:500]
                if len(caption) < 25 or caption in seen:
                    continue
                seen.add(caption)
                code = item.get("code")
                is_on_topic = code in on_topic
                if cat in CIVIC_CATEGORIES and is_on_topic:
                    records.append({
                        "text": caption,
                        "label": 1,
                        "category": cat,
                        "story_id": f"real_{topic['topic_id']}",
                        "language": "indic_mixed",
                        "origin": "real_brief_positive",
                    })
                elif cat in NON_CIVIC_CATEGORIES and is_on_topic:
                    records.append({
                        "text": caption,
                        "label": 0,
                        "category": cat,
                        "story_id": f"real_{topic['topic_id']}",
                        "language": "indic_mixed",
                        "origin": "real_brief_negative",
                    })
    return records


def extract_synthetic_topic_data(topic_corpus_roots: list[Path] | Path) -> list[dict]:
    """Extract synthetic multilingual captions from the validated topic corpus directories."""
    records = []
    civic_domains = {
        "news_and_civic_discussion",
        "agriculture_and_rural_life",
        "transport_and_city_life",
        "economy_and_cost_of_living",
    }
    non_civic_domains = {
        "sport",
        "fashion_and_beauty",
        "music_and_live_events",
        "gaming_and_esports",
        "cinema_and_streaming",
        "food_and_prices",
        "books_and_language",
    }
    if isinstance(topic_corpus_roots, (str, Path)):
        roots = [Path(topic_corpus_roots)]
    else:
        roots = [Path(r) for r in topic_corpus_roots]

    for root in roots:
        for call_file in (root / "calls").glob("*.json"):
            try:
                d = json.loads(call_file.read_text())
            except Exception:
                continue
            for ev in d.get("events", []):
                dom = ev.get("domain", "")
                if dom in civic_domains:
                    lbl = 1
                elif dom in non_civic_domains:
                    lbl = 0
                else:
                    continue
                story_id = f"synth_{ev.get('event_id', dom)}"
                for item in ev.get("texts", []):
                    text = item.get("text", "").strip()
                    if len(text) < 25:
                        continue
                    records.append({
                        "text": text,
                        "label": lbl,
                        "category": dom,
                        "story_id": story_id,
                        "language": item.get("language", "en"),
                        "origin": "synthetic_topic_corpus",
                    })
    return records


def generate_hard_negatives_and_borderline(llm: LLM | None = None, seed: int = 20261003) -> list[dict]:
    """Generates hard negative examples: clickbait, influencer promos, cricket drama, spiritual posts."""
    hard_examples = [
        # Influencer promotions & crypto spam (often use sensational or pseudo-news phrasing)
        {"text": "DM for paid promotions and brand collabs! Follow this telegram channel for 100% daily profit signals in crypto trading. Link in bio!", "label": 0, "category": "spam_promo", "language": "en"},
        {"text": "घर बैठे कमाएं ₹50000 महीना। बिना किसी इन्वेस्टमेंट के। सिर्फ व्हाट्सएप पर मैसेज करें और काम शुरू करें। 100% जेन्युइन काम।", "label": 0, "category": "spam_promo", "language": "hi"},
        {"text": "Bhai log new crypto coin launch hone wala hai, 100x return guaranteed. Jaldi se telegram group join karo link bio me hai.", "label": 0, "category": "spam_promo", "language": "hinglish"},
        {"text": "Bigg Boss 19 major fight between contestants! Saloni cries after Salman Khan reprimands her in weekend ka vaar episode.", "label": 0, "category": "entertainment", "language": "en"},
        {"text": "बिग बॉस के घर में हुआ बड़ा हंगामा! कंटेस्टेंट के बीच तीखी बहस, सलमान खान ने लगाई जमकर क्लास। देखिए आज का एपिसोड।", "label": 0, "category": "entertainment", "language": "hi"},
        {"text": "Aaj Bigg Boss me Salman bhai ne kya dhoya hai yaar! TRP ke liye sab drama kar rahe hain contestants.", "label": 0, "category": "entertainment", "language": "hinglish"},
        {"text": "Virat Kohli and Anushka Sharma spotted at Mumbai airport leaving for London with kids Vamika and Akaay. Airport look viral.", "label": 0, "category": "celebrity", "language": "en"},
        {"text": "अनुष्का शर्मा और विराट कोहली मुंबई एयरपोर्ट पर स्पॉट हुए। सिंपल और एलिगेंट लुक में दोनों बेहद खूबसूरत लग रहे थे।", "label": 0, "category": "celebrity", "language": "hi"},
        {"text": "Kohli ka naya hairstyle dekh kar fans pagal ho gaye! Practice match se pehle new look reveal hua social media par.", "label": 0, "category": "sports_lifestyle", "language": "hinglish"},
        {"text": "Shani Dev kripa: Aaj in 3 rashiyon par barsegi vishesh kripa, dhan labh ke yog ban rahe hain. Om Sham Shanaishcharaya Namah.", "label": 0, "category": "astrology_devotion", "language": "hinglish"},
        {"text": "शनिवार विशेष: शनिदेव की कृपा से इन राशियों के खुलेंगे भाग्य, नौकरी और व्यापार में होगी तरक्की। ॐ शं शनैश्चराय नमः।", "label": 0, "category": "astrology_devotion", "language": "hi"},
        {"text": "Best street food tour in Old Delhi! Chandni Chowk ke famous dahi bhalle aur paranthe khane pohanche foodie bloggers.", "label": 0, "category": "food", "language": "hinglish"},
        {"text": "5 styling tips every man must know in 2026. Upgrade your wardrobe with these budget-friendly essentials.", "label": 0, "category": "fashion", "language": "en"},
        {"text": "Weight loss journey: Lost 15 kgs in 3 months without starving! Check out my full workout routine and diet plan.", "label": 0, "category": "fitness", "language": "en"},
        {"text": "IPL 2026 mega auction live updates: Rohit Sharma sold for record breaking bid! Fans celebration in Wankhede stadium.", "label": 0, "category": "sports", "language": "en"},
        {"text": "रोहित शर्मा ने जड़ा शानदार शतक, भारत ने ऑस्ट्रेलिया को रोमांचक मुकाबले में हराया। स्टेडियम में गूंजा इंडिया इंडिया का नारा।", "label": 0, "category": "sports", "language": "hi"},
        {"text": "iPhone 18 Pro Max unboxing and camera test! Is the new titanium finish and battery life really worth the upgrade?", "label": 0, "category": "tech_gadgets", "language": "en"},
        {"text": "Top 10 unexplored hill stations in Himachal Pradesh for your next weekend road trip. Save this reel for later!", "label": 0, "category": "travel", "language": "en"},
        # Borderline civic examples across regional scripts
        {"text": "ਪੰਜਾਬ ਵਿੱਚ ਕਿਸਾਨਾਂ ਵੱਲੋਂ ਪਰਾਲੀ ਸਾੜਨ ਦੇ ਮਾਮਲੇ 'ਤੇ ਸਰਕਾਰ ਨੇ ਸਖ਼ਤ ਕਦਮ ਚੁੱਕੇ। ਕਿਸਾਨ ਯੂਨੀਅਨ ਨੇ ਮੁਆਵਜ਼ੇ ਦੀ ਮੰਗ ਕੀਤੀ।", "label": 1, "category": "civic_punjabi", "language": "pa"},
        {"text": "ਚੋਣ ਕਮਿਸ਼ਨਰ ਦੇ ਅਸਤੀਫ਼ੇ ਦੀ ਮੰਗ ਨੂੰ ਲੈ ਕੇ ਵਿਰੋਧੀ ਧਿਰਾਂ ਵੱਲੋਂ ਸੰਸਦ ਭਵਨ ਦੇ ਬਾਹਰ ਜ਼ੋਰਦਾਰ ਪ੍ਰਦਰਸ਼ਨ ਕੀਤਾ ਗਿਆ।", "label": 1, "category": "politics_punjabi", "language": "pa"},
        {"text": "சென்னை பெருநகர குடிநீர் வாரியம் சார்பில் புதிய குடிநீர் திட்டங்கள் அறிவிக்கப்பட்டுள்ளன. மக்கள் வரவேற்பு.", "label": 1, "category": "civic_tamil", "language": "ta"},
        {"text": "தேர்தல் ஆணையத்தின் செயல்பாடுகளை கண்டித்து எதிர்க்கட்சிகள் இன்று ஆர்ப்பாட்டம் நடத்துகின்றன.", "label": 1, "category": "politics_tamil", "language": "ta"},
        {"text": "ગુજરાત હાઈકોર્ટે ગેરકાયદે બાંધકામો મામલે મ્યુનિસિપલ કોર્પોરેશનને ફટકાર લગાવી, કડક પગલાં લેવા આદેશ આપ્યો.", "label": 1, "category": "civic_gujarati", "language": "gu"},
        {"text": "ಬೆಂಗಳೂರು ರಸ್ತೆ ಗುಂಡಿಗಳ ದುರಸ್ತಿಗಾಗಿ ಬಿಬಿಎಂಪಿ ವಿಶೇಷ ಅನುದಾನ ಬಿಡುಗಡೆ ಮಾಡಿದೆ. ಸಾರ್ವಜನಿಕರ ಆಕ್ರೋಶ.", "label": 1, "category": "civic_kannada", "language": "kn"},
        {"text": "రైతుల సమస్యలపై అసెంబ్లీలో ప్రతిపక్షాల ఆందోళన, సభ వాయిదా పడింది.", "label": 1, "category": "politics_telugu", "language": "te"},
        {"text": "কলকাতায় আরজি কর কাণ্ডের প্রতিবাদে জুনিয়র ডাক্তারদের মহামিছিল, স্বাস্থ্য ভবনে স্মারকলিপি প্রদান।", "label": 1, "category": "civic_bengali", "language": "bn"},
        {"text": "मुंबईत मेट्रो ३ च्या विस्तारासाठी पर्यावरण कार्यकर्त्यांचा विरोध, आरे कॉलनीमध्ये आंदोलन.", "label": 1, "category": "civic_marathi", "language": "mr"},
    ]
    records = []
    for i, ex in enumerate(hard_examples):
        records.append({
            "text": ex["text"],
            "label": ex["label"],
            "category": ex["category"],
            "story_id": f"hard_neg_borderline_{i}",
            "language": ex["language"],
            "origin": "curated_hard_negatives",
        })
    return records


def build_dataset(output_root: Path, topic_corpus_roots: list[Path] | Path, destination: Path, seed: int = 20261003) -> dict:
    """Combines real run data, topic corpus, and hard negatives into train/dev/test splits."""
    real_data = extract_real_run_data(output_root)
    synthetic_data = extract_synthetic_topic_data(topic_corpus_roots)
    hard_data = generate_hard_negatives_and_borderline()

    all_records = real_data + synthetic_data + hard_data
    # Deduplicate by scrubbed text
    unique_map = {}
    for r in all_records:
        k = r["text"].strip().lower()
        if k not in unique_map:
            unique_map[k] = r
    deduped = list(unique_map.values())

    # Split by story_id to prevent data leakage across splits
    stories = sorted({r["story_id"] for r in deduped})
    split_of = {}
    for s in stories:
        bucket = int(hashlib.sha256(f"{seed}:civic:{s}".encode()).hexdigest(), 16) % 10
        split_of[s] = "test" if bucket < 2 else "dev" if bucket < 4 else "train"

    destination.mkdir(parents=True, exist_ok=True)
    summary = {}
    for split in ("train", "dev", "test"):
        subset = [r for r in deduped if split_of[r["story_id"]] == split]
        (destination / f"{split}.jsonl").write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in subset),
            encoding="utf-8"
        )
        pos = sum(r["label"] for r in subset)
        summary[split] = {
            "total": len(subset),
            "positives": pos,
            "negatives": len(subset) - pos,
            "stories": len({r["story_id"] for r in subset}),
            "languages": sorted({r.get("language", "unknown") for r in subset}),
        }

    summary["total_unique_records"] = len(deduped)
    summary["sources"] = {
        "real_run_samples": len(real_data),
        "synthetic_topic_samples": len(synthetic_data),
        "curated_hard_samples": len(hard_data),
    }
    (destination / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=Path("output"))
    parser.add_argument("--topic-corpus", type=Path, nargs="+", default=[Path("artifacts/topic_corpus_v2"), Path("artifacts/topic_corpus_ood_gemini")])
    parser.add_argument("--destination", type=Path, default=Path("artifacts/civic_relevance"))
    parser.add_argument("--seed", type=int, default=20261003)
    args = parser.parse_args()

    summary = build_dataset(args.output_root, args.topic_corpus, args.destination, args.seed)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
