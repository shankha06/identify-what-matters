"""LLM-driven large-scale dataset generator for demographics and mental state classifiers.

Generates thousands of realistic Indian social media comments with labels:
  - age_bracket: 13-17, 18-24, 25-34, 35-54, 55+
  - gender: male, female, neutral
  - mental_state: 6 archetypes
  - impatience: impatient, neutral, patient
  - irrationality: irrational, neutral, rational
  - party_support: BJP, Congress, AAP, TMC, Other, None

Uses OpenRouter with reasoning continuation to produce high-quality, diverse samples.
"""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
import requests

API_URL = "https://openrouter.ai/api/v1/chat/completions"
ENV_FILE = Path(__file__).resolve().parents[2] / ".env"

load_dotenv(ENV_FILE, override=False)
API_KEY = os.environ.get("OPENROUTER_API_KEY")
MODEL = os.environ.get("OPENROUTER_MODEL", "openai/gpt-6.1-sol")

BATCH_PROMPTS = [
    "Generate 30 diverse Indian social media comments in English, Hinglish, Hindi, Tamil, Telugu, Bengali, or Gujarati. Each comment must reflect one of these age brackets: 18-24 (college students, job aspirants, Gen Z slang), 25-34 (working professionals, parents), 35-54 (middle-aged, civic concerns), 55+ (seniors, retirees). Mix genders and topics. Cover politics, economy, civic issues, entertainment, sports, technology, education, health, festivals, and daily life. Return a JSON array of objects with keys: text, age_bracket, gender, mental_state, impatience, irrationality, party_support. Keep each text 15-80 words. Ensure diversity. No duplicates.",

    "Generate 30 Indian social media comments that show strong civic anger and institutional skepticism. Topics: corruption, police inaction, pollution, potholes, waterlogging, delays in government services, tax harassment. Mix ages 25-55, both genders. Some should be patient and rational, some impatient and angry, some irrational rants. Vary political leanings. Return JSON array with objects: text, age_bracket, gender, mental_state, impatience, irrationality, party_support.",

    "Generate 30 Indian social media comments from Gen Z (18-24) using viral slang, memes, and ironic detachment. Topics: placements, exams, college life, reels, gaming, cricket, movies, dating, hostel life. Mix male, female, neutral. Mental states: ironic detachment, financial anxiety, social empathy, objective observant. Some impatient, some patient. Return JSON array with objects: text, age_bracket, gender, mental_state, impatience, irrationality, party_support.",

    "Generate 30 Indian social media comments from working professionals (25-34). Topics: salary, taxes, EMIs, layoffs, WFH, office politics, startup life, stock market, SIP, GST, corporate grind, work-life balance. Mix male, female, neutral. Mental states: financial anxiety, civic anger, objective observant, community empathy. Some impatient, some rational. Return JSON array with objects: text, age_bracket, gender, mental_state, impatience, irrationality, party_support.",

    "Generate 30 Indian social media comments from parents and senior citizens (35-54 and 55+). Topics: children education, school fees, health, retirement, pension, property, inflation, family values, cricket, national pride, farmers protest, corruption. Mix genders. Mental states: civic anger, patriotic pride, community empathy, financial anxiety. Some impatient, some patient. Return JSON array with objects: text, age_bracket, gender, mental_state, impatience, irrationality, party_support.",

    "Generate 30 Indian social media comments with strong political leanings: BJP supporters (development, nationalism, Ram Mandir, Article 370, Hindutva), Congress supporters (secularism, democracy, welfare, Gandhi legacy), AAP supporters (anti-corruption, education, health, Delhi model), TMC supporters (Bengali pride, federalism, Didi). Mix ages 22-60. Some rational arguments, some emotional, some impatient. Return JSON array with objects: text, age_bracket, gender, mental_state, impatience, irrationality, party_support.",

    "Generate 30 Indian social media comments about entertainment, movies, web series, cricket, IPL, Bollywood, Kollywood, Tollywood. Ages 16-50. Mix genders. Mental states: viral cynicism, community empathy, objective observant. Some impatient about delays, some patient fans. Return JSON array with objects: text, age_bracket, gender, mental_state, impatience, irrationality, party_support.",

    "Generate 30 Indian social media comments about technology, gadgets, AI, coding, startups, internet, data privacy, social media bans. Ages 18-40, mostly male and neutral. Mental states: objective observant, financial anxiety, civic anger, ironic detachment. Some irrational conspiracy theories about tech, some rational analysis. Return JSON array with objects: text, age_bracket, gender, mental_state, impatience, irrationality, party_support.",

    "Generate 30 Indian social media comments about religion, festivals, temples, mosques, churches, garba, diwali, eid, christmas, interfaith issues. Ages 18-70. Mix all demographics. Mental states: patriotic pride, civic anger, community empathy, viral cynicism. Some very patient and tolerant, some impatient and divisive. Party leanings mixed. Return JSON array with objects: text, age_bracket, gender, mental_state, impatience, irrationality, party_support.",

    "Generate 30 Indian social media comments about education, NEET, JEE, UPSC, coaching mafias, fee hikes, reservation, campus politics. Ages 16-35. Mix genders. Mental states: civic anger, financial anxiety, objective observant, social empathy. Many impatient about system failures, some rational. Return JSON array with objects: text, age_bracket, gender, mental_state, impatience, irrationality, party_support.",
]


def _call_openrouter(messages: list[dict], timeout: int = 120) -> str:
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": MODEL,
        "messages": messages,
        "reasoning": {"enabled": True},
        "temperature": 0.85,
        "max_tokens": 4096,
    }
    for attempt in range(3):
        try:
            r = requests.post(API_URL, headers=headers, json=payload, timeout=timeout)
            r.raise_for_status()
            data = r.json()
            msg = data["choices"][0]["message"]
            content = msg.get("content", "")
            if not content:
                messages.extend([
                    {"role": "assistant", "content": "", "reasoning_details": msg.get("reasoning_details")},
                    {"role": "user", "content": "Please output the JSON array now."},
                ])
                r2 = requests.post(API_URL, headers=headers, json=payload, timeout=timeout)
                r2.raise_for_status()
                msg = r2.json()["choices"][0]["message"]
                content = msg.get("content", "")
            return content
        except Exception as e:
            if attempt == 2:
                raise
            time.sleep(2 ** attempt)
    return ""


def generate_batch(prompt: str) -> list[dict]:
    raw = _call_openrouter([{"role": "user", "content": prompt}])
    if not raw:
        return []
    # extract JSON array from markdown fences or raw text
    m = re.search(r"```json\s*(\[.*?\])\s*```", raw, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            pass
    m = re.search(r"(\[\s*\{.*?\}\s*\])", raw, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            pass
    return []


def generate_full_dataset(output_path: Path, target_samples: int = 2500) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    seen = set()
    samples = []
    batches_needed = max(1, (target_samples + 29) // 30)

    print(f"Generating ~{target_samples} samples using OpenRouter ({MODEL})...")
    for i, prompt in enumerate(BATCH_PROMPTS * max(1, batches_needed // len(BATCH_PROMPTS) + 1)):
        if len(samples) >= target_samples:
            break
        print(f"  Batch {i+1}: fetching...")
        batch = generate_batch(prompt)
        for item in batch:
            if not isinstance(item, dict):
                continue
            text = str(item.get("text", "")).strip()
            if not text or len(text) < 15:
                continue
            key = text[:80]
            if key in seen:
                continue
            seen.add(key)
            samples.append({
                "text": text,
                "age_bracket": item.get("age_bracket", "unknown"),
                "gender": item.get("gender", "neutral"),
                "mental_state": item.get("mental_state", "Objective Deliberation & Fact-Checking"),
                "impatience": item.get("impatience", "neutral"),
                "irrationality": item.get("irrationality", "neutral"),
                "party_support": item.get("party_support", "None"),
            })
        print(f"  Batch {i+1}: got {len(batch)} valid, total {len(samples)}")
        time.sleep(0.3)

    with open(output_path, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"Saved {len(samples)} samples to {output_path}")


if __name__ == "__main__":
    out = Path(__file__).resolve().parents[2] / "artifacts" / "demographics_models" / "synthetic_dataset.jsonl"
    generate_full_dataset(out, target_samples=2500)
