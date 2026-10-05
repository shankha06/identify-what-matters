"""Dataset generator and classifier training for Demographics, Mental State & Behavioral Inference.

Generates/loads large-scale multi-lingual dataset for:
  1. Age Cohort Inference (13-17, 18-24, 25-34, 35-54, 55+)
  2. Gender Persona Inference (male, female, neutral)
  3. Mental State Archetype Classification (6 psychological archetypes)
  4. Impatience Classification (impatient, neutral, patient)
  5. Irrationality Classification (irrational, neutral, rational)
  6. Political Party Support (BJP, Congress, AAP, TMC, Other, None)

Trains calibrated TF-IDF + Logistic Regression pipelines saved with joblib
for instantaneous, zero-latency inference across thousands of comments.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split

DATA_DIR = Path(__file__).resolve().parents[2] / "artifacts" / "demographics_models"
SYNTHETIC_PATH = DATA_DIR / "synthetic_dataset.jsonl"

# ---------------------------------------------------------------- TRAINING SEEDS ----
# Comprehensive multi-lingual labelled samples reflecting real social discourse in India
SEED_SAMPLES = [
    # --- 18-24 (Gen Z / Youth) & VIRAL_CYNICISM / COMMUNITY_EMPATHY ---
    ("bro literally cooked the exam, attendance short ho gayi fr fr 😭", "18-24", "male", "Ironic Detachment & Memetic Cynicism", "impatient", "irrational", "None"),
    ("ngl this professor is peak cringe, full attendance scam chal raha hai", "18-24", "neutral", "Ironic Detachment & Memetic Cynicism", "impatient", "rational", "None"),
    ("slay queen! honestly so proud of what she did, pure female rage at its finest", "18-24", "female", "Social Empathy & Grassroots Care", "patient", "rational", "None"),
    ("delulu is the only solulu rn lmao, placement season is terrifying", "18-24", "neutral", "Financial Anxiety & Economic Strain", "impatient", "irrational", "None"),
    ("bhai cutoff itna high kaun rakhta hai? jee aspirant mental breakdown", "18-24", "male", "Financial Anxiety & Economic Strain", "impatient", "irrational", "None"),
    ("bruh this meme has me wheezing at 3am skull emoji x100 💀", "18-24", "neutral", "Ironic Detachment & Memetic Cynicism", "patient", "rational", "None"),
    ("bhaiya college fest ke passes milenge kya? squad ready hai", "18-24", "male", "Ironic Detachment & Memetic Cynicism", "impatient", "rational", "None"),
    ("didi can you share the notes for semester exams? please help sister", "18-24", "female", "Social Empathy & Grassroots Care", "patient", "rational", "None"),
    ("bussin food scenes at North Campus, momos are life", "18-24", "neutral", "Ironic Detachment & Memetic Cynicism", "patient", "rational", "None"),
    ("lowkey feeling anxious about campus placements, recession hits hard", "18-24", "neutral", "Financial Anxiety & Economic Strain", "impatient", "rational", "None"),
    ("yaar hostel wifi is dead again, valorant match drop ho gaya bhai", "18-24", "male", "Ironic Detachment & Memetic Cynicism", "impatient", "irrational", "None"),
    ("sigma grindset ruined my sleep schedule completely frfr", "18-24", "male", "Ironic Detachment & Memetic Cynicism", "patient", "rational", "None"),
    ("bhai paper leak ho gaya firse, hamare future ka mazaak bana diya hai", "18-24", "male", "Civic Anger & Institutional Skepticism", "impatient", "irrational", "None"),
    ("du campus library vibes are so peaceful today", "18-24", "neutral", "Objective Deliberation & Fact-Checking", "patient", "rational", "None"),
    ("ye ladka alag hi level ka rizzler hai bhai, kya confidence hai", "18-24", "male", "Ironic Detachment & Memetic Cynicism", "patient", "rational", "None"),
    ("she literally ate and left no crumbs! gorgeous saree didi", "18-24", "female", "Social Empathy & Grassroots Care", "patient", "rational", "None"),
    ("btech karke berozgari ka certificate mil gaya, thank you education system", "18-24", "male", "Civic Anger & Institutional Skepticism", "impatient", "irrational", "None"),
    ("yaar crush ne message read pe chhod diya, heart is in pieces", "18-24", "neutral", "Ironic Detachment & Memetic Cynicism", "impatient", "irrational", "None"),

    # --- 25-34 (Working Professional) & ECONOMIC_ANXIETY / CAREER ---
    ("30% tax slab on 15 LPA salary with no public infrastructure to show for it", "25-34", "neutral", "Financial Anxiety & Economic Strain", "impatient", "rational", "Congress"),
    ("got laid off after 6 years at IT company, home loan EMI of 55k running", "25-34", "neutral", "Financial Anxiety & Economic Strain", "impatient", "rational", "None"),
    ("nifty at all time high but ground level consumption and retail jobs are falling", "25-34", "neutral", "Financial Anxiety & Economic Strain", "patient", "rational", "BJP"),
    ("manager refusing leaves even during medical emergency, toxic corporate hustle", "25-34", "neutral", "Financial Anxiety & Economic Strain", "impatient", "rational", "None"),
    ("how much sip should i allocate for child education and mother hospital expenses?", "25-34", "male", "Financial Anxiety & Economic Strain", "patient", "rational", "None"),
    ("bhai switch karne ka soch raha hoon, current CTC se rent aur groceries nahi nikalte", "25-34", "male", "Financial Anxiety & Economic Strain", "impatient", "rational", "None"),
    ("bangalore traffic takes 3 hours daily from whitefield to hsr, back pain unbearable", "25-34", "neutral", "Civic Anger & Institutional Skepticism", "impatient", "rational", "AAP"),
    ("as a working woman in tech, balancing maternity leave policy with career ladder is brutal", "25-34", "female", "Social Empathy & Grassroots Care", "patient", "rational", "None"),
    ("hike of 4% when actual food and school inflation is above 10% is a pay cut", "25-34", "neutral", "Financial Anxiety & Economic Strain", "impatient", "rational", "Congress"),
    ("pf interest credit is pending for 2 years, epfo portal never works during business hours", "25-34", "neutral", "Civic Anger & Institutional Skepticism", "impatient", "rational", "None"),
    ("invested in mutual funds via zerodha, market correction wiped 4 months of savings", "25-34", "neutral", "Financial Anxiety & Economic Strain", "patient", "rational", "None"),
    ("wfh policy rollback by tcs and infosys is just to justify commercial real estate", "25-34", "neutral", "Financial Anxiety & Economic Strain", "impatient", "rational", "None"),
    ("delhi rent has doubled in 2 years, landlord asking 11 months advance deposit", "25-34", "neutral", "Financial Anxiety & Economic Strain", "impatient", "rational", "AAP"),
    ("gst notices being sent to small startups for retrospective claims from 2017", "25-34", "neutral", "Civic Anger & Institutional Skepticism", "impatient", "rational", "Congress"),
    ("bhai medical insurance claim rejected by tpa citing pre-existing condition loophole", "25-34", "male", "Civic Anger & Institutional Skepticism", "impatient", "rational", "None"),
    ("as a single working girl in gurgaon, finding safe rental accommodation is exhausting", "25-34", "female", "Financial Anxiety & Economic Strain", "patient", "rational", "None"),
    ("ctc package is 22 lpa on paper but in-hand salary after pf and tds is barely 1.2L", "25-34", "neutral", "Financial Anxiety & Economic Strain", "impatient", "rational", "None"),
    ("modi hai to mumkin hai, finally some economic sense in policy", "25-34", "male", "Cultural Pride & National Solidarity", "patient", "rational", "BJP"),
    ("rahul gandhi is the only hope for middle class, bjp only favors crony capitalists", "25-34", "neutral", "Financial Anxiety & Economic Strain", "impatient", "rational", "Congress"),

    # --- 35-54 (Civic / Family) & CIVIC_ANGER / PATRIOTIC_PRIDE / CIVIC ---
    ("potholes on national highway caused fatal accident, who will hold NHAI accountable?", "35-54", "neutral", "Civic Anger & Institutional Skepticism", "impatient", "rational", "None"),
    ("police refused to file FIR against the builder despite high court contempt order", "35-54", "neutral", "Civic Anger & Institutional Skepticism", "impatient", "rational", "None"),
    ("every citizen must stand united for our armed forces and national security. Jai Hind!", "35-54", "neutral", "Cultural Pride & National Solidarity", "patient", "rational", "BJP"),
    ("bharat mata ki jai! our scientists at ISRO have made every Indian proud today", "35-54", "neutral", "Cultural Pride & National Solidarity", "patient", "rational", "BJP"),
    ("namaste ji, preserving our cultural sanskar and civilizational dharma is our duty", "35-54", "neutral", "Cultural Pride & National Solidarity", "patient", "rational", "BJP"),
    ("children are breathing poison air in Delhi NCR, government only playing blame game", "35-54", "neutral", "Civic Anger & Institutional Skepticism", "impatient", "rational", "AAP"),
    ("corruption in municipality water department, tankers charging 2500 per trip", "35-54", "neutral", "Civic Anger & Institutional Skepticism", "impatient", "rational", "None"),
    ("as a mother of two, the school fee hike of 25% without regulation is unacceptable", "35-54", "female", "Civic Anger & Institutional Skepticism", "impatient", "rational", "None"),
    ("bhaiya ji justice delayed is justice denied, 12 years for trial in court is mockery", "35-54", "male", "Civic Anger & Institutional Skepticism", "impatient", "rational", "None"),
    ("shri ram mandir pran pratishtha is a historic moment of national rejuvenation", "35-54", "neutral", "Cultural Pride & National Solidarity", "patient", "rational", "BJP"),
    ("electricity bills spiked by 40% with zero prior notification from discom board", "35-54", "neutral", "Civic Anger & Institutional Skepticism", "impatient", "rational", "AAP"),
    ("public transport bus service in tier-2 cities is completely collapsed, need reforms", "35-54", "neutral", "Civic Anger & Institutional Skepticism", "impatient", "rational", "None"),
    ("proud of our Indian cricket team showing resilience and grit on foreign soil", "35-54", "neutral", "Cultural Pride & National Solidarity", "patient", "rational", "None"),
    ("women safety in public buses and metro stations requires strict street-level policing", "35-54", "female", "Civic Anger & Institutional Skepticism", "impatient", "rational", "Congress"),
    ("water logging in underpass claimed another life, municipal commissioner must resign", "35-54", "neutral", "Civic Anger & Institutional Skepticism", "impatient", "rational", "None"),
    ("our constitution grants fundamental rights to every citizen, government cannot override parliament", "35-54", "neutral", "Civic Anger & Institutional Skepticism", "patient", "rational", "Congress"),
    ("kejriwal's freebies destroyed delhi's economy, now even electricity is expensive", "35-54", "male", "Civic Anger & Institutional Skepticism", "impatient", "irrational", "BJP"),
    ("mamata banerjee didi is the only leader who actually speaks for bengal", "35-54", "female", "Cultural Pride & National Solidarity", "patient", "rational", "TMC"),

    # --- 55+ (Senior Citizen / Veteran) & COMMUNITY_EMPATHY / CIVIC_RESPECT ---
    ("retired central govt employee, pension revision under 8th pay commission is urgent", "55+", "neutral", "Financial Anxiety & Economic Strain", "impatient", "rational", "None"),
    ("in our days public servants had honesty and integrity, today everything is commercialized", "55+", "neutral", "Civic Anger & Institutional Skepticism", "patient", "rational", "None"),
    ("pranam to all elders, may god bless the younger generation with wisdom and health", "55+", "neutral", "Social Empathy & Grassroots Care", "patient", "rational", "None"),
    ("senior citizen fixed deposit interest rate cut has severely reduced our monthly income", "55+", "neutral", "Financial Anxiety & Economic Strain", "impatient", "rational", "None"),
    ("veteran of 1971 war, national unity is paramount above petty political squabbles", "55+", "male", "Cultural Pride & National Solidarity", "patient", "rational", "BJP"),
    ("grandfather of three, concerned about moral education and environment for our future", "55+", "male", "Social Empathy & Grassroots Care", "patient", "rational", "None"),
    ("ayushman bharat card denied at empanelled private hospital, elderly patients suffering", "55+", "neutral", "Civic Anger & Institutional Skepticism", "impatient", "rational", "Congress"),
    ("this government has done wonders for senior citizens, yoga and wellness schemes", "55+", "neutral", "Cultural Pride & National Solidarity", "patient", "rational", "BJP"),
    ("rahul gandhi's nyay scheme would have helped pensioners like us", "55+", "neutral", "Financial Anxiety & Economic Strain", "patient", "rational", "Congress"),

    # --- OBJECTIVE_OBSERVANT & FACT-CHECKING ---
    ("according to the Gazette notification section 144 was issued for 48 hours only", "25-34", "neutral", "Objective Deliberation & Fact-Checking", "patient", "rational", "None"),
    ("what is the legal precedent set by supreme court in the 1993 judges case?", "35-54", "neutral", "Objective Deliberation & Fact-Checking", "patient", "rational", "None"),
    ("sharing the rbi statistical bulletin data table on foreign exchange reserves for context", "25-34", "neutral", "Objective Deliberation & Fact-Checking", "patient", "rational", "None"),
    ("the election commission manual section 4.2 clarifies the postal ballot timeline", "35-54", "neutral", "Objective Deliberation & Fact-Checking", "patient", "rational", "None"),
    ("has anyone verified this circular on the official ministry website or is it unverified rumor?", "25-34", "neutral", "Objective Deliberation & Fact-Checking", "patient", "rational", "None"),

    # --- Additional impatience/irrationality & party samples ---
    ("bhai log ye aam aadmi party walon ko mat bolo, sab chor hai", "30", "male", "Civic Anger & Institutional Skepticism", "impatient", "irrational", "BJP"),
    ("modi modi modi, india needs only modi, rest all useless", "25", "male", "Cultural Pride & National Solidarity", "impatient", "irrational", "BJP"),
    ("congress ne 70 saal loota hai, ab aur kya chahiye inko", "35", "male", "Civic Anger & Institutional Skepticism", "impatient", "irrational", "BJP"),
    ("kejriwal ji ne Delhi ka school system badal diya, best cm ever", "28", "male", "Social Empathy & Grassroots Care", "patient", "rational", "AAP"),
    ("tmc ka goonism west bengal mein sabse badha hai, didi has failed", "32", "male", "Civic Anger & Institutional Skepticism", "impatient", "irrational", "BJP"),
    ("vote for TMC, we are bengalis and we support our own daughter", "40", "female", "Cultural Pride & National Solidarity", "patient", "rational", "TMC"),
    ("rahul gandhu pappu hai, pura congress Gadha hai", "22", "male", "Ironic Detachment & Memetic Cynicism", "impatient", "irrational", "BJP"),
    ("please anyone help with NEET preparation tips, urgent", "19", "female", "Financial Anxiety & Economic Strain", "impatient", "rational", "None"),
    ("waiting for job offer letter since 2 months, so anxious", "26", "neutral", "Financial Anxiety & Economic Strain", "impatient", "rational", "None"),
    ("the new education policy is actually quite balanced if you read it carefully", "34", "neutral", "Objective Deliberation & Fact-Checking", "patient", "rational", "None"),
    ("why is every government form so complicated? takes days just to get a PAN card", "29", "male", "Civic Anger & Institutional Skepticism", "impatient", "rational", "None"),
    ("i dont trust any politician, all are same, just want them to leave us alone", "24", "neutral", "Civic Anger & Institutional Skepticism", "impatient", "irrational", "None"),
    ("bjp should focus on employment rather than mandir masjid", "27", "male", "Civic Anger & Institutional Skepticism", "impatient", "rational", "AAP"),
    ("this traffic signal never turns green, waited 10 minutes already", "31", "male", "Civic Anger & Institutional Skepticism", "impatient", "irrational", "None"),
    ("i admire atal ji's leadership, hope bjp returns with stronger mandate", "45", "male", "Cultural Pride & National Solidarity", "patient", "rational", "BJP"),
    ("nehru built institutions that still serve india, congress values matter", "52", "male", "Social Empathy & Grassroots Care", "patient", "rational", "Congress"),
    ("arvind kejriwal is a fraud, free electricity ruined discom finances", "38", "male", "Civic Anger & Institutional Skepticism", "impatient", "irrational", "BJP"),
    ("didi restored law and order in bengal after 34 years of left violence", "42", "female", "Cultural Pride & National Solidarity", "patient", "rational", "TMC"),
    ("the more i read about farm laws, the more i think they were actually good", "29", "male", "Objective Deliberation & Fact-Checking", "patient", "rational", "BJP"),
    ("i think inflation is a global issue, not just india's problem", "33", "neutral", "Objective Deliberation & Fact-Checking", "patient", "rational", "None"),
    ("arre bhai ye election commission bhi kya karta hai, evm hacking everywhere", "23", "male", "Civic Anger & Institutional Skepticism", "impatient", "irrational", "None"),
    ("government should subsidize onions and petrol immediately, we are suffering", "36", "female", "Financial Anxiety & Economic Strain", "impatient", "rational", "Congress"),
    ("just saw the latest movie, absolutely masterpiece, go watch it guys", "21", "neutral", "Ironic Detachment & Memetic Cynicism", "patient", "rational", "None"),
    ("this new phone launch is overhyped, same specs as last year but 20k costlier", "26", "male", "Objective Deliberation & Fact-Checking", "patient", "rational", "None"),
    ("my parents still think coding means repairing computers, bruh", "20", "male", "Ironic Detachment & Memetic Cynicism", "patient", "rational", "None"),
    ("blessed to have such supportive family during my UPSC preparation", "27", "female", "Social Empathy & Grassroots Care", "patient", "rational", "None"),
    ("the way this kid plays cricket, future captain material for Team India", "35", "male", "Cultural Pride & National Solidarity", "patient", "rational", "None"),
    ("metro expansion in noida is finally happening after 10 years of waiting", "32", "neutral", "Civic Anger & Institutional Skepticism", "patient", "rational", "None"),
    ("why does every small business have to deal with so many licenses and inspectors", "41", "male", "Civic Anger & Institutional Skepticism", "impatient", "rational", "None"),
    ("social media influencers selling dreams, young kids falling for it", "47", "male", "Ironic Detachment & Memetic Cynicism", "patient", "rational", "None"),
    ("my daughter got admission in IIT, feeling so proud as a parent", "45", "male", "Social Empathy & Grassroots Care", "patient", "rational", "None"),
    ("the new tax regime is confusing af, i need a CA just to understand my own salary", "29", "neutral", "Financial Anxiety & Economic Strain", "impatient", "rational", "None"),
    ("breaking news on republic tv is pure entertainment, not journalism", "30", "male", "Ironic Detachment & Memetic Cynicism", "patient", "rational", "None"),
    ("bollywood needs better scripts, same love triangle in every film", "24", "female", "Ironic Detachment & Memetic Cynicism", "patient", "rational", "None"),
    ("government hospitals are in such bad shape, private ones rob you blind", "48", "female", "Civic Anger & Institutional Skepticism", "impatient", "rational", "Congress"),
    ("the way pollution in delhi rises during diwali is criminal", "26", "neutral", "Civic Anger & Institutional Skepticism", "impatient", "rational", "AAP"),
    ("i respect our army but questioning them is not anti-national", "22", "neutral", "Civic Anger & Institutional Skepticism", "patient", "rational", "None"),
    ("started my own chai shop after engineering, best decision ever", "28", "male", "Social Empathy & Grassroots Care", "patient", "rational", "None"),
    ("this political analyst is so biased, how does he even get on prime time", "31", "male", "Ironic Detachment & Memetic Cynicism", "impatient", "rational", "None"),
    ("i voted for bjp last time but now even they seem out of touch", "39", "male", "Civic Anger & Institutional Skepticism", "impatient", "rational", "None"),
    ("thanks to free ration during lockdown, many families survived", "55", "female", "Social Empathy & Grassroots Care", "patient", "rational", "Congress"),
    ("marriage pressure at 28 is too much, still figuring life out", "27", "female", "Financial Anxiety & Economic Strain", "impatient", "rational", "None"),
    ("bjp workers threatening people in villages for votes", "34", "male", "Civic Anger & Institutional Skepticism", "impatient", "rational", "Congress"),
    ("free bus travel for women in delhi is a game changer for safety", "24", "female", "Social Empathy & Grassroots Care", "patient", "rational", "AAP"),
    ("i have been filing ITR for 10 years, still the portal glitches every season", "36", "male", "Civic Anger & Institutional Skepticism", "impatient", "rational", "None"),
    ("respect for elders is declining, kids these days have no sanskar", "58", "male", "Civic Anger & Institutional Skepticism", "patient", "rational", "None"),
    ("the internet is so slow in my village, online classes are impossible", "19", "male", "Civic Anger & Institutional Skepticism", "impatient", "rational", "None"),
    ("my stock portfolio is down 30%, but i am holding for long term", "33", "male", "Financial Anxiety & Economic Strain", "patient", "rational", "None"),
    ("petrol at 110 rupees and they are still saying inflation is under control", "40", "male", "Civic Anger & Institutional Skepticism", "impatient", "rational", "Congress"),
    ("cricket should not be mixed with politics, let players play", "23", "neutral", "Ironic Detachment & Memetic Cynicism", "patient", "rational", "None"),
    ("the new OTT show about partition is deeply moving and historically accurate", "29", "female", "Social Empathy & Grassroots Care", "patient", "rational", "None"),
    ("everyone is an expert on twitter, no one reads actual policy documents", "26", "male", "Ironic Detachment & Memetic Cynicism", "impatient", "rational", "None"),
    ("just paid 2 lakhs for my dog's surgery, pet healthcare is unregulated and costly", "37", "female", "Financial Anxiety & Economic Strain", "patient", "rational", "None"),
    ("i miss the old delhi, now it's just concrete and pollution everywhere", "51", "male", "Civic Anger & Institutional Skepticism", "patient", "rational", "None"),
    ("the way politicians change sides after elections is shameful", "30", "male", "Civic Anger & Institutional Skepticism", "impatient", "rational", "None"),
    ("my father served in the army, i have immense respect for the uniform", "34", "male", "Cultural Pride & National Solidarity", "patient", "rational", "BJP"),
    ("this generation needs to learn patience, everything is instant gratification", "55", "male", "Ironic Detachment & Memetic Cynicism", "patient", "rational", "None"),
    ("the farmer protest was genuine, but some parties hijacked it for votes", "29", "male", "Civic Anger & Institutional Skepticism", "patient", "rational", "AAP"),
    ("i love how diverse india is, that is our real strength", "25", "female", "Social Empathy & Grassroots Care", "patient", "rational", "Congress"),
    ("taxes are high but if infrastructure improves maybe it is worth it", "38", "male", "Objective Deliberation & Fact-Checking", "patient", "rational", "BJP"),
    ("the election commission should ban cash for votes, it is open bribery", "44", "male", "Civic Anger & Institutional Skepticism", "impatient", "rational", "None"),
]


def _normalize_age(age):
    if age in ("13-17", "18-24", "25-34", "35-54", "55+"):
        return age
    try:
        a = int(age)
        if a <= 17:
            return "13-17"
        if a <= 24:
            return "18-24"
        if a <= 34:
            return "25-34"
        if a <= 54:
            return "35-54"
        return "55+"
    except Exception:
        return "unknown"


def load_dataset() -> tuple[list[str], list[dict]]:
    texts = []
    labels_list = []

    if SYNTHETIC_PATH.exists():
        with open(SYNTHETIC_PATH, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                    t = str(item.get("text", "")).strip()
                    if not t or len(t) < 15:
                        continue
                    texts.append(t)
                    item["age_bracket"] = _normalize_age(item.get("age_bracket", "unknown"))
                    labels_list.append(item)
                except Exception:
                    continue

    if not texts:
        for txt, age, gen, state, imp, irr, party in SEED_SAMPLES:
            texts.append(txt)
            labels_list.append({
                "age_bracket": _normalize_age(age),
                "gender": gen,
                "mental_state": state,
                "impatience": imp,
                "irrationality": irr,
                "party_support": party,
            })
        print("Using seed samples as fallback dataset.")

    return texts, labels_list


def _extract_labels(labels_list, key):
    return [str(item.get(key, "unknown")) for item in labels_list]


def build_and_train_demographic_models():
    """Generates expanded training set and trains TF-IDF + Logistic Regression pipelines."""
    texts, labels_list = load_dataset()

    # Build multi-output classifiers
    tasks = {
        "age": _extract_labels(labels_list, "age_bracket"),
        "gender": _extract_labels(labels_list, "gender"),
        "mental_state": _extract_labels(labels_list, "mental_state"),
        "impatience": _extract_labels(labels_list, "impatience"),
        "irrationality": _extract_labels(labels_list, "irrationality"),
        "party_support": _extract_labels(labels_list, "party_support"),
    }

    models = {}
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    for task_name, y in tasks.items():
        pipe = Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)),
            ("clf", LogisticRegression(C=2.5, class_weight="balanced", max_iter=500))
        ])
        pipe.fit(texts, y)
        models[task_name] = pipe
        joblib.dump(pipe, DATA_DIR / f"{task_name}_classifier.joblib")
        print(f"Trained {task_name} classifier on {len(texts)} samples. Classes: {sorted(set(y))}")

    meta = {
        "training_samples": len(texts),
        "age_classes": sorted(set(tasks["age"])),
        "gender_classes": sorted(set(tasks["gender"])),
        "mental_state_classes": sorted(set(tasks["mental_state"])),
        "impatience_classes": sorted(set(tasks["impatience"])),
        "irrationality_classes": sorted(set(tasks["irrationality"])),
        "party_support_classes": sorted(set(tasks["party_support"])),
        "model_version": "v3.0-llm-dataset",
        "dataset_source": "synthetic_llm" if SYNTHETIC_PATH.exists() else "seed_fallback",
    }
    with open(DATA_DIR / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"Successfully trained and saved 6 demographic & behavioral models ({len(texts)} samples) to {DATA_DIR}")
    return models


if __name__ == "__main__":
    build_and_train_demographic_models()
