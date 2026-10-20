import json, sys, os, time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
sys.path.insert(0, '/Users/s0r0qw6/identify-what-matters-main/src')
from identify_what_matters.llm import make_backend_client
from identify_what_matters.utils import parse_json_payload

SYNTHETIC_PATH = Path("artifacts/demographics_models/synthetic_dataset.jsonl")

PROMPTS = [
    {
        "state": "Social Empathy & Grassroots Care",
        "ages": ["18-24", "25-34", "35-54", "55+"],
        "desc": "Grassroots community mutual aid: flood relief kit packing in Kerala/Assam/Uttarakhand, late night blood donor mobilization for emergency surgeries, helping injured street animals, pooling funds for househelp/maid's child school uniforms and books, community kitchens (langar/annadanam) for daily wage laborers. Warm, deeply compassionate, heartfelt Hinglish and Indian English.",
        "count": 12
    },
    {
        "state": "Social Empathy & Grassroots Care",
        "ages": ["18-24", "25-34", "35-54"],
        "desc": "Empathy for gig workers: Swiggy/Zomato/Blinkit delivery partners working in torrential monsoon rains and 45°C summer heat, offering water bottles and tips to delivery riders, supporting street vendors during harsh municipal crackdowns, peer mental health counseling for stressed students. Grounded, conversational, empathetic tone.",
        "count": 12
    },
    {
        "state": "Social Empathy & Grassroots Care",
        "ages": ["35-54", "55+"],
        "desc": "Elderly and healthcare mutual aid: elder care companion visits, helping bedridden pensioners with bank paperwork, sponsoring cancer chemotherapy drugs for low-income patients, organizing free weekend eye camps and medical consultations in urban slums. Pure social solidarity and altruism.",
        "count": 12
    },
    {
        "state": "Ironic Detachment & Memetic Cynicism",
        "ages": ["13-17", "18-24", "25-34"],
        "desc": "Gen-Z & millennial dark humor, corporate rat race fatigue, 'delulu is the only solulu', Bangalore tech park 3-hour traffic jams, Whitefield water tanker mafia sarcasm, Gurgaon smog post-apocalyptic AQI 999 humor, laughing at return-to-office mandates, relatable burnout fatalism, Discord/Instagram reel banter.",
        "count": 12
    },
    {
        "state": "Ironic Detachment & Memetic Cynicism",
        "ages": ["18-24", "25-34", "35-54"],
        "desc": "Satire on modern Indian life: tech startup founders 'disrupting hydration', 4-crore 2BHK pigeonhole apartments in Bangalore/Mumbai, absurd celebrity wedding news cycles, election promises that sound like sci-fi fantasy, astrology WhatsApp forward mockery, humorous resignation to daily absurdity.",
        "count": 12
    },
    {
        "state": "Financial Anxiety & Economic Strain",
        "ages": ["25-34", "35-54", "55+"],
        "desc": "Crushing middle-class financial squeeze: home loan EMI interest rate spikes adding 8 years to tenure, private school tuition fees touching 1.5 lakhs, health insurance claim rejections with 80% out-of-pocket hospital bills, fear of sudden tech layoffs with dependent parents, rent deposit extortion in Bengaluru, grocery and vegetable inflation eroding monthly savings.",
        "count": 12
    },
    {
        "state": "Financial Anxiety & Economic Strain",
        "ages": ["18-24", "25-34", "55+"],
        "desc": "Generation-specific financial stress: fresh graduates unable to repay education loans due to hiring freezes, young couples delaying marriage and children due to sky-high real estate costs, senior citizens watching fixed deposit interest rates fail to beat chemotherapy/dialysis medical inflation, constant dread of unexpected expenses wiping out life savings.",
        "count": 12
    },
    {
        "state": "Objective Deliberation & Fact-Checking",
        "ages": ["25-34", "35-54", "55+"],
        "desc": "Analytical, data-grounded evaluation: citing CAG audit findings on railway track renewal fund underutilization, comparing RBI Monetary Policy Committee CPI inflation forecasts vs actual MOSPI data, analyzing capital expenditure outlays vs revenue expenditure in Union and State budgets, examining solar capacity megawatt additions factually without partisan hysteria.",
        "count": 12
    },
    {
        "state": "Objective Deliberation & Fact-Checking",
        "ages": ["25-34", "35-54", "55+"],
        "desc": "Fact-checking viral claims and public policy data: reviewing NFHS-5 maternal mortality and stunting metrics across southern vs northern states, calculating per-kilometer deployment costs of railway Kavach automatic train protection, examining direct tax vs indirect GST revenue ratios, comparing EV bus procurement targets with actual fleet deployments objectively.",
        "count": 12
    },
    {
        "state": "Ironic Detachment & Memetic Cynicism",
        "ages": ["13-17"],
        "desc": "Authentic Indian high school life (ages 13-17): 10th/12th board exam anxiety, Kota coaching center pressure, tuition teacher taunts, BGMI gaming vs mock test guilt, reel addiction, generational gap with parents saying 'hamare zamane mein...', dark humor about syllabus revisions, peer banter.",
        "count": 12
    },
    {
        "state": "Civic Anger & Institutional Skepticism",
        "ages": ["55+"],
        "desc": "Senior citizen civic grievances (ages 55+): EPFO digital portal OTP failures making pensioners helpless, bank branch closures forcing elderly to struggle with smartphone banking, fear of cyber/digital banking OTP scams, broken park walking tracks, surging prices of blood pressure and diabetes medicines on fixed pensions, nostalgic indignation at loss of basic human decency in public offices.",
        "count": 12
    }
]

SYSTEM_PROMPT = """You are a high-fidelity synthetic social discourse generator for an advanced NLP demographic analysis system in India.
You output strictly valid JSON matching the exact schema requested."""

TEMPLATE = """Generate {count} hyper-realistic social media comments/reactions embodying the Mental State archetype: '{state}'.
Context: {desc}
Allowed Age Brackets for this batch: {ages}

Ensure varied:
- age_bracket: one of {ages}
- gender: one of "male", "female", "neutral"
- mental_state: exactly '{state}'
- impatience: one of "impatient", "neutral", "patient"
- irrationality: one of "irrational", "neutral", "rational"
- party_support: one of "BJP", "Congress", "AAP", "TMC", "DMK", "SP", "Shiv Sena", "TDP", "None"

Output strictly a JSON object:
{{
  "samples": [
    {{
      "text": "<comment text, 15 to 45 words, authentic Hinglish/English>",
      "age_bracket": "<one of the allowed ages>",
      "gender": "<gender>",
      "mental_state": "{state}",
      "impatience": "<impatience>",
      "irrationality": "<irrationality>",
      "party_support": "<party>"
    }}
  ]
}}
"""

def fetch_prompt(p_def, run_idx):
    client = make_backend_client(max_tokens=2048)
    prompt = TEMPLATE.format(
        count=p_def["count"],
        state=p_def["state"],
        desc=p_def["desc"],
        ages=json.dumps(p_def["ages"])
    )
    for attempt in range(2):
        try:
            resp = client.generate(user_prompt=prompt, request_timeout=40, system_instruction=SYSTEM_PROMPT)
            data = parse_json_payload(resp.text)
            samples = data.get("samples", [])
            valid = []
            for s in samples:
                txt = str(s.get("text", "")).strip()
                if len(txt) >= 20:
                    s["text"] = txt
                    valid.append(s)
            if valid:
                return p_def["state"], run_idx, valid
        except Exception as e:
            time.sleep(1)
    return p_def["state"], run_idx, []

def main():
    tasks = []
    # 2 runs per prompt definition = 11 * 2 = 22 tasks * 12 = ~264 samples per pass
    for idx, p in enumerate(PROMPTS):
        for r in range(2):
            tasks.append((p, r))

    print(f"Generating ~264 targeted demographic samples across {len(tasks)} parallel LLM batches (concurrency=2)...")
    all_new = []
    with ThreadPoolExecutor(max_workers=2) as ex:
        futures = [ex.submit(fetch_prompt, p, r) for p, r in tasks]
        for fut in as_completed(futures):
            state, r, valid = fut.result()
            all_new.extend(valid)
            print(f"[{state[:25]} #{r}] Received {len(valid)} valid samples (Running total: {len(all_new)})")

    print(f"\nTotal new high-quality samples collected: {len(all_new)}")
    if all_new:
        with open(SYNTHETIC_PATH, "a", encoding="utf-8") as f:
            for item in all_new:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"Successfully appended {len(all_new)} samples to {SYNTHETIC_PATH}")
        print(f"New total synthetic dataset size: {sum(1 for _ in open(SYNTHETIC_PATH))}")

if __name__ == "__main__":
    main()
