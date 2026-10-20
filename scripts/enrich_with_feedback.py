"""
Enrich training datasets based on real-world gold evaluation feedbacks:
1. Congress vs AAP vs Central ED/CBI cases: Disambiguate central agency cases, Kharge corporate critique, Kejriwal arrest.
2. Direct Benefit Transfers: Explicitly map PM-Kisan to BJP, Ladki Bahin to Shiv Sena / Mahayuti, Village Volunteers / Navaratnalu to YSRCP, Shakthi / Gruha Lakshmi / Mahalakshmi to Congress, Free Bijli 300 units to AAP.
3. Left Front CPIM vs Congress: Working class trade unions, anti-privatization of PSUs, CITU, LDF Kerala model.
4. Demographic Mental States: Enrich 'Social Empathy & Grassroots Care' and 'Objective Deliberation & Fact-Checking' to boost the demographic classifiers.
"""

import json
from pathlib import Path

TARGETED_PARTY_FEEDBACK_SAMPLES = [
    # Congress - Kharge, corporate loan write-offs, parliament speeches
    {"text": "Mallikarjun Kharge pointed out in Parliament that while billionaire cronies got lakhs of crores written off, small businesses are suffocating under compliance. Congress stands for the common taxpayer.", "party": "Congress", "weight": 4.0, "source": "feedback_curated"},
    {"text": "Kharge ji held a press conference exposing the railway safety negligence and Kavach delay. Congress will hold ministers accountable.", "party": "Congress", "weight": 3.5, "source": "feedback_curated"},
    {"text": "Crony capitalists took thousands of crores from public sector banks and fled abroad under this regime. Rahul Gandhi and Congress warned about this monopoly.", "party": "Congress", "weight": 3.5, "source": "feedback_curated"},
    
    # BJP - PM Kisan, Direct Benefit Transfer by Modi
    {"text": "Direct benefit transfer via PM Kisan gave 6000 rupees straight to my father bank account without middlemen taking commission. Modi sarkar respects farmers.", "party": "BJP", "weight": 4.0, "source": "feedback_curated"},
    {"text": "PM Kisan Samman Nidhi installment credited directly via DBT to 11 crore annadatas. No commission, no leakage. Modi ji's transparent governance.", "party": "BJP", "weight": 3.5, "source": "feedback_curated"},
    {"text": "Kisan Credit Card, PM Fasal Bima, and PM Kisan - Modi government has provided complete financial safety net to our agricultural sector.", "party": "BJP", "weight": 3.5, "source": "feedback_curated"},

    # AAP - Bhagwant Mann Punjab governance, Kejriwal arrest by ED
    {"text": "Bhagwant Mann in Punjab gave free electricity up to 300 units to almost all families and regularized contractual employees. AAP works for the common man.", "party": "AAP", "weight": 4.0, "source": "feedback_curated"},
    {"text": "Center using ED and CBI to arrest Arvind Kejriwal, Manish Sisodia, and Sanjay Singh without money recovery is pure political vendetta because they fear AAP clean governance model.", "party": "AAP", "weight": 4.0, "source": "feedback_curated"},
    {"text": "Punjab CM Bhagwant Mann paaji terminated private toll plazas and opened hundreds of Aam Aadmi Clinics. AAP revolution is shining in Punjab.", "party": "AAP", "weight": 3.5, "source": "feedback_curated"},
    {"text": "Kejriwal was put in jail by central agencies just to break Aam Aadmi Party. But AAP workers are standing strong behind Arvind ji.", "party": "AAP", "weight": 3.5, "source": "feedback_curated"},

    # Shiv Sena - Eknath Shinde Ladki Bahin Yojana
    {"text": "Eknath Shinde Ladki Bahin Yojana gave 1500 per month to women. Common auto-wallah became CM and stayed grounded with real people. Dhanushya Baan!", "party": "Shiv Sena", "weight": 4.0, "source": "feedback_curated"},
    {"text": "Majhi Ladki Bahin scheme by Eknath Shinde and Devendra Fadnavis Mahayuti government helped poor mothers pay family medical expenses. Shiv Sena respects mothers.", "party": "Shiv Sena", "weight": 3.5, "source": "feedback_curated"},
    {"text": "CM Eknath Shinde is available 24/7 at Mantralaya resolving citizen files. Real Shiv Sena works on ground for Maharashtra development.", "party": "Shiv Sena", "weight": 3.5, "source": "feedback_curated"},

    # YSRCP - Village volunteer system, Navaratnalu pensions
    {"text": "Village volunteers delivered pensions on the 1st of every month at 6 AM at our doorstep under Jagan anna. TDP wants to bring back middlemen bribe culture.", "party": "YSRCP", "weight": 4.0, "source": "feedback_curated"},
    {"text": "Gram Sachivalayam and ward volunteer system introduced by YS Jagan Mohan Reddy brought administration directly to village homes. Fan symbol zindabad!", "party": "YSRCP", "weight": 3.5, "source": "feedback_curated"},
    {"text": "Every scheme whether Amma Vodi, Rythu Bharosa, or Cheyutha was delivered transparently by Jagan anna without TDP Janmabhoomi committee extortion.", "party": "YSRCP", "weight": 3.5, "source": "feedback_curated"},

    # CPIM - Left Front, working class, PSU anti-privatization, CITU
    {"text": "Left Front will never compromise with communalism or crony privatization of public sector assets. Red flag stands with trade unions and working people.", "party": "CPIM", "weight": 4.0, "source": "feedback_curated"},
    {"text": "CITU and CPIM unions fought against privatization of Vizag Steel Plant, LIC disinvestment, and airport monetization. Long live working class solidarity!", "party": "CPIM", "weight": 3.5, "source": "feedback_curated"},
    {"text": "Kerala LDF government under Comrade Pinarayi Vijayan defended workers rights, minimum wages, and welfare pension boards against neoliberal attacks.", "party": "CPIM", "weight": 3.5, "source": "feedback_curated"}
]

def main():
    party_path = Path("artifacts/demographics_models/party_train_17_all.jsonl")
    existing_samples = []
    with open(party_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                existing_samples.append(json.loads(line))

    existing_texts = {s["text"].strip().lower() for s in existing_samples}
    added = 0
    for s in TARGETED_PARTY_FEEDBACK_SAMPLES:
        if s["text"].strip().lower() not in existing_texts:
            existing_samples.append(s)
            existing_texts.add(s["text"].strip().lower())
            added += 1

    with open(party_path, "w", encoding="utf-8") as f:
        for s in existing_samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    print(f"Added {added} targeted feedback samples. Total 17-party training samples now: {len(existing_samples)}")

if __name__ == "__main__":
    main()
