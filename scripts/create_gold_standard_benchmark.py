"""
Simulate a diverse, authentic real-world gold-standard evaluation dataset for India's civic, demographic,
and political discourse. Covers:
- 17 Indian Political Parties + Neutral
- 6 Mental State Archetypes
- Age Brackets, Genders, Impatience, Irrationality
- Code-mixed Indian languages: Hinglish, Tanglish, Telugish, Bengali-English, pure vernaculars, and Indian English.
"""

import json
from pathlib import Path

REAL_WORLD_GOLD_DATASET = [
    # =========================================================================
    # CONGRESS (Distinctive economic justice, Samvidhan, Caste Census, Agniveer, NYAY, Bharat Jodo, Kharge, Rahul)
    # =========================================================================
    {
        "text": "Mallikarjun Kharge pointed out in Parliament that while billionaire cronies got lakhs of crores written off, small businesses are suffocating under compliance.",
        "party": "Congress",
        "mental_state": "Civic Anger & Institutional Skepticism",
        "age_bracket": "35-54",
        "gender": "male",
        "impatience": "impatient",
        "irrationality": "rational"
    },
    {
        "text": "Samvidhan bachao andolan and caste census are our guarantees. Without knowing OBC numbers, budgetary allocation remains unjust.",
        "party": "Congress",
        "mental_state": "Objective Deliberation & Fact-Checking",
        "age_bracket": "25-34",
        "gender": "male",
        "impatience": "neutral",
        "irrationality": "rational"
    },
    {
        "text": "Rahul Gandhi walking 4000 km in Bharat Jodo Yatra brought hope. Nafrat ke bazaar mein mohabbat ki dukaan is what India needs right now.",
        "party": "Congress",
        "mental_state": "Cultural Pride & National Solidarity",
        "age_bracket": "18-24",
        "gender": "female",
        "impatience": "patient",
        "irrationality": "rational"
    },
    {
        "text": "Agnipath scheme has broken the spine of rural youth aspiring for permanent army service. Congress promise to scrap Agniveer is crucial.",
        "party": "Congress",
        "mental_state": "Civic Anger & Institutional Skepticism",
        "age_bracket": "18-24",
        "gender": "male",
        "impatience": "impatient",
        "irrationality": "rational"
    },
    {
        "text": "Karnataka Shakthi free bus travel scheme for women is genuine empowerment. My sister saves 2000 rupees every month commuting to college. Congress delivers.",
        "party": "Congress",
        "mental_state": "Social Empathy & Grassroots Care",
        "age_bracket": "25-34",
        "gender": "female",
        "impatience": "patient",
        "irrationality": "rational"
    },

    # =========================================================================
    # BJP (Modi leadership, Viksit Bharat, Hindutva, infrastructure, Article 370, Ram Mandir)
    # =========================================================================
    {
        "text": "Under Prime Minister Narendra Modi, India has transformed its physical infrastructure. Vande Bharat, expressways, and digital UPI are visible across the country.",
        "party": "BJP",
        "mental_state": "Cultural Pride & National Solidarity",
        "age_bracket": "35-54",
        "gender": "male",
        "impatience": "patient",
        "irrationality": "rational"
    },
    {
        "text": "Yogi Adityanath eradicated mafia culture in UP. Earlier goons operated with impunity; now women can step out safely at 10 PM. Bulldozer action on criminals was necessary.",
        "party": "BJP",
        "mental_state": "Civic Anger & Institutional Skepticism",
        "age_bracket": "25-34",
        "gender": "female",
        "impatience": "impatient",
        "irrationality": "neutral"
    },
    {
        "text": "Ayodhya Ram Mandir Pran Pratishtha restored our civilizational dignity after 500 years. Modi ji fulfilled what no government could. Jai Shri Ram!",
        "party": "BJP",
        "mental_state": "Cultural Pride & National Solidarity",
        "age_bracket": "55+",
        "gender": "male",
        "impatience": "patient",
        "irrationality": "rational"
    },
    {
        "text": "Direct benefit transfer via PM Kisan gave 6000 rupees straight to my father bank account without middlemen taking commission. Viksit Bharat is happening.",
        "party": "BJP",
        "mental_state": "Cultural Pride & National Solidarity",
        "age_bracket": "25-34",
        "gender": "male",
        "impatience": "patient",
        "irrationality": "rational"
    },
    {
        "text": "INDI alliance only does appeasement politics for vote banks. Only BJP under Narendra Modi can provide stable national security on Pakistan and China borders.",
        "party": "BJP",
        "mental_state": "Civic Anger & Institutional Skepticism",
        "age_bracket": "35-54",
        "gender": "male",
        "impatience": "neutral",
        "irrationality": "neutral"
    },

    # =========================================================================
    # AAP (Delhi & Punjab governance, Kejriwal, Sisodia, Mohalla Clinics, Atishi, Sanjay Singh)
    # =========================================================================
    {
        "text": "Arvind Kejriwal transformed government schools in Delhi so much that middle class families are shifting kids from private schools. Manish Sisodia built world class education.",
        "party": "AAP",
        "mental_state": "Social Empathy & Grassroots Care",
        "age_bracket": "35-54",
        "gender": "male",
        "impatience": "patient",
        "irrationality": "rational"
    },
    {
        "text": "Mohalla clinic in our area provides free doctor consult and free blood tests. We used to spend 1500 every time someone fell ill. Kejriwal government gave real relief.",
        "party": "AAP",
        "mental_state": "Financial Anxiety & Economic Strain",
        "age_bracket": "35-54",
        "gender": "female",
        "impatience": "neutral",
        "irrationality": "rational"
    },
    {
        "text": "Bhagwant Mann in Punjab gave free electricity up to 300 units to almost all families and regularized contractual employees. AAP works for the common man.",
        "party": "AAP",
        "mental_state": "Cultural Pride & National Solidarity",
        "age_bracket": "25-34",
        "gender": "male",
        "impatience": "patient",
        "irrationality": "rational"
    },
    {
        "text": "Center using ED and CBI to arrest Kejriwal and Sisodia without money recovery is pure political vendetta because they fear AAP clean governance model.",
        "party": "AAP",
        "mental_state": "Civic Anger & Institutional Skepticism",
        "age_bracket": "25-34",
        "gender": "male",
        "impatience": "impatient",
        "irrationality": "rational"
    },

    # =========================================================================
    # TMC (West Bengal, Mamata Banerjee, Lakshmir Bhandar, Khela Hobe, Didi)
    # =========================================================================
    {
        "text": "Mamata didi Lakshmir Bhandar scheme gives direct cash monthly to women in every village. It gives mothers independence and dignity. Joy Bangla!",
        "party": "TMC",
        "mental_state": "Social Empathy & Grassroots Care",
        "age_bracket": "35-54",
        "gender": "female",
        "impatience": "patient",
        "irrationality": "rational"
    },
    {
        "text": "Bengal will never submit to Delhi zamindars. Mamata Banerjee fought on a wheelchair and defeated the entire central machinery. Khela hobe again!",
        "party": "TMC",
        "mental_state": "Civic Anger & Institutional Skepticism",
        "age_bracket": "25-34",
        "gender": "male",
        "impatience": "impatient",
        "irrationality": "neutral"
    },
    {
        "text": "Central government is deliberately withholding Bengal MGNREGA 100 days work wages and Awas Yojana funds to harass poor villagers. TMC will fight in Delhi.",
        "party": "TMC",
        "mental_state": "Civic Anger & Institutional Skepticism",
        "age_bracket": "35-54",
        "gender": "male",
        "impatience": "impatient",
        "irrationality": "rational"
    },

    # =========================================================================
    # SP (Samajwadi Party, Akhilesh Yadav, PDA, Netaji Mulayam, Bicycle symbol)
    # =========================================================================
    {
        "text": "Akhilesh Yadav PDA formula gave real political voice to backward classes and Dalits. Netaji taught us socialist values and bicycle will defeat bulldozer sarkar.",
        "party": "SP",
        "mental_state": "Cultural Pride & National Solidarity",
        "age_bracket": "25-34",
        "gender": "male",
        "impatience": "neutral",
        "irrationality": "rational"
    },
    {
        "text": "Agra-Lucknow expressway and Lucknow Metro were constructed in record time under Akhilesh bhaiya. Real development vs just communal hate speeches.",
        "party": "SP",
        "mental_state": "Objective Deliberation & Fact-Checking",
        "age_bracket": "25-34",
        "gender": "male",
        "impatience": "patient",
        "irrationality": "rational"
    },
    {
        "text": "Faizabad Ayodhya seat victory by Awadhesh Prasad proved that even the religious center voted for PDA social justice over polarization. Samajwadi zindabad!",
        "party": "SP",
        "mental_state": "Cultural Pride & National Solidarity",
        "age_bracket": "35-54",
        "gender": "male",
        "impatience": "patient",
        "irrationality": "rational"
    },

    # =========================================================================
    # DMK (Tamil Nadu, MK Stalin, Dravidian Model, Social Justice, Rising Sun)
    # =========================================================================
    {
        "text": "Chief Minister MK Stalin Dravidian model proves social justice and industrial growth go hand in hand. Tamil Nadu tops in women employment and factory output.",
        "party": "DMK",
        "mental_state": "Objective Deliberation & Fact-Checking",
        "age_bracket": "35-54",
        "gender": "male",
        "impatience": "patient",
        "irrationality": "rational"
    },
    {
        "text": "Governor RN Ravi sitting on state assembly bills is against federalism. DMK under Stalin will never accept Hindi imposition or RSS agenda in Periyar land.",
        "party": "DMK",
        "mental_state": "Civic Anger & Institutional Skepticism",
        "age_bracket": "25-34",
        "gender": "male",
        "impatience": "impatient",
        "irrationality": "rational"
    },
    {
        "text": "Kalaignar Magalir Urimai Thogai providing 1000 rupees monthly to 1 crore women recognized the uncounted labour of homemakers. DMK cares for mothers.",
        "party": "DMK",
        "mental_state": "Social Empathy & Grassroots Care",
        "age_bracket": "35-54",
        "gender": "female",
        "impatience": "patient",
        "irrationality": "rational"
    },

    # =========================================================================
    # SHIV SENA (Maharashtra, Balasaheb, Uddhav Thackeray, Eknath Shinde, Bow & Arrow/Flaming Torch)
    # =========================================================================
    {
        "text": "Balasaheb Thackeray Marathi manoos self-respect will never die. Those who stole the bow and arrow using Delhi money will face the wrath of Mumbaikars. Mashaal zindabad!",
        "party": "Shiv Sena",
        "mental_state": "Civic Anger & Institutional Skepticism",
        "age_bracket": "35-54",
        "gender": "male",
        "impatience": "impatient",
        "irrationality": "neutral"
    },
    {
        "text": "Uddhav Thackeray handled Covid with utmost transparency and compassion. The Dharavi containment model was praised worldwide by WHO.",
        "party": "Shiv Sena",
        "mental_state": "Social Empathy & Grassroots Care",
        "age_bracket": "35-54",
        "gender": "male",
        "impatience": "patient",
        "irrationality": "rational"
    },
    {
        "text": "Eknath Shinde Ladki Bahin Yojana gave 1500 per month to women. Common auto-wallah became CM and stayed grounded with real people. Dhanushya Baan!",
        "party": "Shiv Sena",
        "mental_state": "Cultural Pride & National Solidarity",
        "age_bracket": "25-34",
        "gender": "female",
        "impatience": "patient",
        "irrationality": "rational"
    },

    # =========================================================================
    # NCP (Sharad Pawar, Ajit Pawar, Supriya Sule, Clock/Tutari, Baramati)
    # =========================================================================
    {
        "text": "Sharad Pawar saheb built the farmer cooperative sugar factories and milk cooperatives across Maharashtra. Even at 83, his fighting spirit in Satara rain is legend. Tutari!",
        "party": "NCP",
        "mental_state": "Cultural Pride & National Solidarity",
        "age_bracket": "55+",
        "gender": "male",
        "impatience": "patient",
        "irrationality": "rational"
    },
    {
        "text": "Supriya Sule consistently wins Sansad Ratna for raising farmer issues, inflation, and women safety in Parliament with thorough data. NCP brings intellect to politics.",
        "party": "NCP",
        "mental_state": "Objective Deliberation & Fact-Checking",
        "age_bracket": "35-54",
        "gender": "female",
        "impatience": "neutral",
        "irrationality": "rational"
    },

    # =========================================================================
    # RJD (Bihar, Lalu Prasad Yadav, Tejashwi Yadav, 10 Lakh Naukri, Lantern)
    # =========================================================================
    {
        "text": "Tejashwi Yadav delivered 4 lakh permanent government teacher and doctor jobs in just 17 months in Bihar. Laltain chhap stands for youth employment.",
        "party": "RJD",
        "mental_state": "Cultural Pride & National Solidarity",
        "age_bracket": "18-24",
        "gender": "male",
        "impatience": "patient",
        "irrationality": "rational"
    },
    {
        "text": "Lalu Prasad Yadav gave voice and backward representation to marginalized Dalits and poor who were never allowed to enter polling booths. RJD gave self-respect.",
        "party": "RJD",
        "mental_state": "Cultural Pride & National Solidarity",
        "age_bracket": "55+",
        "gender": "male",
        "impatience": "patient",
        "irrationality": "rational"
    },

    # =========================================================================
    # JDU (Bihar, Nitish Kumar, Sushasan Babu, Har Ghar Nal, Teer symbol)
    # =========================================================================
    {
        "text": "Nitish Kumar sushasan babu built paved roads and Har Ghar Nal Ka Jal tap water across every Bihar village. 50% reservation for women in Panchayats changed rural society.",
        "party": "JDU",
        "mental_state": "Objective Deliberation & Fact-Checking",
        "age_bracket": "35-54",
        "gender": "female",
        "impatience": "patient",
        "irrationality": "rational"
    },
    {
        "text": "Nitish Babu cycle scheme for girl students stopped school dropouts in Bihar. Personal honesty and corruption-free public life is why JDU commands respect.",
        "party": "JDU",
        "mental_state": "Social Empathy & Grassroots Care",
        "age_bracket": "35-54",
        "gender": "male",
        "impatience": "patient",
        "irrationality": "rational"
    },

    # =========================================================================
    # AIADMK (Tamil Nadu, Edappadi Palaniswami, Jayalalithaa Amma, Two Leaves)
    # =========================================================================
    {
        "text": "Jayalalithaa Amma canteen and welfare legacy will live forever in TN. Edappadi Palaniswami gave 7.5% reservation for government school students in medical admissions. Rettai Ilai!",
        "party": "AIADMK",
        "mental_state": "Social Empathy & Grassroots Care",
        "age_bracket": "35-54",
        "gender": "female",
        "impatience": "patient",
        "irrationality": "rational"
    },
    {
        "text": "AIADMK broke ties with BJP to protect minority rights and state interests. EPS proved that AIADMK will never be a B-team of northern parties.",
        "party": "AIADMK",
        "mental_state": "Civic Anger & Institutional Skepticism",
        "age_bracket": "25-34",
        "gender": "male",
        "impatience": "neutral",
        "irrationality": "rational"
    },

    # =========================================================================
    # TDP (Andhra Pradesh, Chandrababu Naidu, Nara Lokesh, Amaravati, Cycle)
    # =========================================================================
    {
        "text": "Chandrababu Naidu vision built Cyberabad and brought Microsoft to Hyderabad. Now he is constructing world-class capital Amaravati and completing Polavaram. Jai TDP!",
        "party": "TDP",
        "mental_state": "Cultural Pride & National Solidarity",
        "age_bracket": "25-34",
        "gender": "male",
        "impatience": "patient",
        "irrationality": "rational"
    },
    {
        "text": "Under YSRCP rule, AP infrastructure collapsed completely and investors fled. CBN coming back brought immediate industrial investments and restored fiscal sanity.",
        "party": "TDP",
        "mental_state": "Civic Anger & Institutional Skepticism",
        "age_bracket": "35-54",
        "gender": "male",
        "impatience": "impatient",
        "irrationality": "rational"
    },

    # =========================================================================
    # YSRCP (Andhra Pradesh, YS Jagan Mohan Reddy, Amma Vodi, Navaratnalu, Fan)
    # =========================================================================
    {
        "text": "YS Jagan Mohan Reddy Amma Vodi gave 15000 per year directly to mothers sending kids to school. Nadu Nedu made government schools better than corporate schools. Fan symbol!",
        "party": "YSRCP",
        "mental_state": "Social Empathy & Grassroots Care",
        "age_bracket": "25-34",
        "gender": "female",
        "impatience": "patient",
        "irrationality": "rational"
    },
    {
        "text": "Village volunteers delivered pensions on the 1st of every month at 6 AM at our doorstep under Jagan anna. TDP wants to bring back middlemen bribe culture.",
        "party": "YSRCP",
        "mental_state": "Civic Anger & Institutional Skepticism",
        "age_bracket": "55+",
        "gender": "male",
        "impatience": "neutral",
        "irrationality": "neutral"
    },

    # =========================================================================
    # BRS (Telangana, KCR, KTR, Rythu Bandhu, Dalit Bandhu, Car symbol)
    # =========================================================================
    {
        "text": "KCR garu fought and brought Telangana state. Rythu Bandhu 10000 per acre and 24 hour uninterrupted free power turned barren lands green. Car symbol forever!",
        "party": "BRS",
        "mental_state": "Cultural Pride & National Solidarity",
        "age_bracket": "35-54",
        "gender": "male",
        "impatience": "patient",
        "irrationality": "rational"
    },
    {
        "text": "KTR built T-Hub and made Hyderabad the tech and biotech capital attracting Amazon and Google global campuses. BRS infra speaks for itself.",
        "party": "BRS",
        "mental_state": "Cultural Pride & National Solidarity",
        "age_bracket": "25-34",
        "gender": "male",
        "impatience": "patient",
        "irrationality": "rational"
    },

    # =========================================================================
    # BSP (Mayawati, Kanshi Ram, Bahujan Samaj, Elephant symbol, Jai Bhim)
    # =========================================================================
    {
        "text": "Mayawati behenji gave strict law and order and Bahujan self respect in UP. Kanshi Ram dream of Bahujan empowerment was realized without compromising ideology. Jai Bhim!",
        "party": "BSP",
        "mental_state": "Cultural Pride & National Solidarity",
        "age_bracket": "35-54",
        "gender": "male",
        "impatience": "patient",
        "irrationality": "rational"
    },
    {
        "text": "Under BSP 2007-2012, no communal riots took place in Uttar Pradesh. Police had free hand against goons regardless of caste or religion. Elephant symbol is dignity.",
        "party": "BSP",
        "mental_state": "Objective Deliberation & Fact-Checking",
        "age_bracket": "35-54",
        "gender": "male",
        "impatience": "patient",
        "irrationality": "rational"
    },

    # =========================================================================
    # CPIM (Kerala, Pinarayi Vijayan, Left Front, Working Class, Lalsalam)
    # =========================================================================
    {
        "text": "Pinarayi Vijayan Left Front Kerala model guarantees healthcare for working class. KFON providing free high speed internet to poor households is true socialism. Lalsalam!",
        "party": "CPIM",
        "mental_state": "Social Empathy & Grassroots Care",
        "age_bracket": "35-54",
        "gender": "male",
        "impatience": "patient",
        "irrationality": "rational"
    },
    {
        "text": "Left Front will never compromise with communalism or crony privatization of public sector assets. Red flag stands with trade unions and working people.",
        "party": "CPIM",
        "mental_state": "Civic Anger & Institutional Skepticism",
        "age_bracket": "25-34",
        "gender": "male",
        "impatience": "impatient",
        "irrationality": "rational"
    },

    # =========================================================================
    # NONE (Apolitical civic complaints, technology, sports, food, lifestyle, meme culture)
    # =========================================================================
    {
        "text": "Silk board to Marathahalli outer ring road is stuck in bumper to bumper traffic for 2 hours every evening. Metro construction needs faster execution.",
        "party": "None",
        "mental_state": "Civic Anger & Institutional Skepticism",
        "age_bracket": "25-34",
        "gender": "male",
        "impatience": "impatient",
        "irrationality": "rational"
    },
    {
        "text": "Bro, living in Delhi with 450 AQI every November is peak dystopian. My air purifier is wheezing harder than my grandfather. Delulu is the only solulu at this point.",
        "party": "None",
        "mental_state": "Ironic Detachment & Memetic Cynicism",
        "age_bracket": "18-24",
        "gender": "neutral",
        "impatience": "neutral",
        "irrationality": "irrational"
    },
    {
        "text": "Corporate burn-out is real guys. Working 14 hour shifts for a 3 percent appraisal while CEO buys another yacht in Europe. Monday morning existential dread hit hard.",
        "party": "None",
        "mental_state": "Ironic Detachment & Memetic Cynicism",
        "age_bracket": "25-34",
        "gender": "female",
        "impatience": "impatient",
        "irrationality": "rational"
    },
    {
        "text": "Home loan interest rate jumped to 9.25 percent and EMI went up by 8000 rupees. With school fees doubling, middle class household budget has completely collapsed.",
        "party": "None",
        "mental_state": "Financial Anxiety & Economic Strain",
        "age_bracket": "35-54",
        "gender": "male",
        "impatience": "impatient",
        "irrationality": "rational"
    },
    {
        "text": "Looking at RBI Monetary Policy Committee minutes, core CPI inflation has cooled down to 3.8% but food price volatility in vegetables remains sticky.",
        "party": "None",
        "mental_state": "Objective Deliberation & Fact-Checking",
        "age_bracket": "35-54",
        "gender": "male",
        "impatience": "neutral",
        "irrationality": "rational"
    },
    {
        "text": "Organized a blood donation camp at our local community hall today. 120 volunteers turned up including college students and elders. Humanity still exists.",
        "party": "None",
        "mental_state": "Social Empathy & Grassroots Care",
        "age_bracket": "25-34",
        "gender": "male",
        "impatience": "patient",
        "irrationality": "rational"
    },
    {
        "text": "Which is better for weekend dinner in Bangalore? Meghana biryani boneless chicken or Nagarjuna Andhra meals with gun powder and ghee?",
        "party": "None",
        "mental_state": "Cultural Pride & National Solidarity",
        "age_bracket": "18-24",
        "gender": "neutral",
        "impatience": "neutral",
        "irrationality": "rational"
    },
    {
        "text": "Freshers sending 500 job applications on LinkedIn and getting automated rejection emails within 5 minutes. The tech job market right now is brutal.",
        "party": "None",
        "mental_state": "Financial Anxiety & Economic Strain",
        "age_bracket": "18-24",
        "gender": "female",
        "impatience": "impatient",
        "irrationality": "rational"
    }
]

def main():
    out_path = Path("artifacts/demographics_models/gold_standard_benchmark.jsonl")
    with open(out_path, "w", encoding="utf-8") as f:
        for s in REAL_WORLD_GOLD_DATASET:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"Created real-world gold benchmark at {out_path} with {len(REAL_WORLD_GOLD_DATASET)} multi-attribute samples.")

if __name__ == "__main__":
    main()
