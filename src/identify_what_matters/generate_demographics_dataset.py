"""LLM-driven large-scale dataset generator for demographics and mental state classifiers.

Generates thousands of realistic Indian social media comments with labels:
  - age_bracket: 13-17, 18-24, 25-34, 35-54, 55+
  - gender: male, female, neutral
  - mental_state: 6 archetypes
  - impatience: impatient, neutral, patient
  - irrationality: irrational, neutral, rational
  - party_support: BJP, Congress, AAP, TMC, Other, None

Runs all LLM calls in parallel (up to MAX_WORKERS concurrent) for speed.
"""

from __future__ import annotations

import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from dotenv import load_dotenv
import requests

API_URL = "https://openrouter.ai/api/v1/chat/completions"
ENV_FILE = Path(__file__).resolve().parents[2] / ".env"

load_dotenv(ENV_FILE, override=False)
API_KEY = os.environ.get("OPENROUTER_API_KEY")
MODEL = "anthropic/claude-haiku-5.5"
MAX_WORKERS = 100

# ---------------------------------------------------------------------------
# LABEL NORMALIZERS — enforce canonical class names at ingestion time
# ---------------------------------------------------------------------------

def _norm_party(raw: str) -> str:
    s = str(raw or "").strip().lower()
    if not s or s in ("none","null","n/a","na","unknown","unspecified","neutral","apolitical",
                      "undecided","uncertain","independent","swing","swing voter","centrist",
                      "moderate","nota","nota supporter","mixed","transactional","disillusioned",
                      "skeptical","neutral/none","none / independent","none (critical of all)"):
        return "None"
    if any(x in s for x in ("bjp","pro-bjp","pro_bjp","pro-modi","ruling party (bjp)",
                              "nationalist","right-leaning","pro-incumbent","pro-establishment",
                              "pro-government","pro-govt","strongly pro-government","ruling party",
                              "ruling","loyal base","pro-modi","likely bjp")):
        return "BJP"
    if any(x in s for x in ("congress","inc","indian national congress","pro-congress","likely congress")):
        return "Congress"
    if any(x in s for x in ("aap","aam aadmi","pro-aap")):
        return "AAP"
    if any(x in s for x in ("tmc","trinamool","pro-tmc","mamata")):
        return "TMC"
    if any(x in s for x in ("dmk","dravida","dravida munnetra","pro-dmk","mk stalin","karunanidhi",
                              "tamil party","regional party (dmk)")):
        return "DMK"
    if re.fullmatch(r"sp", s) or any(x in s for x in ("samajwadi","pro-sp","akhilesh","sp supporter")):
        return "SP"
    if any(x in s for x in ("bsp","brs","shiv sena","shivsena","cpi","left-leaning","aimim",
                              "anti-bjp","anti-congress","opposition","anti-incumbent",
                              "anti-establishment","anti-government","anti-govt")):
        return "Other"
    return "None"

def _norm_age(raw: str) -> str:
    s = str(raw or "").strip().lower()
    if s in ("teen","teenager","teenage","school","13-17","14-17","15-17"):
        return "13-17"
    if s in ("gen z","genz","young adult"):
        return "18-24"
    if "30s" in s or "thirties" in s:
        return "25-34"
    if "40s" in s or "forties" in s:
        return "35-54"
    if any(x in s for x in ("50s","fifties","60s","senior","elderly")):
        return "55+"
    nums = re.findall(r"\d+", s)
    if not nums:
        return "25-34"
    vals = [int(n) for n in nums[:2]]
    mid = sum(vals) / len(vals)
    if mid < 18:
        return "13-17"
    elif mid < 25:
        return "18-24"
    elif mid < 35:
        return "25-34"
    elif mid < 55:
        return "35-54"
    return "55+"

_CANONICAL_MENTAL = {
    "Civic Anger & Institutional Skepticism",
    "Financial Anxiety & Economic Strain",
    "Ironic Detachment & Memetic Cynicism",
    "Cultural Pride & National Solidarity",
    "Social Empathy & Grassroots Care",
    "Objective Deliberation & Fact-Checking",
}
_MENTAL_KWS = [
    ("Civic Anger & Institutional Skepticism",
     ["frustrated","angry","anger","annoyed","exasperated","furious","outraged",
      "civic","corruption","betrayed","protest","enraged","disappointed","resentful"]),
    ("Financial Anxiety & Economic Strain",
     ["financial","economic","anxiety","anxious","worried","distressed","fearful",
      "inflation","unemployment","economic strain"]),
    ("Ironic Detachment & Memetic Cynicism",
     ["ironic","cynical","cynicism","detached","sarcastic","sarcasm","meme",
      "disillusioned","nihilist","apathetic","mockery","satirical"]),
    ("Cultural Pride & National Solidarity",
     ["proud","pride","patriotic","patriotism","confident","solidarity","cultural",
      "enthusiastic","nostalgic","celebratory","triumphant","national"]),
    ("Social Empathy & Grassroots Care",
     ["empathy","empathetic","compassion","compassionate","caring","community","grassroots"]),
    ("Objective Deliberation & Fact-Checking",
     ["analytical","fact","objective","deliberative","pragmatic","rational","reflective",
      "thoughtful","measured","balanced","logical","evidence","data"]),
]
def _norm_mental(raw: str) -> str:
    if raw in _CANONICAL_MENTAL:
        return raw
    s = raw.lower()
    for canonical, keywords in _MENTAL_KWS:
        if any(k in s for k in keywords):
            return canonical
    return "Civic Anger & Institutional Skepticism"

def _norm_impatience(raw: str) -> str:
    if raw in ("impatient","neutral","patient"):
        return raw
    s = str(raw or "").strip().lower()
    if s in ("high","very high","extreme","very impatient","extremely impatient","highly impatient","5","4"):
        return "impatient"
    if s in ("low","very low","patient","very patient","1","2"):
        return "patient"
    if s in ("medium","moderate","3"):
        return "neutral"
    if re.fullmatch(r"\d+", s):
        n = int(s)
        return "impatient" if n >= 4 else ("patient" if n <= 2 else "neutral")
    return "neutral"

def _norm_irrationality(raw: str) -> str:
    if raw in ("irrational","neutral","rational"):
        return raw
    s = str(raw or "").strip().lower()
    if s in ("high","very high","extreme","irrational","emotional","5","4"):
        return "irrational"
    if s in ("low","very low","none","rational","logical","1","2"):
        return "rational"
    if s in ("medium","moderate","mild","3"):
        return "neutral"
    if re.fullmatch(r"\d+", s):
        n = int(s)
        return "irrational" if n >= 4 else ("rational" if n <= 2 else "neutral")
    return "neutral"

def _norm_gender(raw: str) -> str:
    s = str(raw or "").strip().lower()
    if s in ("male","m","man","boy","he","him"):
        return "male"
    if s in ("female","f","woman","girl","she","her"):
        return "female"
    return "neutral"

# ---------------------------------------------------------------------------
# BATCH PROMPTS — 40 targeted slices, each requesting 30 samples
# Covers the weaknesses identified in the current training data:
#   1. Short 3-10 word real Instagram comments
#   2. Sarcasm and coded language (not just explicit party labels)
#   3. Regional parties (DMK, SP, BSP, BRS, Shiv Sena, NC, PDP)
#   4. Hinglish and Hindi-dominant comments
#   5. Devotional / tribal identity comments
#   6. Whataboutism patterns
#   7. Hard negatives (ambiguous, mixed-signal comments)
# ---------------------------------------------------------------------------

BATCH_PROMPTS = [
    # --- AGE: SHORT / REAL INSTAGRAM STYLE (3-15 words) ---
    """Generate 30 SHORT (3-15 words) Indian Instagram comments that feel completely natural and authentic.
Mix Gen Z (18-24) and teens (13-17). Use slang, emoji, lowercase, typos. Topics: cricket, movies, memes,
college, exams, crush, hostel, reels, gaming. Examples of tone: "lol same bro 💀", "yaar ye kya tha",
"ratio + L + no rizz", "dead at this 😭😭". Return JSON array:
[{"text":"...","age_bracket":"...","gender":"...","mental_state":"...","impatience":"...","irrationality":"...","party_support":"..."}]
NO duplicates. Each text must look like an actual Instagram comment, not a paragraph.""",

    # --- AGE: PROFESSIONAL SHORT REACTIONS ---
    """Generate 30 SHORT (5-20 words) Indian comments from working professionals (25-34) reacting to news
headlines. Natural, realistic, mix of English and Hinglish. Topics: layoffs, salary, EMI, startup, tax,
inflation, WFH. Examples: "bhai iska matlab mera pf fir duba", "honestly same, salary hike 5% but rent
went 30%", "gg layoffs again". Return JSON array:
[{"text":"...","age_bracket":"25-34","gender":"...","mental_state":"...","impatience":"...","irrationality":"...","party_support":"..."}]""",

    # --- POLITICAL: BJP INDIRECT / CODED (not explicit "I support BJP") ---
    """Generate 30 Indian social media comments showing BJP support WITHOUT saying "BJP" or "I support BJP".
Use coded language real supporters use: "modiji", "vikas", "strong leadership", "Hindu rashtra", "desh",
"anti-nationals", "urban naxals", "paid media", "Ram Mandir", "Article 370", "surgical strike pride",
"Pakistan ko jawaab", "har har Modi". Mix ages 22-65, both genders. Vary between confident and emotionally
charged. Return JSON array:
[{"text":"...","age_bracket":"...","gender":"...","mental_state":"...","impatience":"...","irrationality":"...","party_support":"BJP"}]""",

    # --- POLITICAL: CONGRESS INDIRECT / CODED ---
    """Generate 30 Indian social media comments showing Congress support WITHOUT saying "Congress" directly.
Use language real Congress supporters use: "nehru ke sapne", "secularism", "constitution ka hatyara",
"democracy under threat", "rahul ji", "nyay yojana", "bharat jodo", "adani ambani", "RSS propaganda",
"divide karo raj karo", "EVMs hacked", "PM ki guarantee jhooth". Ages 20-65, both genders. Return JSON:
[{"text":"...","age_bracket":"...","gender":"...","mental_state":"...","impatience":"...","irrationality":"...","party_support":"Congress"}]""",

    # --- POLITICAL: SARCASM (hard negatives — tone opposite to stated view) ---
    """Generate 30 Indian political comments that use SARCASM. The surface reading is positive about a party
but actually it's criticism, OR vice versa. Examples: "haan ji modiji ne bahut vikas kiya, 2 rupees ka
tamatar 200 ka ho gaya" (BJP criticism through sarcasm), "waah congress waah 70 saal mein sirf garibi
badhai" (BJP lean through sarcasm). Clearly label mental_state as "Ironic Detachment & Memetic Cynicism".
Return JSON:
[{"text":"...","age_bracket":"...","gender":"...","mental_state":"Ironic Detachment & Memetic Cynicism","impatience":"...","irrationality":"...","party_support":"..."}]
IMPORTANT: party_support should reflect ACTUAL lean, not surface text.""",

    # --- POLITICAL: AAP SUPPORTERS ---
    """Generate 30 Indian social media comments from AAP (Aam Aadmi Party) supporters. Cover: Delhi model,
free electricity, Mohalla clinics, Kejriwal, Raghav Chadha, school transformation, anti-corruption,
Punjab and Delhi governance. Mix coded and explicit. Ages 20-50. Return JSON:
[{"text":"...","age_bracket":"...","gender":"...","mental_state":"...","impatience":"...","irrationality":"...","party_support":"AAP"}]""",

    # --- POLITICAL: TMC / REGIONAL BENGAL ---
    """Generate 30 Indian social media comments from TMC (Trinamool Congress) supporters and West Bengal
context. Themes: Didi, Bengali pride, federalism, "Khela Hobe", anti-BJP narrative in Bengal, Durga Puja
disputes, post-poll violence controversy, Sandeshkhali, Mamata Banerjee defending. Mix Bengali pride
with political commentary. Ages 22-65. Return JSON:
[{"text":"...","age_bracket":"...","gender":"...","mental_state":"...","impatience":"...","irrationality":"...","party_support":"TMC"}]""",

    # --- POLITICAL: DMK (Tamil Nadu regional party) ---
    """Generate 30 Indian social media comments from DMK supporters or Tamil Nadu voters aligned with Dravidian politics.
Include: MK Stalin, Tamil pride, anti-Hindi imposition, NEP opposition, Periyar legacy, caste equity, Tamil identity,
Dravidian model of governance, state autonomy. Label party_support as "DMK". Vary ages/genders. Return JSON:
[{"text":"...","age_bracket":"...","gender":"...","mental_state":"...","impatience":"...","irrationality":"...","party_support":"DMK"}]""",

    # --- POLITICAL: SP (Samajwadi Party, UP) ---
    """Generate 30 Indian social media comments from SP (Samajwadi Party) supporters or UP voters aligned with Akhilesh Yadav.
Include: Akhilesh Yadav, OBC/Muslim identity politics, Lohiaite socialism, UP development, Expressway projects,
caste-based mobilisation, PDA alliance, Mulayam Singh legacy, anti-BJP in UP. Label party_support as "SP". Vary ages/genders. Return JSON:
[{"text":"...","age_bracket":"...","gender":"...","mental_state":"...","impatience":"...","irrationality":"...","party_support":"SP"}]""",

    # --- POLITICAL: OTHER REGIONAL (BSP/BRS/ShivSena) ---
    """Generate 30 Indian social media comments from supporters of smaller regional parties: BSP (Ambedkar, Dalit pride,
Mayawati, SC/ST rights), BRS/TRS (Telangana pride, KCR, irrigation, Hyderabad IT), Shiv Sena (Maharashtra identity,
Marathi manoos). Label party_support as "Other". Return JSON:
[{"text":"...","age_bracket":"...","gender":"...","mental_state":"...","impatience":"...","irrationality":"...","party_support":"Other"}]""",

    # --- MENTAL STATE: DEVOTIONAL / TRIBAL IDENTITY ---
    """Generate 30 Indian social media comments expressing DEVOTIONAL FERVOR or TRIBAL IDENTITY around
religion, caste, or regional identity. Not necessarily political party support. Examples: "Jai Shree Ram 🙏",
"Ambedkar amar rahe", "Jai Bhim", "Dravidian pride never dies", "Sikhi meri jaan hai", "Islam zindabad meri
pehchaan", "Marathi manoos utha". Label mental_state as "Cultural Pride & National Solidarity". Vary ages
and genders. Return JSON:
[{"text":"...","age_bracket":"...","gender":"...","mental_state":"Cultural Pride & National Solidarity","impatience":"...","irrationality":"...","party_support":"..."}]""",

    # --- MENTAL STATE: WHATABOUTISM ---
    """Generate 30 Indian political comments that use WHATABOUTISM ("but what about X?"). Examples:
"bhai BJP ki baat karte ho, Congress ne 70 saal mein kya kiya?", "TMC ke gunde marte hain tab media
kahan tha?", "Delhi mein crime badhta hai to Kejriwal pe blame, UP mein hota hai to Modi pe blame kyun
nahi?". Label mental_state as "Civic Anger & Institutional Skepticism" and irrationality as "irrational".
Return JSON:
[{"text":"...","age_bracket":"...","gender":"...","mental_state":"Civic Anger & Institutional Skepticism","impatience":"impatient","irrationality":"irrational","party_support":"..."}]""",

    # --- HINGLISH HEAVY (UP / DELHI / BIHAR) ---
    """Generate 30 comments in HINGLISH (Hindi + English mixed) from commenters in UP, Bihar, Uttarakhand,
Delhi NCR. Authentic code-switching, Roman script Hindi. Topics: local elections, yogi adityanath,
UP development, roads, power, crime, UPSC aspirants, farmer issues. Examples: "bhai sarkaar ka system hi
kharab hai yaar", "modiji ne jo bola woh karke dikhaya na", "UP mein itna crime toh hua hai". Return JSON:
[{"text":"...","age_bracket":"...","gender":"...","mental_state":"...","impatience":"...","irrationality":"...","party_support":"..."}]""",

    # --- HINGLISH: PUNJAB / HARYANA ---
    """Generate 30 comments in Punjabi-inflected Hinglish from Punjab, Haryana commenters. Topics:
farmer protests, AAP Punjab, Bhagwant Mann, drug menace, NRI connections, agricultural policy, Captain
Amarinder, Sukhbir Badal, SAD. Real Punjabi expressions in Roman: "oye yaar", "balle balle", "ki haal
hai", "sat sri akal ji". Return JSON:
[{"text":"...","age_bracket":"...","gender":"...","mental_state":"...","impatience":"...","irrationality":"...","party_support":"..."}]""",

    # --- REGIONAL: SOUTH INDIA IN ROMAN SCRIPT ---
    """Generate 30 comments from South Indian commenters writing in ROMAN SCRIPT English or Roman-transliterated
Tamil/Telugu/Kannada/Malayalam. Topics: anti-Hindi imposition, NEP concerns, DMK vs BJP in Tamil Nadu,
Kerala communism vs BJP, Telangana statehood pride, Kannada-Tamil border disputes. Include phrases like
"Tamil pride endrum vazhga", "namma Karnataka", "orey raa", "ayyo", "machcha". Return JSON:
[{"text":"...","age_bracket":"...","gender":"...","mental_state":"...","impatience":"...","irrationality":"...","party_support":"..."}]""",

    # --- AGE: TEENS 13-17 ---
    """Generate 30 comments from Indian teenagers (13-17) who are still in school. Topics: board exams,
CBSE stress, YouTube, Instagram reels, gaming (BGMI/Free Fire/Valorant), crush drama, school fights,
coaching classes, pocket money. Writing style: very short, emoji-heavy, lowercase, typos, slang.
Examples: "bro ye teacher ka dimaag hai hi nahi 💀", "aaj paper itna tough tha yaar 😭". Return JSON:
[{"text":"...","age_bracket":"13-17","gender":"...","mental_state":"...","impatience":"...","irrationality":"...","party_support":"None"}]""",

    # --- AGE: SENIORS 55+ ---
    """Generate 30 comments from Indian senior citizens (55+, retired, grandparents). Writing style:
formal, respectful tone, sometimes using honorifics. Topics: pension, health, grandchildren's education,
neighborhood safety, religious practice, nostalgia ("in our time..."), cricket, national pride.
Examples: "I served this nation for 30 years as a teacher, today the values have degraded completely",
"pranam to all mothers on this sacred occasion". Return JSON:
[{"text":"...","age_bracket":"55+","gender":"...","mental_state":"...","impatience":"...","irrationality":"...","party_support":"..."}]""",

    # --- MENTAL STATE: FINANCIAL ANXIETY (VERY SPECIFIC) ---
    """Generate 30 Indian comments showing ACUTE financial anxiety about very specific real expenses:
school fees, medical bills, home loan EMI, car loan, rent hikes, vegetable prices, cooking gas cylinder
cost, electricity bills, tuition fees. Real amounts, real product names, real banks mentioned. Ages 25-54.
Return JSON:
[{"text":"...","age_bracket":"...","gender":"...","mental_state":"Financial Anxiety & Economic Strain","impatience":"...","irrationality":"...","party_support":"..."}]""",

    # --- MENTAL STATE: OBJECTIVE / FACT-CHECKER ---
    """Generate 30 Indian comments from people who cite DATA, SOURCES, LAWS, or STATISTICS. These people
share RBI data, CAG reports, election commission data, supreme court orders, RTI findings, news links,
fact-checks. Calm, measured tone. Ages 25-45. Examples: "According to NCRB 2023 data, crime rate has
actually declined by 3.2% in metro cities", "The farm loan waiver that Congress claims was implemented
only for 60% of eligible farmers per CAG audit". Return JSON:
[{"text":"...","age_bracket":"...","gender":"...","mental_state":"Objective Deliberation & Fact-Checking","impatience":"patient","irrationality":"rational","party_support":"..."}]""",

    # --- GENDER: FEMALE VOICE (distinct issues) ---
    """Generate 30 comments specifically from WOMEN commenters on Indian social media, writing from female
perspective. Topics: safety (rape cases, eve teasing, workplace harassment), women reservation bill, Beti
Bachao scheme performance, self-help groups, maternal health, women entrepreneurs, domestic violence, hijab
controversy, women in sports. Mix of ages 18-55. Identify gender from "as a woman/girl/mother/daughter"
statements. Return JSON:
[{"text":"...","age_bracket":"...","gender":"female","mental_state":"...","impatience":"...","irrationality":"...","party_support":"..."}]""",

    # --- IRRATIONALITY: CONSPIRACY THEORIES ---
    """Generate 30 Indian social media comments expressing CONSPIRACY THEORIES or clearly irrational beliefs.
Examples: EVM hacking, chemtrails, COVID vaccine microchips, "George Soros funding Congress", "RSS behind
every riot", "Tukde tukde gang in universities", "CIA running opposition parties", "Hindus are being
systematically replaced". Mix parties. These should be labeled irrationality: "irrational". Return JSON:
[{"text":"...","age_bracket":"...","gender":"...","mental_state":"Civic Anger & Institutional Skepticism","impatience":"impatient","irrationality":"irrational","party_support":"..."}]""",

    # --- APOLITICAL / NONE (entertainment, sports, lifestyle) ---
    """Generate 30 Indian social media comments that are COMPLETELY APOLITICAL — about entertainment,
sports, food, travel, relationships, fitness, pets, fashion, daily life. Zero political content. These
are the "None" party_support ground truth. Ages 15-50, all genders. Topics: IPL, Virat Kohli, new
Bollywood movie, restaurant review, chai vs coffee debate, monsoon mood. Return JSON:
[{"text":"...","age_bracket":"...","gender":"...","mental_state":"...","impatience":"...","irrationality":"...","party_support":"None"}]""",

    # --- CIVIC ANGER: INFRASTRUCTURE ---
    """Generate 30 Indian comments expressing anger at SPECIFIC infrastructure failures: potholes causing
deaths, waterlogging in Mumbai/Chennai, power cuts in summer, broken streetlights, garbage disposal
failures, broken metro escalators, delayed train cancellations, NHAI highway deaths. Real city names,
real problems. Ages 25-55. Return JSON:
[{"text":"...","age_bracket":"...","gender":"...","mental_state":"Civic Anger & Institutional Skepticism","impatience":"impatient","irrationality":"...","party_support":"..."}]""",

    # --- CIVIC ANGER: POLICE / JUDICIARY ---
    """Generate 30 Indian comments expressing frustration with police or judiciary: FIR refused, bail
granted to criminals, delayed justice (12-15 year trial period), custodial deaths, encounter killings,
judge shortage, undertrial prisoners rotting in jail. Real incidents where possible. Ages 28-60. Return JSON:
[{"text":"...","age_bracket":"...","gender":"...","mental_state":"Civic Anger & Institutional Skepticism","impatience":"impatient","irrationality":"...","party_support":"..."}]""",

    # --- SOCIAL EMPATHY: GRASSROOTS CARE ---
    """Generate 30 Indian comments showing GENUINE EMPATHY for others — migrant workers, street vendors,
tribal communities, flood victims, farmers in distress, disabled people, cancer patients seeking help.
People sharing GoFundMe links, calling for donations, sharing grief. Ages 22-55. Return JSON:
[{"text":"...","age_bracket":"...","gender":"...","mental_state":"Social Empathy & Grassroots Care","impatience":"patient","irrationality":"rational","party_support":"..."}]""",

    # --- ELECTION TIME: VOTER BEHAVIOR ---
    """Generate 30 Indian comments made during election season about VOTING behavior. Mix: first-time
voters excited, swing voters uncertain, loyal base voters, EVM doubters, NOTA supporters, booth capture
allegations, candidates door-to-door visits, biryani votes, cash for votes. Ages 18-65. Return JSON:
[{"text":"...","age_bracket":"...","gender":"...","mental_state":"...","impatience":"...","irrationality":"...","party_support":"..."}]""",

    # --- CASTE / RESERVATION DISCOURSE ---
    """Generate 30 Indian comments about caste and reservation — from multiple perspectives: those who
support reservation (Dalit, OBC empowerment), those who oppose it (merit argument, general category
frustration), those who want it extended (EWS), academic debates. Sensitive but realistic. Ages 20-50.
Return JSON:
[{"text":"...","age_bracket":"...","gender":"...","mental_state":"...","impatience":"...","irrationality":"...","party_support":"..."}]""",

    # --- RELIGIOUS POLARIZATION ---
    """Generate 30 Indian comments touching religious identity in political context. Cover: Ram Mandir
reactions (supportive and critical), Waqf Board controversy, Minority appeasement criticism, secularism
defense, Love Jihad conspiracy claims, interfaith marriage, mosque vs temple disputes. Spread across all
political leanings. Ages 20-60. Return JSON:
[{"text":"...","age_bracket":"...","gender":"...","mental_state":"...","impatience":"...","irrationality":"...","party_support":"..."}]""",

    # --- MEDIA / PROPAGANDA DISCOURSE ---
    """Generate 30 Indian comments about media bias and propaganda. Both sides: "godi media working for
BJP", "presstitutes", "NDTV anti-national", "Republic TV TRP manipulation", "paid news", "Ravish Kumar
is the last honest journalist", "Arnab Goswami is a fraud". Ages 22-50. Return JSON:
[{"text":"...","age_bracket":"...","gender":"...","mental_state":"Civic Anger & Institutional Skepticism","impatience":"...","irrationality":"...","party_support":"..."}]""",

    # --- STUDENT / EXAM STRESS (NEET/JEE/UPSC) ---
    """Generate 30 comments from students dealing with NEET, JEE, UPSC, SSC, Banking exams. Paper leaks,
coaching mafia fees (2-5 lakh for Kota coaching), reservation cutoffs, age limit extensions demanded,
mental health of students, Kota suicides, exam center malpractice. Ages 18-28. Return JSON:
[{"text":"...","age_bracket":"...","gender":"...","mental_state":"Financial Anxiety & Economic Strain","impatience":"...","irrationality":"...","party_support":"..."}]""",

    # --- FARMER / RURAL INDIA ---
    """Generate 30 comments from farmers or rural India perspective: MSP demands, crop insurance fraud,
moneylender debt, irrigation failures, middlemen exploitation, tractor march to Delhi, agri-loan waivers,
Lakhimpur Kheri incident, PM-KISAN scheme satisfaction or dissatisfaction. Ages 30-60. Mix Hindi/Hinglish.
Return JSON:
[{"text":"...","age_bracket":"...","gender":"...","mental_state":"...","impatience":"...","irrationality":"...","party_support":"..."}]""",

    # --- MINORITY PERSPECTIVE ---
    """Generate 30 comments from Indian MINORITY community perspectives (Muslim, Christian, Sikh, Buddhist)
on political events. Citizenship Amendment Act fears, minority institutions under pressure, communal
violence, love jihad laws, church vandalism, Sikh farmer protests. Realistic voices, not stereotypes.
Return JSON:
[{"text":"...","age_bracket":"...","gender":"...","mental_state":"...","impatience":"...","irrationality":"...","party_support":"..."}]""",

    # --- HEALTH / COVID AFTERMATH ---
    """Generate 30 Indian comments about healthcare failures and COVID aftermath: oxygen shortage deaths,
private hospital overcharging during COVID, government hospital conditions, AIIMS wait times, generic
medicine vs branded, Ayushman Bharat rejections, mental health taboo, post-COVID long illness. Ages 25-65.
Return JSON:
[{"text":"...","age_bracket":"...","gender":"...","mental_state":"...","impatience":"...","irrationality":"...","party_support":"..."}]""",

    # --- ECONOMY: STARTUP / FINTECH ---
    """Generate 30 comments from India's startup and tech community about economic policy: RBI's fintech
crackdown, Paytm/PhonePe regulation, startup layoffs, unicorn valuations crashing, SEBI F&O regulations,
UPI dominance, DPDP bill concerns, AI job disruption fears, crypto ban discussion. Ages 22-38. Return JSON:
[{"text":"...","age_bracket":"...","gender":"...","mental_state":"...","impatience":"...","irrationality":"...","party_support":"..."}]""",

    # --- ENVIRONMENT / CLIMATE ---
    """Generate 30 Indian comments on environment and climate: Delhi air quality AQI 400+, Yamuna
pollution, plastic ban enforcement, deforestation for highways, coal plant expansion vs renewables,
Uttarakhand floods blaming infrastructure greed, stubble burning, EV subsidy policy. Ages 18-50. Return JSON:
[{"text":"...","age_bracket":"...","gender":"...","mental_state":"Civic Anger & Institutional Skepticism","impatience":"...","irrationality":"...","party_support":"..."}]""",

    # --- CULTURE / SPORTS / FILM (apolitical + soft political) ---
    """Generate 30 Indian comments mixing CRICKET and soft political pride: Team India wins, Rohit-Kohli
debate, MS Dhoni legacy, IPL team loyalties, Neeraj Chopra Olympic gold, Indian women's cricket rise,
Batter economy IPL vs Tests debate. Some comments bring in nationalism/pride that bleeds political.
Ages 15-55. Return JSON:
[{"text":"...","age_bracket":"...","gender":"...","mental_state":"...","impatience":"...","irrationality":"...","party_support":"..."}]""",

    # --- AMBIGUOUS / HARD NEGATIVES (intentionally unclear) ---
    """Generate 30 AMBIGUOUS Indian social media comments where political lean is genuinely UNCLEAR.
These are the hardest cases: someone frustrated with both parties, centrist analysis, a comment that
could be read multiple ways depending on context. These should be labeled party_support: "None".
Ages 22-50. The comments should NOT be explicitly political but might reference economics, corruption,
governance in a non-partisan way. Return JSON:
[{"text":"...","age_bracket":"...","gender":"...","mental_state":"...","impatience":"...","irrationality":"...","party_support":"None"}]""",

    # --- IMPATIENCE: HIGH IMPATIENCE BURST ---
    """Generate 30 Indian comments that are SHORT EMOTIONAL BURSTS expressing extreme impatience —
people who have lost patience completely. Very short, angry, frustrated. Examples: "ab ye nahi sehna!",
"ENOUGH. FIX THE ROADS.", "kitni baar bolunga bhai???", "3 hours line mein khada tha, counter band
ho gaya. koi sun raha hai??". Ages 18-55. Return JSON:
[{"text":"...","age_bracket":"...","gender":"...","mental_state":"Civic Anger & Institutional Skepticism","impatience":"impatient","irrationality":"...","party_support":"..."}]""",

    # --- PATIENCE / RATIONAL (contrasting) ---
    """Generate 30 Indian comments showing unusual PATIENCE and RATIONALITY in political/civic discourse.
People who say "let's wait for data", "both sides have valid points", "change takes time", "I disagree but
I understand why they did this". Rare but important class. Ages 28-55. Return JSON:
[{"text":"...","age_bracket":"...","gender":"...","mental_state":"Objective Deliberation & Fact-Checking","impatience":"patient","irrationality":"rational","party_support":"..."}]""",

    # --- DIASPORA / NRI PERSPECTIVE ---
    """Generate 30 comments from Indian DIASPORA (NRI in US/UK/Canada/Gulf) commenting on Indian politics.
Mix of nostalgia, external perspective, "back home we had better", "India has improved so much since I left",
sending remittances, OCI card issues, dual citizenship debate, Indians abroad proud of ISRO/Modi at G20.
Ages 28-55. Return JSON:
[{"text":"...","age_bracket":"...","gender":"...","mental_state":"...","impatience":"...","irrationality":"...","party_support":"..."}]""",

    # --- ANTI-ESTABLISHMENT (no party preference) ---
    """Generate 30 Indian comments from people who are ANTI-ALL-PARTIES — disgusted with the entire
political class. NOTA voters, anarchists, nihilists, "sab chor hain" camp, people who say "I won't
vote", complete disillusionment. Very important None class. Ages 22-45. Return JSON:
[{"text":"...","age_bracket":"...","gender":"...","mental_state":"Civic Anger & Institutional Skepticism","impatience":"impatient","irrationality":"...","party_support":"None"}]""",

    # --- BJP: MUSCULAR NATIONALISM ---
    """Generate 30 comments from BJP-leaning commenters expressing MUSCULAR NATIONALISM: surgical strikes
pride, Balakot air strike, China border confrontation, Pakistan as enemy, anti-terrorism stance, AFSPA
defense, Article 370 revocation celebration, 26/11 memory, soldier martyrdom tributes. Ages 20-60.
Return JSON:
[{"text":"...","age_bracket":"...","gender":"...","mental_state":"Cultural Pride & National Solidarity","impatience":"...","irrationality":"...","party_support":"BJP"}]""",

    # --- CONGRESS: SECULARISM DEFENSE ---
    """Generate 30 comments from Congress-leaning commenters defending secularism, minority rights,
pluralism, Nehruvian institutions: IITs, IIMs, ISRO, Election Commission under threat narrative,
"idea of India", Ambedkar's constitution being dismantled, "silence is not an option", freedom of press
concern. Ages 22-65. Return JSON:
[{"text":"...","age_bracket":"...","gender":"...","mental_state":"...","impatience":"...","irrationality":"...","party_support":"Congress"}]""",

    # --- URBAN VS RURAL DIVIDE ---
    """Generate 30 comments that show the URBAN-RURAL divide in India: urban commenters dismissing rural
concerns, rural voices about being ignored, tier-2 and tier-3 city residents feeling forgotten, reverse
migration stories, smart city scheme vs reality, MGNREGA vs factory jobs debate. Ages 22-55. Return JSON:
[{"text":"...","age_bracket":"...","gender":"...","mental_state":"...","impatience":"...","irrationality":"...","party_support":"..."}]""",

    # --- YOUTH: UNEMPLOYMENT / MIGRATION ---
    """Generate 30 comments from young Indians (18-30) dealing with UNEMPLOYMENT and migration.
Those who left India for opportunities abroad ("brain drain"), those who stayed and are struggling,
those criticizing "job data" from government, aspirants preparing for 3-5 years for govt jobs, gig
economy workers (Zomato, Swiggy, Uber drivers commenting). Ages 18-32. Return JSON:
[{"text":"...","age_bracket":"...","gender":"...","mental_state":"Financial Anxiety & Economic Strain","impatience":"...","irrationality":"...","party_support":"..."}]""",
]


def _call_openrouter(prompt: str, timeout: int = 90) -> str:
    if not API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY not set")
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "reasoning": {"enabled": True},
        "temperature": 0.90,
        "max_tokens": 4096,
    }
    for attempt in range(3):
        try:
            r = requests.post(API_URL, headers=headers, json=payload, timeout=timeout)
            r.raise_for_status()
            content = r.json()["choices"][0]["message"].get("content", "")
            return content
        except Exception as e:
            if attempt == 2:
                raise
            time.sleep(2 ** attempt)
    return ""


def _parse_batch(raw: str) -> list[dict]:
    if not raw:
        return []
    for pattern in (
        r"```json\s*(\[.*?\])\s*```",
        r"```\s*(\[.*?\])\s*```",
        r"(\[\s*\{.*?\}\s*\])",
    ):
        m = re.search(pattern, raw, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(1))
            except Exception:
                pass
    return []


def _run_prompt(idx_prompt: tuple[int, str]) -> tuple[int, list[dict]]:
    idx, prompt = idx_prompt
    try:
        raw = _call_openrouter(prompt)
        return idx, _parse_batch(raw)
    except Exception as e:
        print(f"  [prompt {idx}] error: {e}")
        return idx, []


def generate_full_dataset(output_path: Path, target_samples: int = 10_000) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    samples: list[dict] = []

    # Load existing samples to avoid re-generating
    if output_path.exists():
        with open(output_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                    key = str(item.get("text", ""))[:80]
                    if key and key not in seen:
                        seen.add(key)
                        samples.append(item)
                except Exception:
                    pass
        print(f"Loaded {len(samples)} existing samples from {output_path}")

    print(f"Generating up to {target_samples} total samples using {MODEL} with {MAX_WORKERS} parallel workers...")

    passes = 0
    while len(samples) < target_samples:
        passes += 1
        # Cycle through prompts with an offset each pass for variety
        offset = (passes - 1) * 7
        prompts_this_pass = [(i, BATCH_PROMPTS[(i + offset) % len(BATCH_PROMPTS)])
                             for i in range(min(len(BATCH_PROMPTS), target_samples - len(samples)))]

        print(f"\nPass {passes}: firing {len(prompts_this_pass)} parallel LLM calls...")

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
            futures = {pool.submit(_run_prompt, ip): ip[0] for ip in prompts_this_pass}
            for fut in as_completed(futures):
                idx, batch = fut.result()
                new_count = 0
                for item in batch:
                    if not isinstance(item, dict):
                        continue
                    text = str(item.get("text", "")).strip()
                    if not text or len(text) < 8:
                        continue
                    key = text[:80]
                    if key in seen:
                        continue
                    seen.add(key)
                    samples.append({
                        "text": text,
                        "age_bracket": _norm_age(item.get("age_bracket", "")),
                        "gender": _norm_gender(item.get("gender", "")),
                        "mental_state": _norm_mental(item.get("mental_state", "")),
                        "impatience": _norm_impatience(item.get("impatience", "")),
                        "irrationality": _norm_irrationality(item.get("irrationality", "")),
                        "party_support": _norm_party(item.get("party_support", "")),
                    })
                    new_count += 1
                print(f"  [prompt {idx}] +{new_count} samples → total {len(samples)}")

        # Checkpoint write after each pass
        with open(output_path, "w", encoding="utf-8") as f:
            for s in samples:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")
        print(f"Checkpoint: {len(samples)} samples saved to {output_path}")

        if len(samples) >= target_samples:
            break

    print(f"\nDone. {len(samples)} samples saved to {output_path}")


if __name__ == "__main__":
    out = Path(__file__).resolve().parents[2] / "artifacts" / "demographics_models" / "synthetic_dataset.jsonl"
    generate_full_dataset(out, target_samples=20_000)
