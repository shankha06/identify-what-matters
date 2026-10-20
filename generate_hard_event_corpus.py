import json, re, random
from pathlib import Path
from collections import defaultdict
import itertools

# 1. High-frequency political / civic entities in Indian discourse
# For each entity/theme, define multiple DISTINCT events (True Negatives between each other)
# and multiple paraphrased descriptions/headlines/captions/comment-reactions (True Positives within each event).

DISTINCT_EVENTS = [
    # --- RAHUL GANDHI & CONGRESS THEME ---
    {
        "event_id": "cong_rahul.detained_barakhamba",
        "title": "Rahul Gandhi and Priyanka Gandhi Detained at Barakhamba Road During Delhi Protest",
        "texts": [
            "Rahul Gandhi and Priyanka Gandhi Vadra detained by Delhi Police at Barakhamba Road during march to Election Commission",
            "Congress leaders Rahul and Priyanka Gandhi taken into police custody during protest near Mandi House",
            "Delhi Police detains Rahul Gandhi and Congress MPs marching against alleged voter list irregularities",
            "Rahul Gandhi detained again at Barakhamba Road police station along with Priyanka Gandhi",
            "Breaking: Rahul Gandhi, Priyanka Gandhi bundled into police bus at Barakhamba Road Delhi",
            "Mandi house se Barakhamba road tak Congress march par police lathicharge aur detention, Rahul Gandhi arrested",
            "Rahul Gandhi detained during opposition march over voter list fraud allegations"
        ]
    },
    {
        "event_id": "cong_rahul.police_clash_allegations",
        "title": "Congress Alleges Physical Assault and Police Misconduct Against Rahul Gandhi During Scuffle",
        "texts": [
            "Congress alleges Delhi Police physically assaulted and pushed Rahul Gandhi during protest scuffle",
            "Rahul and Priyanka Gandhi police clash allegations: Congress files complaint over manhandling",
            "Congress MPs allege physical assault by Delhi Police during confrontation outside Parliament",
            "Video emerges showing scuffle between Delhi Police personnel and Congress security detail protecting Rahul Gandhi",
            "Delhi Police denies assaulting Rahul Gandhi, calls scuffle unfortunate outcome of barricade breach"
        ]
    },
    {
        "event_id": "cong_rahul.rajghat_satyagraha",
        "title": "Opposition MPs Hold Silent Satyagraha at Rajghat Against Election Commission",
        "texts": [
            "Opposition MPs stage silent protest and Satyagraha at Mahatma Gandhi memorial Rajghat",
            "Opposition MPs Protest at Rajghat Against Chief Election Commissioner Gyanesh Kumar",
            "INDIA bloc leaders gather at Rajghat for day-long hunger strike against electoral rolls",
            "Silent demonstration by Congress and Samajwadi Party MPs at Gandhi samadhi Rajghat",
            "Rajghat par opposition parties ka dharna, CEC ke khilaf pradarshan"
        ]
    },
    {
        "event_id": "cong_rahul.kolhapur_statue_unveil",
        "title": "Rahul Gandhi Unveils Chhatrapati Shivaji Maharaj Statue in Kolhapur Maharashtra",
        "texts": [
            "Rahul Gandhi unveils grand bronze statue of Chhatrapati Shivaji Maharaj in Kolhapur",
            "Congress MP Rahul Gandhi addresses massive rally in Kolhapur after unveiling Shivaji Maharaj statue",
            "Kolhapur visit: Rahul Gandhi bows before Shivaji Maharaj statue, criticizes Sindhudurg statue collapse",
            "Maharashtra: Rahul Gandhi felicitates descendants of Shahu Maharaj in Kolhapur rally"
        ]
    },
    {
        "event_id": "cong_rahul.vote_theft_allegations",
        "title": "Opposition Formally Submits Memorandum to ECI Alleging 65 Lakh Voter Name Deletions",
        "texts": [
            "Opposition Alleges Electoral Roll Irregularities and Vote Theft across Maharashtra and Haryana",
            "INDIA bloc delegation submits 200-page dossier on deleted voter names to Election Commission",
            "Congress claims 65 lakh legitimate voter names purged from voter lists ahead of assembly polls",
            "Opposition parties submit memorandum to CEC alleging targeted deletion of minority and poor voters"
        ]
    },

    # --- BENGALURU PROTESTS & CEC / SIR THEME ---
    {
        "event_id": "bengaluru.freedom_park_sir_erupt",
        "title": "Violent Clashes Erupt at Bengaluru Freedom Park During Demonstration Against SIR",
        "texts": [
            "Protests Against SIR Erupt at Bengaluru Freedom Park and Delhi",
            "Massive crowd clashes with Bengaluru police at Freedom Park during demonstration against SIR",
            "Bengaluru Freedom Park Protest Called Against CEC and SIR turns tense after barricades toppled",
            "Freedom park Bengaluru me SIR aur CEC ke khilaf bawal, police ne kiya lathicharge",
            "Police use water cannons on student protestors shouting slogans against SIR at Freedom Park Bengaluru"
        ]
    },
    {
        "event_id": "bengaluru.cjp_protest_oct10_plan",
        "title": "Cockroach Janta Party Plans State-Wide Protest Against SIR on October 10",
        "texts": [
            "CJP protest against SIR planned for October 10 in Bengaluru and district headquarters",
            "Cockroach Janta Party calls for statewide bandh and sit-in on October 10 over SIR rollout",
            "CJP announces upcoming mega rally in Karnataka scheduled for October 10 against SIR policies",
            "October 10 protest call by CJP against SIR receives support from farmer unions in Karnataka",
            "CJP party workers plan peaceful picketing of district collector offices on October 10"
        ]
    },
    {
        "event_id": "bengaluru.dhruv_rathee_freedom_park",
        "title": "YouTuber Dhruv Rathee Joins Demonstration at Bengaluru Freedom Park",
        "texts": [
            "Dhruv Rathee joins Bengaluru Freedom Park protest against election machinery",
            "Popular YouTuber Dhruv Rathee arrives in Bengaluru to attend citizen agitation at Freedom Park",
            "Crowds gather around Dhruv Rathee as he speaks on stage at Freedom Park rally in Bengaluru",
            "Dhruv Rathee addresses protesters in Bengaluru Freedom Park, urges youth to check electoral rolls",
            "Dhruv Rathee ka Bengaluru Freedom Park me bhashan, hazaron log jama hue"
        ]
    },
    {
        "event_id": "bengaluru.women_safety_chappal_protest",
        "title": "Women Rights Activists Stage Chappal Protest at Freedom Park Demanding Safety Measures",
        "texts": [
            "Women Safety Chappal Protest at Freedom Park Bengaluru over delayed fast-track courts",
            "Women activists wave chappals and footwear at Freedom Park Bengaluru demanding immediate street lighting and safety",
            "Bengaluru: Women rights groups organize unique footwear protest at Freedom Park over rising harassment cases",
            "Mahila suraksha ko lekar Freedom Park me chappal morcha, activists ne government par lagaye arop"
        ]
    },

    # --- ELECTION COMMISSION & CEC GYANESH KUMAR THEME ---
    {
        "event_id": "cec.security_beefed_up",
        "title": "Delhi Police Upgrades Security Cover for Chief Election Commissioner Gyanesh Kumar",
        "texts": [
            "Security Beefed Up at Jantar Mantar and Nirvachan Sadan Ahead of Protests",
            "Delhi Police provides Z-category security detail to Chief Election Commissioner Gyanesh Kumar",
            "Opposition Mobilisation and Security Upgrade for CEC Gyanesh Kumar following death threat whispers",
            "Paramilitary deployed outside residence of CEC Gyanesh Kumar amid escalating political tension",
            "Nirvachan Sadan turned into fortress with multi-layer barricades and CRPF deployment"
        ]
    },
    {
        "event_id": "cec.cjp_mumbai_shivaji_park",
        "title": "CJP Stages Demonstration Demanding Resignation of CEC at Mumbai Shivaji Park",
        "texts": [
            "CJP protest at Mumbai Shivaji Park over CEC Gyanesh Kumar resignation demand",
            "Shabana Azmi, Naseeruddin Shah join CJP protest at Mumbai's Shivaji Park against CEC",
            "Mumbai police denies permission for CJP rally at Shivaji Park; thousands gather peacefully",
            "Shivaji Park Mumbai protest: Artists and activists seek Chief Election Commissioner ouster"
        ]
    },

    # --- JUDICIARY & LEGAL THEMES ---
    {
        "event_id": "judiciary.punjab_civic_polls_quashed",
        "title": "Supreme Court Quashes Elections in 50 Punjab Municipalities Over Delimitation Flaws",
        "texts": [
            "Supreme Court quashes Punjab civic poll notification citing arbitrary ward reservation",
            "Supreme Court Quashes Elections in 50 Punjab Municipalities and directs fresh delimitation",
            "Apex court strikes down Punjab municipal election process, slams state government for procedural violations",
            "Punjab civic polls cancelled by Supreme Court over irregular voter ward boundaries",
            "SC cancels Punjab local body elections, mandates 4-week timeline for rectifying ward rosters"
        ]
    },
    {
        "event_id": "judiciary.gst_council_arrest_powers",
        "title": "GST Council Curbs Arrest Powers of Tax Officials and Hikes Prosecution Limit",
        "texts": [
            "GST Council approves higher prosecution threshold to ease compliance burden on traders",
            "GST Council Curbs Tax Officials' Arrest Powers and Sets Monetary Limit to 5 Crores",
            "Finance Minister announces major relief: GST officers cannot arrest without commissioner approval",
            "GST Council meeting: Monetary threshold for criminal prosecution raised from 2 cr to 5 cr",
            "Taxmen arrest powers severely restricted under new GST Council directive approved today"
        ]
    },
    {
        "event_id": "judiciary.delhi_building_collapse_seemapuri",
        "title": "Four-Storey Residential Building Collapses in Northeast Delhi Seemapuri",
        "texts": [
            "Building Collapse in Northeast Delhi's Seemapuri leaves two dead and six injured",
            "Four-storey building collapses in Seemapuri area of East Delhi; rescue operations underway",
            "Tragic building collapse reported in Seemapuri; NDRF teams clear debris",
            "Seemapuri Delhi me 4 manjila imarat giri, kayi logon ke dabe hone ki aashanka"
        ]
    },

    # --- DEFENCE & CELEBRATIONS ---
    {
        "event_id": "defence.iaf_94th_air_force_day",
        "title": "Indian Air Force Celebrates 94th Anniversary with Grand Air Show at Hindon",
        "texts": [
            "Indian Air Force Celebrates 94th Air Force Day at Hindon Air Base",
            "IAF marks 94th foundation day with breathtaking aerial display over Hindon air force station",
            "Tejas and Rafale jets perform acrobatic maneuvers on 94th Indian Air Force Day",
            "Air Chief Marshal addresses air warriors on the occasion of 94th Air Force Day parade"
        ]
    },
    {
        "event_id": "aviation.indigo_patna_lucknow_lightning",
        "title": "IndiGo Flight from Patna to Lucknow Struck by Lightning Mid-Air, Lands Safely",
        "texts": [
            "IndiGo Patna-Lucknow Flight Struck by Lightning Mid-Air, Emergency Landing in Lucknow",
            "Passengers panic as lightning strikes IndiGo aircraft flying from Patna to Lucknow",
            "Mid-air scare: Lightning hits IndiGo flight 6E-452; safe landing at Lucknow airport",
            "Patna se Lucknow ja rahi IndiGo flight par aasmani bijli giri, safe landing"
        ]
    },

    # --- STUDENT & CAMPUS UNREST ---
    {
        "event_id": "campus.du_oct10_mobilization",
        "title": "Delhi University Student Outfits Mobilize for October 10 Protest on Fee Hike",
        "texts": [
            "Delhi University student mobilization for October 10 protest against tuition fee hike",
            "AISA and NSUI organize joint torchlight march across Arts Faculty ahead of October 10 DU strike",
            "DU students call for complete university shutdown on October 10 demanding hostel accommodation",
            "Delhi University students rally outside Arts faculty against steep fee increases"
        ]
    },
    {
        "event_id": "campus.mumbai_tribal_detention",
        "title": "Tribal Student Protesters Detained in Mumbai During March to Mantralaya",
        "texts": [
            "Tribal Student Protesters Detained in Mumbai near CSMT during scholarship demonstration",
            "Mumbai Police detains over 200 tribal students marching towards Mantralaya demanding fellowship funds",
            "Adivasi student morcha stopped near Azad Maidan in Mumbai; leaders bundled into police vans",
            "Mumbai: Tribal welfare scholarship delay triggers student outrage; detentions at Mantralaya"
        ]
    },
    {
        "event_id": "campus.kerala_cu_vande_mataram",
        "title": "Protests Against Vande Mataram Seminar at Central University of Kerala",
        "texts": [
            "Protests Against Vande Mataram Seminar at Central University Kerala by student unions",
            "Clash between ABVP and SFI at Central University of Kerala over Vande Mataram symposium",
            "Kerala Central University campus tense following demonstrations against national song seminar",
            "Student groups blockade administrative block at Central University of Kerala in Kasaragod"
        ]
    },

    # --- CELEBRITY / POP CULTURE HOAXES ---
    {
        "event_id": "cinema.nana_patekar_death_hoax",
        "title": "Death Hoax Surrounding Veteran Actor Nana Patekar Goes Viral on Social Media",
        "texts": [
            "Reports of Actor Nana Patekar Passing Away at 75 Debunked as Fake News",
            "Fake viral post claiming veteran actor Nana Patekar passed away dismissed by family",
            "Nana Patekar is alive and healthy: Representative clarifies after viral death rumors",
            "Actor Nana Patekar ke nidhan ki afwah viral, parivar ne bataya bilkul theek hain"
        ]
    }
]

print(f"Total curated distinct event clusters: {len(DISTINCT_EVENTS)}")

# Generate triplets:
# An anchor and positive come from the SAME event.
# The hard negative comes from a DIFFERENT event that shares domain/keywords/entities!
triplets = []
pair_positives = []
pair_negatives = []

for ev_i in range(len(DISTINCT_EVENTS)):
    ev = DISTINCT_EVENTS[ev_i]
    # Pick a related event in the same domain/group as hard negative
    # Find event that shares words
    ev_words = set(re.findall(r"\w+", ev["title"].lower()))
    
    other_evs = [DISTINCT_EVENTS[j] for j in range(len(DISTINCT_EVENTS)) if j != ev_i]
    # score overlap
    def overlap(o):
        owords = set(re.findall(r"\w+", o["title"].lower()))
        return len(ev_words & owords)
    
    other_evs.sort(key=overlap, reverse=True)
    hard_neg_ev = other_evs[0] # closest semantic neighbor
    
    # Internal positive pairs
    texts = ev["texts"]
    for i, t1 in enumerate(texts):
        for j, t2 in enumerate(texts):
            if i < j:
                pair_positives.append({"text1": t1, "text2": t2, "event1": ev["event_id"], "event2": ev["event_id"], "label": 1})
                # Add hard negative from closest neighbor
                neg_text = random.choice(hard_neg_ev["texts"])
                pair_negatives.append({"text1": t1, "text2": neg_text, "event1": ev["event_id"], "event2": hard_neg_ev["event_id"], "label": 0})
                triplets.append({
                    "anchor": t1,
                    "positive": t2,
                    "negative": neg_text,
                    "events": [ev["event_id"], hard_neg_ev["event_id"]]
                })

print(f"Generated {len(triplets)} targeted hard-negative event triplets.")
print(f"Generated {len(pair_positives)} positive pairs and {len(pair_negatives)} hard-negative pairs.")

out_dir = Path("artifacts/topic_corpus_hard_v5")
out_dir.mkdir(parents=True, exist_ok=True)

with open(out_dir / "hard_triplets.jsonl", "w") as f:
    for tr in triplets:
        f.write(json.dumps(tr, ensure_ascii=False) + "\n")

with open(out_dir / "eval_benchmark_pairs.jsonl", "w") as f:
    for p in pair_positives + pair_negatives:
        f.write(json.dumps(p, ensure_ascii=False) + "\n")

print("Saved targeted hard dataset to", out_dir)
