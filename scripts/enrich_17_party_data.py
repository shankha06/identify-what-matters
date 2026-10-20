"""
Generate high-signal, authentic vernacular training samples to enrich party_train_17_all.jsonl.
Addresses key evaluation weaknesses:
1. Congress vs AAP distinction (Samvidhan, Nyay, Caste Census, Kharge, Rahul Gandhi, Bharat Jodo, Agniveer, MSP legal guarantee).
2. AAP specific governance & struggle (Mohalla clinics, free power/water, Punjab reforms, excise policy witch-hunt, Atishi, Sanjay Singh, Raghav Chadha).
3. Regional party nuances (TDP Amaravati vs YSRCP Navaratnalu/DBT, BRS Rythu Bandhu vs Revanth Reddy critique, Shiv Sena Balasaheb/Uddhav vs Shinde, NCP Pawar legacy, RJD employment/Tejashwi vs JDU Nitish sushasan, DMK Dravidian model/social justice vs AIADMK Amma legacy/EPS, BSP Kanshi Ram/Mayawati Bahujan emancipation, CPIM Pinarayi/Left Front working-class solidarity).
4. Realistic None/Neutral samples (Indian infra complaints without political bias, tech, cricket, cinema, daily life).
"""

import json
from pathlib import Path

ADDITIONAL_DATA = [
    # =========================================================================
    # CONGRESS (Distinctive economic justice, Samvidhan, Caste Census, Agniveer, NYAY, Bharat Jodo, Kharge, Rahul)
    # =========================================================================
    {"text": "Rahul Gandhi is 100% right about caste census. Without knowing the exact population percentage, how can backward classes get fair representation in wealth and budget? Jitni aabadi utna haq!", "party": "Congress", "weight": 2.5, "source": "curated_v2"},
    {"text": "Agniveer scheme has destroyed the dreams of rural youth who prepared for Indian Army. Congress guarantee to scrap Agnipath and restore permanent recruitment is crucial for national security.", "party": "Congress", "weight": 2.5, "source": "curated_v2"},
    {"text": "Mallikarjun Kharge ji rightly questioned the government on why Rs 16 lakh crore corporate loans were written off while farmers commit suicide for Rs 50,000 debt. Congress stands with annadatas.", "party": "Congress", "weight": 2.5, "source": "curated_v2"},
    {"text": "The Constitution of India is in danger under authoritarian rule. Congress Bharat Jodo Nyay Yatra was not for votes, it was to defend the Samvidhan created by Babasaheb Ambedkar.", "party": "Congress", "weight": 2.5, "source": "curated_v2"},
    {"text": "Mahalakshmi scheme giving Rs 1 lakh annually (Rs 8,500/month) to poorest women is true financial empowerment. Congress delivered Gruha Lakshmi in Karnataka and Telangana, and will deliver nationwide.", "party": "Congress", "weight": 2.5, "source": "curated_v2"},
    {"text": "Legal guarantee for MSP based on Swaminathan Commission C2+50% formula is non-negotiable. Modi sarkar promised doubling farm income but gave lathis to farmers. Only Congress will enact MSP law.", "party": "Congress", "weight": 2.5, "source": "curated_v2"},
    {"text": "Unemployment is at a 45-year high. Congress promised Pehli Naukri Pakki - Rs 1 lakh apprenticeship for every graduate and diploma holder. This is the structural reform Indian youth desperately need.", "party": "Congress", "weight": 2.5, "source": "curated_v2"},
    {"text": "Chronology samajhiye: Airports, ports, telecom, coal mines - sab kuch sirf do doston ko de diya gaya. Rahul Gandhi alone is speaking the truth against crony billionaire oligopoly.", "party": "Congress", "weight": 2.5, "source": "curated_v2"},
    {"text": "Kharge sahab's speech in Rajya Sabha tore apart the government's false claims. From Manipur violence to Ladakh Chinese incursions, PM has stayed completely silent. INC is the only fearless opposition.", "party": "Congress", "weight": 2.5, "source": "curated_v2"},
    {"text": "Priyanka Gandhi's grassroots campaigning in Himachal and UP showed real empathy. Ladki Hoon Lad Sakti Hoon empowered thousands of young female leaders across North India.", "party": "Congress", "weight": 2.5, "source": "curated_v2"},
    {"text": "Nyay Patra is not jumla, it is a socio-economic blueprint. From right to apprenticeship to filling 30 lakh vacant central government jobs, Congress manifesto addresses real pain points.", "party": "Congress", "weight": 2.5, "source": "curated_v2"},
    {"text": "Free bus travel for women under Shakthi scheme in Karnataka has allowed women to join the workforce without transport costs. Congress delivers tangible welfarism without communal polarization.", "party": "Congress", "weight": 2.5, "source": "curated_v2"},
    {"text": "Bharat Jodo Yatra walked 4,000 km from Kanyakumari to Kashmir spreading Nafrat ke bazaar mein mohabbat ki dukaan. That spirit is what holds India together.", "party": "Congress", "weight": 2.5, "source": "curated_v2"},
    {"text": "Paper leaks in NEET, UGC-NET, and state exams have ruined millions of hardworking students. Congress promised severe punishment for exam mafia and transparent recruitment boards.", "party": "Congress", "weight": 2.5, "source": "curated_v2"},
    {"text": "Chidambaram and Manmohan Singh built the economic foundation that lifted 270 million people out of poverty between 2004 and 2014. Current GDP growth without jobs is an illusion.", "party": "Congress", "weight": 2.5, "source": "curated_v2"},
    {"text": "Samvidhan bachao andolan is the call of the hour. If opposition loses, secular democracy will be replaced by single-party rule. Vote for Congress and INDIA bloc to protect your fundamental rights.", "party": "Congress", "weight": 2.5, "source": "curated_v2"},
    {"text": "GST was supposed to be simple, but BJP turned it into Gabbar Singh Tax that destroyed MSMEs and small traders. Congress will rationalize GST slabs to single moderate tax rate.", "party": "Congress", "weight": 2.5, "source": "curated_v2"},
    {"text": "Revanth Reddy implement chesina 6 guarantees Telangana lo poor families ki real relief istunnayi. Free electricity up to 200 units, Rs 500 gas cylinder - Congress governance is visible on the ground.", "party": "Congress", "weight": 2.5, "source": "curated_v2"},
    {"text": "Siddaramaiah government Anna Bhagya rice scheme feeds millions of BPL households. While Delhi government withheld FCI grains, Congress arranged direct cash transfer to ensure no family goes hungry.", "party": "Congress", "weight": 2.5, "source": "curated_v2"},
    {"text": "Vote for Congress hand symbol. We need unity, brotherhood, fair wages for laborers, and respect for federalism. Desh ko nafrat nahi, rozgar chahiye!", "party": "Congress", "weight": 2.5, "source": "curated_v2"},

    # =========================================================================
    # AAP (Delhi & Punjab governance, Kejriwal, Sisodia, Mohalla Clinics, Atishi, Sanjay Singh)
    # =========================================================================
    {"text": "Arvind Kejriwal was arrested without a single rupee of recovery because Modi fear AAP's education and health revolution. Satyendar Jain and Manish Sisodia built world-class government schools.", "party": "AAP", "weight": 2.5, "source": "curated_v2"},
    {"text": "Mohalla clinic in our lane provides free doctor consultation, 200+ free diagnostic tests, and medicines. Poor people saved thousands of rupees every month because of Kejriwal government.", "party": "AAP", "weight": 2.5, "source": "curated_v2"},
    {"text": "Manish Sisodia turned Delhi government schools into institutions where rich parents are withdrawing kids from private schools to enroll in sarkari schools. Desh ke shiksha mantri ko jail bhejna sharmnaak hai.", "party": "AAP", "weight": 2.5, "source": "curated_v2"},
    {"text": "Bhagwant Mann government in Punjab has given free 300 units electricity to 90% households, regularized 30,000 contractual workers, and opened 800+ Aam Aadmi Clinics across the state.", "party": "AAP", "weight": 2.5, "source": "curated_v2"},
    {"text": "Delhi model proves that clean politics is possible. Despite having a budget surplus every year, Kejriwal gave free water, free bijli up to 200 units, and free bus rides for women.", "party": "AAP", "weight": 2.5, "source": "curated_v2"},
    {"text": "Atishi Marlena and Saurabh Bharadwaj are managing Delhi brilliantly despite LG continuously blocking administrative files, mohalla clinic doctors' salaries, and hospital medicine tenders.", "party": "AAP", "weight": 2.5, "source": "curated_v2"},
    {"text": "Sanjay Singh roared in Parliament against central vendetta. AAP is the only party that was born out of an anti-corruption movement and genuinely cares for the common man (Aam Aadmi).", "party": "AAP", "weight": 2.5, "source": "curated_v2"},
    {"text": "Kejriwal's arrest before Lok Sabha elections was a deliberate conspiracy to prevent AAP from campaigning. Jhadoo chhap is the symbol of integrity that won hearts in Delhi and Punjab.", "party": "AAP", "weight": 2.5, "source": "curated_v2"},
    {"text": "Farishtey Dilli Ke scheme has saved over 23,000 accident victims by providing free treatment in private hospitals paid for by the AAP government. This is humanity in governance.", "party": "AAP", "weight": 2.5, "source": "curated_v2"},
    {"text": "Raghav Chadha's articulate defense of Delhi services bill in Rajya Sabha exposed how the central ordinance subverted the Supreme Court constitution bench judgment.", "party": "AAP", "weight": 2.5, "source": "curated_v2"},
    {"text": "Punjab Police anti-corruption helpline launched by Bhagwant Mann led to immediate arrests of corrupt officers. AAP walks the talk on zero tolerance to corruption.", "party": "AAP", "weight": 2.5, "source": "curated_v2"},
    {"text": "BJP calls free education and healthcare 'revdi' while waiving loans for corporate cronies. Kejriwal said: public money belongs to the public, not to politicians' personal friends.", "party": "AAP", "weight": 2.5, "source": "curated_v2"},
    {"text": "Our electricity bill in Rohini used to be Rs 3,500 every summer. Under AAP, with zero-bill subsidy, we haven't paid electricity bills for two straight years. Kejriwal zindabad!", "party": "AAP", "weight": 2.5, "source": "curated_v2"},
    {"text": "Specialized Schools of Excellence (SOSE) in Delhi are producing IIT-JEE and NEET toppers from humble auto-driver and laborer households. AAP transformed lives.", "party": "AAP", "weight": 2.5, "source": "curated_v2"},
    {"text": "Kejriwal ki guarantee matlab 100% execution. Pehle Delhi jeeta, phir Punjab jeeta, ab Gujarat aur Goa mein bhi jhadoo chalega. Aam Aadmi Party is the future of alternative politics.", "party": "AAP", "weight": 2.5, "source": "curated_v2"},

    # =========================================================================
    # BJP (Modi leadership, Viksit Bharat, Hindutva, infrastructure, Article 370, Ram Mandir)
    # =========================================================================
    {"text": "Under Prime Minister Narendra Modi, India became the 5th largest economy and will soon be 3rd. UPI digital payments and Bharatmala expressways show the world what Viksit Bharat looks like.", "party": "BJP", "weight": 2.5, "source": "curated_v2"},
    {"text": "Abrogation of Article 370 brought peace, tourist boom, and full constitutional integration to Jammu & Kashmir. Only PM Modi and Home Minister Amit Shah had the political courage to do it.", "party": "BJP", "weight": 2.5, "source": "curated_v2"},
    {"text": "After 500 years of struggle, the grand Ram Lalla Mandir stands proudly in Ayodhya. Modi ji performed Pran Pratishtha and fulfilled the civilizational dream of crores of Sanatanis. Jai Shri Ram!", "party": "BJP", "weight": 2.5, "source": "curated_v2"},
    {"text": "Yogi Adityanath has completely eradicated mafia raj and gangster rule in Uttar Pradesh. Law and order is so secure now that sisters can walk safely at midnight. Bulldozer justice works!", "party": "BJP", "weight": 2.5, "source": "curated_v2"},
    {"text": "80 crore citizens receive free ration under PM Garib Kalyan Anna Yojana, 50 crore have Ayushman Bharat Rs 5 lakh free healthcare, and 12 crore toilets built. BJP's Antyodaya is true social welfare.", "party": "BJP", "weight": 2.5, "source": "curated_v2"},
    {"text": "Operation Ganga, Operation Kaveri, Vande Bharat trains, Tejas fighter jets, defense exports reaching Rs 21,000 crore - India's global stature under Modi has never been higher.", "party": "BJP", "weight": 2.5, "source": "curated_v2"},
    {"text": "Congress and INDI alliance want to snatch SC/ST/OBC reservation and distribute it to minority vote banks. BJP will never allow appeasement politics to harm national unity. Phir ek baar Modi sarkar!", "party": "BJP", "weight": 2.5, "source": "curated_v2"},
    {"text": "PM Kisan Samman Nidhi transfers Rs 6,000 direct into bank accounts of 11 crore small and marginal farmers without cut or commission. No middlemen can touch PM's direct benefit transfer.", "party": "BJP", "weight": 2.5, "source": "curated_v2"},
    {"text": "Kashi Vishwanath Corridor, Mahakal Lok Ujjain, Somnath redevelopment - Modi government has rejuvenated our ancient spiritual heritage while simultaneously building bullet train and chip fabs.", "party": "BJP", "weight": 2.5, "source": "curated_v2"},
    {"text": "National security is non-negotiable. Balakot air strike and Uri surgical strike proved to Pakistan that any terror attack will meet swift and decisive retribution inside enemy territory. Har Har Modi!", "party": "BJP", "weight": 2.5, "source": "curated_v2"},

    # =========================================================================
    # TMC (West Bengal, Mamata Banerjee, Lakshmir Bhandar, Khela Hobe, Didi)
    # =========================================================================
    {"text": "Mamata Didi's Lakshmir Bhandar scheme gives Rs 1,000 to 1,200 every month to every female homemaker in Bengal. It has provided financial dignity to mothers and sisters in every village.", "party": "TMC", "weight": 2.5, "source": "curated_v2"},
    {"text": "BJP used CBI, ED, and money power to conquer Bengal in 2021, but Mamata Banerjee on a wheelchair defeated the entire central cabinet! Khela hobe! Bengal will never bow to Delhi outsiders.", "party": "TMC", "weight": 2.5, "source": "curated_v2"},
    {"text": "Abhishek Banerjee walked hundreds of kilometers in Naba Jowar Yatra connecting youth to grassroots Trinamool politics. TMC is fighting for West Bengal's rightful MGNREGA and Awas funds withheld by Centre.", "party": "TMC", "weight": 2.5, "source": "curated_v2"},
    {"text": "Kanyashree Prakalpa received United Nations public service award for keeping girls in school and eliminating child marriage. Didi's social schemes are unmatched globally. Joy Bangla!", "party": "TMC", "weight": 2.5, "source": "curated_v2"},
    {"text": "Swasthya Sathi card empowers families with Rs 5 lakh free hospitalization in top private hospitals across Bengal and beyond. No one in Bengal dies due to lack of medical funds under TMC rule.", "party": "TMC", "weight": 2.5, "source": "curated_v2"},
    {"text": "Bohiragoto goons from Delhi cannot teach Tagore, Netaji, and Nazrul's land about nationalism. Mamata Banerjee is the lioness defending secular federalism in Eastern India.", "party": "TMC", "weight": 2.5, "source": "curated_v2"},
    {"text": "Sabuj Sathi bicycles given to school students have transformed rural Bengal's mobility. Vote for Trinamool grass-and-flowers symbol to keep communal forces out of our sacred soil.", "party": "TMC", "weight": 2.5, "source": "curated_v2"},

    # =========================================================================
    # SP (Samajwadi Party, Akhilesh Yadav, PDA, Netaji Mulayam, Bicycle symbol)
    # =========================================================================
    {"text": "PDA (Pichhda, Dalit, Alpsankhyak) formula created by Akhilesh Yadav united the oppressed communities of Uttar Pradesh and crushed the arrogance of the ruling party. Samajwadi zindabad!", "party": "SP", "weight": 2.5, "source": "curated_v2"},
    {"text": "Netaji Mulayam Singh Yadav dedicated his entire life to farmers, soldiers, and socialist values. Akhilesh bhaiya's visionary Agra-Lucknow Expressway and Lucknow Metro remain the best infrastructure UP ever saw.", "party": "SP", "weight": 2.5, "source": "curated_v2"},
    {"text": "Bulldozer sarkar has ruined rule of law in UP. Homes of innocent minorities and backward communities are demolished without court orders. Samajwadi Party will bring back rule of Constitution.", "party": "SP", "weight": 2.5, "source": "curated_v2"},
    {"text": "Dial 100 police response system and 108 ambulance were world-class initiatives launched during SP government. BJP only renamed them and degraded their service quality.", "party": "SP", "weight": 2.5, "source": "curated_v2"},
    {"text": "Awadhesh Prasad's historic victory in Faizabad (Ayodhya) proved that the people rejected polarizing religious politics and voted for PDA social justice and Samajwadi bicycle symbol.", "party": "SP", "weight": 2.5, "source": "curated_v2"},
    {"text": "Caste census is the only way to ensure proportionate justice for OBCs and Dalits. Akhilesh Yadav has pledged that Samajwadi government will conduct caste census within three months.", "party": "SP", "weight": 2.5, "source": "curated_v2"},

    # =========================================================================
    # DMK (Tamil Nadu, MK Stalin, Dravidian Model, Social Justice, Rising Sun)
    # =========================================================================
    {"text": "Chief Minister MK Stalin's Dravidian Model ensures economic growth with social justice. Tamil Nadu is India's second largest economy and leader in industrialization without any communal discord.", "party": "DMK", "weight": 2.5, "source": "curated_v2"},
    {"text": "Kalaignar Magalir Urimai Thogai transfers Rs 1,000 monthly to over 1.15 crore women in Tamil Nadu as a basic income right. Kalaignar Karunanidhi's social reform legacy lives on in DMK.", "party": "DMK", "weight": 2.5, "source": "curated_v2"},
    {"text": "Imposition of Hindi and NEP will never be accepted in Periyar and Anna's Tamil soil. DMK's two-language policy protects Tamil pride while equipping students with global English proficiency.", "party": "DMK", "weight": 2.5, "source": "curated_v2"},
    {"text": "Pudhumai Penn scheme gives Rs 1,000 monthly to female government school students pursuing college degrees. Female higher education enrollment in Tamil Nadu is now the highest in India.", "party": "DMK", "weight": 2.5, "source": "curated_v2"},
    {"text": "Udhayanidhi Stalin spoke against Sanatana hierarchical caste inequality to uphold Dr Ambedkar and Thanthai Periyar's egalitarian vision. DMK will always fight against Manuvadi dominance.", "party": "DMK", "weight": 2.5, "source": "curated_v2"},
    {"text": "Governor RN Ravi's unconstitutional delay of bills passed by the elected Tamil Nadu Assembly was slapped down by the Supreme Court. DMK stands as the foremost champion of state autonomy.", "party": "DMK", "weight": 2.5, "source": "curated_v2"},

    # =========================================================================
    # SHIV SENA (Maharashtra, Balasaheb, Uddhav Thackeray, Eknath Shinde, Bow & Arrow/Flaming Torch)
    # =========================================================================
    {"text": "Balasaheb Thackeray established Shiv Sena to protect Marathi manoos self-respect in Mumbai. Those traitors who broke the party using Surat-Guwahati planes will be taught a lesson by the public.", "party": "Shiv Sena", "weight": 2.5, "source": "curated_v2"},
    {"text": "Uddhav Thackeray handled the COVID-19 pandemic with extreme compassion and scientific temper. The Dharavi model was praised by the WHO. Mashaal symbol will triumph across Maharashtra!", "party": "Shiv Sena", "weight": 2.5, "source": "curated_v2"},
    {"text": "Eknath Shinde and Mahayuti government launched Ladki Bahin Yojana giving Rs 1,500 monthly to women. Asli Shiv Sena is working for the common auto driver and farmer under the Dhanushya-baan symbol.", "party": "Shiv Sena", "weight": 2.5, "source": "curated_v2"},
    {"text": "Maharashtra's mega projects like Vedanta Foxconn, Tata Airbus, and bulk drug parks were shifted to Gujarat by the Centre. Only Shiv Sena can fight for Mumbai's financial autonomy and pride.", "party": "Shiv Sena", "weight": 2.5, "source": "curated_v2"},
    {"text": "Aamhi Shiv Sainik aahot! Balasaheb's Hindutva was inclusive and proud, not filled with petty hate. Uddhav Saheb kept his word and stood with farmers during drought.", "party": "Shiv Sena", "weight": 2.5, "source": "curated_v2"},

    # =========================================================================
    # NCP (Sharad Pawar, Ajit Pawar, Supriya Sule, Clock/Tutari, Baramati)
    # =========================================================================
    {"text": "Sharad Pawar saheb is the Bhishma Pitamah of Indian politics. At the age of 83, in pouring rain at Satara, he inspired millions of youth to defeat authoritarian forces. Tutari wajwa!", "party": "NCP", "weight": 2.5, "source": "curated_v2"},
    {"text": "Supriya Sule's stellar performance in Parliament raising issues of agrarian distress, MSP, and women's safety earned her multiple Sansad Ratna awards. NCP represents educated, progressive leadership.", "party": "NCP", "weight": 2.5, "source": "curated_v2"},
    {"text": "Ajit Pawar's administrative efficiency (Dada chi style) in clearing irrigation and development files is unmatched. NCP under Ajitdada brought budget allocations directly to rural cooperative banks.", "party": "NCP", "weight": 2.5, "source": "curated_v2"},
    {"text": "The entire sugar cooperative, dairy, and agricultural market yard revolution in Western Maharashtra was architected by Sharad Pawar. Nobody understands farmer economics better than NCP.", "party": "NCP", "weight": 2.5, "source": "curated_v2"},

    # =========================================================================
    # RJD (Bihar, Lalu Prasad Yadav, Tejashwi Yadav, 10 Lakh Naukri, Lantern)
    # =========================================================================
    {"text": "Tejashwi Yadav gave appointment letters to over 4 lakh teachers and healthcare workers in just 17 months during the Mahagathbandhan government. Real employment is Tejashwi's trademark.", "party": "RJD", "weight": 2.5, "source": "curated_v2"},
    {"text": "Lalu Prasad Yadav gave voice, self-respect, and political representation to the backward, extremely backward, and minority communities of Bihar who were previously forbidden from entering polling booths.", "party": "RJD", "weight": 2.5, "source": "curated_v2"},
    {"text": "Jan Vishwas Yatra led by Tejashwi drew lakhs of enthusiastic youth. Bihar wants jobs, IT parks, and factories, not hollow slogans. Laltain chhap zindabad!", "party": "RJD", "weight": 2.5, "source": "curated_v2"},
    {"text": "Caste survey conducted in Bihar under RJD-JDU government revealed 85% backward population and paved the way for increasing reservations to 65%. Social justice is RJD's core commitment.", "party": "RJD", "weight": 2.5, "source": "curated_v2"},

    # =========================================================================
    # JDU (Bihar, Nitish Kumar, Sushasan Babu, Har Ghar Nal, Teer symbol)
    # =========================================================================
    {"text": "Nitish Kumar transformed Bihar from lawlessness to sushasan. 50% reservation for women in Panchayats, cycle scheme for school girls, and total liquor prohibition made mothers safe.", "party": "JDU", "weight": 2.5, "source": "curated_v2"},
    {"text": "Saat Nischay program brought tap water (Har Ghar Nal Ka Jal), electricity to every hamlet, and pucca roads to remote villages. JDU governance delivered tangible basic amenities.", "party": "JDU", "weight": 2.5, "source": "curated_v2"},
    {"text": "Nitish Babu has maintained a clean, unblemished personal integrity throughout his 50-year public career. No nepotism, no corruption charges against him personally.", "party": "JDU", "weight": 2.5, "source": "curated_v2"},
    {"text": "Special category status and special financial package for Bihar is Nitish Kumar's consistent demand to industrialize the state. Vote for Teer chhap Janata Dal United.", "party": "JDU", "weight": 2.5, "source": "curated_v2"},

    # =========================================================================
    # AIADMK (Tamil Nadu, Edappadi Palaniswami, Jayalalithaa Amma, Two Leaves)
    # =========================================================================
    {"text": "Puratchi Thalaivi Amma Jayalalithaa gave Amma Canteen, Amma Baby Care Kit, free laptops, and gold for marriage. Her welfare vision will never be matched in Tamil Nadu politics.", "party": "AIADMK", "weight": 2.5, "source": "curated_v2"},
    {"text": "Edappadi K Palaniswami (EPS) provided stable governance for four years, built 11 new government medical colleges, and enacted the 7.5% horizontal quota for government school students in MBBS.", "party": "AIADMK", "weight": 2.5, "source": "curated_v2"},
    {"text": "AIADMK broke alliance with BJP to protect minority rights and Tamil identity. EPS proved he will not compromise Tamil Nadu's autonomy for Delhi power. Rettai Ilai (Two Leaves) forever!", "party": "AIADMK", "weight": 2.5, "source": "curated_v2"},
    {"text": "Under EPS, Tamil Nadu topped good governance index in India. Farmers in Cauvery delta were protected when he declared Cauvery delta as Protected Special Agricultural Zone.", "party": "AIADMK", "weight": 2.5, "source": "curated_v2"},

    # =========================================================================
    # TDP (Andhra Pradesh, Chandrababu Naidu, Nara Lokesh, Amaravati, Cycle)
    # =========================================================================
    {"text": "Nara Chandrababu Naidu is the visionary who built Hyderabad's Cyberabad and brought IT giants. Now he is rebuilding world-class green capital city Amaravati and completing Polavaram project.", "party": "TDP", "weight": 2.5, "source": "curated_v2"},
    {"text": "Under YSRCP, Andhra Pradesh was drowned in Rs 10 lakh crore debt with zero capital investment and broken roads. CBN's return restored investor confidence, TCS, and Google investments.", "party": "TDP", "weight": 2.5, "source": "curated_v2"},
    {"text": "Super Six guarantees promised by TDP alliance - Rs 1,500 monthly for women, 20 lakh youth jobs, Rs 20,000 annual farmer support, and free RTC bus travel - are transforming AP.", "party": "TDP", "weight": 2.5, "source": "curated_v2"},
    {"text": "Nara Lokesh Yuva Galam padayatra walked 3,100 km listening to unemployed youth and students. TDP is the true party of Telugu development and institutional governance.", "party": "TDP", "weight": 2.5, "source": "curated_v2"},

    # =========================================================================
    # YSRCP (Andhra Pradesh, YS Jagan Mohan Reddy, Amma Vodi, Navaratnalu, Fan)
    # =========================================================================
    {"text": "YS Jagan Mohan Reddy deposited over Rs 2.7 lakh crore directly into bank accounts of poor mothers, farmers, and students through Navaratnalu DBT without a single paisa of corruption.", "party": "YSRCP", "weight": 2.5, "source": "curated_v2"},
    {"text": "Village and Ward Secretariat system brought 500 government services right to the citizens' doorstep through village volunteers. Poor villagers never had to bribe officers under Jagan anna.", "party": "YSRCP", "weight": 2.5, "source": "curated_v2"},
    {"text": "Nadu-Nedu renovated thousands of government schools with English medium education, interactive flat panels, and CBSE affiliation. Jagan gave poor children world-class education.", "party": "YSRCP", "weight": 2.5, "source": "curated_v2"},
    {"text": "Amma Vodi provided Rs 15,000 every year to every mother who sends her child to school, preventing dropouts. YSRCP proved that welfarism is an investment in human dignity.", "party": "YSRCP", "weight": 2.5, "source": "curated_v2"},

    # =========================================================================
    # BRS (Telangana, KCR, KTR, Rythu Bandhu, Dalit Bandhu, Car symbol)
    # =========================================================================
    {"text": "KCR fought for 14 years and achieved separate Telangana statehood. Under BRS rule, Telangana became the highest per capita income state with 24-hour uninterrupted free agricultural electricity.", "party": "BRS", "weight": 2.5, "source": "curated_v2"},
    {"text": "KTR transformed Hyderabad into India's premier tech and life-sciences hub, attracting Apple, Amazon, and Novartis campuses. BRS created infrastructure like T-Hub and Durgam Cheruvu bridge.", "party": "BRS", "weight": 2.5, "source": "curated_v2"},
    {"text": "Kaleshwaram Lift Irrigation project brought Godavari waters to parched lands, making Telangana the rice bowl of India. Rythu Bandhu gave Rs 10,000 per acre investment support directly to farmers.", "party": "BRS", "weight": 2.5, "source": "curated_v2"},
    {"text": "Dalit Bandhu gave Rs 10 lakh direct non-repayable grant to Dalit families to start businesses. No other state in Indian history launched such an audacious affirmative empowerment scheme.", "party": "BRS", "weight": 2.5, "source": "curated_v2"},

    # =========================================================================
    # BSP (Mayawati, Kanshi Ram, Bahujan Samaj, Elephant symbol, Jai Bhim)
    # =========================================================================
    {"text": "Bahujan Samaj Party under Behenji Mayawati gave administrative dignity and security to Dalits and backwards in Uttar Pradesh. Law and order under BSP 2007-2012 was the best in modern Indian history.", "party": "BSP", "weight": 2.5, "source": "curated_v2"},
    {"text": "Kanshi Ram sahab's philosophy 'Vote hamara, raj tumhara, nahi chalega' awakened millions of marginalized people. Self-respect and political autonomy can never be surrendered to Manuvadi parties.", "party": "BSP", "weight": 2.5, "source": "curated_v2"},
    {"text": "Mayawati built grand memorials, parks, and universities dedicated to Dr BR Ambedkar, Jyotirao Phule, and Manyawar Kanshi Ram so future generations remember Bahujan icons with pride.", "party": "BSP", "weight": 2.5, "source": "curated_v2"},
    {"text": "Sarvajan Hitaya, Sarvajan Sukhaya was the formula that brought Brahmin, Dalit, and Muslim communities together to give UP a full 5-year majority government. Jai Bhim, Jai Bharat!", "party": "BSP", "weight": 2.5, "source": "curated_v2"},

    # =========================================================================
    # CPIM (Kerala, Pinarayi Vijayan, Left Front, Working Class, Lalsalam)
    # =========================================================================
    {"text": "Comrade Pinarayi Vijayan and the Left Democratic Front (LDF) government in Kerala maintained India's lowest infant mortality, highest literacy, and best public healthcare throughout global crises.", "party": "CPIM", "weight": 2.5, "source": "curated_v2"},
    {"text": "CPIM is the only party that relentlessly fights against neoliberal privatization, corporate loot of public sector undertakings, and sectarian communalism without making compromises.", "party": "CPIM", "weight": 2.5, "source": "curated_v2"},
    {"text": "K-FON provided free high-speed internet to 20 lakh BPL families declaring internet as a basic human right. Left front models scientific socialism with real human development index.", "party": "CPIM", "weight": 2.5, "source": "curated_v2"},
    {"text": "Lalsalam! Red flag will always stand with trade unions, unorganized plantation laborers, Anganwadi workers, and farmers protesting against anti-people labor codes.", "party": "CPIM", "weight": 2.5, "source": "curated_v2"},

    # =========================================================================
    # NONE (Apolitical civic complaints, technology, sports, food, lifestyle, daily life)
    # =========================================================================
    {"text": "The metro construction on Outer Ring Road has doubled commute times from Marathahalli to Silk Board. Need better traffic marshals and lane discipline.", "party": "None", "weight": 2.5, "source": "curated_v2"},
    {"text": "Is anyone facing broadband fiber outage in Indiranagar since morning? Airtel customer care says technical team is working on underground optical cable cut.", "party": "None", "weight": 2.5, "source": "curated_v2"},
    {"text": "Virat Kohli's cover drive against Mitchell Starc yesterday was pure poetry. That match-winning partnership took India into the semi-finals comfortably.", "party": "None", "weight": 2.5, "source": "curated_v2"},
    {"text": "Best biryani in Hyderabad? Some people swear by Shadab in Old City, others prefer Bawarchi at RTC Crossroads or Café Bahar. What is your go-to weekend spot?", "party": "None", "weight": 2.5, "source": "curated_v2"},
    {"text": "Tech hiring market is still brutal for freshers. 500 applicants within 30 minutes of a job posting on LinkedIn. Upskilling in PyTorch and system design is essential.", "party": "None", "weight": 2.5, "source": "curated_v2"},
    {"text": "Tomato prices jumped from Rs 30 to Rs 80 per kg at the local sabzi mandi this week due to unseasonal rains affecting harvest in Kolar and Nashik.", "party": "None", "weight": 2.5, "source": "curated_v2"},
    {"text": "Has anyone watched the new Malayalam investigative thriller on OTT? The screenplay and background score in the climax were breathtaking without unnecessary melodrama.", "party": "None", "weight": 2.5, "source": "curated_v2"},
    {"text": "Electricity voltage fluctuation in our building caused our inverter battery to trip. Called the local power sub-station line-man to fix transformer fuse.", "party": "None", "weight": 2.5, "source": "curated_v2"},
    {"text": "Nifty 50 touched an all-time high today led by banking and IT stocks. Good time to rebalance portfolio and book partial profits into debt funds.", "party": "None", "weight": 2.5, "source": "curated_v2"},
    {"text": "The monsoon clouds over Western Ghats are spectacular this morning. Enjoying hot filter coffee with piping hot medu vada.", "party": "None", "weight": 2.5, "source": "curated_v2"},
    {"text": "Public libraries should be open on weekends too. Students living in cramped hostels need quiet air-conditioned study spaces with decent Wi-Fi.", "party": "None", "weight": 2.5, "source": "curated_v2"},
    {"text": "Garbage collection in our ward hasn't happened for three days. Waste is piling up on the corner of the street attracting stray dogs. Ward engineer must intervene.", "party": "None", "weight": 2.5, "source": "curated_v2"},
    {"text": "Electric vehicle charging stations along the expressway need standardized payment QR codes instead of forcing drivers to download 10 different buggy apps.", "party": "None", "weight": 2.5, "source": "curated_v2"},
    {"text": "Doctor advised 45 minutes of brisk walking every day to control border-line cholesterol levels. Lifestyle modifications work better than medicines early on.", "party": "None", "weight": 2.5, "source": "curated_v2"},
    {"text": "Can someone recommend a sturdy water purifier with low maintenance filters for borewell water with TDS over 600? Ro plus UV preferred.", "party": "None", "weight": 2.5, "source": "curated_v2"}
]

def main():
    target_file = Path("artifacts/demographics_models/party_train_17_all.jsonl")
    existing_samples = []
    if target_file.exists():
        with open(target_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    existing_samples.append(json.loads(line))

    existing_texts = {s["text"].strip().lower() for s in existing_samples}
    added_count = 0

    new_samples = list(existing_samples)
    for item in ADDITIONAL_DATA:
        t = item["text"].strip()
        if t.lower() not in existing_texts:
            new_samples.append(item)
            existing_texts.add(t.lower())
            added_count += 1

    print(f"Original samples: {len(existing_samples)}")
    print(f"Added high-signal curated samples: {added_count}")
    print(f"Total new samples: {len(new_samples)}")

    with open(target_file, "w", encoding="utf-8") as f:
        for item in new_samples:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Successfully saved enriched dataset to {target_file}")

if __name__ == "__main__":
    main()
