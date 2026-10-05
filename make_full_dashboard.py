import json
import html
import os
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path

report_paths = sorted(Path('output/runs').glob('*/report.json'))
if not report_paths:
  raise FileNotFoundError('No completed trend reports found under output/runs')

report_records = []
for path in report_paths:
  with path.open(encoding='utf-8') as f:
    report = json.load(f)
  try:
    generated_at = datetime.fromisoformat(str(report.get('generated_at', '')).replace('Z', '+00:00'))
    generated_order = generated_at.timestamp()
  except (TypeError, ValueError, OverflowError):
    generated_order = float('-inf')
  report_records.append((generated_order, path, report))

report_records.sort(key=lambda record: (record[0], record[1].parent.name))
_, report_path, data = report_records[-1]

target_run_dir = report_path.parent
collect_path = target_run_dir / 'collect.json'
latest_collection = None
if collect_path.exists():
  try:
    with collect_path.open(encoding='utf-8') as f:
      latest_collection = json.load(f)
  except Exception:
    latest_collection = None

# If current run doesn't have collect.json, find the latest by run folder / captured_at
if not latest_collection:
  collection_records = []
  for path in Path('output/runs').glob('*/collect.json'):
    with path.open(encoding='utf-8') as f:
      collection = json.load(f)
    try:
      captured_order = datetime.fromisoformat(str(collection.get('captured_at', '')).replace('Z', '+00:00')).timestamp()
    except (TypeError, ValueError, OverflowError):
      captured_order = float('-inf')
    collection_records.append((path.parent.name, captured_order, path, collection))
  if collection_records:
    collection_records.sort(key=lambda r: (r[0], r[1]))
    _, _, collect_path, latest_collection = collection_records[-1]

collection_section_html = '<div class="card" style="margin-bottom: 24px;"><div class="card-body">No public trend collection is available.</div></div>'
if latest_collection:
  collection_path = collect_path
  captured_at = html.escape(str(latest_collection.get('captured_at') or 'unknown'))
  collection_run = html.escape(str(latest_collection.get('run_id') or collection_path.parent.name))
  signal_rows = []
  signals = sorted(latest_collection.get('signals', []),
           key=lambda signal: str(signal.get('published_at') or ''), reverse=True)
  for signal in signals[:30]:
    title = html.escape(str(signal.get('title') or 'Untitled signal'))
    url = str(signal.get('url') or '')
    title_cell = f'<a href="{html.escape(url, quote=True)}" target="_blank" rel="noopener noreferrer">{title}</a>' if url.startswith('https://') else title
    source = html.escape(str(signal.get('feed') or signal.get('source') or 'unknown'))
    published = html.escape(str(signal.get('published_at') or ''))
    signal_rows.append(f'<tr><td>{source}</td><td>{title_cell}</td><td>{published}</td></tr>')
  feed_rows = []
  for feed, result in sorted((latest_collection.get('availability') or {}).items()):
    status = f'{result} items' if isinstance(result, int) else str(result)
    feed_rows.append(f'<tr><td>{html.escape(str(feed))}</td><td>{html.escape(status)}</td></tr>')
  signals_table = (
    '<table style="width:100%; border-collapse:collapse; font-size:12px; margin-top:14px;">'
    '<thead><tr><th align="left">Feed</th><th align="left">Signal</th><th align="left">Published</th></tr></thead>'
    f'<tbody>{"".join(signal_rows)}</tbody></table>'
    if signal_rows else '<p style="margin-top:12px; color:var(--text-dim);">No fresh signals were returned by the available feeds.</p>'
  )
  availability_table = (
    '<table style="width:100%; border-collapse:collapse; font-size:11.5px; margin-top:14px;">'
    '<thead><tr><th align="left">Source check</th><th align="left">Result</th></tr></thead>'
    f'<tbody>{"".join(feed_rows)}</tbody></table>' if feed_rows else ''
  )
  collection_section_html = f'''<div class="card" style="margin-bottom: 24px;">
    <div class="card-header"><div class="card-title">Latest signal collection</div>
    <span class="chip mono">{collection_run} · {captured_at}</span></div>
    <div class="card-body"><div style="font-size:12px; color:var(--text-dim);">{len(signals)} live public signals · Instagram: {html.escape(str(latest_collection.get('transport') or 'not collected'))}</div>
    {availability_table}{signals_table}</div>
  </div>'''

topics = data.get('topics', [])
panel_pulse = data.get('panel_pulse', [])
run_id = data.get('run_id', '20261004T131104Z')
generated_at = data.get('generated_at', '')

panel_cfg_accounts = []
try:
    with open('config/panel.json', encoding='utf-8') as f:
        panel_cfg_accounts = json.load(f).get('accounts', [])
except Exception:
    panel_cfg_accounts = []

from src.identify_what_matters.demographics import (
    GEN_Z_SLANG, PROFESSIONAL_SLANG, MATURE_SLANG,
    FEMALE_CONVERSATIONAL, MALE_CONVERSATIONAL, infer_location, _get_models
)
from src.identify_what_matters.political_evidence import audit_comments
from src.identify_what_matters.bot_detector import bot_detector
from src.identify_what_matters.political_pulse import political_pulse_engine
from collections import Counter

# Global demographic mental state counters across all analyzed comments
global_age_cohorts = Counter()
global_personas = Counter()
global_mental_states = Counter()
global_regions = Counter()
total_analyzed_comments = 0

cat_counts = {}
stage_counts = {}
stance_counts = {'supportive': 0, 'critical': 0, 'neutral': 0, 'humorous': 0}

def infer_topic_mental_state(t):
    brief = t.get('brief') or {}
    pool = brief.get('comment_pool', [])
    if not pool:
        return {
            'has_data': False,
            'cohort': 'Unsampled',
            'gender_persona': 'Mixed Public',
            'top_region': 'Pan-India',
            'mindset': 'General Public',
            'comment_count': 0,
            'age_distribution': {},
            'gender_distribution': {},
            'mental_distribution': {},
            'regions': []
        }
    
    a_cnt = Counter()
    g_cnt = Counter()
    l_cnt = Counter()
    m_cnt = Counter()

    age_m, gen_m, state_m = _get_models()
    
    for c in pool:
        txt = c.get('text', '')
        if not txt.strip():
            continue
        words = set(re.findall(r'[a-z]+', txt.lower()))
        
        # Gender Persona (Lexical + ML fallback)
        if words & FEMALE_CONVERSATIONAL:
            g_cnt['Female Persona'] += 1
            global_personas['Female-Skewed'] += 1
        elif words & MALE_CONVERSATIONAL:
            g_cnt['Male Persona'] += 1
            global_personas['Male-Skewed'] += 1
        elif gen_m is not None and len(txt) > 15:
            try:
                g_p = gen_m.predict([txt])[0]
                if g_p == 'female':
                    g_cnt['Female Persona'] += 1
                    global_personas['Female-Skewed'] += 1
                elif g_p == 'male':
                    g_cnt['Male Persona'] += 1
                    global_personas['Male-Skewed'] += 1
                else:
                    g_cnt['Neutral / Topic-Centered'] += 1
                    global_personas['Neutral / Topic-Centered'] += 1
            except Exception:
                g_cnt['Neutral / Topic-Centered'] += 1
                global_personas['Neutral / Topic-Centered'] += 1
        else:
            g_cnt['Neutral / Topic-Centered'] += 1
            global_personas['Neutral / Topic-Centered'] += 1
            
        # Age Cohort (Lexical + ML fallback)
        if words & GEN_Z_SLANG or any(em in txt for em in ['😂', '🤣', '💀', '🤡', '🔥', '😭', '☠️']):
            a_cnt['18-24 (Gen-Z / Youth)'] += 1
            global_age_cohorts['18-24 (Gen-Z / Youth)'] += 1
        elif words & PROFESSIONAL_SLANG or any(w in txt.lower() for w in ['tax', 'salary', 'job', 'corporate', 'market', 'stock', 'work']):
            a_cnt['25-34 (Working Prof.)'] += 1
            global_age_cohorts['25-34 (Working Prof.)'] += 1
        elif words & MATURE_SLANG or any(w in txt.lower() for w in ['sir', 'ji', 'nation', 'respect', 'desh', 'dharma', 'god', 'prayer']):
            a_cnt['35-54 (Civic / Family)'] += 1
            global_age_cohorts['35-54 (Civic / Family)'] += 1
        elif age_m is not None and len(txt) > 15:
            try:
                a_p = age_m.predict([txt])[0]
                label = f"{a_p} (ML Inferred)"
                a_cnt[label] += 1
                global_age_cohorts[label] += 1
            except Exception:
                a_cnt['25-34 (General Public)'] += 1
                global_age_cohorts['25-34 (General Public)'] += 1
        else:
            a_cnt['25-34 (General Public)'] += 1
            global_age_cohorts['25-34 (General Public)'] += 1
            
        # Psychological Archetype / Mental State (Trained 6-Class ML Classifier)
        if state_m is not None and len(txt) > 10:
            try:
                s_p = state_m.predict([txt])[0]
                m_cnt[str(s_p)] += 1
                global_mental_states[str(s_p)] += 1
            except Exception:
                m_cnt['General Public'] += 1
                global_mental_states['General Public'] += 1
        else:
            m_cnt['General Public'] += 1
            global_mental_states['General Public'] += 1
            
        st, _, _ = infer_location(txt)
        if st:
            l_cnt[st] += 1
            global_regions[st] += 1
            
    top_age = a_cnt.most_common(1)[0][0] if a_cnt else '25-34 (General Public)'
    top_gender = g_cnt.most_common(1)[0][0] if g_cnt else 'Mixed Public'
    top_loc = l_cnt.most_common(1)[0][0] if l_cnt else 'Pan-India'
    top_mindset = m_cnt.most_common(1)[0][0] if m_cnt else 'General Public'
    
    return {
        'has_data': True,
        'cohort': top_age,
        'gender_persona': top_gender,
        'top_region': top_loc,
        'mindset': top_mindset,
        'comment_count': len(pool),
        'age_distribution': {k: round(v/len(pool), 2) for k, v in a_cnt.items()},
        'gender_distribution': {k: round(v/len(pool), 2) for k, v in g_cnt.items()},
        'mental_distribution': {k: round(v/len(pool), 2) for k, v in m_cnt.items()},
        'regions': [k for k, _ in l_cnt.most_common(3)]
    }

LANGUAGE_MAP = {
    'en': 'English', 'hi': 'Hindi', 'ta': 'Tamil', 'te': 'Telugu',
    'ml': 'Malayalam', 'bn': 'Bengali', 'gu': 'Gujarati', 'mr': 'Marathi',
    'pa': 'Punjabi', 'kn': 'Kannada', 'ur': 'Urdu', 'or': 'Odia'
}

# Load collected Reddit discussions if available
reddit_signals = []
reddit_file = Path('output/reddit_signals.json')
if reddit_file.exists():
    try:
        with reddit_file.open(encoding='utf-8') as f:
            reddit_signals = json.load(f)
    except Exception:
        reddit_signals = []

# Pre-index raw signals by headline title for rapid provenance cross-referencing
signals_by_title = {}
for path in Path('output/runs').glob('*/signals.json'):
    try:
        with path.open(encoding='utf-8') as f:
            for s in json.load(f):
                stitle = s.get('title')
                if stitle and stitle not in signals_by_title:
                    signals_by_title[stitle] = s
    except Exception:
        pass

# Also index reddit signals in signals_by_title
for r in reddit_signals:
    rtitle = r.get('title')
    if rtitle and rtitle not in signals_by_title:
        signals_by_title[rtitle] = r

STOPWORDS = {"india", "delhi", "govt", "police", "news", "today", "after", "over", "with", "from", "that", "this", "first", "years", "time", "order", "case"}

topics_client = []
for idx, t in enumerate(topics):
    key = t.get('topic_key') or f"t{idx+1}"
    title = t.get('title', 'Untitled')
    cat = t.get('category', 'other')
    scores = t.get('scores') or {}
    breakout = t.get('breakout') or {}
    stage = breakout.get('stage', 'quiet')
    stage_score = breakout.get('score', 0.0)
    reasons = breakout.get('reasons', [])
    signals = breakout.get('signals', {})
    entities = t.get('entities') or []
    hashtags = t.get('hashtags') or []
    headlines = t.get('headlines') or []
    languages_raw = t.get('languages') or []
    language_labels = [LANGUAGE_MAP.get(l, l.upper()) for l in languages_raw]
    
    ig = t.get('instagram_metrics') or {}
    brief = t.get('brief')

    # Collect matched news/search articles for provenance
    matched_articles = []
    for h in headlines:
        if h in signals_by_title:
            art = signals_by_title[h]
            matched_articles.append({
                'title': art.get('title'),
                'publisher': art.get('publisher') or 'News Outlet',
                'url': art.get('url'),
                'source': art.get('feed') or art.get('source') or 'news'
            })

    # Match Reddit discussion signals
    matched_reddit = []
    t_ents = [e.lower() for e in entities if len(e) > 3 and e.lower() not in STOPWORDS]
    t_title_words = [w for w in re.findall(r"[a-z]+", title.lower()) if len(w) > 3 and w not in STOPWORDS]
    
    for r in reddit_signals:
        rtitle = r.get('title', '').lower()
        rwords = set(re.findall(r"[a-z]+", rtitle))
        entity_hit = any(e in rtitle for e in t_ents)
        word_hits = sum(1 for w in t_title_words if w in rwords)
        
        if entity_hit or word_hits >= 2:
            matched_reddit.append({
                'title': r.get('title'),
                'feed': r.get('feed'),
                'score': (r.get('engagement') or {}).get('score', 0) or 0,
                'comments': (r.get('engagement') or {}).get('comments', 0) or 0,
                'url': r.get('url')
            })

    matched_reddit.sort(key=lambda x: (x['score'] + x['comments']*2), reverse=True)
    reddit_comments_total = sum(r['comments'] for r in matched_reddit)
    
    # 1. Enriched Summary
    summary = brief.get('summary', '') if brief else ''
    if not summary:
        ents_str = ', '.join(entities[:3]) if entities else ''
        if matched_articles:
            best_art = max(matched_articles, key=lambda a: len(a.get('title', '')))
            lead_headline = best_art.get('title')
            pub = best_art.get('publisher') or 'major press wires'
            summary = f"{lead_headline}. Tracked via {pub} and multi-lingual regional coverage"
            summary += f", centering on {ents_str}." if ents_str else "."
        elif matched_reddit:
            lead_r = matched_reddit[0]
            summary = f"{lead_r['title']}. Active debate on {lead_r['feed']} with {lead_r['comments']} comments and {lead_r['score']} upvotes."
        elif headlines:
            summary = f"{headlines[0]}. Developing story circulating across national public search and news reporting."
        else:
            summary = f"Civic conversation and search activity around {title.lower()}, under observation across regional streams."

    # 2. Enriched Why Trending
    why_trending = brief.get('why_trending', '') if brief else ''
    if not why_trending:
        if matched_reddit and (matched_reddit[0]['comments'] > 15 or matched_reddit[0]['score'] > 200):
            lead_r = matched_reddit[0]
            why_trending = f"High engagement discussion thread in {lead_r['feed']} (⬆️ {lead_r['score']:,} upvotes, 💬 {lead_r['comments']:,} comments)."
        elif reasons:
            why_trending = '; '.join(reasons)
        elif scores.get('news_heat', 0) > 1.5:
            pub_cnt = t.get('publishers', len(matched_articles) or 2)
            lang_cnt = len(languages_raw) or 1
            why_trending = f"Multi-source news surge across {pub_cnt} publishers in {lang_cnt} regional language{'s' if lang_cnt > 1 else ''}."
        elif scores.get('search_heat', 0) > 1.5:
            ent_lead = entities[0] if entities else title
            why_trending = f"High search volume velocity and rapid public query surge around {ent_lead}."
        elif ig.get('on_topic_posts', 0) > 0:
            why_trending = f"Social engagement acceleration across verified community posts and discussion threads."
        else:
            why_trending = f"Sustained civic interest monitored across real-time regional indicators."

    stance = brief.get('stance', {}) if brief else {}

    what_people_say = brief.get('what_people_say', '') if brief else ''
    caution = brief.get('caution', '') if brief else ''
    debate = brief.get('debate', []) if brief else []
    measured_opinion = brief.get('measured_opinion') if brief else None

    # Aggregates
    cat_counts[cat] = cat_counts.get(cat, 0) + 1
    stage_counts[stage] = stage_counts.get(stage, 0) + 1
    if stance:
        top_st = max(stance.items(), key=lambda x: x[1])[0]
        stance_counts[top_st] = stance_counts.get(top_st, 0) + 1

    # Check if political/civic
    is_political = (cat in {'politics_governance', 'civic_issues'}) or bool(re.search(
      r'\b(?:modi|bjp|congress|election|electoral|minister|government|parliament|assembly|policy|protest|kejriwal|yogi|dmk|tvk)\b',
      title, re.I))
    has_probed_data = bool((brief or {}).get('comment_pool') or ig.get('on_topic_posts', 0) > 0)
    
    # Demographics and mental state proxy
    demo = infer_topic_mental_state(t)
    if not demo.get('has_data'):
        # Contextual default based on category, matched reddit, and language
        if matched_reddit:
            lead_feed = matched_reddit[0]['feed']
            if lead_feed == 'r/developersIndia':
                demo['cohort'] = '25-34 (Working Prof.)'
                demo['mindset'] = 'Career & Tech Insecurity / Opportunity'
            elif lead_feed == 'r/IndianStockMarket':
                demo['cohort'] = '25-34 (Working Prof.)'
                demo['mindset'] = 'Financial Anxiety & Macro Economics'
            elif lead_feed == 'r/indiasocial':
                demo['cohort'] = '18-24 (Gen-Z / Youth)'
                demo['mindset'] = 'Casual Lifestyle & Community Relatability'
            elif lead_feed == 'r/bollywood':
                demo['cohort'] = '18-24 (Gen-Z / Youth)'
                demo['mindset'] = 'Celebrity Fandom & Entertainment'
            elif lead_feed == 'r/Cricket':
                demo['cohort'] = '18-34 (Sports Enthusiasts)'
                demo['mindset'] = 'National Sporting Pride & Performance'
            elif lead_feed in ('r/india', 'r/IndiaSpeaks'):
                demo['cohort'] = '25-44 (Civic Activists & Observers)'
                demo['mindset'] = 'Civic Discourse & Institutional Skepticism'
        elif cat in ('cinema_tv_ott', 'music_celebrity', 'viral_memes_social'):
            demo['cohort'] = '18-24 (Gen-Z / Youth)'
            demo['mindset'] = 'Pop-Culture & Viral Trends'
        elif cat in ('economy_business', 'technology', 'jobs_education'):
            demo['cohort'] = '25-34 (Working Prof.)'
            demo['mindset'] = 'Career, Policy & Markets'
        elif cat in ('politics_governance', 'civic_issues'):
            demo['cohort'] = '35-54 (Civic / Family)'
            demo['mindset'] = 'Governance & Institutional Stakeholder'
        else:
            demo['cohort'] = '25-34 (General Public)'
            demo['mindset'] = 'General News Observant'

        if 'ml' in languages_raw:
            demo['top_region'] = 'Kerala'
        elif 'ta' in languages_raw:
            demo['top_region'] = 'Tamil Nadu'
        elif 'te' in languages_raw:
            demo['top_region'] = 'Andhra Pradesh / Telangana'
        elif 'bn' in languages_raw:
            demo['top_region'] = 'West Bengal'
        elif 'pa' in languages_raw:
            demo['top_region'] = 'Punjab'
        elif 'mr' in languages_raw:
            demo['top_region'] = 'Maharashtra'
        elif 'gu' in languages_raw:
            demo['top_region'] = 'Gujarat'
        else:
            demo['top_region'] = 'National (Pan-India)'

    if not demo.get('has_data'):
      demo.update(cohort='Unavailable', mindset='Unavailable', gender_persona='Unavailable', top_region='Unavailable')

    topics_client.append({
        'id': key,
        'index': idx,
        'title': title,
        'category': cat,
        'entities': entities[:6],
        'hashtags': hashtags[:6],
        'languages': language_labels,
        'publishers_count': t.get('publishers', len(matched_articles) or 1),
        'matched_articles': matched_articles[:4],
        'origin': t.get('origin', 'news_and_search').replace('_', ' ').title(),
        'first_seen': t.get('first_seen', '')[:16].replace('T', ' '),
        'trend_score': round(scores.get('trend_score', 0.0), 2),
        'instagram_heat': round(scores.get('instagram_heat', 0.0), 2),
        'news_heat': round(scores.get('news_heat', 0.0), 2),
        'search_heat': round(scores.get('search_heat', 0.0), 2),
        'debate_score': round(scores.get('debate_score', 0.0), 2),
        'stage': stage,
        'stage_score': round(stage_score, 2),
        'reasons': reasons,
        'signals': signals,
        'ig': {
            'on_topic_posts': ig.get('on_topic_posts', 0),
            'authors_7d': ig.get('authors_7d', 0),
            'comments_7d': ig.get('comments_7d', 0),
            'likes_7d': ig.get('likes_7d', 0),
            'engagement_per_hour': round(ig.get('engagement_per_hour', 0.0), 1),
            'replies_per_comment': round(ig.get('replies_per_comment', 0.0), 2),
            'top_posts': ig.get('top_posts', [])[:5],
            'hashtag_counts': ig.get('hashtag_counts', {}),
        },
        'stance': stance,
        'summary': summary,
        'what_people_say': what_people_say,
        'why_trending': why_trending,
        'caution': caution,
        'debate': debate,
        'measured_opinion': measured_opinion,
        'is_political': is_political,
        'political_evidence': audit_comments((brief or {}).get('comment_pool') or []),
        'bot_audit': bot_detector.analyze_topic_astroturf(t.get('title', ''), (brief or {}).get('comment_pool') or [], (brief or {}).get('debate', {}).get('sides', [])),
        'political_pulse': political_pulse_engine.evaluate_topic_pulse(t, (brief or {}).get('comment_pool') or [], matched_reddit),
        'has_probed_data': has_probed_data,
        'search_queries': t.get('search_queries', []),
        'demo': demo,
        'matched_reddit': matched_reddit[:4],
        'reddit_comments_total': reddit_comments_total,
    })

total_on_topic_posts = sum(t['ig']['on_topic_posts'] for t in topics_client)
total_comments = sum(t['ig']['comments_7d'] for t in topics_client)
total_likes = sum(t['ig']['likes_7d'] for t in topics_client)
total_reddit_signals = len(reddit_signals)
total_reddit_comments = sum((r.get('engagement') or {}).get('comments', 0) or 0 for r in reddit_signals)
total_probed_topics = sum(1 for t in topics_client if t['has_probed_data'])
total_political_probed = sum(1 for t in topics_client if t['is_political'] and t['has_probed_data'])
total_political_unprobed = sum(1 for t in topics_client if t['is_political'] and not t['has_probed_data'])
political_comments = [comment for index, topic in enumerate(topics_client) if topic['is_political']
            for comment in ((topics[index].get('brief') or {}).get('comment_pool') or [])]
political_audit = audit_comments(political_comments)
global_bot_audit = bot_detector.analyze_topic_astroturf("All Political Comments", political_comments)

# Calculate Macro Political Pulse Across All Political Topics
macro_bjp_points = 0.0
macro_opp_points = 0.0
macro_neu_points = 0.0
community_matrix = defaultdict(lambda: {'bjp': 0.0, 'opp': 0.0, 'neu': 0.0, 'count': 0})

for t in topics_client:
    if t['is_political']:
        p = t.get('political_pulse') or {}
        b = p.get('bjp_favor_pct', 0.0)
        o = p.get('opp_favor_pct', 0.0)
        n = p.get('neutral_pct', 100.0)
        w = max(t.get('trend_score', 1.0), 0.5)
        macro_bjp_points += b * w
        macro_opp_points += o * w
        macro_neu_points += n * w
        c_tab = p.get('community_cross_tab') or {}
        for comm, vals in c_tab.items():
            community_matrix[comm]['bjp'] += vals.get('bjp_favor_pct', 0) * vals.get('volume', 1)
            community_matrix[comm]['opp'] += vals.get('opp_favor_pct', 0) * vals.get('volume', 1)
            community_matrix[comm]['neu'] += vals.get('neutral_pct', 0) * vals.get('volume', 1)
            community_matrix[comm]['count'] += vals.get('volume', 1)

total_macro_points = macro_bjp_points + macro_opp_points + macro_neu_points or 1.0
macro_bjp_pct = round((macro_bjp_points / total_macro_points) * 100, 1)
macro_opp_pct = round((macro_opp_points / total_macro_points) * 100, 1)
macro_neu_pct = round((macro_neu_points / total_macro_points) * 100, 1)

community_table_rows = []
for comm, stats in sorted(community_matrix.items(), key=lambda x: -x[1]['count']):
    tot = stats['bjp'] + stats['opp'] + stats['neu'] or 1.0
    b_p = round((stats['bjp'] / tot) * 100, 1)
    o_p = round((stats['opp'] / tot) * 100, 1)
    n_p = round((stats['neu'] / tot) * 100, 1)
    bar = f'''<div style="display:flex;height:8px;border-radius:4px;overflow:hidden;background:#e2e8f0;margin-top:4px;">
      <div style="width:{b_p}%;background:#f97316;" title="BJP {b_p}%"></div>
      <div style="width:{o_p}%;background:#3b82f6;" title="Opposition {o_p}%"></div>
      <div style="width:{n_p}%;background:#94a3b8;" title="Neutral {n_p}%"></div>
    </div>'''
    community_table_rows.append(
        f'<tr><td><strong>{html.escape(comm)}</strong></td>'
        f'<td><span style="color:#c2410c;font-weight:600;">{b_p}%</span></td>'
        f'<td><span style="color:#1d4ed8;font-weight:600;">{o_p}%</span></td>'
        f'<td><span style="color:#64748b;">{n_p}%</span></td>'
        f'<td>{bar}</td>'
        f'<td><span class="mono" style="font-size:11px;color:var(--text-dim);">{stats["count"]} signals</span></td></tr>'
    )
community_cross_tab_html = ''.join(community_table_rows) if community_table_rows else '<tr><td colspan="6">No demographic lenses identified yet.</td></tr>'

# Format Bot Astroturf Clusters Table
bot_cluster_rows = []
for cl in global_bot_audit.get('template_clusters', []):
    acc_badge = f'<span class="chip mono" style="background:#fef2f2;border-color:#fecaca;color:#991b1b;">{cl["unique_accounts"]} accounts</span>'
    bot_cluster_rows.append(
        f'<tr><td><div style="font-size:12.5px;font-style:italic;">"{html.escape(cl["template"])}"</div></td>'
        f'<td>{acc_badge}</td>'
        f'<td><span class="mono">{cl["occurrences"]}x</span></td>'
        f'<td><span class="chip mono" style="background:#fff7ed;color:#c2410c;border-color:#fed7aa;">{html.escape(cl["suspicion_level"].upper())}</span></td></tr>'
    )
bot_cluster_table_html = ''.join(bot_cluster_rows) if bot_cluster_rows else '<tr><td colspan="4">No suspicious cross-account copy-paste templates identified.</td></tr>'

party_rows = ''.join(
  f'<tr><td>{html.escape(party)}</td><td>{counts.get("favor", 0)}</td><td>{counts.get("oppose", 0)}</td></tr>'
  for party, counts in political_audit['party_positions'].items()
)
party_table = (
  '<table class="evidence-table"><thead><tr><th>Explicit party target</th><th>Favor</th><th>Oppose</th></tr></thead>'
  f'<tbody>{party_rows}</tbody></table>' if party_rows else
  '<p>No explicit, unambiguous party-directed positions in the sampled comments. Party support is unknown.</p>'
)

history_points = []
for _, history_path, history_report in report_records:
  generated = str(history_report.get('generated_at') or history_report.get('run_id') or history_path.parent.name)
  history_points.append({
    'label': generated[:16].replace('T', ' '),
    'signals': history_report.get('signals', 0),
    'topics': len(history_report.get('topics', [])),
  })

topics_json_str = json.dumps(topics_client, ensure_ascii=False)
panel_pulse_json_str = json.dumps(panel_pulse, ensure_ascii=False)
history_json_str = json.dumps(history_points, ensure_ascii=False)
reddit_signals_json_str = json.dumps(reddit_signals, ensure_ascii=False)

html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Civic Pulse | India Conversation Desk</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&family=Newsreader:opsz,wght@6..72,400;6..72,500;6..72,600&display=swap" rel="stylesheet">
  <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
  <style>
    :root {{
      --bg: #f1f5f2;
      --surface: #ffffff;
      --surface-2: #e9efec;
      --surface-3: #dbe5e0;
      --border: #d4dfda;
      --border-focus: #7c978e;
      --text: #182a27;
      --text-dim: #536761;
      --text-faint: #7f928c;
      
      --accent: #087f76;
      --accent-hover: #06675f;
      --accent-light: #e4f2ee;
      --accent-border: #acd4ca;
      
      --pos: #24846d;
      --pos-bg: #e0f0e9;
      --pos-border: #b9d9ca;
      
      --neg: #c84f45;
      --neg-bg: #f8e8e4;
      --neg-border: #edc5bc;
      
      --neu: #71847e;
      --neu-bg: #e9efec;
      --neu-border: #d4dfda;
      
      --warn: #b77b23;
      --warn-bg: #f6edda;
      --warn-border: #e8d6ac;
      
      --purple: #5c6684;
      --purple-bg: #eaecf2;
      --purple-border: #d4d9e4;

      --mono: 'IBM Plex Mono', ui-monospace, Menlo, Consolas, monospace;
      --display: 'Newsreader', Georgia, serif;
    }}

    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    
    body {{
      background: var(--bg);
      color: var(--text);
      font-family: 'DM Sans', sans-serif;
      font-size: 13.5px;
      line-height: 1.5;
      -webkit-font-smoothing: antialiased;
      padding-bottom: 80px;
    }}

    .mono {{
      font-family: var(--mono);
      font-variant-numeric: tabular-nums;
    }}

    .container {{
      max-width: 1440px;
      margin: 0 auto;
      padding: 0 28px;
    }}

    /* Header & Navigation */
    header {{
      position: sticky;
      top: 0;
      z-index: 50;
      background: rgba(248, 250, 248, 0.97);
      border-bottom: 1px solid var(--border);
      border-top: 3px solid var(--accent);
    }}

    .header-inner {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      min-height: 72px;
      gap: 24px;
    }}

    .brand {{
      display: flex;
      align-items: center;
      gap: 10px;
      text-decoration: none;
      color: var(--text);
    }}

    .brand-icon {{
      width: 36px;
      height: 36px;
      background: var(--accent);
      color: #ffffff;
      border-radius: 4px;
      display: grid;
      place-items: center;
      font-family: var(--mono);
      font-weight: 700;
      font-size: 11px;
    }}

    .brand-title {{
      font-family: var(--display);
      font-size: 21px;
      font-weight: 600;
      line-height: 1;
      letter-spacing: 0;
    }}

    .nav-tabs {{
      display: flex;
      gap: 20px;
      background: transparent;
      padding: 0;
      border-radius: 0;
      border: 0;
    }}

    .nav-tab {{
      padding: 10px 2px 9px;
      border-radius: 0;
      border-bottom: 2px solid transparent;
      font-size: 12px;
      font-weight: 600;
      color: var(--text-dim);
      background: transparent;
      border: none;
      cursor: pointer;
      transition: color 0.15s ease, border-color 0.15s ease;
      display: flex;
      align-items: center;
      gap: 6px;
    }}

    .nav-tab:hover {{
      color: var(--text);
    }}

    .nav-tab.active {{
      color: var(--accent);
      border-bottom-color: var(--accent);
    }}

    .header-meta {{
      display: flex;
      align-items: center;
      gap: 8px;
      font-size: 12px;
    }}

    .chip {{
      display: inline-flex;
      align-items: center;
      gap: 6px;
      padding: 4px 8px;
      border-radius: 4px;
      border: 1px solid var(--border);
      background: #ffffff;
      font-size: 11.5px;
      font-weight: 500;
      color: var(--text-dim);
    }}

    .dot {{
      width: 7px;
      height: 7px;
      border-radius: 50%;
      background: var(--warn);
    }}
    .dot.pulse {{
      animation: pulse 2s infinite;
    }}
    @keyframes pulse {{
      0% {{ transform: scale(0.95); box-shadow: 0 0 0 0 rgba(22, 163, 74, 0.7); }}
      70% {{ transform: scale(1); box-shadow: 0 0 0 5px rgba(22, 163, 74, 0); }}
      100% {{ transform: scale(0.95); box-shadow: 0 0 0 0 rgba(22, 163, 74, 0); }}
    }}

    /* Global KPI Cards */
    .kpi-row {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
      gap: 14px;
      margin: 18px 0 22px;
      gap: 0;
      border-block: 1px solid var(--border);
    }}

    .kpi-card {{
      background: transparent;
      border: 0;
      border-right: 1px solid var(--border);
      border-radius: 0;
      padding: 13px 18px;
      box-shadow: none;
    }}

    .kpi-card:last-child {{ border-right: 0; }}

    .kpi-label {{
      font-size: 10px;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0;
      color: var(--text-dim);
      display: flex;
      align-items: center;
      justify-content: space-between;
    }}

    .kpi-value {{
      font-size: 27px;
      font-weight: 700;
      color: var(--text);
      margin-top: 6px;
      letter-spacing: 0;
    }}

    .kpi-subtext {{
      font-size: 11.5px;
      color: var(--text-dim);
      margin-top: 4px;
    }}

    /* Controls Bar */
    .controls-panel {{
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 5px;
      padding: 14px 18px;
      margin-bottom: 20px;
      display: flex;
      flex-wrap: wrap;
      align-items: center;
      justify-content: space-between;
      gap: 14px;
      box-shadow: none;
    }}

    .filter-group {{
      display: flex;
      align-items: center;
      gap: 10px;
      flex-wrap: wrap;
    }}

    .filter-label {{
      font-size: 11px;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      color: var(--text-faint);
    }}

    .search-input {{
      padding: 7px 12px;
      border-radius: 7px;
      border: 1px solid var(--border);
      background: var(--bg);
      font-size: 13px;
      color: var(--text);
      outline: none;
      width: 260px;
      transition: all 0.15s ease;
    }}
    .search-input:focus {{
      border-color: var(--accent);
      background: #ffffff;
      box-shadow: 0 0 0 2px var(--accent-border);
    }}

    .select-dropdown {{
      padding: 6px 12px;
      border-radius: 7px;
      border: 1px solid var(--border);
      background: #ffffff;
      font-size: 12.5px;
      color: var(--text);
      outline: none;
      cursor: pointer;
    }}
    .select-dropdown:focus {{
      border-color: var(--accent);
    }}

    .btn-group {{
      display: flex;
      background: var(--surface-2);
      border: 1px solid var(--border);
      border-radius: 7px;
      padding: 2px;
    }}

    .btn-group-item {{
      border: none;
      background: transparent;
      padding: 5px 11px;
      border-radius: 5px;
      font-size: 12px;
      font-weight: 500;
      color: var(--text-dim);
      cursor: pointer;
      transition: all 0.12s;
    }}
    .btn-group-item.active {{
      background: #ffffff;
      color: var(--text);
      font-weight: 600;
      box-shadow: 0 1px 2px rgba(0,0,0,0.05);
    }}

    /* Card Component */
    .card {{
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 5px;
      overflow: hidden;
      box-shadow: none;
    }}

    .card-header {{
      padding: 14px 18px;
      border-bottom: 1px solid var(--border);
      display: flex;
      align-items: center;
      justify-content: space-between;
      background: var(--surface);
    }}

    .card-title {{
      font-size: 14px;
      font-weight: 600;
      letter-spacing: -0.01em;
      color: var(--text);
      display: flex;
      align-items: center;
      gap: 8px;
    }}

    .card-body {{
      padding: 16px 18px;
    }}

    /* SentimentBar, DeltaBar, StagePills */
    .sentiment-bar {{
      display: flex;
      height: 7px;
      width: 100%;
      border-radius: 9999px;
      overflow: hidden;
      background: var(--surface-2);
    }}
    .sentiment-seg {{
      height: 100%;
      transition: width 0.3s;
    }}
    .sentiment-seg.supportive {{ background: var(--pos); }}
    .sentiment-seg.critical {{ background: var(--neg); }}
    .sentiment-seg.neutral {{ background: var(--neu); }}
    .sentiment-seg.humorous {{ background: var(--warn); }}

    .stage-pill {{
      display: inline-flex;
      align-items: center;
      gap: 5px;
      padding: 2.5px 8px;
      border-radius: 5px;
      font-size: 11px;
      font-weight: 600;
      text-transform: capitalize;
    }}
    .stage-pill.saturated {{ background: var(--neg-bg); color: var(--neg); border: 1px solid var(--neg-border); }}
    .stage-pill.breaking_out {{ background: var(--accent-light); color: var(--accent); border: 1px solid var(--accent-border); }}
    .stage-pill.peak {{ background: var(--warn-bg); color: var(--warn); border: 1px solid var(--warn-border); }}
    .stage-pill.simmering {{ background: #f7ecd9; color: #996213; border: 1px solid #e8d4a8; }}
    .stage-pill.pre_breakout {{ background: var(--pos-bg); color: var(--pos); border: 1px solid var(--pos-border); }}
    .stage-pill.settled {{ background: #f8fafc; color: #475569; border: 1px solid #e2e8f0; }}
    .stage-pill.quiet {{ background: var(--surface-2); color: var(--text-faint); border: 1px solid var(--border); }}

    .cat-pill {{
      display: inline-flex;
      align-items: center;
      padding: 2px 7px;
      border-radius: 4px;
      font-size: 10.5px;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.04em;
      background: var(--surface-2);
      color: var(--text-dim);
      border: 1px solid var(--border);
    }}

    .chart-grid-2 {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(480px, 1fr));
      gap: 16px;
      margin-bottom: 20px;
    }}

    .chart-grid-3 {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(340px, 1fr));
      gap: 16px;
      margin-bottom: 20px;
    }}

    /* Topic Feed */
    .topic-feed {{
      display: flex;
      flex-direction: column;
      gap: 10px;
    }}

    .topic-card {{
      background: #ffffff;
      border: 1px solid var(--border);
      border-radius: 9px;
      transition: all 0.15s ease;
      overflow: hidden;
    }}

    .topic-card:hover {{
      border-color: var(--border-focus);
      box-shadow: 0 2px 6px rgba(0,0,0,0.04);
    }}

    .topic-summary-row {{
      padding: 12px 16px;
      display: grid;
      grid-template-columns: auto 1fr auto auto auto 36px;
      align-items: center;
      gap: 14px;
      cursor: pointer;
      user-select: none;
    }}

    .topic-id-badge {{
      font-size: 11px;
      font-family: var(--mono);
      font-weight: 600;
      color: var(--text-faint);
      width: 32px;
    }}

    .topic-title-box {{
      min-width: 0;
    }}

    .topic-title {{
      font-size: 13.5px;
      font-weight: 600;
      color: var(--text);
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }}

    .topic-meta-line {{
      display: flex;
      align-items: center;
      gap: 8px;
      margin-top: 3px;
      font-size: 11.5px;
      color: var(--text-dim);
    }}

    .topic-metrics-cell {{
      display: flex;
      align-items: center;
      gap: 14px;
      font-size: 12px;
      font-family: var(--mono);
      color: var(--text-dim);
    }}

    .metric-item {{
      display: flex;
      flex-direction: column;
      align-items: flex-end;
    }}
    .metric-val {{
      font-weight: 600;
      color: var(--text);
    }}
    .metric-sub {{
      font-size: 10px;
      color: var(--text-faint);
    }}

    .topic-sentiment-col {{
      width: 110px;
      display: flex;
      flex-direction: column;
      gap: 3px;
    }}

    .expand-chevron {{
      display: grid;
      place-items: center;
      width: 24px;
      height: 24px;
      border-radius: 4px;
      color: var(--text-faint);
      transition: transform 0.2s;
    }}
    .topic-card.open .expand-chevron {{
      transform: rotate(180deg);
      color: var(--accent);
    }}

    /* Drawer */
    .topic-detail-drawer {{
      display: none;
      padding: 16px 20px 20px 20px;
      border-top: 1px solid var(--border);
      background: #fafbfc;
    }}
    .topic-card.open .topic-detail-drawer {{
      display: block;
    }}

    .drawer-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
      gap: 14px;
      margin-bottom: 8px;
    }}

    .drawer-section {{
      background: #ffffff;
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 12px 14px;
    }}

    .drawer-sec-title {{
      font-size: 11px;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      color: var(--text-dim);
      margin-bottom: 10px;
      display: flex;
      align-items: center;
      justify-content: space-between;
    }}

    .signal-list {{
      display: flex;
      flex-direction: column;
      gap: 5px;
    }}
    .signal-item {{
      font-size: 12px;
      display: flex;
      align-items: center;
      gap: 6px;
      color: var(--text);
    }}
    .signal-bullet {{
      width: 5px;
      height: 5px;
      border-radius: 50%;
      background: var(--accent);
      flex-shrink: 0;
    }}

    .side-card {{
      background: var(--surface-2);
      border-radius: 6px;
      padding: 8px 10px;
      margin-top: 8px;
      border-left: 3px solid var(--accent);
    }}
    .side-header {{
      display: flex;
      justify-content: space-between;
      font-size: 11px;
      font-weight: 600;
      color: var(--text-dim);
      margin-bottom: 4px;
    }}
    .side-text {{
      font-size: 12px;
      color: var(--text);
      line-height: 1.4;
    }}
    .side-demographics {{
      margin-top: 5px;
      font-size: 11px;
      font-family: var(--mono);
      color: var(--text-dim);
      display: flex;
      gap: 10px;
    }}

    .evidence-table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 12px;
      margin-top: 6px;
    }}
    .evidence-table th {{
      text-align: left;
      padding: 6px 8px;
      font-size: 10.5px;
      text-transform: uppercase;
      color: var(--text-faint);
      border-bottom: 1px solid var(--border);
    }}
    .evidence-table td {{
      padding: 6px 8px;
      border-bottom: 1px solid var(--surface-2);
      color: var(--text);
    }}
    .action-queue-scroll {{ overflow-x: auto; }}
    .action-queue-scroll .evidence-table {{ min-width: 780px; }}
    .action-queue-scroll .evidence-table td:nth-child(2) {{ min-width: 260px; }}
    .action-queue-scroll .evidence-table td:nth-child(3) {{ min-width: 200px; }}

    .pagination-bar {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 14px 4px;
      font-size: 12.5px;
      color: var(--text-dim);
    }}

    @media (max-width: 900px) {{
      .header-inner {{ height: auto; flex-wrap: wrap; padding-top: 10px; }}
      .nav-tabs {{ order: 3; width: 100%; overflow-x: auto; gap: 18px; }}
      .nav-tab {{ white-space: nowrap; }}
      .header-meta {{ margin-left: auto; }}
      .kpi-row {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
      .kpi-card:nth-child(2) {{ border-right: 0; }}
      .kpi-card:nth-child(-n+2) {{ border-bottom: 1px solid var(--border); }}
      .topic-summary-row {{
        grid-template-columns: 1fr auto 30px;
        gap: 8px;
      }}
      .topic-metrics-cell, .topic-sentiment-col {{
        display: none;
      }}
      .chart-grid-2, .chart-grid-3 {{
        grid-template-columns: 1fr;
      }}
    }}

    .edition-head {{
      display: flex;
      align-items: end;
      justify-content: space-between;
      gap: 20px;
      padding: 12px 0 15px;
      border-bottom: 1px solid var(--text);
    }}
    .edition-kicker {{ color: var(--accent); font: 600 10px var(--mono); text-transform: uppercase; }}
    .edition-title {{ margin-top: 2px; font: 500 30px/1.1 var(--display); color: var(--text); }}
    .edition-meta {{ text-align: right; font-size: 11px; color: var(--text-dim); }}
    .edition-meta span {{ display: block; font: 500 9px var(--mono); color: var(--text-faint); text-transform: uppercase; }}
    .edition-meta strong {{ display: block; margin-top: 3px; font: 500 11px var(--mono); color: var(--text); }}

    @media (max-width: 540px) {{
      .container {{ padding: 0 15px; }}
      .header-meta .chip:first-child {{ display: none; }}
      .nav-tabs {{ gap: 16px; }}
      .nav-tab {{ font-size: 11px; }}
      .edition-head {{ align-items: start; }}
      .edition-title {{ font-size: 26px; }}
      .edition-meta {{ padding-top: 3px; }}
      .kpi-card {{ padding: 11px 10px; }}
      .kpi-value {{ font-size: 22px; }}
      .action-queue-scroll .evidence-table {{ min-width: 0; display: block; }}
      .action-queue-scroll thead {{ display: none; }}
      .action-queue-scroll tbody {{ display: block; }}
      .action-queue-scroll tr {{ display: grid; grid-template-columns: minmax(0, 1fr) auto; gap: 4px 12px; padding: 12px 0; border-bottom: 1px solid var(--border); }}
      .action-queue-scroll .evidence-table td {{ border: 0; padding: 2px 0; min-width: 0; }}
      .action-queue-scroll td:nth-child(1) {{ grid-column: 1 / -1; grid-row: 1; font-size: 11px; color: var(--accent); }}
      .action-queue-scroll td:nth-child(2) {{ grid-column: 1 / -1; grid-row: 2; font-weight: 600; overflow-wrap: anywhere; }}
      .action-queue-scroll td:nth-child(3) {{ grid-column: 1 / -1; grid-row: 3; }}
      .action-queue-scroll td:nth-child(4) {{ grid-column: 1; grid-row: 4; }}
      .action-queue-scroll td:nth-child(5) {{ grid-column: 2; grid-row: 4; }}
    }}
  </style>
</head>
<body>

  <!-- Header -->
  <header>
    <div class="container header-inner">
      <div class="brand">
        <div class="brand-icon">CP</div>
        <div>
          <div class="brand-title">Civic Pulse</div>
          <div style="font: 500 9px var(--mono); color: var(--text-faint);">INDIA / CONVERSATION DESK</div>
        </div>
      </div>

      <div class="nav-tabs">
        <button class="nav-tab active" onclick="switchTab('overview')">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="3" width="7" height="7"></rect><rect x="14" y="3" width="7" height="7"></rect><rect x="14" y="14" width="7" height="7"></rect><rect x="3" y="14" width="7" height="7"></rect></svg>
          Overview
        </button>
        <button class="nav-tab" onclick="switchTab('topics')">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><line x1="8" y1="6" x2="21" y2="6"></line><line x1="8" y1="12" x2="21" y2="12"></line><line x1="8" y1="18" x2="21" y2="18"></line><line x1="3" y1="6" x2="3.01" y2="6"></line><line x1="3" y1="12" x2="3.01" y2="12"></line><line x1="3" y1="18" x2="3.01" y2="18"></line></svg>
          All Topics ({len(topics_client)})
        </button>
        <button class="nav-tab" onclick="switchTab('political')">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M4 15s1-1 4-1 5 2 8 2 4-1 4-1V3s-1 1-4 1-5-2-8-2-4 1-4 1z"></path><line x1="4" y1="22" x2="4" y2="15"></line></svg>
          Civic & Political ({total_political_probed + total_political_unprobed})
        </button>
        <button class="nav-tab" onclick="switchTab('panel')">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2"></path><circle cx="9" cy="7" r="4"></circle><path d="M23 21v-2a4 4 0 0 0-3-3.87"></path><path d="M16 3.13a4 4 0 0 1 0 7.75"></path></svg>
          Panel Pulse ({len(panel_pulse)})
        </button>
        <button class="nav-tab" onclick="switchTab('reddit')">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"></circle><path d="M8 12a2 2 0 1 0 4 0 2 2 0 1 0-4 0"></path><path d="M12 8a2 2 0 1 0 4 0 2 2 0 1 0-4 0"></path><path d="M12 16a2 2 0 1 0 4 0 2 2 0 1 0-4 0"></path></svg>
          Reddit ({total_reddit_signals})
        </button>
      </div>

      <div class="header-meta">
        <span class="chip">
          <span class="dot"></span>
          Report {run_id}
        </span>
        <span class="chip mono">{len(topics_client)} Topics</span>
      </div>
    </div>
  </header>

  <div class="container" style="margin-top: 18px;">

    <div class="edition-head">
      <div>
        <div class="edition-kicker">India / public conversation</div>
        <h1 class="edition-title">Conversation desk</h1>
      </div>
      <div class="edition-meta"><span>Report generated</span><strong>{generated_at[:16].replace('T', ' ')} UTC</strong></div>
    </div>

    <!-- KPI Summary Row -->
    <div class="kpi-row">
      <div class="kpi-card">
        <div class="kpi-label">
          <span>Topics tracked</span>
        </div>
        <div class="kpi-value mono">{len(topics_client)}</div>
        <div class="kpi-subtext"><b>{total_probed_topics}</b> measured on social &middot; <b>{len(topics_client) - total_probed_topics}</b> news/search</div>
      </div>

      <div class="kpi-card">
        <div class="kpi-label">
          <span>Civic & political</span>
        </div>
        <div class="kpi-value mono">{total_political_probed + total_political_unprobed}</div>
        <div class="kpi-subtext"><b>{total_political_probed}</b> deep probed &middot; <b>{total_political_unprobed}</b> emerging in news</div>
      </div>

      <div class="kpi-card">
        <div class="kpi-label">
          <span>Social & Community</span>
        </div>
        <div class="kpi-value mono">{total_comments + total_reddit_comments:,}</div>
        <div class="kpi-subtext"><b>{total_reddit_signals}</b> Reddit threads &middot; <b>{total_on_topic_posts}</b> Instagram posts</div>
      </div>

      <div class="kpi-card">
        <div class="kpi-label">
          <span>Inferred Mental State</span>
        </div>
        <div class="kpi-value mono" style="font-size: 20px;">{global_age_cohorts.most_common(1)[0][0].split('(')[0].strip() if global_age_cohorts else 'General'}</div>
        <div class="kpi-subtext"><b>{sum(global_age_cohorts.values()):,}</b> comments profiled &middot; <b>{global_mental_states.most_common(1)[0][0] if global_mental_states else 'General'}</b></div>
      </div>
    </div>

    <!-- TAB 1: OVERVIEW & CHARTS -->
    <div id="tab-overview" class="tab-content">
      
      <!-- Chart Row 1: Report History & Breakout Distribution -->
      <div class="chart-grid-2">
        <div class="card">
          <div class="card-header">
            <div class="card-title">
              <span>Signals across reports</span>
            </div>
            <span class="chip mono">{len(history_points)} report snapshots</span>
          </div>
          <div class="card-body">
            <canvas id="chart-volume" height="180"></canvas>
          </div>
        </div>

        <div class="card">
          <div class="card-header">
            <div class="card-title">
              <span>Breakout stage</span>
            </div>
            <span class="chip mono">{sum(stage_counts.values())} Total Topics</span>
          </div>
          <div class="card-body">
            <canvas id="chart-stages" height="180"></canvas>
          </div>
        </div>
      </div>

      <!-- Chart Row 2: Data Availability & Category Breakdown -->
      <div class="chart-grid-3">
        <div class="card">
          <div class="card-header">
            <div class="card-title">
              <span>Mental State & Cohort Proxy</span>
            </div>
            <span class="chip mono">{sum(global_age_cohorts.values())} comments</span>
          </div>
          <div class="card-body">
            <canvas id="chart-demographics" height="200"></canvas>
          </div>
        </div>

        <div class="card">
          <div class="card-header">
            <div class="card-title">
              <span>Stance signals</span>
            </div>
          </div>
          <div class="card-body">
            <canvas id="chart-stance" height="200"></canvas>
          </div>
        </div>

        <div class="card">
          <div class="card-header">
            <div class="card-title">
              <span>Topic mix</span>
            </div>
          </div>
          <div class="card-body">
            <canvas id="chart-categories" height="200"></canvas>
          </div>
        </div>
      </div>

      {collection_section_html}

      <!-- Top 10 Breakout Topics Spotlight -->
      <div class="card" style="margin-bottom: 24px;">
        <div class="card-header">
          <div class="card-title">
            <span>Topics drawing attention</span>
          </div>
          <button class="btn-group-item active" onclick="switchTab('topics')">View All {len(topics_client)} Topics &rarr;</button>
        </div>
        <div class="card-body" style="padding: 0;">
          <div id="top-breakouts-list" class="topic-feed" style="gap: 0;"></div>
        </div>
      </div>
    </div>

    <!-- TAB 2: ALL TOPICS ACCORDION FEED -->
    <div id="tab-topics" class="tab-content" style="display: none;">
      
      <!-- Interactive Filter Controls -->
      <div class="controls-panel">
        <div class="filter-group">
          <span class="filter-label">Search</span>
          <input type="text" id="topic-search" class="search-input" placeholder="Search title, keywords, names..." oninput="applyFilters()">
        </div>

        <div class="filter-group">
          <span class="filter-label">Category</span>
          <select id="cat-select" class="select-dropdown" onchange="applyFilters()">
            <option value="all">All Categories ({len(topics_client)})</option>
            <option value="politics_governance">Politics & Governance ({cat_counts.get('politics_governance', 0)})</option>
            <option value="crime_law">Crime & Law ({cat_counts.get('crime_law', 0)})</option>
            <option value="civic_issues">Civic Issues ({cat_counts.get('civic_issues', 0)})</option>
            <option value="economy_business">Economy & Business ({cat_counts.get('economy_business', 0)})</option>
            <option value="sports">Sports ({cat_counts.get('sports', 0)})</option>
            <option value="cinema_tv_ott">Cinema & OTT ({cat_counts.get('cinema_tv_ott', 0)})</option>
            <option value="technology">Technology ({cat_counts.get('technology', 0)})</option>
            <option value="viral_memes_social">Viral & Social ({cat_counts.get('viral_memes_social', 0)})</option>
          </select>
        </div>

        <div class="filter-group">
          <span class="filter-label">Stage</span>
          <select id="stage-select" class="select-dropdown" onchange="applyFilters()">
            <option value="all">All Stages</option>
            <option value="saturated">Saturated</option>
            <option value="breaking_out">Breaking Out</option>
            <option value="peak">Peak</option>
            <option value="simmering">Simmering</option>
            <option value="pre_breakout">Pre-Breakout</option>
            <option value="quiet">Quiet / Emerging</option>
          </select>
        </div>

        <div class="filter-group">
          <span class="filter-label">Sort</span>
          <div class="btn-group">
            <button class="btn-group-item active" onclick="setSort('trend_score', this)">Trend Score</button>
            <button class="btn-group-item" onclick="setSort('comments', this)">Comments</button>
            <button class="btn-group-item" onclick="setSort('heat', this)">IG Heat</button>
            <button class="btn-group-item" onclick="setSort('debate', this)">Debate</button>
          </div>
        </div>
      </div>

      <!-- Feed Container -->
      <div id="topics-feed" class="topic-feed"></div>

      <div class="pagination-bar">
        <span id="results-count">Showing 0 of {len(topics_client)} topics</span>
        <div style="font-size: 11.5px; color: var(--text-faint);">Click any row to expand Breakout Signals, Stance, Demographics, and Evidence</div>
      </div>
    </div>

    <!-- TAB 3: POLITICAL & CIVIC ISSUES -->
    <div id="tab-political" class="tab-content" style="display: none;">
      
      <!-- MACRO POLITICAL PULSE & PARTY FAVORABILITY -->
      <div class="card" style="margin-bottom: 20px; border-left: 4px solid #f97316;">
        <div class="card-header">
          <div class="card-title">⚖️ Political Pulse Check & Multi-Party Favorability Matrix</div>
          <span class="chip mono" style="background:#fff7ed;color:#c2410c;border-color:#fed7aa;">
            Macro Alignment: BJP {macro_bjp_pct}% vs Opp {macro_opp_pct}%
          </span>
        </div>
        <div class="card-body">
          <p style="font-size:13px; color:var(--text-dim); margin-bottom: 14px;">
            Continuous quantitative tracking of public sentiment, comment arguments, and news coverage evaluating favourability toward the <strong>Ruling Government / BJP (NDA)</strong>, the <strong>Opposition (INDIA / Regional Parties)</strong>, or <strong>Neutral / Civic Non-partisan issues</strong>.
          </p>

          <!-- Macro Bar Visual -->
          <div style="background:var(--surface-2); padding:16px; border-radius:8px; margin-bottom:18px;">
            <div style="display:flex; justify-content:space-between; margin-bottom:8px; font-weight:600; font-size:13px;">
              <span style="color:#c2410c;">🟠 Favoring BJP / NDA: {macro_bjp_pct}%</span>
              <span style="color:#64748b;">⚪ Neutral / Non-partisan: {macro_neu_pct}%</span>
              <span style="color:#1d4ed8;">🔵 Favoring Opposition / INDIA: {macro_opp_pct}%</span>
            </div>
            <div style="display:flex; height:14px; border-radius:7px; overflow:hidden; background:#e2e8f0;">
              <div style="width:{macro_bjp_pct}%; background:#f97316;" title="BJP {macro_bjp_pct}%"></div>
              <div style="width:{macro_neu_pct}%; background:#94a3b8;" title="Neutral {macro_neu_pct}%"></div>
              <div style="width:{macro_opp_pct}%; background:#3b82f6;" title="Opposition {macro_opp_pct}%"></div>
            </div>
          </div>

          <!-- Cross-tabulation Table across socio-religious communities -->
          <div style="margin-top: 15px;">
            <div style="font-weight:600; font-size:13px; margin-bottom:8px; color:var(--text);">Socio-Religious & Community Lens Cross-Tabulation:</div>
            <div class="action-queue-scroll">
              <table class="evidence-table">
                <thead>
                  <tr>
                    <th>Community / Demographic Lens</th>
                    <th>BJP / NDA Favor</th>
                    <th>Opposition Favor</th>
                    <th>Neutral / Non-partisan</th>
                    <th style="min-width: 140px;">Distribution</th>
                    <th>Sample Volume</th>
                  </tr>
                </thead>
                <tbody>
                  {community_cross_tab_html}
                </tbody>
              </table>
            </div>
            <p style="font-size:11.5px; color:var(--text-dim); margin-top:6px;">
              * Grounded in contextual discourse, verified vernacular terms, and matched community subreddit threads (e.g., r/IndiaSpeaks nationalist perspective vs r/india reformist perspective).
            </p>
          </div>
        </div>
      </div>

      <!-- BOT & ASTROTURF COORDINATION RADAR -->
      <div class="card" style="margin-bottom: 20px; border-left: 4px solid #ef4444;">
        <div class="card-header">
          <div class="card-title">🤖 Bot, Astroturf & Inauthentic Coordination Radar</div>
          <span class="chip mono" style="background:#fef2f2;color:#991b1b;border-color:#fecaca;">
            Astroturf Index: {global_bot_audit['astroturf_index']}/100 &middot; ~{global_bot_audit['bot_share_pct']}% Synthetic Accounts
          </span>
        </div>
        <div class="card-body">
          <p style="font-size:13px; color:var(--text-dim); margin-bottom: 12px;">
            Multi-signal ML heuristic detecting inorganic debate manipulation: identifies accounts posting identical verbatim comment scripts across multiple IDs, zero-post throwaway profiles, canned campaign slogan clusters, and sudden comment bursts.
          </p>

          <div style="display:grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 12px; margin-bottom: 16px;">
            <div style="background:var(--surface-2); padding:12px; border-radius:6px; text-align:center;">
              <div style="font-size:11px; color:var(--text-dim); text-transform:uppercase;">Identified Bot Comments</div>
              <div style="font-size:20px; font-weight:700; color:#ef4444;" class="mono">{global_bot_audit['bot_comment_count']} / {global_bot_audit['total_comments_audited']}</div>
            </div>
            <div style="background:var(--surface-2); padding:12px; border-radius:6px; text-align:center;">
              <div style="font-size:11px; color:var(--text-dim); text-transform:uppercase;">Cross-Account Clusters</div>
              <div style="font-size:20px; font-weight:700; color:#f97316;" class="mono">{len(global_bot_audit.get('template_clusters', []))}</div>
            </div>
            <div style="background:var(--surface-2); padding:12px; border-radius:6px; text-align:center;">
              <div style="font-size:11px; color:var(--text-dim); text-transform:uppercase;">Coordination Status</div>
              <div style="font-size:14px; font-weight:700; color:#991b1b; margin-top:4px;" class="mono">{global_bot_audit['coordination_status'].upper()}</div>
            </div>
          </div>

          <div style="font-weight:600; font-size:13px; margin-bottom:8px;">Suspicious Repeated Narrative Templates Across Distinct Handles:</div>
          <div class="action-queue-scroll">
            <table class="evidence-table">
              <thead>
                <tr>
                  <th>Repeated Script / Copy-Paste Template</th>
                  <th>Unique Handles</th>
                  <th>Occurrences</th>
                  <th>Suspicion Gate</th>
                </tr>
              </thead>
              <tbody>
                {bot_cluster_table_html}
              </tbody>
            </table>
          </div>
          <p style="font-size:11.5px; color:var(--text-dim); margin-top:6px;">
            Verdict: {html.escape(global_bot_audit['verdict_summary'])}
          </p>
        </div>
      </div>

      <!-- RADAR BANNER -->
      <div class="card" style="margin-bottom: 20px; background: #faf5ff; border-color: #e9d5ff;">
        <div class="card-body" style="padding: 14px 18px; display: flex; align-items: center; justify-content: space-between;">
          <div>
            <div style="font-weight: 600; color: #6b21a8; font-size: 14px;">🏛️ Dedicated Political, Electoral & Civic Intelligence Feed</div>
            <div style="font-size: 12.5px; color: #7e22ce; margin-top: 2px;">
              Live intelligence feed displaying per-topic BJP vs Opposition alignment, bot concentration scores, and debate conflict.
            </div>
          </div>
          <span class="chip mono" style="background: #ffffff; border-color: #d8b4fe; color: #6b21a8;">
            {total_political_probed + total_political_unprobed} Topics Tracked
          </span>
        </div>
      </div>

      <div class="controls-panel">
        <div class="filter-group">
          <span class="filter-label">Filter Section</span>
          <div class="btn-group">
            <button class="btn-group-item active" onclick="filterPoliticalTab('all', this)">All Political ({total_political_probed + total_political_unprobed})</button>
            <button class="btn-group-item" onclick="filterPoliticalTab('probed', this)">Instagram/comment evidence ({total_political_probed})</button>
            <button class="btn-group-item" onclick="filterPoliticalTab('unprobed', this)">Headlines/Reddit titles only ({total_political_unprobed})</button>
          </div>
        </div>
      </div>

      <div id="political-feed" class="topic-feed"></div>
    </div>

    <!-- TAB 4: PANEL PULSE -->
    <div id="tab-panel" class="tab-content" style="display: none;">
      <div class="card" style="margin-bottom: 20px; background: #f8fafc; border-color: #e2e8f0;">
        <div class="card-body" style="padding: 14px 18px; display: flex; align-items: center; justify-content: space-between;">
          <div>
            <div style="font-weight: 600; color: var(--text); font-size: 14px;">📡 High-Impact Influencer & Media Sentinel Panel</div>
            <div style="font-size: 12.5px; color: var(--text-dim); margin-top: 2px;">
              Tracking {len(panel_cfg_accounts)} curated political commentators, investigative journalists, satire creators, and news networks. Posts are scored against each handle's running 30-day engagement baseline to detect anomalous spikes.
            </div>
          </div>
          <span class="chip mono" style="background: var(--surface); color: var(--accent); border-color: var(--border);">
            {len(panel_pulse)} Spiking Posts Active
          </span>
        </div>
      </div>

      <!-- Panel Filters -->
      <div class="controls-panel">
        <div class="filter-group">
          <span class="filter-label">Search Influencer / Post</span>
          <input type="text" id="panel-search" class="search-input" placeholder="Search by @handle, caption text, comment..." oninput="renderPanelTable()">
        </div>

        <div class="filter-group">
          <span class="filter-label">Page Group</span>
          <select id="panel-group-select" class="select-dropdown" onchange="renderPanelTable()">
            <option value="all">All Groups ({len(panel_pulse)})</option>
            <option value="creator_commentary">Independent Creators & Commentators</option>
            <option value="political_commentary">Political Commentary & Ideological Desks</option>
            <option value="political_satire">Political Satire & Memes</option>
            <option value="factcheck_civic">Fact-Check & Civic Watchdogs</option>
            <option value="news_hindi">Hindi News Broadcasters</option>
            <option value="news_english">National English Media</option>
            <option value="news_regional">Regional State Broadcasters</option>
            <option value="memes_commentary">Youth Culture & Social Commentary</option>
          </select>
        </div>

        <div class="filter-group">
          <span class="filter-label">Sort By</span>
          <div class="btn-group">
            <button class="btn-group-item active" onclick="setPanelSort('heat', this)">Heat</button>
            <button class="btn-group-item" onclick="setPanelSort('comments', this)">Comments</button>
            <button class="btn-group-item" onclick="setPanelSort('ratio_lift', this)">Ratio Lift</button>
            <button class="btn-group-item" onclick="setPanelSort('likes', this)">Likes</button>
          </div>
        </div>
      </div>

      <div class="card">
        <div class="card-header">
          <div class="card-title">
            <span>📡 Emerging & Spiking Posts Across Sentinel Accounts</span>
          </div>
          <span class="chip mono" id="panel-filtered-count">{len(panel_pulse)} Posts Showing</span>
        </div>
        <div class="card-body" style="padding: 0;">
          <table class="evidence-table" style="margin: 0;">
            <thead>
              <tr style="background: var(--surface-2);">
                <th style="padding-left: 18px;">Account Handle & Spiking Post</th>
                <th>Group / Category</th>
                <th>Post Age</th>
                <th>Comments (vs Usual)</th>
                <th>Likes</th>
                <th>Comment:Like Lift</th>
                <th>Anomalous Heat</th>
                <th style="padding-right: 18px;">Top Public Comment</th>
              </tr>
            </thead>
            <tbody id="panel-table-body">
            </tbody>
          </table>
        </div>
      </div>
    </div>

    <!-- TAB 5: REDDIT DISCUSSIONS -->
    <div id="tab-reddit" class="tab-content" style="display: none;">
      <div class="card" style="margin-bottom: 20px; background: #fff7ed; border-color: #ffedd5;">
        <div class="card-body" style="padding: 14px 18px; display: flex; align-items: center; justify-content: space-between;">
          <div>
            <div style="font-weight: 600; color: #c2410c; font-size: 14px;">💬 Indian Community Reddit Discourse Streams</div>
            <div style="font-size: 12.5px; color: #9a3412; margin-top: 2px;">
              Live discussions mined from major Indian subreddits: r/india, r/IndiaSpeaks, r/developersIndia, r/IndianStockMarket, r/Cricket, r/indiasocial, and r/bollywood.
            </div>
          </div>
          <span class="chip mono" style="background: #ffffff; border-color: #fed7aa; color: #c2410c;">
            {total_reddit_signals} Threads &middot; {total_reddit_comments:,} Comments
          </span>
        </div>
      </div>

      <!-- Reddit Subreddit Filters -->
      <div class="controls-panel">
        <div class="filter-group">
          <span class="filter-label">Search Discussions</span>
          <input type="text" id="reddit-search" class="search-input" placeholder="Search Reddit posts, topics, keywords..." oninput="renderRedditFeed()">
        </div>

        <div class="filter-group">
          <span class="filter-label">Subreddit</span>
          <select id="reddit-sub-select" class="select-dropdown" onchange="renderRedditFeed()">
            <option value="all">All Subreddits ({total_reddit_signals})</option>
            <option value="r/india">r/india</option>
            <option value="r/IndiaSpeaks">r/IndiaSpeaks</option>
            <option value="r/developersIndia">r/developersIndia (Tech & Careers)</option>
            <option value="r/IndianStockMarket">r/IndianStockMarket (Finance & Economy)</option>
            <option value="r/Cricket">r/Cricket (Sports)</option>
            <option value="r/indiasocial">r/indiasocial (Culture & Casual)</option>
            <option value="r/bollywood">r/bollywood (Cinema & OTT)</option>
          </select>
        </div>

        <div class="filter-group">
          <span class="filter-label">Sort By</span>
          <div class="btn-group">
            <button class="btn-group-item active" onclick="setRedditSort('score', this)">Upvotes</button>
            <button class="btn-group-item" onclick="setRedditSort('comments', this)">Comments</button>
            <button class="btn-group-item" onclick="setRedditSort('newest', this)">Newest</button>
          </div>
        </div>
      </div>

      <div id="reddit-feed" class="topic-feed"></div>
    </div>

  </div>

  <script>
    const TOPICS = {topics_json_str};
    const PANEL = {panel_pulse_json_str};
    const REPORT_HISTORY = {history_json_str};
    const REDDIT_POSTS = {reddit_signals_json_str};

    let currentTab = 'overview';
    let currentSort = 'trend_score';
    let currentRedditSort = 'score';
    let politicalSubFilter = 'all';

    function switchTab(tabId) {{
      currentTab = tabId;
      document.querySelectorAll('.tab-content').forEach(el => el.style.display = 'none');
      document.querySelectorAll('.nav-tab').forEach(el => el.classList.remove('active'));
      
      document.getElementById('tab-' + tabId).style.display = 'block';
      event?.currentTarget?.classList?.add('active');

      if (tabId === 'topics') {{
        applyFilters();
      }} else if (tabId === 'political') {{
        renderPoliticalFeed();
      }} else if (tabId === 'panel') {{
        renderPanelTable();
      }} else if (tabId === 'reddit') {{
        renderRedditFeed();
      }}
    }}

    function setSort(sortKey, btn) {{
      currentSort = sortKey;
      document.querySelectorAll('.btn-group-item').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      applyFilters();
    }}

    function filterPoliticalTab(type, btn) {{
      politicalSubFilter = type;
      btn.parentElement.querySelectorAll('.btn-group-item').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      renderPoliticalFeed();
    }}

    function toggleCard(rowEl) {{
      const cardEl = rowEl.closest(".topic-card");
      cardEl.classList.toggle('open');
    }}

    // Render Compact Card HTML
    function buildTopicCardHtml(t, prefix="") {{
      const st = t.stance || {{}};
      const sup = Math.round((st.supportive || 0) * 100);
      const crit = Math.round((st.critical || 0) * 100);
      const neu = Math.round((st.neutral || 0) * 100);
      const hum = Math.round((st.humorous || 0) * 100);
      const hasStance = (sup + crit + neu + hum) > 0;

      const sides = (t.measured_opinion && t.measured_opinion.sides) || [];

      return `
        <div class="topic-card" id="card-${{t.id}}">
          <div class="topic-summary-row" onclick="toggleCard(document.getElementById('card-${{t.id}}'))">
            <span class="topic-id-badge mono">${{t.id}}</span>
            
            <div class="topic-title-box">
              <div class="topic-title" title="${{t.title}}">${{t.title}}</div>
              <div class="topic-meta-line">
                <span class="cat-pill">${{t.category.replace(/_/g, ' ')}}</span>
                <span class="stage-pill ${{t.stage}}">${{t.stage.replace(/_/g, ' ')}}</span>
                ${{t.is_political ? '<span class="cat-pill" style="background:#fef2f2; color:#b91c1c; border-color:#fecaca;">🏛️ Political</span>' : ''}}
                ${{t.matched_reddit && t.matched_reddit.length > 0 ? '<span class="cat-pill" style="background:#fff7ed; color:#c2410c; border-color:#ffedd5;">💬 Reddit (' + t.matched_reddit.length + ')</span>' : ''}}
                <span>&bull;</span>
                <span class="mono" style="font-size:11px; color:var(--text-faint);">${{t.ig.on_topic_posts}} verified posts</span>
              </div>
            </div>

            <div class="topic-metrics-cell">
              <div class="metric-item">
                <span class="metric-val mono">${{t.trend_score}}</span>
                <span class="metric-sub">Trend Score</span>
              </div>
              <div class="metric-item">
                <span class="metric-val mono">${{t.ig.comments_7d.toLocaleString()}}</span>
                <span class="metric-sub">Comments</span>
              </div>
              <div class="metric-item">
                <span class="metric-val mono">${{t.ig.engagement_per_hour > 0 ? t.ig.engagement_per_hour.toLocaleString() : '-'}}/h</span>
                <span class="metric-sub">Velocity</span>
              </div>
            </div>

            <div class="topic-sentiment-col">
              <div class="sentiment-bar">
                ${{hasStance ? `
                  <div class="sentiment-seg supportive" style="width: ${{sup}}%;" title="Supportive: ${{sup}}%"></div>
                  <div class="sentiment-seg critical" style="width: ${{crit}}%;" title="Critical: ${{crit}}%"></div>
                  <div class="sentiment-seg humorous" style="width: ${{hum}}%;" title="Humorous: ${{hum}}%"></div>
                  <div class="sentiment-seg neutral" style="width: ${{neu}}%;" title="Neutral: ${{neu}}%"></div>
                ` : `
                  <div class="sentiment-seg neutral" style="width: 100%;" title="Pending social probe"></div>
                `}}
              </div>
              <div style="display:flex; justify-content:space-between; font-size:10px; font-family:var(--mono); color:var(--text-faint);">
                <span>${{hasStance ? '+' + sup + '%' : 'Pending'}}</span>
                <span>${{hasStance ? '-' + crit + '%' : ''}}</span>
              </div>
            </div>

            <div class="expand-chevron">
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><polyline points="6 9 12 15 18 9"></polyline></svg>
            </div>
          </div>

          <!-- Expanded Detail Drawer -->
          <div class="topic-detail-drawer">
            <div class="drawer-grid">
              
              <!-- 1. Breakout Signals & Acceleration -->
              <div class="drawer-section">
                <div class="drawer-sec-title">
                  <span>⚡ 1. Breakout Signals & Velocity</span>
                  <span class="stage-pill ${{t.stage}}">${{t.stage}}</span>
                </div>
                ${{t.summary ? `<p style="font-size: 12px; color: var(--text-dim); margin-bottom: 10px; line-height: 1.45;">${{t.summary}}</p>` : ''}}
                <div class="signal-list">
                  ${{t.reasons.length > 0 ? t.reasons.map(r => `
                    <div class="signal-item">
                      <span class="signal-bullet"></span>
                      <span>${{r}}</span>
                    </div>
                  `).join('') : '<div style="font-size:11.5px; color:var(--text-faint);">No anomalous surge signals triggered yet.</div>'}}
                </div>
                ${{t.why_trending ? `
                  <div style="margin-top: 10px; padding: 6px 8px; background: var(--surface-2); border-radius: 5px; font-size: 11.5px; color: var(--text-dim);">
                    <b>Why Trending:</b> ${{t.why_trending}}
                  </div>
                ` : ''}}
              </div>

              <!-- 2. Stance & Debate Positions -->
              <div class="drawer-section">
                <div class="drawer-sec-title">
                  <span>⚖️ 2. Tone toward story (not party preference)</span>
                  <span class="mono" style="font-size: 10.5px;">${{t.ig.comments_7d}} comments</span>
                </div>
                <div style="margin-bottom: 8px;">
                  ${{!hasStance ? '<div style="font-size:11px; color:var(--text-faint);">No measured comment stance for this topic.</div>' : ''}}
                  <div class="sentiment-bar" style="height: 10px;">
                    <div class="sentiment-seg supportive" style="width: ${{sup}}%;" title="Supportive: ${{sup}}%"></div>
                    <div class="sentiment-seg critical" style="width: ${{crit}}%;" title="Critical: ${{crit}}%"></div>
                    <div class="sentiment-seg humorous" style="width: ${{hum}}%;" title="Humorous: ${{hum}}%"></div>
                    <div class="sentiment-seg neutral" style="width: ${{neu}}%;" title="Neutral: ${{neu}}%"></div>
                  </div>
                  <div style="display:flex; justify-content:space-between; font-size:11px; margin-top:4px; font-family:var(--mono);">
                    ${{hasStance ? `<span style="color:var(--pos);">Support: ${{sup}}%</span>
                    <span style="color:var(--neg);">Critical: ${{crit}}%</span>
                    <span style="color:var(--warn);">Humor: ${{hum}}%</span>
                    <span style="color:var(--neu);">Neutral: ${{neu}}%</span>` : ''}}
                  </div>
                </div>

                ${{sides.length > 0 ? `
                  <div style="margin-top: 10px;">
                    <div style="font-size: 10.5px; font-weight:700; color:var(--text-faint); text-transform:uppercase;">Debate Positions:</div>
                    ${{sides.map((s, idx) => `
                      <div class="side-card">
                        <div class="side-header">
                          <span>Side #${{idx+1}}</span>
                          <span>Position</span>
                        </div>
                        <div class="side-text">${{s.position}}</div>
                      </div>
                    `).join('')}}
                  </div>
                ` : '<div style="font-size:11.5px; color:var(--text-faint); margin-top:8px;">General social conversation without factional polarization.</div>'}}
              </div>

              <!-- 3. Demographic & Mental State Profile -->
              <div class="drawer-section">
                <div class="drawer-sec-title">
                  <span>👥 3. Inferred Demographics & Mental State</span>
                  <span class="mono" style="font-size: 10.5px;">${{t.demo && t.demo.has_data ? t.demo.comment_count + ' comments' : 'Proxy Model'}}</span>
                </div>
                ${{t.demo && t.demo.has_data ? `
                  <div style="font-size: 12px; color: var(--text-dim); margin-bottom: 8px;">
                    <div><b>Age Cohort Proxy:</b> <span class="mono" style="color:var(--text);">${{t.demo.cohort}}</span></div>
                    <div style="margin-top: 3px;"><b>Mental State / Mindset:</b> <span class="mono" style="color:var(--text);">${{t.demo.mindset}}</span></div>
                    <div style="margin-top: 3px;"><b>Conversational Persona:</b> <span class="mono" style="color:var(--text);">${{t.demo.gender_persona}}</span></div>
                    <div style="margin-top: 3px;"><b>Regional / Linguistic Focus:</b> <span class="mono" style="color:var(--text);">${{t.demo.top_region}}</span></div>
                  </div>
                  <div style="font-size: 11px; color: var(--text-faint); line-height: 1.4; padding: 6px 8px; background: var(--surface-2); border-radius: 4px;">
                    Inferred from user activity, slang markers, script expressions, and conversational discourse patterns.
                  </div>
                ` : `
                  <p style="font-size: 12px; color: var(--text-dim);">No active comment sample available for individual discourse profiling on this topic. General public news/search interest applies.</p>
                `}}

                <!-- Political Pulse & Bot Audit in Drawer -->
                ${{t.political_pulse ? `
                  <div style="margin-top: 12px; padding-top: 10px; border-top: 1px dashed var(--border);">
                    <div style="font-size: 11px; font-weight:700; color:var(--text); margin-bottom: 4px;">⚖️ Political Party Favorability & Lens:</div>
                    <div style="font-size: 12px; margin-bottom: 6px;">
                      <span class="chip mono" style="background:#fff7ed; color:#c2410c; border-color:#fed7aa;">${{t.political_pulse.overall_leaning}}</span>
                      <span class="mono" style="font-size:11px; margin-left:6px;">BJP: <b>${{t.political_pulse.bjp_favor_pct}}%</b> &middot; Opp: <b>${{t.political_pulse.opp_favor_pct}}%</b></span>
                    </div>
                    <div style="font-size: 11.5px; color: var(--text-dim);">${{t.political_pulse.narrative_summary}}</div>
                  </div>
                ` : ''}}

                ${{t.bot_audit && t.bot_audit.total_comments_audited > 0 ? `
                  <div style="margin-top: 10px; padding: 8px; background: #fef2f2; border-radius: 5px; border: 1px solid #fee2e2;">
                    <div style="display:flex; justify-content:space-between; align-items:center;">
                      <span style="font-size:11px; font-weight:700; color:#991b1b;">🤖 Bot / Synthetic Manipulation Risk:</span>
                      <span class="chip mono" style="font-size:10px; background:#fff; color:#991b1b; border-color:#fecaca;">
                        Index: ${{t.bot_audit.astroturf_index}}/100
                      </span>
                    </div>
                    <div style="font-size: 11px; color:#b91c1c; margin-top:3px;">
                      ~${{t.bot_audit.bot_share_pct}}% synthetic / bot accounts &middot; ${{t.bot_audit.template_clusters.length}} copy-paste clusters
                    </div>
                  </div>
                ` : ''}}
              </div>

              <!-- 4. Evidence & Source Provenance -->
              <div class="drawer-section">
                <div class="drawer-sec-title">
                  <span>🔗 4. Multi-Source Evidence & Provenance</span>
                  <span class="mono" style="font-size: 10.5px;">${{t.matched_articles.length + (t.matched_reddit ? t.matched_reddit.length : 0) + t.ig.top_posts.length}} Sources</span>
                </div>
                
                <table class="evidence-table">
                  <tbody>
                    <tr>
                      <td style="color: var(--text-faint); width: 110px;">Primary Origin</td>
                      <td class="mono"><b>${{t.origin}}</b></td>
                    </tr>
                    <tr>
                      <td style="color: var(--text-faint);">Coverage Scope</td>
                      <td class="mono"><b>${{t.publishers_count}}</b> publishers &middot; ${{t.languages.length > 0 ? t.languages.join(', ') : 'National'}}</td>
                    </tr>
                    ${{t.entities.length > 0 ? `
                    <tr>
                      <td style="color: var(--text-faint);">Key Entities</td>
                      <td>
                        <div style="display:flex; flex-wrap:wrap; gap:4px;">
                          ${{t.entities.map(e => `<span class="cat-pill" style="font-size:10px; padding:1px 6px;">${{e}}</span>`).join('')}}
                        </div>
                      </td>
                    </tr>
                    ` : ''}}
                    ${{t.matched_reddit && t.matched_reddit.length > 0 ? `
                    <tr>
                      <td style="color: var(--text-faint);"><span style="color:#ea580c; font-weight:600;">Reddit</span> (${{t.matched_reddit.length}})</td>
                      <td>
                        ${{t.matched_reddit.map(r => `
                          <div style="margin-bottom: 6px; padding: 4px 6px; background: #fff7ed; border-radius: 4px; border: 1px solid #ffedd5;">
                            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 2px;">
                              <span class="mono" style="font-size: 10.5px; font-weight: 700; color: #c2410c;">${{r.feed}}</span>
                              <span class="mono" style="font-size: 10px; color: #9a3412;">⬆️ ${{r.score}} &middot; 💬 ${{r.comments}} comments</span>
                            </div>
                            <a href="${{r.url || '#'}}" target="_blank" rel="noopener noreferrer" style="color: #431407; text-decoration: none; font-size: 11.5px; font-weight: 500; display: block; line-height: 1.3;">
                              ${{r.title}} &nearr;
                            </a>
                          </div>
                        `).join('')}}
                      </td>
                    </tr>
                    ` : ''}}
                    ${{t.matched_articles.length > 0 ? `
                    <tr>
                      <td style="color: var(--text-faint);">Reporting Outlets</td>
                      <td>
                        ${{t.matched_articles.map((art, i) => `
                          <div style="margin-bottom: 4px;">
                            <a href="${{art.url || '#'}}" target="_blank" rel="noopener noreferrer" style="color: var(--accent); text-decoration: none; font-size: 11.5px; font-weight: 500;">
                              ${{art.publisher}} &nearr;
                            </a>
                            <div style="font-size: 11px; color: var(--text-dim); line-height: 1.3;">${{art.title}}</div>
                          </div>
                        `).join('')}}
                      </td>
                    </tr>
                    ` : ''}}
                    ${{t.ig.top_posts.length > 0 ? `
                    <tr>
                      <td style="color: var(--text-faint);">Social Posts</td>
                      <td>
                        ${{t.ig.top_posts.map((url, i) => `
                          <div><a href="${{url}}" target="_blank" rel="noopener noreferrer" style="color: var(--accent); text-decoration: none; font-size: 11.5px;">Instagram Post #${{i+1}} &nearr;</a></div>
                        `).join('')}}
                      </td>
                    </tr>
                    ` : ''}}
                    ${{t.matched_articles.length === 0 && (!t.matched_reddit || t.matched_reddit.length === 0) && t.ig.top_posts.length === 0 ? `
                    <tr>
                      <td style="color: var(--text-faint);">Verification</td>
                      <td>
                        <a href="https://news.google.com/search?q=${{encodeURIComponent(t.title)}}" target="_blank" rel="noopener noreferrer" style="color: var(--accent); text-decoration: none; font-size: 11.5px;">
                          Search Wire Records &nearr;
                        </a>
                      </td>
                    </tr>
                    ` : ''}}
                  </tbody>
                </table>
              </div>

            </div>
          </div>
        </div>
      `;
    }}

    function applyFilters() {{
      const search = document.getElementById('topic-search').value.toLowerCase();
      const cat = document.getElementById('cat-select').value;
      const stage = document.getElementById('stage-select').value;

      let filtered = TOPICS.filter(t => {{
        if (cat !== 'all' && t.category !== cat) return false;
        if (stage !== 'all' && t.stage !== stage) return false;
        if (search) {{
          const matchTitle = t.title.toLowerCase().includes(search);
          const matchId = t.id.toLowerCase().includes(search);
          const matchSum = (t.summary || '').toLowerCase().includes(search);
          if (!matchTitle && !matchId && !matchSum) return false;
        }}
        return true;
      }});

      filtered.sort((a, b) => {{
        if (currentSort === 'trend_score') return b.trend_score - a.trend_score;
        if (currentSort === 'comments') return b.ig.comments_7d - a.ig.comments_7d;
        if (currentSort === 'heat') return b.instagram_heat - a.instagram_heat;
        if (currentSort === 'debate') return b.debate_score - a.debate_score;
        return 0;
      }});

      document.getElementById('topics-feed').innerHTML = filtered.map(buildTopicCardHtml).join('');
      document.getElementById('results-count').innerText = `Showing ${{filtered.length}} of ${{TOPICS.length}} topics`;
    }}

    function renderPoliticalFeed() {{
      let filtered = TOPICS.filter(t => t.is_political);
      if (politicalSubFilter === 'probed') {{
        filtered = filtered.filter(t => t.has_probed_data);
      }} else if (politicalSubFilter === 'unprobed') {{
        filtered = filtered.filter(t => !t.has_probed_data);
      }}
      filtered.sort((a, b) => b.trend_score - a.trend_score);
      document.getElementById('political-feed').innerHTML = filtered.map(buildTopicCardHtml).join('');
    }}

    let currentPanelSort = 'heat';

    function setPanelSort(sortKey, btn) {{
      currentPanelSort = sortKey;
      btn.parentElement.querySelectorAll('.btn-group-item').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      renderPanelTable();
    }}

    function renderPanelTable() {{
      const search = (document.getElementById('panel-search')?.value || '').toLowerCase();
      const group = document.getElementById('panel-group-select')?.value || 'all';

      let filtered = PANEL.filter(p => {{
        if (group !== 'all' && p.group !== group) return false;
        if (search) {{
          const acc = (p.account || '').toLowerCase();
          const cap = (p.caption || '').toLowerCase();
          const top = (p.top_comment || '').toLowerCase();
          if (!acc.includes(search) && !cap.includes(search) && !top.includes(search)) return false;
        }}
        return true;
      }});

      filtered.sort((a, b) => {{
        if (currentPanelSort === 'heat') return (b.heat || 0) - (a.heat || 0);
        if (currentPanelSort === 'comments') return (b.comments || 0) - (a.comments || 0);
        if (currentPanelSort === 'ratio_lift') return (b.ratio_lift || 0) - (a.ratio_lift || 0);
        if (currentPanelSort === 'likes') return (b.likes || 0) - (a.likes || 0);
        return 0;
      }});

      const countEl = document.getElementById('panel-filtered-count');
      if (countEl) countEl.innerText = `${{filtered.length}} Posts Showing`;

      const tbody = document.getElementById('panel-table-body');
      if (!tbody) return;

      tbody.innerHTML = filtered.map(p => `
        <tr>
          <td style="padding-left: 18px; font-weight: 600; max-width: 300px;">
            <a href="${{p.url || ('https://instagram.com/' + p.account)}}" target="_blank" rel="noopener noreferrer" style="color:var(--accent); text-decoration:none; font-size:13.5px;">
              @${{p.account}} &nearr;
            </a>
            <div style="font-size:11.5px; color:var(--text); margin-top:3px; line-height:1.35; max-height:48px; overflow:hidden; text-overflow:ellipsis;">
              ${{p.caption || 'No caption available'}}
            </div>
          </td>
          <td>
            <span class="cat-pill" style="font-size:10.5px;">${{p.group ? p.group.replace(/_/g, ' ') : 'news'}}</span>
          </td>
          <td class="mono" style="font-size:12px; color:var(--text-faint);">${{p.age_hours}}h ago</td>
          <td class="mono">
            <b>${{(p.comments || 0).toLocaleString()}}</b>
            <div style="font-size:10.5px; color:${{p.over_comments >= 5 ? 'var(--neg)' : 'var(--text-dim)'}};">
              ${{p.over_comments ? p.over_comments.toFixed(1) + 'x avg' : '-'}}
            </div>
          </td>
          <td class="mono" style="font-size:12px;">${{(p.likes || 0).toLocaleString()}}</td>
          <td class="mono">
            <span class="chip" style="font-size:10.5px; font-weight:600; color:${{p.ratio_lift >= 2.0 ? '#c2410c' : 'inherit'}}; background:${{p.ratio_lift >= 2.0 ? '#fff7ed' : 'var(--surface-2)'}};">
              ${{p.ratio_lift ? p.ratio_lift.toFixed(1) + 'x' : '-'}}
            </span>
          </td>
          <td class="mono">
            <span class="chip" style="font-weight:700; color:#b91c1c; background:#fef2f2; border-color:#fecaca;">
              ${{p.heat ? p.heat.toFixed(2) : '-'}} 🔥
            </span>
          </td>
          <td style="padding-right: 18px; max-width: 320px;">
            ${{p.top_comment ? `
              <div style="font-size:11.5px; color:var(--text-dim); line-height:1.35; background:var(--surface-2); padding:5px 8px; border-radius:4px;">
                "${{p.top_comment}}"
                <div class="mono" style="font-size:10px; color:var(--text-faint); margin-top:2px;">
                  💬 ${{p.top_comment_likes ? p.top_comment_likes.toLocaleString() + ' likes' : 'Top comment'}}
                </div>
              </div>
            ` : '<span style="font-size:11px; color:var(--text-faint);">No comment sample</span>'}}
          </td>
        </tr>
      `).join('');
    }}

    function setRedditSort(sortKey, btn) {{
      currentRedditSort = sortKey;
      btn.parentElement.querySelectorAll('.btn-group-item').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      renderRedditFeed();
    }}

    function renderRedditFeed() {{
      const search = (document.getElementById('reddit-search')?.value || '').toLowerCase();
      const sub = document.getElementById('reddit-sub-select')?.value || 'all';

      let filtered = REDDIT_POSTS.filter(p => {{
        if (sub !== 'all' && p.feed !== sub) return false;
        if (search) {{
          const t = (p.title || '').toLowerCase();
          const f = (p.feed || '').toLowerCase();
          if (!t.includes(search) && !f.includes(search)) return false;
        }}
        return true;
      }});

      filtered.sort((a, b) => {{
        const scoreA = (a.engagement && a.engagement.score) || 0;
        const scoreB = (b.engagement && b.engagement.score) || 0;
        const commA = (a.engagement && a.engagement.comments) || 0;
        const commB = (b.engagement && b.engagement.comments) || 0;

        if (currentRedditSort === 'score') return scoreB - scoreA;
        if (currentRedditSort === 'comments') return commB - commA;
        if (currentRedditSort === 'newest') return (b.published_at || '').localeCompare(a.published_at || '');
        return 0;
      }});

      const feedEl = document.getElementById('reddit-feed');
      if (!feedEl) return;

      feedEl.innerHTML = filtered.map((p, idx) => `
        <div class="topic-card" style="margin-bottom: 8px;">
          <div class="topic-summary-row" style="cursor: default;">
            <span class="topic-id-badge mono" style="background:#fff7ed; color:#c2410c; border-color:#ffedd5;">#${{idx+1}}</span>
            
            <div class="topic-title-box">
              <a href="${{p.url || '#'}}" target="_blank" rel="noopener noreferrer" style="font-size: 14.5px; font-weight: 600; color: var(--text); text-decoration: none; line-height: 1.35; display: inline-block;">
                ${{p.title}} &nearr;
              </a>
              <div class="topic-meta-line" style="margin-top: 4px;">
                <span class="cat-pill" style="background: #fff7ed; color: #c2410c; border-color: #fed7aa; font-weight: 600;">${{p.feed}}</span>
                <span class="mono" style="font-size: 11px; color: var(--text-faint);">${{p.published_at ? p.published_at.substring(0, 16).replace('T', ' ') : ''}} UTC</span>
              </div>
            </div>

            <div class="topic-metrics-cell" style="gap: 16px;">
              <div class="metric-item">
                <span class="metric-val mono" style="color: #ea580c;">${{((p.engagement && p.engagement.score) || 0).toLocaleString()}}</span>
                <span class="metric-sub">Upvotes</span>
              </div>
              <div class="metric-item">
                <span class="metric-val mono">${{((p.engagement && p.engagement.comments) || 0).toLocaleString()}}</span>
                <span class="metric-sub">Comments</span>
              </div>
            </div>
          </div>
        </div>
      `).join('');
    }}

    function initCharts() {{
      Chart.defaults.font.family = "'DM Sans', sans-serif";
      Chart.defaults.color = '#536761';
      const ctxVol = document.getElementById('chart-volume').getContext('2d');
      new Chart(ctxVol, {{
        type: 'line',
        data: {{
          labels: REPORT_HISTORY.map(run => run.label),
          datasets: [
            {{
              label: 'Signals recorded',
              data: REPORT_HISTORY.map(run => run.signals),
              borderColor: '#087f76',
              backgroundColor: 'rgba(8, 127, 118, 0.09)',
              fill: true,
              tension: 0,
              borderWidth: 2,
            }},
            {{
              label: 'Topics in report',
              data: REPORT_HISTORY.map(run => run.topics),
              borderColor: '#c84f45',
              backgroundColor: 'transparent',
              borderDash: [4, 4],
              tension: 0,
              borderWidth: 1.5,
            }}
          ]
        }},
        options: {{
          responsive: true,
          maintainAspectRatio: false,
          plugins: {{ legend: {{ position: 'bottom' }} }},
          scales: {{
            y: {{ grid: {{ color: '#f1f5f9' }} }},
            x: {{ grid: {{ display: false }} }}
          }}
        }}
      }});

      const ctxStages = document.getElementById('chart-stages').getContext('2d');
      new Chart(ctxStages, {{
        type: 'bar',
        data: {{
          labels: ['Saturated', 'Peak', 'Breaking Out', 'Simmering', 'Pre-Breakout', 'Settled'],
          datasets: [{{
            data: [{stage_counts.get('saturated', 0)}, {stage_counts.get('peak', 0)}, {stage_counts.get('breaking_out', 0)}, {stage_counts.get('simmering', 0)}, {stage_counts.get('pre_breakout', 0)}, {stage_counts.get('settled', 0)}],
            backgroundColor: ['#c84f45', '#b77b23', '#087f76', '#d7a84d', '#24846d', '#8b9b96'],
            borderRadius: 6
          }}]
        }},
        options: {{
          responsive: true,
          maintainAspectRatio: false,
          plugins: {{ legend: {{ display: false }} }},
          scales: {{
            y: {{ grid: {{ color: '#f1f5f9' }} }},
            x: {{ grid: {{ display: false }} }}
          }}
        }}
      }});

      const ctxStance = document.getElementById('chart-stance').getContext('2d');
      new Chart(ctxStance, {{
        type: 'pie',
        data: {{
          labels: ['Supportive', 'Critical', 'Neutral', 'Humorous'],
          datasets: [{{
            data: [{stance_counts['supportive']}, {stance_counts['critical']}, {stance_counts['neutral']}, {stance_counts['humorous']}],
            backgroundColor: ['#24846d', '#c84f45', '#71847e', '#b77b23']
          }}]
        }},
        options: {{
          responsive: true,
          maintainAspectRatio: false,
          plugins: {{ legend: {{ position: 'bottom', labels: {{ boxWidth: 12 }} }} }}
        }}
      }});

      const ctxCat = document.getElementById('chart-categories').getContext('2d');
      new Chart(ctxCat, {{
        type: 'bar',
        data: {{
          labels: ['Politics', 'Crime/Law', 'Cinema', 'Sports', 'Economy', 'Tech', 'Civic'],
          datasets: [{{
            data: [{cat_counts.get('politics_governance', 0)}, {cat_counts.get('crime_law', 0)}, {cat_counts.get('cinema_tv_ott', 0)}, {cat_counts.get('sports', 0)}, {cat_counts.get('economy_business', 0)}, {cat_counts.get('technology', 0)}, {cat_counts.get('civic_issues', 0)}],
            backgroundColor: ['#087f76', '#c84f45', '#b77b23', '#24846d', '#397b96', '#71847e', '#a66a54'],
            borderRadius: 4
          }}]
        }},
        options: {{
          indexAxis: 'y',
          responsive: true,
          maintainAspectRatio: false,
          plugins: {{ legend: {{ display: false }} }},
          scales: {{
            x: {{ grid: {{ color: '#f1f5f9' }} }},
            y: {{ grid: {{ display: false }} }}
          }}
        }}
      }});

      const ctxDemo = document.getElementById('chart-demographics').getContext('2d');
      new Chart(ctxDemo, {{
        type: 'doughnut',
        data: {{
          labels: [
            'Civic Anger & Skepticism',
            'Financial Anxiety',
            'Cultural Pride & Solidarity',
            'Ironic Detachment / Cynicism',
            'Social Empathy & Care',
            'Objective Deliberation'
          ],
          datasets: [{{
            data: [
              {global_mental_states.get('Civic Anger & Institutional Skepticism', 0)},
              {global_mental_states.get('Financial Anxiety & Economic Strain', 0)},
              {global_mental_states.get('Cultural Pride & National Solidarity', 0)},
              {global_mental_states.get('Ironic Detachment & Memetic Cynicism', 0)},
              {global_mental_states.get('Social Empathy & Grassroots Care', 0)},
              {global_mental_states.get('Objective Deliberation & Fact-Checking', 0)}
            ],
            backgroundColor: ['#c84f45', '#b77b23', '#24846d', '#087f76', '#397b96', '#71847e']
          }}]
        }},
        options: {{
          responsive: true,
          maintainAspectRatio: false,
          plugins: {{ legend: {{ position: 'bottom', labels: {{ boxWidth: 10, font: {{ size: 9.5 }} }} }} }}
        }}
      }});

      const top10 = [...TOPICS].sort((a,b) => (Number(b.is_political) - Number(a.is_political)) || b.trend_score - a.trend_score).slice(0, 10);
      document.getElementById('top-breakouts-list').innerHTML = top10.map(buildTopicCardHtml).join('');
    }}

    window.onload = function() {{
      initCharts();
      applyFilters();
    }};
  </script>
</body>
</html>
"""

output_path = report_path.parent / 'dashboard.html'
with open(output_path, 'w', encoding='utf-8') as f:
    f.write(html_content)

print(f"Successfully generated dashboard at {output_path} ({os.path.getsize(output_path)} bytes)")
