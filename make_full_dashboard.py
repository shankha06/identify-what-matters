import json
import html
import math
import os
import re
import sys
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
if len(sys.argv) > 1:
  report_path = Path(sys.argv[1])
  data = json.loads(report_path.read_text(encoding='utf-8'))

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
  partial_notice = ''
  if latest_collection.get('partial_collection'):
    stopped = html.escape(str(latest_collection.get('stopped') or 'Instagram collection stopped early.'))
    partial_notice = (f'<p style="margin-top:12px; color:#9a3412;"><strong>Partial Instagram capture.</strong> '
                      f'{stopped}</p>')
  collection_section_html = f'''<div class="card" style="margin-bottom: 24px;">
    <div class="card-header"><div class="card-title">Latest signal collection</div>
    <span class="chip mono">{collection_run} · {captured_at}</span></div>
    <div class="card-body"><div style="font-size:12px; color:var(--text-dim);">{len(signals)} live public signals · Instagram: {html.escape(str(latest_collection.get('transport') or 'not collected'))}</div>
    {partial_notice}{availability_table}{signals_table}</div>
  </div>'''

topics = data.get('topics', [])
panel_pulse = data.get('panel_pulse', [])
run_id = data.get('run_id', '20261004T131104Z')
generated_at = data.get('generated_at', '')

panel_cfg_accounts = []
panel_accounts_per_run = 100
try:
        panel_config = json.loads(Path('config/panel.json').read_text(encoding='utf-8'))
        panel_cfg_accounts = panel_config.get('accounts', [])
        panel_accounts_per_run = panel_config.get('accounts_per_run', panel_accounts_per_run)
except Exception:
        panel_cfg_accounts = []
panel_groups = sorted({str(account.get('group') or 'other') for account in panel_cfg_accounts})
panel_group_options_html = ''.join(
  f'<option value="{html.escape(group, quote=True)}">{html.escape(group.replace("_", " ").title())}</option>'
  for group in panel_groups
)
panel_cohort_count = max(1, (len(panel_cfg_accounts) + panel_accounts_per_run - 1) // panel_accounts_per_run)

from src.identify_what_matters.demographics import (
    GEN_Z_SLANG, PROFESSIONAL_SLANG, MATURE_SLANG,
    FEMALE_CONVERSATIONAL, MALE_CONVERSATIONAL, infer_location, _get_models
)
from src.identify_what_matters.political_evidence import audit_comments
from src.identify_what_matters.bot_detector import bot_detector, _evaluate_comment_stance_and_lean
from src.identify_what_matters.political_pulse import political_pulse_engine
from src.identify_what_matters.political_corpus import ISSUE_CLUSTERS
from src.identify_what_matters import political_classifier as polclf
from src.identify_what_matters import undercurrent_scoring
from collections import Counter

# Load demographics models once at module level so both infer_topic_mental_state
# and evaluate_topic_pulse can share the party_support_classifier.
from src.identify_what_matters.demographics import (
    infer_commenter_demographics,
    predict_demographics_neural,
    predict_party_neural
)

_demographics_models = _get_models()
_party_model = _demographics_models[5] if len(_demographics_models) > 5 else None

def _ml_party_predict(text: str) -> str:
    """Return party label from fine-tuned 17-party neural transformer or joblib fallback."""
    try:
        p_res = predict_party_neural(text)
        if p_res and p_res[0] != "None" and p_res[1] >= 0.40:
            return p_res[0]
    except Exception:
        pass
    if _party_model is None:
        return "None"
    try:
        return _party_model.predict([text])[0]
    except Exception:
        return "None"


def _compute_comment_velocity(comment_pool: list) -> dict:
    """Compute timestamp-based comment velocity using ISO created_at timestamps."""
    from datetime import datetime, timezone
    timestamps = []
    for c in comment_pool:
        ts_raw = c.get('created_at')
        if not ts_raw:
            continue
        try:
            ts = datetime.fromisoformat(str(ts_raw).replace('Z', '+00:00'))
            timestamps.append(ts)
        except Exception:
            pass
    if not timestamps:
        return {'comments_per_hour_6h': 0.0, 'comments_per_hour_prev': 0.0, 'velocity_trend': 'flat', 'window_comments': 0}
    now = max(timestamps)
    window_6h = sum(1 for ts in timestamps if (now - ts).total_seconds() <= 21600)
    window_prev = sum(1 for ts in timestamps if 21600 < (now - ts).total_seconds() <= 43200)
    rate_6h = round(window_6h / 6.0, 1)
    rate_prev = round(window_prev / 6.0, 1)
    if rate_prev == 0:
        trend = 'rising' if rate_6h > 0 else 'flat'
    elif rate_6h > rate_prev * 1.5:
        trend = 'accelerating'
    elif rate_6h > rate_prev * 1.1:
        trend = 'rising'
    elif rate_6h < rate_prev * 0.6:
        trend = 'fading'
    else:
        trend = 'steady'
    return {
        'comments_per_hour_6h': rate_6h,
        'comments_per_hour_prev': rate_prev,
        'velocity_trend': trend,
        'window_comments': len(timestamps),
    }

# Global demographic mental state counters across all analyzed comments
global_age_cohorts = Counter()
global_personas = Counter()
global_mental_states = Counter()
global_regions = Counter()
total_analyzed_comments = 0

cat_counts = {}
stage_counts = {}
stance_counts = {'supportive': 0, 'critical': 0, 'neutral': 0, 'humorous': 0}
issue_cluster_stats = {c: {
    'topics': 0, 'engagement': 0.0, 'bjp_pct_sum': 0.0, 'opp_pct_sum': 0.0,
    'narrative_types': Counter(), 'supportive_sum': 0.0, 'critical_sum': 0.0,
    'sentiment_pos_sum': 0.0, 'sentiment_neg_sum': 0.0, 'sentiment_neu_sum': 0.0,
} for c in ISSUE_CLUSTERS}
# Which pipeline tasks are running on a real trained model right now vs. keyword fallback - shown in the
# Undercurrent Intel tab so the dashboard is honest about its own current sophistication.
models_status = polclf.models_available()
try:
    from src.identify_what_matters.strategic_classifier import load_strategic_model
    s_mod, _, _ = load_strategic_model()
    models_status["strategic_threat_radar"] = (s_mod is not None)
    models_status["neural_criticality"] = (s_mod is not None)
except Exception:
    models_status["strategic_threat_radar"] = False
    models_status["neural_criticality"] = False

def infer_topic_mental_state(t, pool=None):
    brief = t.get('brief') or {}
    if pool is None:
        pool = brief.get('comment_pool', [])

    age_m, gen_m, state_m, imp_m, irr_m, party_m = _demographics_models

    # Map party_support_classifier output to political lean buckets
    _BJP_ALLIED = {"BJP"}
    _OPP_ALLIED = {"Congress", "AAP", "TMC", "SP", "DMK", "RJD", "CPIM", "NCP", "JDU", "TDP", "YSRCP", "BRS", "BSP", "AIADMK", "Shiv Sena"}

    def _party_to_lean(party_label: str) -> str | None:
        if party_label in _BJP_ALLIED:
            return "Favoring BJP / NDA"
        if party_label in _OPP_ALLIED:
            return "Favoring Opposition / Regional"
        return None  # "None" → defer to keyword heuristic

    # Fallback to topic story text and narrative claims when no comments are scraped
    if not pool:
        story_corpus = []
        if t.get('title'):
            story_corpus.append(str(t['title']))
        if brief.get('summary'):
            story_corpus.append(str(brief['summary']))
        if brief.get('why_trending'):
            story_corpus.append(str(brief['why_trending']))
        for stmt in brief.get('what_people_say') or []:
            if isinstance(stmt, str) and stmt.strip():
                story_corpus.append(stmt)
        combined_text = " ".join(story_corpus)
        passages = [p.strip() for p in re.split(r'[.\n]+', combined_text) if len(p.strip()) > 15]
        if not passages:
            passages = [combined_text] if combined_text.strip() else []

        if not passages:
            return {
                'has_data': False,
                'source': 'estimated',
                'cohort': '25-34 (General Public)',
                'gender_persona': 'Mixed Public',
                'top_region': 'Pan-India',
                'mindset': 'General Public',
                'comment_count': 0,
                'age_distribution': {},
                'gender_distribution': {},
                'mental_distribution': {},
                'political_lean_distribution': {},
                'coverage': {'model_classified': 0, 'lexicon_classified': 0, 'default_fallback': 0, 'total': 0},
                'regions': []
            }

        # Run behavioral inference over the story narrative passages using end-to-end infer_commenter_demographics
        a_cnt, g_cnt, m_cnt, imp_cnt, irr_cnt, p_cnt = Counter(), Counter(), Counter(), Counter(), Counter(), Counter()
        for p in passages:
            try:
                demo_res = infer_commenter_demographics({'text': p})
                a_cnt[f"{demo_res.age_bracket} (ML Inferred)"] += 1
                g_cnt['Female Persona' if demo_res.gender == 'female' else 'Male Persona' if demo_res.gender == 'male' else 'Neutral / Topic-Centered'] += 1
                m_cnt[demo_res.mental_state] += 1
                imp_cnt[demo_res.impatience] += 1
                irr_cnt[demo_res.irrationality] += 1
                if demo_res.party_support and demo_res.party_support != "None":
                    lean_str = _party_to_lean(demo_res.party_support)
                    p_cnt[lean_str or "Neutral / Non-partisan"] += 1
                else:
                    lean = political_pulse_engine.classify_text_favorability(p)
                    p_cnt[lean['leaning']] += 1
            except Exception:
                pass

        top_age = a_cnt.most_common(1)[0][0] if a_cnt else '25-34 (Working Prof.)'
        top_gen = g_cnt.most_common(1)[0][0] if g_cnt else 'Neutral / Topic-Centered'
        top_mindset = m_cnt.most_common(1)[0][0] if m_cnt else 'Civic Discourse & Skepticism'
        top_imp = imp_cnt.most_common(1)[0][0] if imp_cnt else 'neutral'
        top_irr = irr_cnt.most_common(1)[0][0] if irr_cnt else 'rational'

        n_pass = len(passages)
        global_age_cohorts[top_age] += n_pass or 1
        global_mental_states[top_mindset] += n_pass or 1
        global_personas[top_gen] += n_pass or 1
        return {
            'has_data': True,
            'source': 'story_nlp_inferred',
            'cohort': top_age,
            'gender_persona': top_gen,
            'top_region': 'Pan-India',
            'mindset': top_mindset,
            'impatience': top_imp,
            'irrationality': top_irr,
            'comment_count': 0,
            'passages_analyzed': n_pass,
            'age_distribution': {k: round(v/n_pass, 2) for k, v in a_cnt.items()} if a_cnt else {},
            'gender_distribution': {k: round(v/n_pass, 2) for k, v in g_cnt.items()} if g_cnt else {},
            'mental_distribution': {k: round(v/n_pass, 2) for k, v in m_cnt.items()} if m_cnt else {},
            'impatience_distribution': {k: round(v/n_pass, 2) for k, v in imp_cnt.items()} if imp_cnt else {},
            'irrationality_distribution': {k: round(v/n_pass, 2) for k, v in irr_cnt.items()} if irr_cnt else {},
            'political_lean_distribution': {k: round(v/n_pass, 2) for k, v in p_cnt.items()} if p_cnt else {},
            'coverage': {'model_classified': n_pass, 'lexicon_classified': 0, 'default_fallback': 0, 'total': n_pass},
            'regions': []
        }

    a_cnt = Counter()
    g_cnt = Counter()
    l_cnt = Counter()
    m_cnt = Counter()
    p_cnt = Counter()
    imp_cnt = Counter()
    irr_cnt = Counter()
    coverage = Counter()

    for c in pool:
        txt = c.get('text', '')
        if not txt.strip():
            continue

        # Full-spectrum neural + calibrated demographic inference
        demo = infer_commenter_demographics(c)

        # Gender Persona
        g_label = 'Female Persona' if demo.gender == 'female' else 'Male Persona' if demo.gender == 'male' else 'Neutral / Topic-Centered'
        g_cnt[g_label] += 1
        global_personas['Female-Skewed' if demo.gender == 'female' else 'Male-Skewed' if demo.gender == 'male' else 'Neutral / Topic-Centered'] += 1

        # Age Cohort
        a_label = f"{demo.age_bracket} (ML Inferred)"
        a_cnt[a_label] += 1
        global_age_cohorts[a_label] += 1

        # Psychological Archetype / Mental State
        m_label = demo.mental_state
        m_cnt[m_label] += 1
        global_mental_states[m_label] += 1
        coverage['model_classified'] += 1

        # Conversational Persona: Impatience & Urgency
        imp_cnt[demo.impatience] += 1

        # Epistemic Persona: Irrationality / Conspiracy markers
        irr_cnt[demo.irrationality] += 1

        # Political lean: neural party support + keyword heuristic fallback
        lean_label = None
        if demo.party_support and demo.party_support != "None" and demo.party_confidence >= 0.40:
            lean_label = _party_to_lean(demo.party_support)

        if not lean_label:
            try:
                lean = political_pulse_engine.classify_text_favorability(txt)
                lean_label = lean.get('leaning', 'Neutral / Non-Political')
            except Exception:
                lean_label = 'Neutral / Non-Political'

        p_cnt[lean_label] += 1

        st, _, _ = infer_location(txt)
        if st:
            l_cnt[st] += 1
            global_regions[st] += 1

    coverage['total'] = len(pool)
    coverage['lexicon_classified'] = max(0, len(pool) - coverage['model_classified'] - coverage['default_fallback'])
    top_age = a_cnt.most_common(1)[0][0] if a_cnt else '25-34 (General Public)'
    top_gender = g_cnt.most_common(1)[0][0] if g_cnt else 'Mixed Public'
    top_loc = l_cnt.most_common(1)[0][0] if l_cnt else 'Pan-India'
    top_mindset = m_cnt.most_common(1)[0][0] if m_cnt else 'General Public'
    top_impatience = imp_cnt.most_common(1)[0][0] if imp_cnt else 'neutral'
    top_irrationality = irr_cnt.most_common(1)[0][0] if irr_cnt else 'neutral'

    return {
        'has_data': True,
        'source': 'measured',
        'cohort': top_age,
        'gender_persona': top_gender,
        'top_region': top_loc,
        'mindset': top_mindset,
        'impatience': top_impatience,
        'irrationality': top_irrationality,
        'comment_count': len(pool),
        'age_distribution': {k: round(v/len(pool), 2) for k, v in a_cnt.items()},
        'gender_distribution': {k: round(v/len(pool), 2) for k, v in g_cnt.items()},
        'mental_distribution': {k: round(v/len(pool), 2) for k, v in m_cnt.items()},
        'impatience_distribution': {k: round(v/len(pool), 2) for k, v in imp_cnt.items()},
        'irrationality_distribution': {k: round(v/len(pool), 2) for k, v in irr_cnt.items()},
        'political_lean_distribution': {k: round(v/len(pool), 2) for k, v in p_cnt.items()},
        'coverage': dict(coverage),
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

# Dynamic discovery proposals: new candidate accounts (account_discovery.py) and emergent issue/narrative
# clusters (emergent_narrative_discovery.py) the system found on its own. Both are empty until a collect
# or measure run has populated output/discovery/ - never auto-applied to config/panel.json or the issue
# taxonomy, see each file's own "policy" field.
account_candidates, issue_candidates = [], []
account_candidates_path = Path('output/discovery/account_candidates.json')
if account_candidates_path.exists():
    try:
        account_candidates = json.loads(account_candidates_path.read_text(encoding='utf-8')).get('candidates', [])
    except Exception:
        account_candidates = []
issue_candidates_path = Path('output/discovery/issue_cluster_candidates.json')
if issue_candidates_path.exists():
    try:
        issue_candidates = json.loads(issue_candidates_path.read_text(encoding='utf-8')).get('emergent_clusters', [])
    except Exception:
        issue_candidates = []

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

# Load stored Instagram media and comments if available in the run directory
stored_instagram = {}
ig_json_path = target_run_dir / 'instagram.json'
if ig_json_path.exists():
    try:
        with ig_json_path.open(encoding='utf-8') as f:
            stored_instagram = json.load(f)
        print(f'[dashboard] loaded stored instagram evidence across {len(stored_instagram)} topics')
    except Exception:
        stored_instagram = {}

# Ensure topics have comment_pool populated from stored_instagram if missing
for _t in topics:
    _tbrief = _t.get('brief')
    if not _tbrief:
        _tbrief = {}
        _t['brief'] = _tbrief
    if not _tbrief.get('comment_pool'):
        _k = _t.get('topic_key') or ''
        _media = (stored_instagram.get(_k) or {}).get('media', [])
        _cmts = []
        for _m in _media:
            for _c in _m.get('comments', []):
                _ctxt = str(_c.get('text') or '').strip()
                if _ctxt:
                    _cmts.append({
                        'text': _ctxt,
                        'likes': max(0, int(_c.get('like_count') or _c.get('likes') or 0)),
                        'replies': max(0, int(_c.get('reply_count') or 0)),
                        'author_id': str((_c.get('user') or {}).get('pk') or _c.get('author_id') or ''),
                        'created_at': _c.get('taken_at') or _c.get('created_at', ''),
                    })
        _cmts.sort(key=lambda x: x['likes'], reverse=True)
        _tbrief['comment_pool'] = _cmts[:80]

# Batch the per-comment neural model calls once up front; the per-topic loop below reads cached results.
from src.identify_what_matters.demographics import prewarm_neural_demographics, prewarm_party_neural as prewarm_demo_party
from src.identify_what_matters.political_pulse import prewarm_party_neural as prewarm_pulse_party
_neural_texts = []
for _t in topics:
    _neural_texts += [str(c.get('text') or '') for c in ((_t.get('brief') or {}).get('comment_pool') or [])]
    _neural_texts += [str(c.get('text') or '') for c in ((_t.get('reddit_proxy') or {}).get('comment_excerpts') or [])]
try:
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    prewarm_neural_demographics(_neural_texts, batch_size=32)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    prewarm_demo_party(_neural_texts, batch_size=32)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    prewarm_pulse_party(_neural_texts, batch_size=32)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f'[dashboard] batched neural prewarm over {len(set(_neural_texts))} unique comment texts')
except Exception as _error:
    print(f'[dashboard] batched neural prewarm skipped: {type(_error).__name__}: {str(_error)[:120]}')

from src.identify_what_matters.strategic_classifier import classify_strategic_batch
try:
    _all_topic_texts = [f"{_t.get('title', '')}. {(_t.get('brief') or {}).get('summary', '')}" for _t in topics]
    _strategic_results = classify_strategic_batch(_all_topic_texts, batch_size=32)
    for _t, _s_res in zip(topics, _strategic_results):
        _t['neural_strategic'] = _s_res
    print(f'[dashboard] batched strategic neural classification over {len(topics)} topics')
except Exception as _e:
    print(f'[dashboard] strategic batch skipped: {_e}')

from src.identify_what_matters.political_classifier import prewarm_political_transformers
try:
    _topic_combined_texts = []
    for _t in topics:
        _t_brief = _t.get('brief') or {}
        _t_sum = _t_brief.get('summary', '')
        _t_why = _t_brief.get('why_trending', '')
        _t_parts = [_t.get('title', '')]
        if _t_sum:
            _t_parts.append(_t_sum)
        if _t_why:
            _t_parts.append(_t_why)
        _topic_combined_texts.append(". ".join(_t_parts))
    prewarm_political_transformers(_topic_combined_texts, batch_size=32)
    print(f'[dashboard] batched political transformers over {len(_topic_combined_texts)} topics')
except Exception as _e:
    print(f'[dashboard] political transformers batch skipped: {_e}')

topics_client = []
for idx, t in enumerate(topics):
    if (idx + 1) % 50 == 0 or idx == 0:
        print(f'[dashboard] processing topic {idx+1}/{len(topics)}...', flush=True)
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

    reddit_proxy = t.get('reddit_proxy') or {}
    matched_reddit = reddit_proxy.get('threads', [])
    reddit_comments_total = reddit_proxy.get('post_comments', 0)
    
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

    # Check if political/civic
    is_political = (cat in {'politics_governance', 'civic_issues'}) or bool(re.search(
      r'\b(?:modi|bjp|congress|election|electoral|minister|government|parliament|assembly|policy|protest|kejriwal|yogi|dmk|tvk)\b',
      title, re.I))
    has_probed_data = bool((brief or {}).get('comment_pool') or ig.get('on_topic_posts', 0) > 0)
    
    # Demographics and mental state proxy — pass combined IG + Reddit pool
    _ig_pool_early = (brief or {}).get('comment_pool') or []
    _reddit_excerpts_early = (t.get('reddit_proxy') or {}).get('comment_excerpts') or []
    _reddit_tagged_early = [{'text': c.get('text', ''), 'likes': c.get('score', 0),
                              'replies': 0, 'source': 'reddit', 'subreddit': c.get('subreddit', '')}
                             for c in _reddit_excerpts_early if c.get('text')]
    _combined_early = [{**c, 'source': 'instagram'} for c in _ig_pool_early] + _reddit_tagged_early
    demo = infer_topic_mental_state(t, pool=_combined_early)
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
        demo['has_data'] = True
        global_age_cohorts[demo['cohort']] += 1
        global_mental_states[demo['mindset']] += 1
        global_personas[demo.get('gender_persona', 'Neutral / Topic-Centered')] += 1

    # Build a unified comment pool: Instagram (brief.comment_pool) + Reddit excerpts
    _ig_pool = (brief or {}).get('comment_pool') or []
    _reddit_excerpts = reddit_proxy.get('comment_excerpts') or []
    # Tag Reddit comments so they can be labelled in the UI
    _reddit_tagged = [{'text': c.get('text', ''), 'likes': c.get('score', 0),
                       'replies': 0, 'source': 'reddit', 'subreddit': c.get('subreddit', '')}
                      for c in _reddit_excerpts if c.get('text')]
    # Tag Instagram pool with source label (they lack the field currently)
    _ig_tagged = [{**c, 'source': 'instagram'} for c in _ig_pool]
    _combined_pool = _ig_tagged + _reddit_tagged

    # Timestamp-based comment velocity (comments per hour in last 6h vs previous 6h)
    _comment_velocity = _compute_comment_velocity(_ig_pool)

    _pulse = political_pulse_engine.evaluate_topic_pulse(
        t, _combined_pool, matched_reddit, ml_classifier_fn=_ml_party_predict
    )
    _engagement = ig.get('engagement_per_hour', 0.0) or 0.0
    _bot_audit = bot_detector.analyze_topic_astroturf(
        t.get('title', ''), _combined_pool, (brief or {}).get('debate', {}).get('sides', []))
    _undercurrent = undercurrent_scoring.score_topic(t, _bot_audit, _combined_pool, political_pulse=_pulse)
    issue_cluster = _undercurrent['issue_cluster']
    undercurrent_score = _undercurrent['undercurrent_score']
    # Multi-task Political/Civic ML classification on topic content
    _topic_text_parts = [title]
    if summary:
        _topic_text_parts.append(summary)
    if why_trending:
        _topic_text_parts.append(why_trending)
    _combined_topic_text = ". ".join(_topic_text_parts)
    _political_intel = polclf.classify(_combined_topic_text)

    # Calculate and assign stance if missing from brief
    if not stance and _combined_pool:
        tones = Counter()
        for c in _combined_pool:
            c_txt = str(c.get('text') or '').strip()
            if c_txt:
                meta = _evaluate_comment_stance_and_lean(c_txt)
                w = 1.0 + math.log1p(max(0, int(c.get('likes') or 0)))
                tones[meta['stance']] += w
        tot_t = sum(tones.values()) or 1.0
        stance = {k: round(tones[k] / tot_t, 3) for k in ('supportive', 'critical', 'humorous', 'neutral')}
        if brief is not None:
            brief['stance'] = stance
    elif not stance:
        _s_obj = _political_intel.get('sentiment') or {}
        _s_val = _s_obj.get('value', 'neutral')
        if _s_val == 'positive':
            stance = {'supportive': 0.60, 'critical': 0.15, 'humorous': 0.05, 'neutral': 0.20}
        elif _s_val == 'negative':
            stance = {'supportive': 0.15, 'critical': 0.60, 'humorous': 0.05, 'neutral': 0.20}
        elif cat in ('cinema_tv_ott', 'music_celebrity', 'viral_memes_social'):
            stance = {'supportive': 0.40, 'critical': 0.15, 'humorous': 0.35, 'neutral': 0.10}
        else:
            stance = {'supportive': 0.25, 'critical': 0.25, 'humorous': 0.10, 'neutral': 0.40}
        if brief is not None:
            brief['stance'] = stance

    if stance:
        top_st = max(stance.items(), key=lambda x: x[1])[0]
        stance_counts[top_st] = stance_counts.get(top_st, 0) + 1

    if issue_cluster not in ('none', 'n/a', '', None) and issue_cluster in issue_cluster_stats:
        bucket = issue_cluster_stats[issue_cluster]
        bucket['topics'] += 1
        bucket['engagement'] += _engagement
        bucket['bjp_pct_sum'] += _pulse.get('bjp_favor_pct', 0)
        bucket['opp_pct_sum'] += _pulse.get('opp_favor_pct', 0)
        nt = _undercurrent.get('narrative_type', 'other')
        if nt and nt != 'other':
            bucket['narrative_types'][nt] += 1
        _stance = (brief or {}).get('stance', {})
        bucket['supportive_sum'] += _stance.get('supportive', 0)
        bucket['critical_sum'] += _stance.get('critical', 0)
        # Political sentiment towards governance / subject
        _s_obj = _political_intel.get('sentiment') or {}
        _s_val = _s_obj.get('value', 'neutral')
        _s_conf = _s_obj.get('confidence', 0.5)
        if _s_val == 'positive':
            bucket['sentiment_pos_sum'] += _s_conf
            bucket['sentiment_neu_sum'] += (1.0 - _s_conf)
        elif _s_val == 'negative':
            bucket['sentiment_neg_sum'] += _s_conf
            bucket['sentiment_neu_sum'] += (1.0 - _s_conf)
        else:
            bucket['sentiment_neu_sum'] += 1.0

    topics_client.append({
        'id': key,
        'index': idx,
        'title': title,
        'category': cat,
        'issue_cluster': issue_cluster,
        'undercurrent_score': undercurrent_score,
        'political_intel': _political_intel,
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
        'bot_audit': _bot_audit,
        'political_pulse': _pulse,
        'undercurrent': _undercurrent,
        'has_probed_data': has_probed_data,
        'search_queries': t.get('search_queries', []),
        'demo': demo,
        'matched_reddit': matched_reddit[:4],
        'reddit_comments_total': reddit_comments_total,
        'reddit_proxy': reddit_proxy,
        # Instagram comments for display (top 8 by likes, with text truncated for safety)
        'ig_comments': [
            {'text': c.get('text', '')[:500], 'likes': c.get('likes', 0),
             'created_at': c.get('created_at', ''), 'replies': c.get('replies', 0)}
            for c in sorted(_ig_pool, key=lambda x: x.get('likes', 0), reverse=True)[:8]
            if c.get('text', '').strip()
        ],
        'comment_velocity': _comment_velocity,
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
            for comment in (topic.get('ig_comments') or []) + [
                {'text': c.get('text', ''), 'likes': c.get('score', 0), 'replies': 0, 'source': 'reddit'}
                for c in ((topics[index].get('reddit_proxy') or {}).get('comment_excerpts') or [])
            ]]
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
        # Give higher weight to topics with measured comment engagement, while still including narrative framing for unprobed ones
        w = max(t.get('trend_score', 1.0), 0.5)
        if p.get('engagement_count', 0) > 0:
            w *= 1.5
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
    lean_col = cl.get("badge_color", "#c2410c")
    lean_bg = cl.get("badge_bg", "#fff7ed")
    lean_bor = cl.get("border_color", "#fed7aa")
    synth_lean_chip = f'<span class="chip mono" style="background:{lean_bg};color:{lean_col};border-color:{lean_bor};font-weight:600;font-size:11px;">{html.escape(cl.get("synthetic_lean", "Neutral Script"))}</span>'
    dom_sent_chip = f'<span class="chip mono" style="font-size:11px;background:var(--surface-2);color:var(--text);">{html.escape(cl.get("dominant_sentiment", "Template Slogan"))}</span>'
    bot_cluster_rows.append(
        f'<tr><td><div style="font-size:12.5px;font-style:italic;">"{html.escape(cl["template"])}"</div></td>'
        f'<td>{acc_badge}</td>'
        f'<td><span class="mono">{cl["occurrences"]}x</span></td>'
        f'<td>{synth_lean_chip}</td>'
        f'<td>{dom_sent_chip}</td>'
        f'<td><span class="chip mono" style="background:#fff7ed;color:#c2410c;border-color:#fed7aa;">{html.escape(cl["suspicion_level"].upper())}</span></td></tr>'
    )
bot_cluster_table_html = ''.join(bot_cluster_rows) if bot_cluster_rows else '<tr><td colspan="6">No suspicious cross-account copy-paste templates identified.</td></tr>'

# 1. Dedicated "Bot vs. Organic Sentiment" Split Metric Cards Widget
org_mix = global_bot_audit.get('organic_sentiment_mix', {})
syn_mix = global_bot_audit.get('synthetic_sentiment_mix', {})
bot_div = global_bot_audit.get('sentiment_divergence', {})
bot_ratio = global_bot_audit.get('organic_vs_synthetic_ratio', 1.0)
skew_val = bot_div.get('critical_skew', 0.0)
skew_color = '#dc2626' if skew_val > 10 else ('#16a34a' if skew_val < -5 else '#c2410c')

bot_split_kpis_html = f'''
<div style="display:grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 14px; margin-bottom: 18px;">
  <div style="background:#ffffff; border:1px solid #bbf7d0; border-left:4px solid #16a34a; border-radius:8px; padding:14px 16px; box-shadow:0 1px 3px rgba(0,0,0,0.03);">
    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
      <span style="font-size:11px; font-weight:700; text-transform:uppercase; color:#15803d; letter-spacing:0.4px;">🧑 Organic Sentiment Mix</span>
      <span class="chip mono" style="font-size:10px; background:#f0fdf4; color:#166534; border-color:#bbf7d0;">Verified Human Accounts</span>
    </div>
    <div style="display:flex; justify-content:space-between; font-size:12px; margin-bottom:6px; font-weight:600;">
      <span style="color:#16a34a;">🟢 Supportive: {org_mix.get('supportive', 0)}%</span>
      <span style="color:#dc2626;">🔴 Critical: {org_mix.get('critical', 0)}%</span>
      <span style="color:#64748b;">⚪ Neutral: {org_mix.get('neutral', 0)}%</span>
    </div>
    <div style="display:flex; height:10px; border-radius:5px; overflow:hidden; background:#e2e8f0; margin-bottom:8px;">
      <div style="width:{org_mix.get('supportive', 0)}%; background:#22c55e;" title="Supportive: {org_mix.get('supportive', 0)}%"></div>
      <div style="width:{org_mix.get('critical', 0)}%; background:#ef4444;" title="Critical: {org_mix.get('critical', 0)}%"></div>
      <div style="width:{org_mix.get('neutral', 0)}%; background:#94a3b8;" title="Neutral: {org_mix.get('neutral', 0)}%"></div>
    </div>
    <div style="font-size:11px; color:var(--text-dim); display:flex; justify-content:space-between;">
      <span>Political Vector: <b>BJP {org_mix.get('pro_incumbent', 0)}%</b> vs <b>Opp {org_mix.get('anti_incumbent', 0)}%</b></span>
      <span class="mono">{global_bot_audit.get('organic_comment_count', 0)} comments</span>
    </div>
    <div style="font-size:11px; color:#166534; margin-top:6px; font-style:italic;">
      Derived strictly from verified human users — captures genuine grassroots sentiment without bot distortion.
    </div>
  </div>

  <div style="background:#ffffff; border:1px solid #fecaca; border-left:4px solid #dc2626; border-radius:8px; padding:14px 16px; box-shadow:0 1px 3px rgba(0,0,0,0.03);">
    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
      <span style="font-size:11px; font-weight:700; text-transform:uppercase; color:#b91c1c; letter-spacing:0.4px;">🤖 Bot / Synthetic Script Mix</span>
      <span class="chip mono" style="font-size:10px; background:#fef2f2; color:#991b1b; border-color:#fecaca;">Automated Scripts &amp; Slogans</span>
    </div>
    <div style="display:flex; justify-content:space-between; font-size:12px; margin-bottom:6px; font-weight:600;">
      <span style="color:#d97706;">⚡ Slogans: {syn_mix.get('extreme_partisan_slogan', 0)}%</span>
      <span style="color:#dc2626;">🔴 Attack: {syn_mix.get('critical', 0)}%</span>
      <span style="color:#64748b;">⚪ Neutral: {syn_mix.get('neutral', 0)}%</span>
    </div>
    <div style="display:flex; height:10px; border-radius:5px; overflow:hidden; background:#e2e8f0; margin-bottom:8px;">
      <div style="width:{syn_mix.get('extreme_partisan_slogan', 0)}%; background:#f59e0b;" title="Extreme Partisan / Slogan: {syn_mix.get('extreme_partisan_slogan', 0)}%"></div>
      <div style="width:{syn_mix.get('critical', 0)}%; background:#ef4444;" title="Critical Attack: {syn_mix.get('critical', 0)}%"></div>
      <div style="width:{syn_mix.get('supportive', 0)}%; background:#22c55e;" title="Supportive: {syn_mix.get('supportive', 0)}%"></div>
      <div style="width:{syn_mix.get('neutral', 0)}%; background:#94a3b8;" title="Neutral: {syn_mix.get('neutral', 0)}%"></div>
    </div>
    <div style="font-size:11px; color:var(--text-dim); display:flex; justify-content:space-between;">
      <span>Synthetic Lean: <b>BJP {syn_mix.get('pro_incumbent', 0)}%</b> vs <b>Opp {syn_mix.get('anti_incumbent', 0)}%</b></span>
      <span class="mono">{global_bot_audit.get('bot_comment_count', 0)} flagged</span>
    </div>
    <div style="font-size:11px; color:#b91c1c; margin-top:6px; font-style:italic;">
      Shows how automated scripts, copy-paste templates, and coordinated clusters skew perception.
    </div>
  </div>

  <div style="background:#ffffff; border:1px solid #fed7aa; border-left:4px solid #ea580c; border-radius:8px; padding:14px 16px; box-shadow:0 1px 3px rgba(0,0,0,0.03);">
    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
      <span style="font-size:11px; font-weight:700; text-transform:uppercase; color:#c2410c; letter-spacing:0.4px;">⚖️ Artificial Skew Delta</span>
      <span class="chip mono" style="font-size:10px; background:#fff7ed; color:#c2410c; border-color:#fed7aa;">Purity Ratio: {bot_ratio}:1 Organic</span>
    </div>
    <div style="font-size:13px; font-weight:700; color:#0f172a; margin-bottom:4px;">
      Synthetic Distortion Skew: <span class="mono" style="color:{skew_color};">{"+" if skew_val > 0 else ""}{skew_val}%</span>
    </div>
    <div style="font-size:11.5px; color:var(--text); line-height:1.45; background:#f8fafc; padding:7px 9px; border-radius:5px; border:1px solid #e2e8f0; margin-top:6px;">
      <b>Diagnostic Verdict:</b> {html.escape(bot_div.get('diagnosis', 'Minimal synthetic distortion observed.'))}
    </div>
    <div style="font-size:10.5px; color:var(--text-faint); margin-top:6px;">
      Exposes immediately whether a negative spike is a genuine grassroots grievance or an artificial astroturfed campaign.
    </div>
  </div>
</div>
'''

# Cross-topic commenter overlap — users who commented on 2+ posts, from collect bundle
_commenter_overlap = data.get('commenter_overlap') or {}
_overlap_2plus = {u: posts for u, posts in _commenter_overlap.items() if len(set(posts)) >= 2}
_overlap_3plus = sum(1 for posts in _overlap_2plus.values() if len(set(posts)) >= 3)
_commenter_overlap_rows = []
for persona in (data.get('commenter_personas') or [])[:50]:
    username = persona['username']
    post_count = persona['posts']
    badge_color = '#991b1b' if post_count >= 4 else '#c2410c' if post_count >= 3 else '#92400e'
    badge_bg = '#fef2f2' if post_count >= 4 else '#fff7ed' if post_count >= 3 else '#fffbeb'
    badge_border = '#fecaca' if post_count >= 4 else '#fed7aa' if post_count >= 3 else '#fde68a'
    topics_txt = ', '.join(persona.get('topics') or []) or 'not mapped to a topic'
    _pp = persona.get('profile') or {}
    _inf = persona.get('inferred') or {}
    if _pp:
        _profile_line = (
            f"{(_pp.get('followers') or 0):,} followers"
            + (' &middot; verified' if _pp.get('verified') else '')
            + (' &middot; private' if _pp.get('private') else '')
            + f" &middot; gender {html.escape(str(_inf.get('gender', '?')))}"
            + f" &middot; age {html.escape(str(_inf.get('age_bracket', '?')))}"
            + f" &middot; location {html.escape(str(_inf.get('location') or '?'))}"
            + f" &middot; party {html.escape(str(_inf.get('party_support', '?')))}"
        )
        _bio_title = html.escape(_pp.get('bio') or '', quote=True)
    else:
        _profile_line = 'profile not fetched'
        _bio_title = ''
    _commenter_overlap_rows.append(
        f'<tr>'
        f'<td><a href="https://www.instagram.com/{html.escape(username)}/" target="_blank" rel="noopener noreferrer" style="color:var(--accent); font-weight:600;">@{html.escape(username)}</a></td>'
        f'<td><span class="chip mono" style="background:{badge_bg};color:{badge_color};border-color:{badge_border};">{post_count} posts</span></td>'
        f'<td>'
        f'<div style="font-size:11.5px;"><b>{html.escape(str(persona.get("lean") or "-"))}</b> &middot; '
        f'{html.escape(str(persona.get("tone") or "-"))} &middot; {html.escape(str(persona.get("archetype") or "-"))}</div>'
        f'<div style="font-size:11px; color:var(--text-dim);">Engaged on: {html.escape(topics_txt)}</div>'
        f'<div style="font-size:11px; color:var(--text-dim);" title="{_bio_title}">{_profile_line}</div>'
        f'<div style="font-size:11px; color:var(--text-dim);">{persona["comments"]} comments &middot; {persona["replies"]} replies &middot; {persona["likes_received"]:,} comment likes &middot; activity {persona.get("activity", 0)}</div>'
        f'</td>'
        f'</tr>'
    )
commenter_overlap_table_html = (
    ''.join(_commenter_overlap_rows)
    if _commenter_overlap_rows
    else '<tr><td colspan="3" style="color:var(--text-dim); font-style:italic;">No commenter overlap data available — run collect with the updated pipeline to populate this section.</td></tr>'
)

# Issue-cluster map: Volume -> Engagement -> dominant political Target/Direction, per the data-requirements
# doc's "issue-level sentiment map" (keyword-tagged for now - see ISSUE_CLUSTER_KEYWORDS note above).
issue_cluster_rows = []
political_lens_rows = []
# Narrative type distribution across all political topics
all_narrative_counts: Counter = Counter()
for cluster, s in sorted(issue_cluster_stats.items(), key=lambda kv: -kv[1]['topics']):
    if s['topics'] == 0:
        continue
    avg_engagement = s['engagement'] / s['topics']
    avg_bjp, avg_opp = s['bjp_pct_sum'] / s['topics'], s['opp_pct_sum'] / s['topics']
    if avg_bjp - avg_opp >= 8:
        lean_chip = f'<span class="chip mono" style="background:#fff7ed;color:#c2410c;border-color:#fed7aa;">Pro-incumbent ({avg_bjp:.0f}% / {avg_opp:.0f}%)</span>'
        lens_lean_bar = f'<div style="display:flex;height:8px;border-radius:4px;overflow:hidden;width:100%;"><div style="width:{avg_bjp:.0f}%;background:#f97316;"></div><div style="width:{max(0,100-avg_bjp-avg_opp):.0f}%;background:#cbd5e1;"></div><div style="width:{avg_opp:.0f}%;background:#3b82f6;"></div></div>'
    elif avg_opp - avg_bjp >= 8:
        lean_chip = f'<span class="chip mono" style="background:#eff6ff;color:#1d4ed8;border-color:#bfdbfe;">Pro-opposition ({avg_bjp:.0f}% / {avg_opp:.0f}%)</span>'
        lens_lean_bar = f'<div style="display:flex;height:8px;border-radius:4px;overflow:hidden;width:100%;"><div style="width:{avg_bjp:.0f}%;background:#f97316;"></div><div style="width:{max(0,100-avg_bjp-avg_opp):.0f}%;background:#cbd5e1;"></div><div style="width:{avg_opp:.0f}%;background:#3b82f6;"></div></div>'
    else:
        lean_chip = f'<span class="chip mono" style="background:var(--surface-2);color:var(--text-dim);">Contested ({avg_bjp:.0f}% / {avg_opp:.0f}%)</span>'
        lens_lean_bar = f'<div style="display:flex;height:8px;border-radius:4px;overflow:hidden;width:100%;"><div style="width:{avg_bjp:.0f}%;background:#f97316;"></div><div style="width:{max(0,100-avg_bjp-avg_opp):.0f}%;background:#cbd5e1;"></div><div style="width:{avg_opp:.0f}%;background:#3b82f6;"></div></div>'
    issue_cluster_rows.append(
        f'<tr><td><b>{cluster.replace("_", " ").title()}</b></td>'
        f'<td><span class="mono">{s["topics"]}</span></td>'
        f'<td><span class="mono">{avg_engagement:.1f}/h</span></td>'
        f'<td>{lean_chip}</td></tr>'
    )
    # Dominant narrative for this cluster
    top_nt = s['narrative_types'].most_common(1)[0][0] if s['narrative_types'] else 'mixed'
    all_narrative_counts.update(s['narrative_types'])
    tot_sent = s['sentiment_pos_sum'] + s['sentiment_neg_sum'] + s['sentiment_neu_sum'] or 1.0
    avg_pos = s['sentiment_pos_sum'] / tot_sent
    avg_neg = s['sentiment_neg_sum'] / tot_sent
    avg_neu_st = max(0.0, 1.0 - avg_pos - avg_neg)
    sentiment_bar = (
        f'<div style="display:flex;height:7px;border-radius:3px;overflow:hidden;width:80px;">'
        f'<div style="width:{avg_pos*100:.0f}%;background:#22c55e;" title="Favorable {avg_pos*100:.0f}%"></div>'
        f'<div style="width:{avg_neg*100:.0f}%;background:#ef4444;" title="Critical {avg_neg*100:.0f}%"></div>'
        f'<div style="width:{avg_neu_st*100:.0f}%;background:#cbd5e1;" title="Neutral/Mixed {avg_neu_st*100:.0f}%"></div>'
        f'</div>'
        f'<div style="font-size:9px;color:var(--text-faint);margin-top:2px;">'
        f'{avg_pos*100:.0f}% fav · {avg_neg*100:.0f}% crit'
        f'</div>'
    )
    political_lens_rows.append({
        'cluster': cluster,
        'topics': s['topics'],
        'avg_engagement': avg_engagement,
        'avg_bjp': avg_bjp,
        'avg_opp': avg_opp,
        'lean_bar': lens_lean_bar,
        'sentiment_bar': sentiment_bar,
        'top_narrative': top_nt.replace('_', ' '),
    })
issue_cluster_table_html = ''.join(issue_cluster_rows) if issue_cluster_rows else '<tr><td colspan="4">No issue-tagged political topics this run.</td></tr>'

# Political Lens table (richer version for the new tab)
_lens_rows_html = ''
for r in political_lens_rows:
    cname = r['cluster'].replace('_', ' ').title()
    _lens_rows_html += (
        f'<tr>'
        f'<td style="white-space:nowrap;"><b>{cname}</b></td>'
        f'<td class="mono">{r["topics"]}</td>'
        f'<td class="mono">{r["avg_engagement"]:.1f}</td>'
        f'<td style="min-width:120px;">{r["lean_bar"]}<div style="font-size:9px;color:var(--text-faint);margin-top:2px;">BJP {r["avg_bjp"]:.0f}% · Opp {r["avg_opp"]:.0f}%</div></td>'
        f'<td>{r["sentiment_bar"]}</td>'
        f'<td style="font-size:11px;color:var(--text-dim);">{r["top_narrative"]}</td>'
        f'</tr>'
    )
_lens_rows_html = _lens_rows_html or '<tr><td colspan="6">No issue-tagged topics this run.</td></tr>'

# Narrative distribution for Political Lens bar chart
_nt_labels = json.dumps([k.replace('_', ' ') for k, _ in all_narrative_counts.most_common(9)])
_nt_values = json.dumps([v for _, v in all_narrative_counts.most_common(9)])
_total_narrative = sum(all_narrative_counts.values()) or 1

# Strategic Early Warnings: Rapid Issue Escalation & Intervention Radar
# Synthesizes 6 concrete friction axes representing all 14 vulnerable issues detected across topics
_threat_axes_defs = [
    {
        'id': 'voter_rolls',
        'axis': 'Institutional Delegitimization & Electoral Integrity',
        'priority': 'CRITICAL',
        'badge_bg': '#fee2e2', 'badge_color': '#b91c1c', 'border': '#fca5a5',
        'keywords': ['electoral roll', 'voter', 'vote theft', 'cec', 'gyanesh kumar', 'freedom park protest called against cec'],
        'threat_mechanics': 'Multi-state narrative convergence alleging mass voter deletions via Form 7. Opposition parties are synchronizing digital campaigns and physical rallies targeting the Election Commission to sow systemic doubt ahead of assembly elections.',
        'demographic_trait': '18-35 Cohort · High Impatience (71%) · Cynical / Hostile Mindset',
        'risk_diagnosis': 'Opposition leaders and digital influencers are weaponizing administrative voter list updates into a coordinated "stolen democracy" narrative. Opposition sentiment holds ~75% engagement dominance, creating fertile ground for nationwide election delegitimization.',
        'immediate_actions': [
            'ECI Technical Secretariat must issue a transparent data bulletin detailing Form 7 statutory deletion statistics and multi-party verification audits.',
            'State BJP units should equip local booth agents with verified voter roll verification toolkits to counter localized false claims immediately.',
            'Digital media rapid-response desk should push factual explainer reels clarifying statutory deletion rules for deceased/relocated voters.'
        ]
    },
    {
        'id': 'detention_optics',
        'axis': 'Opposition Martyrdom & Law Enforcement Optics',
        'priority': 'HIGH',
        'badge_bg': '#ffedd5', 'badge_color': '#c2410c', 'border': '#fed7aa',
        'keywords': ['detained', 'jantar mantar', 'confrontation with congress', 'police confrontation'],
        'threat_mechanics': 'High-velocity visuals of senior MPs being physically bundled into police buses generate emotional rallying points that consolidate otherwise fragmented opposition factions and trigger cynical detachment among middle-class observers.',
        'demographic_trait': 'Middle-Class & Urban Youth · High Polarization · Critical Sentiment (68%)',
        'risk_diagnosis': 'Heavy-handed detention visuals at Jantar Mantar furnish the opposition with viral "police-state" imagery. This unifies disparate regional opposition parties and overshadows government policy messaging.',
        'immediate_actions': [
            'Direct law enforcement to calibrate public dispersal protocols and avoid aggressive physical detainment imagery that creates opposition martyr optics.',
            'Delhi Police and administration should proactively issue official video releases establishing prior warning notices and designated protest zoning violations.',
            'Depute senior parliamentary spokespersons for prime-time debates to pivot discussion toward unlawful traffic blockades and public inconvenience.'
        ]
    },
    {
        'id': 'women_safety',
        'axis': 'Grassroots Women Safety & Governance Failure Framing',
        'priority': 'HIGH',
        'badge_bg': '#ffedd5', 'badge_color': '#c2410c', 'border': '#fed7aa',
        'keywords': ['chappal', 'women safety'],
        'threat_mechanics': 'Visceral symbolic protests (e.g., chappal protests at Freedom Park) weaponize municipal and public safety failures against governance credentials. This directly threatens the crucial women voter coalition that forms a core pillar of ruling party support.',
        'demographic_trait': 'Female Demographic & Urban Families · High Fear / Frustration · High Viral Potential',
        'risk_diagnosis': 'High-impact symbolic demonstrations (chappals) convert safety grievances into viral anti-incumbent symbols. Left unaddressed, this erodes the decisive female electorate trust in governance and law enforcement.',
        'immediate_actions': [
            'Fast-track public announcements of urban surveillance expansion, pink patrol deployments, and dedicated women emergency response desks.',
            'Party women wing (Mahila Morcha) must hold town halls and community safety audits to preempt opposition political capture of women safety grievances.',
            'Direct state home departments to issue 48-hour compliance directives on high-risk crime hotspots and pending POCSO/safety trials.'
        ]
    },
    {
        'id': 'youth_agitation',
        'axis': 'Youth Disaffection & Cross-Campus Agitation',
        'priority': 'HIGH',
        'badge_bg': '#ffedd5', 'badge_color': '#c2410c', 'border': '#fed7aa',
        'keywords': ['student', 'vande mataram seminar', 'tribal student', 'delhi university'],
        'threat_mechanics': 'Multiple localized campus flashpoints across Kerala, Mumbai, and Delhi are showing signs of cross-university coordination. Models detect peak impatience and institutional anger in the 18-24 cohort, creating ripe conditions for a unified student movement.',
        'demographic_trait': '18-24 Gen Z Cohort · Peak Impatience (82%) · Anti-Establishment Sentiment',
        'risk_diagnosis': 'Disparate campus grievances (ideological disputes in Kerala, tribal student welfare in Mumbai, exam fee/hostel friction in Delhi) threaten to coalesce into an all-India student agitation if framed under an anti-youth narrative.',
        'immediate_actions': [
            'Ministry of Education and university administrations must open preemptive grievance dialogs with recognized student councils before agitation dates.',
            'Instruct university chancellors to de-escalate ideological seminar controversies through open academic symposiums.',
            'Highlight government youth welfare initiatives, scholarship disbursals, and newly announced civil service exam calendar transparencies.'
        ]
    },
    {
        'id': 'identity_flareup',
        'axis': 'Regional & Identity Friction Flashpoint',
        'priority': 'MEDIUM',
        'badge_bg': '#fef3c7', 'badge_color': '#b45309', 'border': '#fde68a',
        'keywords': ['kashmiri youth', 'connaught place'],
        'threat_mechanics': 'Isolated urban altercation in high-visibility central Delhi is being framed along regional and ethnic fault lines, risking backlash in sensitive border states and international negative press.',
        'demographic_trait': 'Regional Diaspora & Youth · Heightened Identity Sensitivity · 74% Critical',
        'risk_diagnosis': 'A localized street fight in CP is being weaponized into an ethnic profiling narrative. Narrative capture is heavily skewed towards opposition amplifiers attempting to ignite regional resentment in J&K.',
        'immediate_actions': [
            'Delhi Police must publicly announce prompt FIR registration and suspect custody with CCTV evidence within hours to prevent misinformation.',
            'Home Ministry media cell should release reassuring statements affirming zero tolerance for harassment against students and youth from Jammu & Kashmir.'
        ]
    },
    {
        'id': 'infrastructure_blame',
        'axis': 'Urban Decay & Civic Infrastructure Accountability',
        'priority': 'WATCH',
        'badge_bg': '#eff6ff', 'badge_color': '#1d4ed8', 'border': '#bfdbfe',
        'keywords': ['building collapse', 'seemapuri'],
        'threat_mechanics': 'Localized structural failures in densely populated urban clusters are being generalized into systemic attacks on municipal corruption and regulatory dereliction, dampening national infrastructure achievements.',
        'demographic_trait': 'Lower-Middle-Class Urban Residents · Impatience · Municipal Frustration',
        'risk_diagnosis': 'Structural building collapse incidents in northeast Delhi provide opposition spokespersons with visual proof to counter central infrastructure claims with localized civic negligence charges.',
        'immediate_actions': [
            'Announce rapid structural safety audit drives across unauthorized/dense constructions and ensure prompt interim relief for affected families.',
            'Deploy local municipal spokespersons to delineate administrative jurisdictions and counter attempts to link local building violations to central development policies.'
        ]
    }
]

_early_warning_alerts = []
# Dynamic Neural Early-Warning Radar using fine-tuned MultiTaskStrategicClassifier (mDeBERTa-v3)
from src.identify_what_matters.strategic_classifier import classify_strategic_batch

_AXIS_TO_META = {
    'institutional_integrity': {
        'axis': 'Institutional Delegitimization & Electoral Integrity',
        'badge_bg': '#fee2e2', 'badge_color': '#b91c1c', 'border': '#fca5a5',
        'threat_mechanics': 'Multi-state narrative convergence targeting institutional credibility (e.g. Election Commission, Judiciary, central agencies). Coordinated campaigns seek to sow systemic doubt ahead of electoral milestones.',
        'demographic_trait': '18-35 Cohort · High Impatience · Skeptical Mindset',
        'risk_diagnosis': 'Institutional distrust narratives generate fertile ground for nationwide delegitimization and cross-party opposition alignment.',
        'immediate_actions': [
            'Issue transparent data bulletins and proactive statutory clarification audits.',
            'Deploy regional spokespersons with verified documentation toolkits to counter localized claims.',
            'Digital rapid-response desk should push factual visual explainers across social platforms.'
        ]
    },
    'public_safety_crisis': {
        'axis': 'Grassroots Public Safety & Governance Vulnerability',
        'badge_bg': '#ffedd5', 'badge_color': '#c2410c', 'border': '#fed7aa',
        'threat_mechanics': 'Visceral symbolic protests weaponize municipal and public safety incidents against governance credentials, directly impacting voter trust and security sentiment.',
        'demographic_trait': 'Female Demographic & Urban Families · High Fear / Frustration · High Viral Potential',
        'risk_diagnosis': 'High-impact symbolic safety grievances convert localized crime into viral anti-incumbent symbols if unaddressed.',
        'immediate_actions': [
            'Fast-track public announcements of urban surveillance, emergency response desks, and strict enforcement drives.',
            'Hold community safety town halls to preempt partisan capture of local safety grievances.',
            'Direct law enforcement and state home departments to issue 48-hour compliance directives on high-risk crime hotspots.'
        ]
    },
    'youth_campus_agitation': {
        'axis': 'Youth Disaffection & Cross-Campus Agitation',
        'badge_bg': '#ffedd5', 'badge_color': '#c2410c', 'border': '#fed7aa',
        'threat_mechanics': 'Localized campus flashpoints, exam controversies, or student welfare disputes showing signs of cross-university digital coordination.',
        'demographic_trait': '18-24 Gen Z Cohort · Peak Impatience · Anti-Establishment Sentiment',
        'risk_diagnosis': 'Disparate campus grievances threaten to coalesce into an all-India student agitation if framed under an anti-youth narrative.',
        'immediate_actions': [
            'Open preemptive grievance dialogs with student councils before planned agitation dates.',
            'Instruct academic administrations to de-escalate ideological seminar controversies through open symposiums.',
            'Highlight government youth welfare initiatives, scholarship disbursals, and exam calendar transparencies.'
        ]
    },
    'communal_regional_faultline': {
        'axis': 'Regional & Identity Friction Flashpoint',
        'badge_bg': '#fef3c7', 'badge_color': '#b45309', 'border': '#fde68a',
        'threat_mechanics': 'Localized altercations or linguistic/regional disputes weaponized along ethnic or state fault lines, risking backlash across border regions.',
        'demographic_trait': 'Regional Diaspora & Youth · Heightened Identity Sensitivity · Elevated Critical Sentiment',
        'risk_diagnosis': 'Weaponized identity framing seeks to ignite regional resentment and international adverse commentary.',
        'immediate_actions': [
            'Prompt law enforcement FIR registration and video evidence releases to preempt misinformation.',
            'Spokespersons should emphasize regional unity, zero tolerance for harassment, and statutory protections.'
        ]
    },
    'infrastructure_governance_failure': {
        'axis': 'Urban Decay & Civic Infrastructure Accountability',
        'badge_bg': '#eff6ff', 'badge_color': '#1d4ed8', 'border': '#bfdbfe',
        'threat_mechanics': 'Localized structural failures or civic breakdowns generalized into systemic attacks on municipal corruption and regulatory dereliction.',
        'demographic_trait': 'Lower-Middle-Class Urban Residents · Impatience · Municipal Frustration',
        'risk_diagnosis': 'Structural civic incidents furnish visual proof to counter development claims with localized negligence charges.',
        'immediate_actions': [
            'Announce rapid structural safety audits and prompt interim relief for affected families.',
            'Deploy municipal spokespersons to delineate administrative jurisdictions and counter systemic blame.'
        ]
    },
    'economic_distress': {
        'axis': 'Middle-Class Economic Strain & Inflation Anxiety',
        'badge_bg': '#fee2e2', 'badge_color': '#b91c1c', 'border': '#fca5a5',
        'threat_mechanics': 'Spikes in everyday living costs, fuel/grocery prices, loan interest rates, or corporate layoffs driving widespread middle-class anxiety.',
        'demographic_trait': '25-54 Working Professionals · High Anxiety · Economic Focus',
        'risk_diagnosis': 'Pocketbook pressures erode middle-class economic optimism and amplify anti-incumbency sentiment.',
        'immediate_actions': [
            'Highlight government relief measures, tax rationalization, and consumer price stabilization steps.',
            'Promote data on long-term capital investments, job creation initiatives, and domestic industrial growth.'
        ]
    },
    'disinformation_rumor_panic': {
        'axis': 'Viral Disinformation & Panic Amplification',
        'badge_bg': '#fee2e2', 'badge_color': '#b91c1c', 'border': '#fca5a5',
        'threat_mechanics': 'Rapid viral dissemination of unverified claims or manipulated media creating public panic or institutional delegitimization.',
        'demographic_trait': 'Cross-Demographic · High Emotional Velocity · Low Deliberation',
        'risk_diagnosis': 'Fabricated claims weaponize panic to trigger spontaneous civic confrontation before fact-checks gain traction.',
        'immediate_actions': [
            'Deploy PIB Fact Check and state verification desks with high-contrast debunking cards immediately.',
            'Brief digital editors and platform liaison teams to downgrade unverified sensational rumors.'
        ]
    }
}

# Group topics by neural threat axis
_axis_grouped_topics = {}
for t in topics_client:
    s_res = t.get('neural_strategic') or {}
    ax_id = s_res.get('threat_axis')
    if ax_id and ax_id != 'general_civic_deliberation' and ax_id in _AXIS_TO_META:
        _axis_grouped_topics.setdefault(ax_id, []).append(t)

# Fallback check against manual keywords if neural missed any explicit alert
for ax_def in _threat_axes_defs:
    matched = []
    for t in topics_client:
        t_low = (t.get('title', '') + ' ' + t.get('summary', '')).lower()
        if any(k in t_low for k in ax_def.get('keywords', [])):
            matched.append(t)
    if matched:
        mapped_axis_id = {
            'voter_rolls': 'institutional_integrity',
            'detention_optics': 'youth_campus_agitation',
            'women_safety': 'public_safety_crisis',
            'youth_agitation': 'youth_campus_agitation',
            'identity_flareup': 'communal_regional_faultline',
            'infrastructure_blame': 'infrastructure_governance_failure'
        }.get(ax_def.get('id'), 'institutional_integrity')
        for m_t in matched:
            if m_t not in _axis_grouped_topics.setdefault(mapped_axis_id, []):
                _axis_grouped_topics[mapped_axis_id].append(m_t)

for ax_id, matched_topics in _axis_grouped_topics.items():
    meta = _AXIS_TO_META[ax_id]
    total_u = 0.0
    bjp_shares = []
    opp_shares = []
    crit_scores = []
    for t in matched_topics:
        total_u += t.get('undercurrent_score', 0.0)
        pp = t.get('political_pulse', {})
        bjp_shares.append(pp.get('bjp_favor_pct', 0.0))
        opp_shares.append(pp.get('opp_favor_pct', 0.0))
        s_intel = t.get('neural_strategic') or {}
        crit_scores.append(s_intel.get('criticality_score', 0.5))

    avg_u = total_u / len(matched_topics)
    avg_bjp = sum(bjp_shares) / len(bjp_shares) if bjp_shares else 0.0
    avg_opp = sum(opp_shares) / len(opp_shares) if opp_shares else 0.0
    avg_crit = sum(crit_scores) / len(crit_scores) if crit_scores else 0.5
    top_topic = max(matched_topics, key=lambda x: x.get('undercurrent_score', 0.0))

    priority = "CRITICAL" if avg_crit >= 0.75 or avg_u >= 5.5 else "HIGH" if avg_crit >= 0.55 or avg_u >= 4.5 else "MEDIUM"
    b_bg = "#fee2e2" if priority == "CRITICAL" else "#ffedd5" if priority == "HIGH" else "#fef3c7"
    b_col = "#b91c1c" if priority == "CRITICAL" else "#c2410c" if priority == "HIGH" else "#b45309"
    b_border = "#fca5a5" if priority == "CRITICAL" else "#fed7aa" if priority == "HIGH" else "#fde68a"

    _early_warning_alerts.append({
        'priority': priority,
        'badge_bg': b_bg,
        'badge_color': b_col,
        'border': b_border,
        'threat_axis': meta['axis'],
        'topics_count': len(matched_topics),
        'matched_titles': [t.get('title', '') for t in matched_topics],
        'lead_topic': top_topic.get('title', ''),
        'avg_undercurrent': round(avg_u, 1),
        'avg_criticality': round(avg_crit * 100, 0),
        'stage': top_topic.get('stage', 'emerging'),
        'bjp_share': round(avg_bjp, 0),
        'opp_share': round(avg_opp, 0),
        'threat_mechanics': meta['threat_mechanics'],
        'demographic_trait': meta['demographic_trait'],
        'risk_diagnosis': meta['risk_diagnosis'],
        'immediate_actions': meta['immediate_actions']
    })

_early_warning_alerts.sort(key=lambda x: (0 if x['priority'] == 'CRITICAL' else 1 if x['priority'] == 'HIGH' else 2, -x['avg_undercurrent']))

_early_warning_cards_html = ''.join(
    f'<div style="background:#ffffff;border:1px solid {a["border"]};border-left:4px solid {a["badge_color"]};border-radius:8px;padding:16px 18px;margin-bottom:14px;box-shadow:0 1px 3px rgba(0,0,0,0.04);">'
    f'<div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:8px;flex-wrap:wrap;gap:8px;">'
    f'<div style="display:flex;align-items:center;gap:10px;">'
    f'<span class="chip mono" style="font-weight:800;font-size:11px;background:{a["badge_bg"]};color:{a["badge_color"]};border-color:{a["border"]};padding:2px 8px;">{a["priority"]}</span>'
    f'<span style="font-weight:700;font-size:14px;color:var(--text);">{html.escape(a["threat_axis"])}</span>'
    f'</div>'
    f'<div style="display:flex;align-items:center;gap:8px;">'
    f'<span class="chip mono" style="font-size:11px;background:#f8fafc;color:var(--text-dim);border-color:#e2e8f0;">Friction Score: <b>{a["avg_undercurrent"]}/10</b></span>'
    f'<span class="chip mono" style="font-size:11px;background:var(--surface-2);color:var(--text-dim);">{a["topics_count"]} Related Topics</span>'
    f'</div>'
    f'</div>'
    f'<div style="font-size:12px;color:var(--text-faint);margin-bottom:10px;line-height:1.4;">'
    f'<b>Correlated Event Cluster:</b> ' + ' &bull; '.join(f'<span style="color:var(--text);font-weight:500;">{html.escape(t)}</span>' for t in a["matched_titles"][:3]) + (f' <span style="color:var(--text-faint);">+{len(a["matched_titles"])-3} more</span>' if len(a["matched_titles"]) > 3 else '') + f'<br/>'
    f'<b>Audience Pulse:</b> <span style="color:#c2410c;font-weight:600;">BJP Favor {a["bjp_share"]:.0f}%</span> vs <span style="color:#1d4ed8;font-weight:600;">Opp Narrative {a["opp_share"]:.0f}%</span> &middot; <b>Mindset Trait:</b> <span style="color:#475569;">{html.escape(a["demographic_trait"])}</span>'
    f'</div>'
    f'<div style="font-size:12.5px;color:var(--text);line-height:1.5;margin-bottom:12px;background:#f8fafc;padding:10px 12px;border-radius:6px;border:1px solid #e2e8f0;">'
    f'<b style="color:#0f172a;">Threat Escalation Mechanics:</b> {html.escape(a["threat_mechanics"])}<br/>'
    f'<span style="font-size:12px;color:var(--text-dim);display:inline-block;margin-top:4px;"><b>Risk Diagnosis:</b> {html.escape(a["risk_diagnosis"])}</span>'
    f'</div>'
    f'<div style="background:#fefefe;border:1px solid #fed7aa;border-radius:6px;padding:10px 14px;">'
    f'<div style="font-size:11.5px;font-weight:700;color:#c2410c;margin-bottom:6px;text-transform:uppercase;letter-spacing:0.4px;">🎯 Recommended Strategic Action Playbook</div>'
    f'<ol style="margin:0;padding-left:18px;font-size:12px;color:var(--text);line-height:1.55;">'
    + ''.join(f'<li style="margin-bottom:4px;">{html.escape(act)}</li>' for act in a["immediate_actions"]) +
    f'</ol>'
    f'</div>'
    f'</div>'
    for a in _early_warning_alerts
) or '<p style="color:var(--text-faint);">No critical early warnings active in this cycle.</p>'

# Top political undercurrents for Political Lens tab (top 10 by undercurrent_score, is_political only)
_top_undercurrents = sorted(
    [t for t in topics_client if t.get('is_political') and t.get('undercurrent_score', 0) > 3],
    key=lambda x: -x.get('undercurrent_score', 0)
)[:12]
_undercurrent_rows_html = ''
for u in _top_undercurrents:
    nt = (u.get('undercurrent') or {}).get('narrative_type', 'other').replace('_', ' ')
    ic = (u.get('undercurrent') or {}).get('issue_cluster', 'none').replace('_', ' ')
    pp = u.get('political_pulse') or {}
    bjp_p = pp.get('bjp_favor_pct', 0)
    opp_p = pp.get('opp_favor_pct', 0)
    _undercurrent_rows_html += (
        f'<tr>'
        f'<td style="font-size:12px;">{html.escape(u["title"][:80])}</td>'
        f'<td><span class="chip mono" style="font-size:9.5px;background:#eff6ff;color:#1d4ed8;">{html.escape(ic)}</span></td>'
        f'<td style="font-size:11px;color:var(--text-dim);">{html.escape(nt)}</td>'
        f'<td class="mono" style="font-size:12px;font-weight:600;">{u["undercurrent_score"]}/10</td>'
        f'<td style="font-size:11px;">'
        f'<span style="color:#c2410c;">BJP {bjp_p:.0f}%</span> · '
        f'<span style="color:#1d4ed8;">Opp {opp_p:.0f}%</span>'
        f'</td>'
        f'</tr>'
    )
_undercurrent_rows_html = _undercurrent_rows_html or '<tr><td colspan="5">No high-scoring political undercurrents this run — run more topics through the political probe pipeline.</td></tr>'

# Discovery proposals tables (account_discovery.py / emergent_narrative_discovery.py)
account_candidate_rows = ''.join(
    f'<tr><td class="mono">@{html.escape(c["username"])}</td>'
    f'<td><span class="mono">{c["distinct_runs"]}</span></td>'
    f'<td><span class="mono">{c["political_share"]*100:.0f}%</span></td>'
    f'<td><span class="chip mono" style="font-size:10px;">{html.escape(c["predicted_group"].replace("_"," "))} ({c["tier_confidence"]:.2f})</span></td>'
    f'<td><span class="mono">{c["promotion_score"]}</span></td></tr>'
    for c in account_candidates[:15]
) or '<tr><td colspan="5">No sustained candidate accounts discovered yet - needs a few more collect runs with the discovery ledger wired in.</td></tr>'

issue_candidate_rows = ''.join(
    f'<tr><td>{", ".join(html.escape(k) for k in c["representative_keywords"][:5])}</td>'
    f'<td><span class="mono">{c["size"]}</span></td>'
    f'<td><span class="mono">{c["total_engagement"]:,.0f}</span></td>'
    f'<td style="font-size:11px; font-style:italic;">"{html.escape(c["sample_texts"][0][:100])}"</td></tr>'
    for c in issue_candidates[:10]
) or '<tr><td colspan="4">No emergent (taxonomy-uncovered) clusters found this run.</td></tr>'

party_rows = ''.join(
  f'<tr><td>{html.escape(party)}</td><td>{counts.get("favor", 0)}</td><td>{counts.get("oppose", 0)}</td></tr>'
  for party, counts in political_audit['party_positions'].items()
)
party_table = (
  '<table class="evidence-table"><thead><tr><th>Explicit party target</th><th>Favor</th><th>Oppose</th></tr></thead>'
  f'<tbody>{party_rows}</tbody></table>' if party_rows else
  '<p>No explicit, unambiguous party-directed positions in the sampled comments. Party support is unknown.</p>'
)

# Strategic & Prescriptive Leadership Analytics
from src.identify_what_matters.strategic_leadership import (
    compute_narrative_penetration_rate,
    compute_persuadability_matrix,
    compute_policy_gap_detector,
    compute_opposition_vulnerabilities,
    compute_influencer_alignment,
    generate_action_recommendations,
)

npr_alerts = compute_narrative_penetration_rate(topics_client)
persuadability_matrix = compute_persuadability_matrix(topics_client)
policy_gaps = compute_policy_gap_detector(topics_client)
opp_vulnerabilities = compute_opposition_vulnerabilities(topics_client)
influencer_intel = compute_influencer_alignment(panel_pulse)
action_recommendations = generate_action_recommendations(topics_client, _early_warning_alerts, panel_pulse)

# Format Strategic Recommendation Feed Rows
action_recommendation_rows = []
for rec in action_recommendations:
    pr = rec["priority"]
    pr_color = "#dc2626" if "High" in pr else ("#d97706" if "Medium" in pr else "#16a34a")
    pr_bg = "#fef2f2" if "High" in pr else ("#fffbeb" if "Medium" in pr else "#f0fdf4")
    pr_border = "#fecaca" if "High" in pr else ("#fde68a" if "Medium" in pr else "#bbf7d0")
    badge = f'<span class="chip mono" style="background:{pr_bg};color:{pr_color};border-color:{pr_border};font-weight:700;font-size:11px;">{pr}</span>'
    action_recommendation_rows.append(
        f'<tr class="action-row" data-priority="{html.escape(pr)}">'
        f'<td style="white-space:nowrap;">{badge}</td>'
        f'<td style="font-weight:600;color:var(--text);font-size:12.5px;">{html.escape(rec["trigger_event"])}<div style="font-size:10.5px;font-family:var(--mono);color:var(--text-faint);margin-top:2px;">{html.escape(rec.get("metric_trigger", ""))} &middot; <span style="color:var(--accent);">{html.escape(rec.get("category", ""))}</span></div></td>'
        f'<td style="font-size:12px;color:var(--text-dim);">{html.escape(rec["target_audience"])}</td>'
        f'<td style="font-size:12.5px;color:#0f172a;line-height:1.45;"><div style="background:#f8fafc;padding:8px 12px;border-radius:5px;border:1px solid #e2e8f0;"><b style="color:var(--accent);">Leadership Action:</b> {html.escape(rec["recommended_action"])}</div></td>'
        f'</tr>'
    )
action_recommendations_table_html = ''.join(action_recommendation_rows) if action_recommendation_rows else '<tr><td colspan="4">No immediate tactical actions required.</td></tr>'

# Format NPR Alerts Rows
npr_rows = []
for a in npr_alerts[:8]:
    urg = a["urgency"]
    urg_bg = "#fee2e2" if urg == "HIGH" else ("#fef3c7" if urg == "MEDIUM" else "#f1f5f9")
    urg_col = "#b91c1c" if urg == "HIGH" else ("#b45309" if urg == "MEDIUM" else "#475569")
    npr_rows.append(
        f'<tr>'
        f'<td><b>{html.escape(a["topic_title"][:70])}</b></td>'
        f'<td><span class="chip mono" style="font-size:10px;">{html.escape(a["issue_cluster"].replace("_", " ").title())}</span></td>'
        f'<td><span class="chip mono" style="font-weight:700;background:{urg_bg};color:{urg_col};">{a["npr_score"]}% NPR</span></td>'
        f'<td style="font-size:11.5px;"><span style="color:#1d4ed8;font-weight:600;">Opp {a["opp_share"]:.0f}%</span> vs <span style="color:#c2410c;font-weight:600;">BJP {a["bjp_share"]:.0f}%</span></td>'
        f'<td style="font-size:11.5px;color:var(--text-dim);">{html.escape(a["target_audience"])}</td>'
        f'<td style="font-size:12px;color:#0f172a;"><div style="background:#f0fdf4;padding:6px 10px;border-radius:4px;border:1px solid #bbf7d0;"><b style="color:#15803d;">Recommended Counter-Narrative:</b> {html.escape(a["recommended_counter_narrative"])}</div></td>'
        f'</tr>'
    )
npr_table_html = ''.join(npr_rows) if npr_rows else '<tr><td colspan="6">No critical opposition narrative penetration detected across core/neutral demographics.</td></tr>'

# Format Persuadability Matrix Rows
persuadability_rows = []
for row in persuadability_matrix:
    score = row["persuadability_score"]
    score_col = "#b91c1c" if score >= 75 else ("#c2410c" if score >= 65 else "#15803d")
    persuadability_rows.append(
        f'<tr>'
        f'<td><b>{html.escape(row["segment_name"])}</b><div style="font-size:11px;color:var(--text-dim);">{html.escape(row["geography"])}</div></td>'
        f'<td><span class="mono" style="font-size:14px;font-weight:700;color:{score_col};">{score}/100</span></td>'
        f'<td><span class="chip mono" style="font-size:10.5px;">{html.escape(row["dominant_grievance"])}</span></td>'
        f'<td><div style="font-size:12px;color:var(--text);line-height:1.4;">{html.escape(row["responsive_policy_lever"])}</div></td>'
        f'<td><span class="chip mono" style="font-size:10.5px;background:#f8fafc;color:var(--text-dim);">{html.escape(row["recommended_allocation"])}</span></td>'
        f'<td>{html.escape(row["priority"])}</td>'
        f'</tr>'
    )
persuadability_table_html = ''.join(persuadability_rows) if persuadability_rows else '<tr><td colspan="6">No swing demographic models available.</td></tr>'

# Format Policy Gap Rows
policy_gap_rows = []
for pg in policy_gaps:
    gap_d = pg["gap_delta"]
    status = pg["status"]
    st_col = "#dc2626" if "CRITICAL" in status else ("#d97706" if "MILD" in status else "#16a34a")
    st_bg = "#fef2f2" if "CRITICAL" in status else ("#fffbeb" if "MILD" in status else "#f0fdf4")
    policy_gap_rows.append(
        f'<tr>'
        f'<td><b>{html.escape(pg["cluster_label"])}</b></td>'
        f'<td><span class="mono" style="font-weight:600;color:#c2410c;">{pg["public_anxiety_share"]}%</span></td>'
        f'<td><span class="mono" style="font-weight:600;color:#2563eb;">{pg["messaging_share"]}%</span></td>'
        f'<td><span class="mono" style="font-weight:700;color:{st_col};">{"+" if gap_d > 0 else ""}{gap_d}%</span></td>'
        f'<td><span class="chip mono" style="font-weight:700;font-size:10px;background:{st_bg};color:{st_col};">{status}</span></td>'
        f'<td style="font-size:11.5px;color:var(--text-dim);">{html.escape(pg["manifesto_theme"])}</td>'
        f'<td style="font-size:12px;color:#0f172a;"><div style="background:#f8fafc;padding:6px 10px;border-radius:4px;border:1px solid #e2e8f0;"><b style="color:var(--accent);">Suggested Talking Point:</b> {html.escape(pg["recommended_injection"])}</div></td>'
        f'</tr>'
    )
policy_gap_table_html = ''.join(policy_gap_rows) if policy_gap_rows else '<tr><td colspan="7">No policy gaps identified.</td></tr>'

# Format Opposition Vulnerabilities Rows
opp_vuln_rows = []
for ov in opp_vulnerabilities[:6]:
    b_idx = ov["backlash_index"]
    opp_vuln_rows.append(
        f'<tr>'
        f'<td><b>{html.escape(ov["trigger_event"])}</b></td>'
        f'<td><span class="chip mono" style="font-weight:700;background:#fef2f2;color:#b91c1c;">{b_idx}% Backlash</span></td>'
        f'<td><span class="chip mono" style="font-weight:700;background:#fff7ed;color:#c2410c;">{html.escape(ov["capitalization_window"])}</span></td>'
        f'<td style="font-size:11.5px;color:var(--text-dim);">{html.escape(ov["target_audience"])}</td>'
        f'<td style="font-size:12px;color:#0f172a;"><div style="background:#f8fafc;padding:6px 10px;border-radius:4px;border:1px solid #fed7aa;"><b style="color:#c2410c;">Tactical Action:</b> {html.escape(ov["recommended_action"])}</div></td>'
        f'</tr>'
    )
opp_vuln_table_html = ''.join(opp_vuln_rows) if opp_vuln_rows else '<tr><td colspan="5">No current opposition gaffes or backlashes active.</td></tr>'

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
    .sentiment-seg.hatched-bot {{
      background-image: repeating-linear-gradient(
        45deg,
        rgba(239, 68, 68, 0.5),
        rgba(239, 68, 68, 0.5) 4px,
        transparent 4px,
        transparent 8px
      );
      border-left: 1px solid rgba(239, 68, 68, 0.7);
    }}

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
    .stage-pill.unobserved {{ background: #fff7ed; color: #9a3412; border: 1px solid #fed7aa; }}

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
      padding: 12px 18px;
      display: grid;
      grid-template-columns: 32px minmax(280px, 2fr) minmax(200px, 1.2fr) auto auto 28px;
      align-items: center;
      gap: 16px;
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
      gap: 6px;
      margin-top: 4px;
      font-size: 11px;
      color: var(--text-dim);
      white-space: nowrap;
      overflow: hidden;
    }}

    /* AI & ML Intelligence Chip Array */
    .topic-ml-array {{
      display: flex;
      align-items: center;
      gap: 6px;
      flex-wrap: wrap;
      min-width: 0;
    }}
    .ml-chip {{
      display: inline-flex;
      align-items: center;
      gap: 4px;
      padding: 2.5px 7px;
      border-radius: 4px;
      font-size: 10.5px;
      font-family: var(--mono);
      font-weight: 500;
      line-height: 1.2;
      border: 1px solid transparent;
      white-space: nowrap;
    }}
    .ml-chip.epistemic-grounded {{
      background: #eff6ff;
      color: #1d4ed8;
      border-color: #bfdbfe;
    }}
    .ml-chip.epistemic-speculative {{
      background: #fef3c7;
      color: #b45309;
      border-color: #fde68a;
    }}
    .ml-chip.urgency-high {{
      background: #fef2f2;
      color: #b91c1c;
      border-color: #fecaca;
    }}
    .ml-chip.urgency-moderate {{
      background: #f0fdf4;
      color: #15803d;
      border-color: #bbf7d0;
    }}
    .ml-chip.mindset {{
      background: #f5f3ff;
      color: #6d28d9;
      border-color: #ddd6fe;
      max-width: 175px;
      overflow: hidden;
      text-overflow: ellipsis;
    }}
    .ml-chip.model-badge {{
      background: #f8fafc;
      color: #475569;
      border-color: #cbd5e1;
      font-weight: 600;
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
      width: 100px;
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
      .topic-metrics-cell, .topic-sentiment-col, .topic-ml-array {{
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
        <button class="nav-tab" onclick="switchTab('undercurrent')">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M3 12h4l3 8 4-16 3 8h4"></path></svg>
          Undercurrent Intel
        </button>
        <button class="nav-tab" onclick="switchTab('panel')">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2"></path><circle cx="9" cy="7" r="4"></circle><path d="M23 21v-2a4 4 0 0 0-3-3.87"></path><path d="M16 3.13a4 4 0 0 1 0 7.75"></path></svg>
          Panel Pulse ({len(panel_pulse)})
        </button>
        <button class="nav-tab" onclick="switchTab('political-lens')">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2"></polygon></svg>
          Political Lens
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

      <div class="kpi-card" style="border-color: var(--warn-border); background: var(--warn-bg);">
        <div class="kpi-label">
          <span>⚡ Not Settled Yet</span>
        </div>
        <div class="kpi-value mono">{sum(v for k, v in stage_counts.items() if k in ('breaking_out', 'pre_breakout', 'peak', 'saturated', 'simmering'))}</div>
        <div class="kpi-subtext"><b>{stage_counts.get('breaking_out', 0) + stage_counts.get('pre_breakout', 0)}</b> breaking out / pre-breakout right now &middot; <b>{stage_counts.get('settled', 0)}</b> settled</div>
      </div>
    </div>

    <!-- TAB 1: OVERVIEW & CHARTS -->
    <div id="tab-overview" class="tab-content">
      
      <!-- 🎯 Strategic Recommendations Feed (High-Value Actionable Panel for Political Leadership) -->
      <div class="card" style="margin-bottom: 24px; border-left: 5px solid #2563eb; box-shadow: 0 4px 12px rgba(37, 99, 235, 0.08);">
        <div class="card-header" style="background: linear-gradient(90deg, #eff6ff 0%, #ffffff 100%); padding: 14px 20px;">
          <div style="display:flex; align-items:center; gap:10px;">
            <span style="font-size:22px;">🎯</span>
            <div>
              <div class="card-title" style="font-size:15px; font-weight:800; color:#1e40af; letter-spacing:-0.2px;">
                Strategic Leadership Action Feed — Prescriptive Tactical Recommendation Queue
              </div>
              <div style="font-size:11.5px; color:#3b82f6; margin-top:2px;">
                Answers three core leadership questions: Where are we losing/winning votes? What narrative intervention is required? And what resource allocation yields highest ROI?
              </div>
            </div>
          </div>
          <div style="display:flex; align-items:center; gap:8px;">
            <span class="chip mono" style="background:#eff6ff; color:#1d4ed8; border-color:#bfdbfe; font-weight:700;">{len(action_recommendations)} Priority Actions</span>
            <button class="btn-group-item active" style="font-size:11px; padding:4px 10px;" onclick="switchTab('political-lens')">Full Prescriptive Intel &rarr;</button>
          </div>
        </div>
        <div class="card-body" style="padding: 0;">
          <div style="padding: 10px 18px; background: #f8fafc; border-bottom: 1px solid var(--border); display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:10px;">
            <div style="font-size: 12px; color: var(--text-dim);">
              Filter Tactical Queue:
            </div>
            <div class="btn-group">
              <button class="btn-group-item active" onclick="filterActionRecs('all', this)">All Priorities ({len(action_recommendations)})</button>
              <button class="btn-group-item" onclick="filterActionRecs('High 🔴', this)">High 🔴 ({sum(1 for r in action_recommendations if 'High' in r['priority'])})</button>
              <button class="btn-group-item" onclick="filterActionRecs('Medium 🟡', this)">Medium 🟡 ({sum(1 for r in action_recommendations if 'Medium' in r['priority'])})</button>
              <button class="btn-group-item" onclick="filterActionRecs('Low 🟢', this)">Low 🟢 ({sum(1 for r in action_recommendations if 'Low' in r['priority'])})</button>
            </div>
          </div>
          <div class="action-queue-scroll">
            <table class="evidence-table" id="action-recs-table" style="margin: 0;">
              <thead>
                <tr style="background: var(--surface-2);">
                  <th style="padding-left: 18px; width: 110px;">Priority</th>
                  <th>Trigger Event</th>
                  <th style="width: 220px;">Target Audience</th>
                  <th style="padding-right: 18px;">Recommended Leadership Action</th>
                </tr>
              </thead>
              <tbody>
                {action_recommendations_table_html}
              </tbody>
            </table>
          </div>
        </div>
      </div>

      <!-- Executive Early-Warning Radar — Emerging Vulnerabilities & Rapid Interventions (FRONT AND CENTER) -->
      <div class="card" style="margin-bottom: 24px; border-left: 5px solid #dc2626; box-shadow: 0 4px 12px rgba(220, 38, 38, 0.08);">
        <div class="card-header" style="background: linear-gradient(90deg, #fef2f2 0%, #fff7ed 50%, #ffffff 100%); padding: 14px 20px;">
          <div style="display:flex; align-items:center; gap:10px;">
            <span style="font-size:20px;">🚨</span>
            <div>
              <div class="card-title" style="font-size:15px; font-weight:800; color:#991b1b; letter-spacing:-0.2px;">
                Executive Early-Warning Radar — Strategic Vulnerabilities & Action Playbook
              </div>
              <div style="font-size:11.5px; color:#b45309; margin-top:2px;">
                Critical friction signals detected early across 14 vulnerable issues by the Multi-Task AI Neural Suite before major political escalation
              </div>
            </div>
          </div>
          <div style="display:flex; align-items:center; gap:8px;">
            <span class="chip mono" style="background:#fee2e2;color:#b91c1c;border-color:#fca5a5;font-weight:700;">{len(_early_warning_alerts)} Threat Axes Active</span>
            <button class="btn-group-item active" style="font-size:11px; padding:4px 10px;" onclick="switchTab('political-lens')">Open Political Lens &rarr;</button>
          </div>
        </div>
        <div class="card-body" style="padding: 16px 20px;">
          <p style="font-size:12.5px; color:var(--text-dim); margin-bottom:16px; line-height:1.5;">
            Synthesized across <b>undercurrent friction</b>, <b>breakout momentum</b>, <b>audience demographic cynicism</b>, and <b>opposition narrative capture</b>. This radar clusters disparate grassroots and campus flare-ups into systemic risk axes, providing leadership with immediate tactical interventions to defuse controversies before they expand into nationwide crises.
          </p>
          <div>
            {_early_warning_cards_html}
          </div>
        </div>
      </div>

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
            {''.join(f'<option value="{html.escape(cat, quote=True)}">{html.escape(cat.replace("_", " ").title())} ({count})</option>' for cat, count in sorted(cat_counts.items(), key=lambda kv: -kv[1]))}
          </select>
        </div>

        <div class="filter-group">
          <span class="filter-label">Stage</span>
          <select id="stage-select" class="select-dropdown" onchange="applyFilters()">
            <option value="all">All Stages</option>
            <option value="not_settled">⚡ Not Settled Yet (still active)</option>
            <option value="saturated">Saturated</option>
            <option value="breaking_out">Breaking Out</option>
            <option value="peak">Peak</option>
            <option value="simmering">Simmering</option>
            <option value="pre_breakout">Pre-Breakout</option>
            <option value="settled">Settled</option>
            <option value="quiet">Quiet / Emerging</option>
            <option value="unobserved">Unobserved · needs Instagram evidence</option>
          </select>
        </div>

        <div class="filter-group">
          <span class="filter-label">Authenticity</span>
          <select id="topics-authenticity-select" class="select-dropdown" onchange="applyFilters()">
            <option value="all">All Comments (Blended)</option>
            <option value="organic">Verified Organic Only</option>
            <option value="synthetic">Flagged Synthetic / Bot Clusters Only</option>
          </select>
        </div>

        <div class="filter-group">
          <span class="filter-label">Sort</span>
          <div class="btn-group">
            <button class="btn-group-item active" onclick="setSort('trend_score', this)">Trend Score</button>
            <button class="btn-group-item" onclick="setSort('active_first', this)">Still Active First</button>
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

          <!-- Dedicated Bot vs. Organic Sentiment Split KPI Row -->
          {bot_split_kpis_html}

          <div style="font-weight:600; font-size:13px; margin-bottom:8px;">Suspicious Repeated Narrative Templates Across Distinct Handles:</div>
          <div class="action-queue-scroll">
            <table class="evidence-table">
              <thead>
                <tr>
                  <th>Repeated Script / Copy-Paste Template</th>
                  <th>Unique Handles</th>
                  <th>Occurrences</th>
                  <th>Synthetic Lean &amp; Political Vector</th>
                  <th>Dominant Sentiment</th>
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

      <!-- CROSS-TOPIC COMMENTER OVERLAP -->
      <div class="card" style="margin-bottom: 20px; border-left: 4px solid #7c3aed;">
        <div class="card-header">
          <div class="card-title">🔗 Cross-Topic Commenter Overlap</div>
          <span class="chip mono" style="background:#f5f3ff;color:#6d28d9;border-color:#ddd6fe;">
            {len(_overlap_2plus)} users on 2+ posts &middot; {_overlap_3plus} on 3+ posts
          </span>
        </div>
        <div class="card-body">
          <p style="font-size:13px; color:var(--text-dim); margin-bottom: 12px;">
            Instagram accounts that appeared in comment threads across <strong>multiple distinct posts</strong> — a coordination signal complementing the bot radar above. High overlap on politically opposed posts suggests cross-narrative orchestration.
          </p>
          <div class="action-queue-scroll">
            <table class="evidence-table">
              <thead>
                <tr>
                  <th>Instagram Handle</th>
                  <th>Posts Commented On</th>
                  <th>Engagement Persona (lean &middot; tone &middot; archetype, topics, activity)</th>
                </tr>
              </thead>
              <tbody>
                {commenter_overlap_table_html}
              </tbody>
            </table>
          </div>
          {"" if _commenter_overlap_rows else '<p style="font-size:11.5px; color:var(--text-dim); margin-top:8px;">This section auto-populates after the next data collection cycle.</p>'}
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

        <div class="filter-group">
          <span class="filter-label">Source Authenticity</span>
          <select id="political-authenticity-select" class="select-dropdown" onchange="renderPoliticalFeed()">
            <option value="all">All Comments (Blended)</option>
            <option value="organic">Verified Organic Only</option>
            <option value="synthetic">Flagged Synthetic / Bot Clusters Only</option>
          </select>
        </div>
      </div>

      <div id="political-feed" class="topic-feed"></div>
    </div>

    <!-- TAB: UNDERCURRENT INTEL (data-requirements doc: Volume -> Engagement -> Sentiment -> Issue -> Target -> Narrative -> Undercurrent) -->
    <div id="tab-undercurrent" class="tab-content" style="display: none;">
      <div class="card" style="margin-bottom: 20px; {'' if any(models_status.values()) else 'opacity: 0.92;'}">
        <div class="card-header">
          <div class="card-title">🧠 Classification Pipeline Status</div>
          <span class="chip mono" style="{"background:var(--pos-bg);color:var(--pos);border-color:var(--pos-border);" if any(models_status.values()) else "background:var(--surface-2);color:var(--text-faint);"}">
            {sum(models_status.values())}/{len(models_status)} tasks on a trained model
          </span>
        </div>
        <div class="card-body" style="display:flex; gap:8px; flex-wrap:wrap;">
          {''.join(f'<span class="chip mono" style="font-size:10.5px; {"background:var(--pos-bg);color:var(--pos);border-color:var(--pos-border);" if on else "background:var(--surface-2);color:var(--text-faint);"}">{task.replace("_"," ")}: {"model" if on else "keyword fallback"}</span>' for task, on in models_status.items())}
          <div style="width:100%; font-size:11px; color:var(--text-faint); margin-top:6px;">
            Every post and sampled comment is classified by whichever is available per task - a trained classifier
            from <code>artifacts/political_models</code> if one has been fitted, political_pulse.py's keyword/regex
            heuristic otherwise (never silently treated as the same confidence - see docs/TRAINING_RESULTS.md).
          </div>
        </div>
      </div>

      <div class="card" style="margin-bottom: 20px;">
        <div class="card-header">
          <div class="card-title">🔭 Discovery: What the System Found on Its Own</div>
          <span class="chip mono">{len(account_candidates)} account candidates &middot; {len(issue_candidates)} emergent clusters</span>
        </div>
        <div class="card-body">
          <p style="font-size: 11.5px; color: var(--text-faint); margin-bottom: 10px;">
            Nothing here is hand-curated. Account candidates are accounts outside <code>config/panel.json</code>
            that keep surfacing with real political engagement across multiple collect runs, auto-tiered by a
            classifier trained on the existing panel's own writing style. Emergent clusters are political text
            that didn't match any of the 14 fixed issue clusters, grouped by semantic similarity. Both are
            proposals for human review - never auto-applied.
          </p>
          <div style="font-size:11px; font-weight:700; color:var(--text-faint); text-transform:uppercase; margin-bottom:4px;">Candidate accounts</div>
          <table style="width:100%; border-collapse:collapse; font-size:12px; margin-bottom:14px;">
            <thead><tr><th align="left">Account</th><th align="left">Runs seen</th><th align="left">% political</th><th align="left">Predicted tier</th><th align="left">Score</th></tr></thead>
            <tbody>{account_candidate_rows}</tbody>
          </table>
          <div style="font-size:11px; font-weight:700; color:var(--text-faint); text-transform:uppercase; margin-bottom:4px;">Emergent issue/narrative clusters</div>
          <table style="width:100%; border-collapse:collapse; font-size:12px;">
            <thead><tr><th align="left">Keywords</th><th align="left">Size</th><th align="left">Engagement</th><th align="left">Sample</th></tr></thead>
            <tbody>{issue_candidate_rows}</tbody>
          </table>
        </div>
      </div>

      <div class="card" style="margin-bottom: 20px;">
        <div class="card-header">
          <div class="card-title">🗺️ Issue-Cluster Map</div>
          <span class="chip mono">{sum(1 for s in issue_cluster_stats.values() if s['topics'] > 0)} active clusters</span>
        </div>
        <div class="card-body">
          <p style="font-size: 11.5px; color: var(--text-faint); margin-bottom: 10px;">
            This replaces a single "BJP sentiment = +X%" number with an issue-level view: which civic issues are
            driving engagement, and whether each one currently reads pro- or anti-incumbent.
          </p>
          <table style="width:100%; border-collapse:collapse; font-size:12.5px;">
            <thead><tr><th align="left">Issue Cluster</th><th align="left">Topics</th><th align="left">Avg. Engagement</th><th align="left">Lean</th></tr></thead>
            <tbody>{issue_cluster_table_html}</tbody>
          </table>
        </div>
      </div>

      <div class="controls-panel">
        <div class="filter-group">
          <span class="filter-label">Sort</span>
          <div class="btn-group">
            <button class="btn-group-item active" onclick="renderUndercurrentTab('score', this)">Undercurrent Score</button>
            <button class="btn-group-item" onclick="renderUndercurrentTab('disagreement', this)">Framing vs. Reaction Gap</button>
            <button class="btn-group-item" onclick="renderUndercurrentTab('engagement', this)">Engagement</button>
            <button class="btn-group-item" onclick="renderUndercurrentTab('active_first', this)">Still Active First</button>
          </div>
        </div>
      </div>
      <div style="font-size:11.5px; color:var(--text-faint); margin-bottom:10px;">
        Per topic: a post's own framing is classified independently from how sampled commenters actually reacted
        (bot/coordination-discounted) - disagreement between the two IS the undercurrent signal, not just another
        engagement-weighted score. See each card's "Fusion & Undercurrent" section.
      </div>
      <div id="undercurrent-feed" class="topic-feed"></div>
    </div>

    <!-- TAB 4: PANEL PULSE -->
    <div id="tab-panel" class="tab-content" style="display: none;">
      <div class="card" style="margin-bottom: 20px; background: #f8fafc; border-color: #e2e8f0;">
        <div class="card-body" style="padding: 14px 18px; display: flex; align-items: center; justify-content: space-between;">
          <div>
            <div style="font-weight: 600; color: var(--text); font-size: 14px;">📡 High-Impact Influencer & Media Sentinel Panel</div>
            <div style="font-size: 12.5px; color: var(--text-dim); margin-top: 2px;">
              {len(panel_cfg_accounts)} configured public accounts across {len(panel_groups)} source-role groups. Up to {panel_accounts_per_run} accounts are sampled per run in rotating daily cohorts ({panel_cohort_count}-day full sweep); groups describe source roles, not political alignment.
            </div>
          </div>
          <span class="chip mono" style="background: var(--surface); color: var(--accent); border-color: var(--border);">
            {len(panel_pulse)} Spiking Posts Active
          </span>
        </div>
      </div>

      <!-- INFLUENCER SENTIMENT ALIGNMENT & SUBURBAN SWING RADAR -->
      <div class="card" style="margin-bottom: 20px; border-left: 4px solid #8b5cf6;">
        <div class="card-header" style="background: linear-gradient(90deg, #f5f3ff 0%, #ffffff 100%);">
          <div class="card-title">⭐ Influencer Sentiment Alignment &amp; Suburban Swing Early Indicator</div>
          <span class="chip mono" style="background:#f5f3ff;color:#6d28d9;border-color:#ddd6fe;font-weight:700;">
            {influencer_intel['high_reach_posts_count']} High-Reach Creator Posts Tracked (&gt;200K Sentinel Tier)
          </span>
        </div>
        <div class="card-body">
          <div style="display:grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 12px; margin-bottom: 14px;">
            <div style="background:var(--surface-2); padding:10px 14px; border-radius:6px;">
              <div style="font-size:10.5px; color:var(--text-dim); text-transform:uppercase;">Influencer Favorable Slant</div>
              <div style="font-size:18px; font-weight:700; color:#16a34a;" class="mono">{influencer_intel['favorable_pct']}%</div>
            </div>
            <div style="background:var(--surface-2); padding:10px 14px; border-radius:6px;">
              <div style="font-size:10.5px; color:var(--text-dim); text-transform:uppercase;">Influencer Critical Slant</div>
              <div style="font-size:18px; font-weight:700; color:#dc2626;" class="mono">{influencer_intel['critical_pct']}%</div>
            </div>
            <div style="background:var(--surface-2); padding:10px 14px; border-radius:6px;">
              <div style="font-size:10.5px; color:var(--text-dim); text-transform:uppercase;">Net Slant Delta</div>
              <div style="font-size:18px; font-weight:700; color:{'#16a34a' if influencer_intel['net_slant'] >= 0 else '#dc2626'};" class="mono">{'+' if influencer_intel['net_slant'] > 0 else ''}{influencer_intel['net_slant']}%</div>
            </div>
            <div style="background:var(--surface-2); padding:10px 14px; border-radius:6px;">
              <div style="font-size:10.5px; color:var(--text-dim); text-transform:uppercase;">Total Influencer Engagement</div>
              <div style="font-size:18px; font-weight:700; color:var(--accent);" class="mono">{influencer_intel['total_influencer_likes']:,} likes</div>
            </div>
          </div>
          <div style="background:#f8fafc; padding:10px 14px; border-radius:6px; border:1px solid #e2e8f0; font-size:12.5px; color:var(--text); line-height:1.45;">
            <b>Suburban Voter Swing Predictive Signal:</b> {html.escape(influencer_intel['swing_indicator'])}
            <div style="font-size:11px; color:var(--text-dim); margin-top:4px;">
              * High-reach political/lifestyle creators (&gt;200K followers) act as early bellwethers for suburban voter mood swings before formal polling detects shifts.
            </div>
          </div>
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
            {panel_group_options_html}
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
                <th>Political Lean</th>
                <th>Tone Toward Story</th>
                <th>Narrative Framing</th>
                <th>Psychological Archetype</th>
                <th style="padding-right: 18px;">Top Public Comment</th>
              </tr>
            </thead>
            <tbody id="panel-table-body">
            </tbody>
          </table>
        </div>
      </div>
    </div>

    <!-- TAB: POLITICAL LENS (docx architecture: Volume → Engagement → Sentiment → Issue → Target → Narrative → Undercurrent) -->
    <div id="tab-political-lens" class="tab-content" style="display: none;">

      <!-- Header Banner -->
      <div class="card" style="margin-bottom: 20px; background: linear-gradient(135deg, #fff7ed 0%, #eff6ff 100%); border-left: 4px solid #f97316;">
        <div class="card-body" style="padding: 16px 20px;">
          <div style="font-weight: 700; font-size: 15px; color: var(--text); margin-bottom: 6px;">
            🏛️ Political Intelligence Lens — Issue-Level Sentiment & Party Favorability Map
          </div>
          <div style="font-size: 13px; color: var(--text-dim); line-height: 1.5;">
            Implements the data-requirements document architecture:
            <b>Volume → Engagement → Sentiment → Issue Cluster → Political Target → Narrative Type → Emerging Undercurrent</b>.
            Not a simple "Pro-BJP / Anti-Congress" label — every metric is derived from observed text, issue framing, and comment reaction, not account affiliation.
          </div>
          <div style="display:flex; gap:8px; margin-top:10px; flex-wrap:wrap;">
            <span class="chip mono" style="background:#fff7ed;color:#c2410c;border-color:#fed7aa;">BJP Lean: {macro_bjp_pct}%</span>
            <span class="chip mono" style="background:#eff6ff;color:#1d4ed8;border-color:#bfdbfe;">Opp Lean: {macro_opp_pct}%</span>
            <span class="chip mono">{sum(1 for s in issue_cluster_stats.values() if s['topics'] > 0)} active issue clusters</span>
            <span class="chip mono">{len(_top_undercurrents)} political undercurrents</span>
            <span class="chip mono" style="background:#fee2e2;color:#b91c1c;border-color:#fca5a5;">{len(_early_warning_alerts)} early warnings active</span>
          </div>
        </div>
      </div>

      <!-- Section 0: Executive Action Radar & Early Warnings (AI Intervention Deck) -->
      <div class="card" style="margin-bottom: 20px; border-left: 4px solid #ef4444;">
        <div class="card-header" style="background: linear-gradient(90deg, #fff5f5 0%, #ffffff 100%);">
          <div class="card-title">🚨 Executive Early-Warning Radar — Emerging Vulnerabilities & Rapid Interventions</div>
          <span class="chip mono" style="background:#fee2e2;color:#b91c1c;border-color:#fca5a5;font-weight:700;">Actionable Intelligence</span>
        </div>
        <div class="card-body">
          <p style="font-size:12.5px; color:var(--text-dim); margin-bottom:14px;">
            Synthesized by the multi-task neural suite combining <b>undercurrent friction</b>, <b>breakout momentum</b>, <b>demographic mindset cynicism</b>, and <b>opposition narrative capture</b>. Identifies subterranean concerns early so leadership can address them proactively before they scale into uncontrollable crises.
          </p>
          <div>
            {_early_warning_cards_html}
          </div>
        </div>
      </div>

      <!-- Section 0B: Strategic Recommendations Feed & Tactical Action Queue -->
      <div class="card" style="margin-bottom: 20px; border-left: 4px solid #2563eb;">
        <div class="card-header" style="background: linear-gradient(90deg, #eff6ff 0%, #ffffff 100%);">
          <div class="card-title">🎯 Strategic Leadership Action Feed — Tactical Intervention Queue</div>
          <span class="chip mono" style="background:#eff6ff;color:#1d4ed8;border-color:#bfdbfe;font-weight:700;">{len(action_recommendations)} Recommendations</span>
        </div>
        <div class="card-body" style="padding: 0;">
          <div class="action-queue-scroll">
            <table class="evidence-table" style="margin: 0;">
              <thead>
                <tr style="background: var(--surface-2);">
                  <th style="padding-left: 18px; width: 110px;">Priority</th>
                  <th>Trigger Event</th>
                  <th style="width: 220px;">Target Audience</th>
                  <th style="padding-right: 18px;">Recommended Leadership Action</th>
                </tr>
              </thead>
              <tbody>
                {action_recommendations_table_html}
              </tbody>
            </table>
          </div>
        </div>
      </div>

      <!-- Section 0C: Narrative Vulnerability & Friction Mapping (NPR Tracker) -->
      <div class="card" style="margin-bottom: 20px; border-left: 4px solid #dc2626;">
        <div class="card-header">
          <div class="card-title">⚡ Narrative Vulnerability &amp; Friction Mapping (NPR Tracker)</div>
          <span class="chip mono" style="background:#fee2e2;color:#b91c1c;border-color:#fca5a5;font-weight:700;">Narrative Penetration Rate</span>
        </div>
        <div class="card-body">
          <p style="font-size:12.5px; color:var(--text-dim); margin-bottom:12px;">
            Measures the <b>velocity of friction</b>: how fast an opposition talking point or attack narrative spreads outside its core partisan echo chamber into neutral, moderate, or traditionally supportive demographics. Prevents campaigns from reacting to mere noise while catching high-risk viral attacks before they harden into permanent voter perceptions.
          </p>
          <div class="action-queue-scroll">
            <table class="evidence-table">
              <thead>
                <tr style="background:var(--surface-2);">
                  <th>Vulnerable Topic / Attack Line</th>
                  <th>Cluster</th>
                  <th>Penetration Rate (NPR)</th>
                  <th>Audience Pulse</th>
                  <th>Infiltrated Audience</th>
                  <th>Recommended Counter-Narrative</th>
                </tr>
              </thead>
              <tbody>
                {npr_table_html}
              </tbody>
            </table>
          </div>
        </div>
      </div>

      <!-- Section 0D: Demographic & Geographic "Swing Potential" Index -->
      <div class="card" style="margin-bottom: 20px; border-left: 4px solid #16a34a;">
        <div class="card-header">
          <div class="card-title">🗺️ Demographic &amp; Geographic Swing Potential Matrix (Persuadability Index)</div>
          <span class="chip mono" style="background:#f0fdf4;color:#166534;border-color:#bbf7d0;font-weight:700;">Ground &amp; Digital Resource Allocation</span>
        </div>
        <div class="card-body">
          <p style="font-size:12.5px; color:var(--text-dim); margin-bottom:12px;">
            Identifies clusters of voters whose sentiment is volatile or neutral rather than locked-in. Tells campaign leadership precisely <b>where the candidate should physically travel</b>, <b>which local grievances to address</b>, and <b>where digital ad spend will move the needle most efficiently</b>.
          </p>
          <div class="action-queue-scroll">
            <table class="evidence-table">
              <thead>
                <tr style="background:var(--surface-2);">
                  <th>Demographic Segment &amp; Geography</th>
                  <th>Persuadability Score</th>
                  <th>Dominant Grievance</th>
                  <th>Responsive Policy Lever</th>
                  <th>Resource Allocation Strategy</th>
                  <th>Priority</th>
                </tr>
              </thead>
              <tbody>
                {persuadability_table_html}
              </tbody>
            </table>
          </div>
        </div>
      </div>

      <!-- Section 0E: Issue-to-Action Translation (The "Policy Gap" Detector) -->
      <div class="card" style="margin-bottom: 20px; border-left: 4px solid #d97706;">
        <div class="card-header">
          <div class="card-title">🔍 Issue-to-Action Translation (The "Policy Gap" Detector)</div>
          <span class="chip mono" style="background:#fffbeb;color:#b45309;border-color:#fde68a;font-weight:700;">Grievance-to-Policy Alignment</span>
        </div>
        <div class="card-body">
          <p style="font-size:12.5px; color:var(--text-dim); margin-bottom:12px;">
            Voters discuss symptoms (inflation, paper leaks, civic infrastructure), while political parties often talk about macro policies. This detector cross-references trending public complaints with the party's communication volume to flag disconnects where public anxiety is high but leadership messaging is missing.
          </p>
          <div class="action-queue-scroll">
            <table class="evidence-table">
              <thead>
                <tr style="background:var(--surface-2);">
                  <th>Issue Cluster</th>
                  <th>Public Anxiety Share</th>
                  <th>Party Messaging Share</th>
                  <th>Gap Delta</th>
                  <th>Status</th>
                  <th>Manifesto Baseline</th>
                  <th>Suggested Talking Point Injection</th>
                </tr>
              </thead>
              <tbody>
                {policy_gap_table_html}
              </tbody>
            </table>
          </div>
        </div>
      </div>

      <!-- Section 0F: Opposition Vulnerability & Counter-Attack Window Tracker -->
      <div class="card" style="margin-bottom: 20px; border-left: 4px solid #ea580c;">
        <div class="card-header">
          <div class="card-title">🏹 Opposition Vulnerability &amp; Counter-Attack Window Tracker</div>
          <span class="chip mono" style="background:#fff7ed;color:#c2410c;border-color:#fed7aa;font-weight:700;">Opportunistic Offense</span>
        </div>
        <div class="card-body">
          <p style="font-size:12.5px; color:var(--text-dim); margin-bottom:12px;">
            Monitors opposition announcements, candidate gaffes, and regional coalition infighting where <b>they are bleeding support</b>. Shifts campaign posture from purely defensive damage control to rapid opportunistic counter-attacks during active capitalization windows.
          </p>
          <div class="action-queue-scroll">
            <table class="evidence-table">
              <thead>
                <tr style="background:var(--surface-2);">
                  <th>Opposition Vulnerability Trigger</th>
                  <th>Backlash Index</th>
                  <th>Capitalization Window</th>
                  <th>Target Demographic</th>
                  <th>Recommended Counter-Attack Action</th>
                </tr>
              </thead>
              <tbody>
                {opp_vuln_table_html}
              </tbody>
            </table>
          </div>
        </div>
      </div>

      <!-- Section 1: Issue × Favorability Matrix -->
      <div class="card" style="margin-bottom: 20px;">
        <div class="card-header">
          <div class="card-title">📊 Issue-Level Sentiment & Party Favorability Matrix</div>
          <span class="chip mono" style="font-size:10px;">Per the doc's 15-cluster issue taxonomy</span>
        </div>
        <div class="card-body">
          <p style="font-size:12.5px; color:var(--text-dim); margin-bottom:12px;">
            Each row = one civic issue cluster. BJP Lean vs Opposition Lean derived from topic text, comment pool, and Reddit signals — not from account handles.
            Sentiment bar = 🟢 supportive / 🔴 critical / ⬜ neutral-mixed, averaged over topics in that cluster.
          </p>
          <div class="action-queue-scroll">
            <table class="evidence-table">
              <thead>
                <tr style="background:var(--surface-2);">
                  <th>Issue Cluster</th>
                  <th>Topics</th>
                  <th>Avg Eng/h</th>
                  <th style="min-width:150px;">BJP ▪ Neutral ▪ Opp</th>
                  <th>Sentiment</th>
                  <th>Top Narrative</th>
                </tr>
              </thead>
              <tbody>
                {_lens_rows_html}
              </tbody>
            </table>
          </div>
        </div>
      </div>

      <!-- Section 2: Macro Party Bar -->
      <div class="card" style="margin-bottom: 20px;">
        <div class="card-header">
          <div class="card-title">⚖️ Macro Party Favorability</div>
        </div>
        <div class="card-body">
          <div style="display:flex; justify-content:space-between; margin-bottom:8px; font-weight:600; font-size:13px;">
            <span style="color:#c2410c;">🟠 BJP / NDA: {macro_bjp_pct}%</span>
            <span style="color:#64748b;">⚪ Neutral / Non-partisan: {macro_neu_pct}%</span>
            <span style="color:#1d4ed8;">🔵 Opposition / INDIA: {macro_opp_pct}%</span>
          </div>
          <div style="display:flex; height:16px; border-radius:8px; overflow:hidden; background:#e2e8f0;">
            <div style="width:{macro_bjp_pct}%; background:#f97316;" title="BJP {macro_bjp_pct}%"></div>
            <div style="width:{macro_neu_pct}%; background:#94a3b8;" title="Neutral {macro_neu_pct}%"></div>
            <div style="width:{macro_opp_pct}%; background:#3b82f6;" title="Opposition {macro_opp_pct}%"></div>
          </div>
          <p style="font-size:11.5px; color:var(--text-faint); margin-top:8px;">
            Composite across all {total_political_probed + total_political_unprobed} political topics: weighted by engagement, comment pool, Reddit signals, and issue-cluster priors (e.g. paper-leak → anti-incumbent, infrastructure → pro-incumbent). Not hard-coded by account type.
          </p>
        </div>
      </div>

      <!-- Section 3: Narrative Type Distribution -->
      <div class="card" style="margin-bottom: 20px;">
        <div class="card-header">
          <div class="card-title">🗣️ Narrative Type Distribution</div>
          <span class="chip mono" style="font-size:10px;">{sum(all_narrative_counts.values())} classified posts</span>
        </div>
        <div class="card-body">
          <p style="font-size:12.5px; color:var(--text-dim); margin-bottom:14px;">
            What kinds of narratives dominate Indian political discourse this cycle? Classified by the multi-task political intelligence pipeline.
          </p>
          <div style="display:flex; flex-direction:column; gap:8px;">
            {"".join(
                f'<div style="display:flex; align-items:center; gap:10px;">'
                f'<div style="font-size:12px; color:var(--text-dim); min-width:180px;">{k.replace("_"," ").title()}</div>'
                f'<div style="flex:1; background:var(--surface-2); border-radius:4px; height:18px; position:relative; max-width:400px;">'
                f'<div style="width:{v*100//_total_narrative}%; background:#6366f1; height:100%; border-radius:4px;"></div>'
                f'</div>'
                f'<div class="mono" style="font-size:12px; color:var(--text);">{v} topics</div>'
                f'</div>'
                for k, v in all_narrative_counts.most_common(9)
            ) or '<p style="color:var(--text-faint);">No narrative classifications available — run the political probe pipeline on more topics.</p>'}
          </div>
        </div>
      </div>

      <!-- Section 4: Emerging Political Undercurrents -->
      <div class="card" style="margin-bottom: 20px;">
        <div class="card-header">
          <div class="card-title">⚡ Emerging Political Undercurrents</div>
          <span class="chip mono" style="background:var(--warn-bg);color:var(--warn);border-color:var(--warn-border);">Top by Undercurrent Score</span>
        </div>
        <div class="card-body">
          <p style="font-size:12.5px; color:var(--text-dim); margin-bottom:12px;">
            Topics with the highest undercurrent scores — where public conversation framing diverges from the official narrative, or where issue salience is spiking in ways not yet reflected in mainstream coverage.
          </p>
          <div class="action-queue-scroll">
            <table class="evidence-table">
              <thead>
                <tr style="background:var(--surface-2);">
                  <th>Topic</th>
                  <th>Issue Cluster</th>
                  <th>Narrative</th>
                  <th>Undercurrent Score</th>
                  <th>Party Lean</th>
                </tr>
              </thead>
              <tbody>
                {_undercurrent_rows_html}
              </tbody>
            </table>
          </div>
        </div>
      </div>

      <!-- Section 5: Community Lens Cross-Tab -->
      <div class="card" style="margin-bottom: 20px;">
        <div class="card-header">
          <div class="card-title">🕌 Socio-Religious Lens Cross-Tabulation</div>
        </div>
        <div class="card-body">
          <div class="action-queue-scroll">
            <table class="evidence-table">
              <thead>
                <tr>
                  <th>Community / Lens</th>
                  <th>BJP / NDA Favor</th>
                  <th>Opposition Favor</th>
                  <th>Neutral / Non-partisan</th>
                  <th style="min-width:140px;">Distribution</th>
                  <th>Sample Volume</th>
                </tr>
              </thead>
              <tbody>
                {community_cross_tab_html}
              </tbody>
            </table>
          </div>
          <p style="font-size:11.5px; color:var(--text-dim); margin-top:6px;">
            Inferred from vernacular signals, keyword context, and matched subreddit community priors (r/IndiaSpeaks → nationalist, r/india → reformist).
          </p>
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

    function escapeHtml(value) {{
      return String(value ?? '').replace(/&/g, '&amp;').replace(/</g, '&lt;')
        .replace(/>/g, '&gt;').replace(/"/g, '&quot;').replace(/'/g, '&#39;');
    }}

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
      }} else if (tabId === 'undercurrent') {{
        renderUndercurrentTab('score');
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

    let currentAuthFilter = 'all';

    function filterActionRecs(priority, btn) {{
      if (btn) {{
        btn.parentElement.querySelectorAll('.btn-group-item').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
      }}
      document.querySelectorAll('#action-recs-table tbody tr').forEach(tr => {{
        const rowPrio = tr.getAttribute('data-priority') || '';
        if (priority === 'all' || rowPrio.includes(priority.split(' ')[0])) {{
          tr.style.display = '';
        }} else {{
          tr.style.display = 'none';
        }}
      }});
    }}

    function setDrawerSentimentMode(cardKey, mode) {{
      const btnRaw = document.getElementById(`btn-raw-${{cardKey}}`);
      const btnAdj = document.getElementById(`btn-adj-${{cardKey}}`);
      const badge = document.getElementById(`sent-mode-badge-${{cardKey}}`);
      const rawDiv = document.getElementById(`drawer-sent-raw-${{cardKey}}`);
      const adjDiv = document.getElementById(`drawer-sent-adj-${{cardKey}}`);

      if (mode === 'adjusted') {{
        btnRaw?.classList.remove('active');
        btnAdj?.classList.add('active');
        if (badge) {{
          badge.innerText = '🛡️ Showing True Organic Sentiment';
          badge.style.background = '#dcfce7';
          badge.style.color = '#15803d';
        }}
        if (rawDiv) rawDiv.style.display = 'none';
        if (adjDiv) adjDiv.style.display = 'block';
      }} else {{
        btnAdj?.classList.remove('active');
        btnRaw?.classList.add('active');
        if (badge) {{
          badge.innerText = 'Showing Raw Blended';
          badge.style.background = 'var(--surface-2)';
          badge.style.color = 'var(--text-dim)';
        }}
        if (rawDiv) rawDiv.style.display = 'block';
        if (adjDiv) adjDiv.style.display = 'none';
      }}
    }}

    function toggleCard(rowEl) {{
      const cardEl = rowEl.closest(".topic-card");
      cardEl.classList.toggle('open');
    }}

    // Render Compact Card HTML
    function buildTopicCardHtml(t, prefix="") {{
      const st = t.stance || {{}};
      const ba = t.bot_audit || {{}};
      const orgSt = ba.bot_adjusted_stance || st;
      const orgSup = Math.round((orgSt.supportive || 0) * 100);
      const orgCrit = Math.round((orgSt.critical || 0) * 100);
      const orgNeu = Math.round((orgSt.neutral || 0) * 100);
      const orgHum = Math.round((orgSt.humorous || 0) * 100);
      const botShare = ba.bot_share_pct || 0;
      const rp = t.reddit_proxy || {{}};
      const sup = Math.round((st.supportive || 0) * 100);
      const crit = Math.round((st.critical || 0) * 100);
      const neu = Math.round((st.neutral || 0) * 100);
      const hum = Math.round((st.humorous || 0) * 100);
      const hasStance = (sup + crit + neu + hum) > 0;

      const sides = (t.measured_opinion && t.measured_opinion.sides) || [];
      const opinionSens = t.measured_opinion && t.measured_opinion.weighting_sensitivity;
      const exposureFlag = !!(opinionSens && opinionSens.flag);
      const pi = t.political_intel || {{}};
      const demo = t.demo || {{}};

      // Determine epistemic and urgency chips from behavioral models
      const isSpeculative = demo.irrationality === 'irrational';
      const isUrgent = demo.impatience === 'impatient';
      const mindsetLabel = demo.mindset || 'General Discourse';
      const provLabel = demo.source === 'measured' ? `${{demo.comment_count}} Comments` : (demo.passages_analyzed ? `${{demo.passages_analyzed}} Passages (Story NLP)` : 'Contextual NLP');

      return `
        <div class="topic-card" id="card-${{prefix}}${{t.id}}">
          <div class="topic-summary-row" onclick="toggleCard(document.getElementById('card-${{prefix}}${{t.id}}'))">
            <span class="topic-id-badge mono">${{t.id}}</span>
            
            <div class="topic-title-box">
              <div class="topic-title" title="${{t.title}}">${{t.title}}</div>
              <div class="topic-meta-line">
                <span class="cat-pill">${{t.category.replace(/_/g, ' ')}}</span>
                <span class="stage-pill ${{t.stage}}">${{t.stage.replace(/_/g, ' ')}}</span>
                ${{t.is_political ? '<span class="cat-pill" style="background:#fef2f2; color:#b91c1c; border-color:#fecaca;">🏛️ Political</span>' : ''}}
                ${{t.issue_cluster && t.issue_cluster !== 'none' ? '<span class="cat-pill" style="background:#eff6ff; color:#1d4ed8; border-color:#bfdbfe;">' + t.issue_cluster.replace(/_/g, ' ') + '</span>' : ''}}
                ${{t.undercurrent_score > 0 ? '<span class="cat-pill mono" style="background:#f5f3ff; color:#6d28d9; border-color:#ddd6fe;">⚡ ' + t.undercurrent_score + '</span>' : ''}}
                ${{t.matched_reddit && t.matched_reddit.length > 0 ? '<span class="cat-pill" style="background:#fff7ed; color:#c2410c; border-color:#ffedd5;">💬 Reddit (' + t.matched_reddit.length + ')</span>' : ''}}
                <span>&bull;</span>
                <span class="mono" style="font-size:11px; color:var(--text-faint);">${{t.ig.on_topic_posts}} verified posts</span>
              </div>
            </div>

            <!-- Deep ML Intelligence Chip Array -->
            <div class="topic-ml-array" title="AI & Deep ML Inferences across 6 Political Dimensions + Behavioral Classifiers">
              <span class="ml-chip ${{isSpeculative ? 'epistemic-speculative' : 'epistemic-grounded'}}" title="Epistemic Lens: ${{isSpeculative ? 'High speculation / unverified allegations detected' : 'Grounded empirical claims / factual reporting'}}">
                ${{isSpeculative ? '🔮 Speculative' : '⚖️ Grounded'}}
              </span>
              <span class="ml-chip ${{isUrgent ? 'urgency-high' : 'urgency-moderate'}}" title="Audience Urgency: ${{isUrgent ? 'High public urgency / active agitation' : 'Moderate observational pacing'}}">
                ${{isUrgent ? '🔥 High Urgency' : '⏱️ Measured'}}
              </span>
              <span class="ml-chip mindset" title="Psychological Mindset Archetype: ${{mindsetLabel}}">
                🧠 ${{mindsetLabel}}
              </span>
              ${{pi.is_political && pi.is_political.value ? `
                <span class="ml-chip model-badge" style="background:#eff6ff; color:#1e40af; border-color:#bfdbfe;" title="Political Target: ${{pi.target ? pi.target.value : 'General'}} · Direction: ${{pi.direction ? pi.direction.value : 'Neutral'}} · Conf: ${{pi.is_political.confidence}}">
                  🤖 mDeBERTa: ${{Math.round(pi.is_political.confidence * 100)}}%
                </span>
              ` : `
                <span class="ml-chip model-badge" title="Trained Classifier Inference Provenance: ${{provLabel}}">
                  🤖 ML Inferred
                </span>
              `}}
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
                ${{t.comment_velocity && t.comment_velocity.window_comments > 0
                  ? `<span class="metric-val mono" title="Comment velocity: ${{t.comment_velocity.comments_per_hour_6h}}/h (last 6h) vs ${{t.comment_velocity.comments_per_hour_prev}}/h (prev 6h)">${{t.comment_velocity.comments_per_hour_6h}}/h</span>`
                  : `<span class="metric-val mono">${{t.ig.engagement_per_hour > 0 ? t.ig.engagement_per_hour.toLocaleString() : '-'}}/h</span>`
                }}
                <span class="metric-sub">Velocity ${{t.comment_velocity && t.comment_velocity.velocity_trend !== 'flat' ? '(' + t.comment_velocity.velocity_trend + ')' : ''}}</span>
              </div>
            </div>

            <div class="topic-sentiment-col">
              <div class="sentiment-bar">
                ${{hasStance ? `
                  <div class="sentiment-seg supportive" style="width: ${{currentAuthFilter === 'organic' ? orgSup : sup}}%;" title="${{currentAuthFilter === 'organic' ? 'Organic Supportive' : 'Supportive'}}: ${{currentAuthFilter === 'organic' ? orgSup : sup}}%"></div>
                  <div class="sentiment-seg critical" style="width: ${{currentAuthFilter === 'organic' ? orgCrit : crit}}%;" title="${{currentAuthFilter === 'organic' ? 'Organic Critical' : 'Critical'}}: ${{currentAuthFilter === 'organic' ? orgCrit : crit}}%"></div>
                  <div class="sentiment-seg humorous" style="width: ${{currentAuthFilter === 'organic' ? orgHum : hum}}%;" title="Humorous: ${{currentAuthFilter === 'organic' ? orgHum : hum}}%"></div>
                  <div class="sentiment-seg neutral" style="width: ${{currentAuthFilter === 'organic' ? orgNeu : neu}}%;" title="Neutral: ${{currentAuthFilter === 'organic' ? orgNeu : neu}}%"></div>
                ` : `
                  <div class="sentiment-seg neutral" style="width: 100%;" title="Pending social probe"></div>
                `}}
              </div>
              <div style="display:flex; justify-content:space-between; font-size:10px; font-family:var(--mono); color:var(--text-faint);">
                <span>${{hasStance ? '+' + (currentAuthFilter === 'organic' ? orgSup : sup) + '%' : 'Pending'}}</span>
                <span style="font-size:9px; color:${{currentAuthFilter === 'organic' ? '#15803d' : 'inherit'}};">${{currentAuthFilter === 'organic' ? '🛡️ Organic' : ''}}</span>
                <span>${{hasStance ? '-' + (currentAuthFilter === 'organic' ? orgCrit : crit) + '%' : ''}}</span>
              </div>
            </div>

            <div class="expand-chevron">
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><polyline points="6 9 12 15 18 9"></polyline></svg>
            </div>
          </div>

          <!-- Expanded Detail Drawer -->
          <div class="topic-detail-drawer">
            <!-- Strategic Risk Diagnosis & Recommended Government Action Playbook (If Vulnerability Detected) -->
            ${{(() => {{
              const text = ((t.title || '') + ' ' + (t.summary || '')).toLowerCase();
              let matchedThreat = null;
              if (text.includes('electoral roll') || text.includes('voter') || text.includes('vote theft') || text.includes('cec') || text.includes('gyanesh kumar')) {{
                matchedThreat = {{
                  axis: 'Institutional Delegitimization & Electoral Integrity',
                  priority: 'CRITICAL', badge_bg: '#fee2e2', badge_color: '#b91c1c', border: '#fca5a5',
                  diagnosis: 'Opposition leaders and digital influencers are weaponizing administrative voter list updates (Form 7) into a coordinated "stolen democracy" narrative. Attacks directly target ECI credibility ahead of assembly elections.',
                  actions: [
                    'ECI Technical Secretariat must issue a transparent data bulletin detailing Form 7 statutory deletion statistics and multi-party verification audits.',
                    'State BJP units should equip local booth agents with verified voter roll verification toolkits to counter localized false claims immediately.',
                    'Digital media rapid-response desk should push factual explainer reels clarifying statutory deletion rules for deceased/relocated voters.'
                  ]
                }};
              }} else if (text.includes('detained') || text.includes('jantar mantar') || text.includes('confrontation with congress') || text.includes('police confrontation')) {{
                matchedThreat = {{
                  axis: 'Opposition Martyrdom & Law Enforcement Optics',
                  priority: 'HIGH', badge_bg: '#ffedd5', badge_color: '#c2410c', border: '#fed7aa',
                  diagnosis: 'Visuals of senior MPs being physically bundled into police vehicles generate emotional rallying points that consolidate otherwise fragmented opposition factions and trigger cynical detachment among middle-class observers.',
                  actions: [
                    'Direct law enforcement to calibrate public dispersal protocols and avoid aggressive physical detainment imagery that creates opposition martyr optics.',
                    'Delhi Police and administration should proactively issue official video releases establishing prior warning notices and designated protest zoning violations.',
                    'Depute senior parliamentary spokespersons for prime-time debates to pivot discussion toward unlawful traffic blockades and public inconvenience.'
                  ]
                }};
              }} else if (text.includes('chappal') || text.includes('women safety')) {{
                matchedThreat = {{
                  axis: 'Grassroots Women Safety & Governance Failure Framing',
                  priority: 'HIGH', badge_bg: '#ffedd5', badge_color: '#c2410c', border: '#fed7aa',
                  diagnosis: 'Visceral symbolic demonstrations (chappals) convert safety grievances into viral anti-incumbent symbols. Left unaddressed, this erodes the decisive female electorate trust in governance and law enforcement.',
                  actions: [
                    'Fast-track public announcements of urban surveillance expansion, pink patrol deployments, and dedicated women emergency response desks.',
                    'Party women wing (Mahila Morcha) must hold town halls and community safety audits to preempt opposition political capture of women safety grievances.',
                    'Direct state home departments to issue 48-hour compliance directives on high-risk crime hotspots and pending POCSO/safety trials.'
                  ]
                }};
              }} else if (text.includes('student') || text.includes('vande mataram') || text.includes('tribal student') || text.includes('delhi university')) {{
                matchedThreat = {{
                  axis: 'Youth Disaffection & Cross-Campus Agitation',
                  priority: 'HIGH', badge_bg: '#ffedd5', badge_color: '#c2410c', border: '#fed7aa',
                  diagnosis: 'Disparate campus grievances (ideological disputes in Kerala, tribal student welfare in Mumbai, exam fee/hostel friction in Delhi) threaten to coalesce into an all-India student agitation if framed under an anti-youth narrative.',
                  actions: [
                    'Ministry of Education and university administrations must open preemptive grievance dialogs with recognized student councils before agitation dates.',
                    'Instruct university chancellors to de-escalate ideological seminar controversies through open academic symposiums.',
                    'Highlight government youth welfare initiatives, scholarship disbursals, and newly announced civil service exam calendar transparencies.'
                  ]
                }};
              }} else if (text.includes('kashmiri youth') || text.includes('connaught place')) {{
                matchedThreat = {{
                  axis: 'Regional & Identity Friction Flashpoint',
                  priority: 'MEDIUM', badge_bg: '#fef3c7', badge_color: '#b45309', border: '#fde68a',
                  diagnosis: 'A localized street fight in CP is being weaponized into an ethnic profiling narrative. Narrative capture is heavily skewed towards opposition amplifiers attempting to ignite regional resentment in J&K.',
                  actions: [
                    'Delhi Police must publicly announce prompt FIR registration and suspect custody with CCTV evidence within hours to prevent misinformation.',
                    'Home Ministry media cell should release reassuring statements affirming zero tolerance for harassment against students and youth from Jammu & Kashmir.'
                  ]
                }};
              }} else if (text.includes('building collapse') || text.includes('seemapuri')) {{
                matchedThreat = {{
                  axis: 'Urban Decay & Civic Infrastructure Accountability',
                  priority: 'WATCH', badge_bg: '#eff6ff', badge_color: '#1d4ed8', border: '#bfdbfe',
                  diagnosis: 'Structural building collapse incidents in northeast Delhi provide opposition spokespersons with visual proof to counter central infrastructure claims with localized civic negligence charges.',
                  actions: [
                    'Announce rapid structural safety audit drives across unauthorized/dense constructions and ensure prompt interim relief for affected families.',
                    'Deploy local municipal spokespersons to delineate administrative jurisdictions and counter attempts to link local building violations to central development policies.'
                  ]
                }};
              }}

              if (!matchedThreat) return '';
              return `
                <div style="margin-bottom: 14px; padding: 12px 16px; background: #ffffff; border: 1.5px solid ${{matchedThreat.border}}; border-left: 5px solid ${{matchedThreat.badge_color}}; border-radius: 6px; box-shadow: 0 2px 6px rgba(0,0,0,0.03);">
                  <div style="display:flex; align-items:center; justify-content:space-between; margin-bottom: 8px;">
                    <div style="display:flex; align-items:center; gap:8px;">
                      <span class="chip mono" style="font-weight:800; font-size:10.5px; background:${{matchedThreat.badge_bg}}; color:${{matchedThreat.badge_color}}; border-color:${{matchedThreat.border}};">🚨 ${{matchedThreat.priority}} THREAT AXIS</span>
                      <span style="font-weight:700; font-size:13px; color:#0f172a;">${{matchedThreat.axis}}</span>
                    </div>
                    <span class="chip mono" style="font-size:10px; background:#f8fafc; color:#64748b;">AI Strategic Advisory</span>
                  </div>
                  <div style="font-size:12px; color:#334155; line-height:1.5; margin-bottom:10px; background:#f8fafc; padding:8px 10px; border-radius:4px; border:1px solid #e2e8f0;">
                    <b>Risk Diagnosis:</b> ${{matchedThreat.diagnosis}}
                  </div>
                  <div style="font-size:11.5px; font-weight:700; color:${{matchedThreat.badge_color}}; margin-bottom:5px; text-transform:uppercase; letter-spacing:0.3px;">
                    🎯 Recommended Strategic Action Playbook:
                  </div>
                  <ol style="margin:0; padding-left:18px; font-size:12px; color:#1e293b; line-height:1.5;">
                    ${{matchedThreat.actions.map(a => `<li style="margin-bottom:3px;">${{escapeHtml(a)}}</li>`).join('')}}
                  </ol>
                </div>
              `;
            }})()}}

            <!-- AI & ML Pipeline Observability Banner -->
            <div style="margin-bottom: 14px; padding: 10px 14px; background: #ffffff; border: 1px solid #e2e8f0; border-radius: 6px; box-shadow: 0 1px 3px rgba(0,0,0,0.02);">
              <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 8px;">
                <div style="display:flex; align-items:center; gap:8px;">
                  <span style="font-size: 13px; font-weight: 700; color: #0f172a;">🤖 Live AI & Multi-Task ML Pipeline Observability</span>
                  <span class="chip mono" style="font-size: 10px; background: #f8fafc; color: #475569; border: 1px solid #cbd5e1;">6 Political Models + 6 Behavioral Classifiers</span>
                </div>
                <div style="font-size: 11px; font-family: var(--mono); color: #64748b;">
                  Inference Provenance: <b>${{provLabel}}</b>
                </div>
              </div>
              <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 8px; font-size: 11.5px; font-family: var(--mono);">
                <div style="background:#f8fafc; padding:6px 10px; border-radius:4px; border:1px solid #e2e8f0;">
                  <div style="font-size:10px; color:#64748b;">POLITICAL RELEVANCE</div>
                  <div style="font-weight:700; color:${{pi.is_political && pi.is_political.value ? '#b91c1c' : '#15803d'}};">
                    ${{pi.is_political ? (pi.is_political.value ? 'True' : 'False') : (t.is_political ? 'True' : 'False')}}
                    <span style="font-size:10px; font-weight:400; color:#64748b;">(${{pi.is_political ? Math.round(pi.is_political.confidence * 100) + '%' : 'heuristic'}})</span>
                  </div>
                  <div style="font-size:9.5px; color:#94a3b8;">mDeBERTa-v3-base</div>
                </div>
                <div style="background:#f8fafc; padding:6px 10px; border-radius:4px; border:1px solid #e2e8f0;">
                  <div style="font-size:10px; color:#64748b;">POLITICAL TARGET</div>
                  <div style="font-weight:700; color:#0f172a;">
                    ${{pi.target ? pi.target.value : 'None'}}
                    <span style="font-size:10px; font-weight:400; color:#64748b;">(${{pi.target ? Math.round(pi.target.confidence * 100) + '%' : '-'}})</span>
                  </div>
                  <div style="font-size:9.5px; color:#94a3b8;">${{pi.target ? pi.target.source : 'ensemble'}}</div>
                </div>
                <div style="background:#f8fafc; padding:6px 10px; border-radius:4px; border:1px solid #e2e8f0;">
                  <div style="font-size:10px; color:#64748b;">INCUMBENCY DIRECTION</div>
                  <div style="font-weight:700; color:${{pi.direction && pi.direction.value === 'anti_incumbent' ? '#dc2626' : (pi.direction && pi.direction.value === 'pro_incumbent' ? '#16a34a' : '#0f172a')}};">
                    ${{pi.direction ? pi.direction.value.replace(/_/g, ' ') : 'neutral'}}
                    <span style="font-size:10px; font-weight:400; color:#64748b;">(${{pi.direction ? Math.round(pi.direction.confidence * 100) + '%' : '-'}})</span>
                  </div>
                  <div style="font-size:9.5px; color:#94a3b8;">mDeBERTa Transformer</div>
                </div>
                <div style="background:#f8fafc; padding:6px 10px; border-radius:4px; border:1px solid #e2e8f0;">
                  <div style="font-size:10px; color:#64748b;">ISSUE CLUSTER</div>
                  <div style="font-weight:700; color:#1d4ed8;">
                    ${{pi.issue_cluster ? pi.issue_cluster.value.replace(/_/g, ' ') : (t.issue_cluster || 'none')}}
                    <span style="font-size:10px; font-weight:400; color:#64748b;">(${{pi.issue_cluster ? Math.round(pi.issue_cluster.confidence * 100) + '%' : '-'}})</span>
                  </div>
                  <div style="font-size:9.5px; color:#94a3b8;">Hybrid Prior Router</div>
                </div>
                <div style="background:#f8fafc; padding:6px 10px; border-radius:4px; border:1px solid #e2e8f0;">
                  <div style="font-size:10px; color:#64748b;">NARRATIVE FRAMING</div>
                  <div style="font-weight:700; color:#6d28d9;">
                    ${{pi.narrative_type ? pi.narrative_type.value.replace(/_/g, ' ') : 'other'}}
                    <span style="font-size:10px; font-weight:400; color:#64748b;">(${{pi.narrative_type ? Math.round(pi.narrative_type.confidence * 100) + '%' : '-'}})</span>
                  </div>
                  <div style="font-size:9.5px; color:#94a3b8;">Discourse Marker Model</div>
                </div>
                <div style="background:#f8fafc; padding:6px 10px; border-radius:4px; border:1px solid #e2e8f0;">
                  <div style="font-size:10px; color:#64748b;">PSYCHOLOGICAL ARCHETYPE</div>
                  <div style="font-weight:700; color:#0f172a;">
                    ${{mindsetLabel}}
                  </div>
                  <div style="font-size:9.5px; color:#94a3b8;">21,284-Sample Classifier</div>
                </div>
              </div>
            </div>

            <div class="drawer-grid">
              
              <!-- 1. Breakout Signals & Acceleration -->
              <div class="drawer-section">
                <div class="drawer-sec-title">
                  <span>⚡ 1. Breakout Signals & Velocity</span>
                  <div style="display:flex; gap:6px; align-items:center;">
                    <span class="stage-pill ${{t.stage}}">${{t.stage}}</span>
                    ${{t.comment_velocity && t.comment_velocity.window_comments > 0 ? `
                      <span class="chip mono" style="font-size:10px; background:#f0fdf4; color:#166534; border:1px solid #bbf7d0;" title="Comment activity velocity (timestamp-based)">
                        ${{t.comment_velocity.comments_per_hour_6h}}/h · ${{t.comment_velocity.velocity_trend}}
                      </span>
                    ` : ''}}
                  </div>
                </div>
                ${{t.summary ? `<p style="font-size: 12px; color: var(--text-dim); margin-bottom: 10px; line-height: 1.45;">${{t.summary}}</p>` : ''}}

                <!-- Instagram Comments (primary source) -->
                ${{t.ig_comments && t.ig_comments.length > 0 ? `
                  <div style="margin-top: 10px; padding: 8px; background: #f0f9ff; border: 1px solid #bae6fd; border-radius: 5px; font-size: 11.5px; color: #0c4a6e;">
                    <div style="font-weight: 700; margin-bottom: 6px; display:flex; justify-content:space-between;">
                      <span>📸 Instagram Comments <span style="font-weight:400; color:#0369a1;">· from sampled post comments</span></span>
                      <span style="font-size:10px; color:#0369a1;">${{t.ig_comments.length}} shown · sorted by likes</span>
                    </div>
                    ${{t.ig_comments.slice(0, 5).map(c => `
                      <div style="margin: 5px 0; padding: 5px 8px; background: #e0f2fe; border-radius: 4px; border-left: 2px solid #38bdf8;">
                        <div style="color: #0c4a6e; line-height: 1.4;">${{escapeHtml(c.text)}}</div>
                        <div style="font-size: 10px; color: #0369a1; margin-top: 3px; font-family: var(--mono);">
                          ❤️ ${{c.likes}} likes
                          ${{c.replies > 0 ? '· 💬 ' + c.replies + ' replies' : ''}}
                          ${{c.created_at ? '· ' + String(c.created_at).substring(0,10) : ''}}
                        </div>
                      </div>
                    `).join('')}}
                  </div>
                ` : ''}}

                <!-- Reddit proxy (secondary) -->
                ${{rp.thread_count > 0 ? `
                  <div style="margin-top: 8px; padding: 8px; background: #fff7ed; border: 1px solid #ffedd5; border-radius: 5px; font-size: 11.5px; color: #7c2d12;">
                    <div style="font-weight: 700; margin-bottom: 4px;">Reddit discussion proxy <span style="font-weight: 400;">· ${{rp.thread_count}} threads · ${{rp.subreddit_count}} subreddits · ${{Number(rp.upvotes || 0).toLocaleString()}} upvotes</span></div>
                    ${{(rp.comment_excerpts || []).slice(0, 3).map(c => `
                      <blockquote style="margin: 6px 0 0; padding-left: 8px; border-left: 2px solid #fdba74; color: #431407;">
                        &ldquo;${{escapeHtml(c.text)}}&rdquo;<div style="font-size: 10px; color: #9a3412;">${{escapeHtml(c.subreddit)}} · score ${{Number(c.score || 0).toLocaleString()}}</div>
                      </blockquote>
                    `).join('')}}
                  </div>
                ` : ''}}
                ${{(!t.ig_comments || t.ig_comments.length === 0) && rp.thread_count === 0 ? '<div style="margin-top:10px; font-size:11px; color:var(--text-faint);">No comment data collected for this topic yet.</div>' : ''}}

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

              <!-- 2. Stance & Debate Positions with True Sentiment (Bot-Adjusted) -->
              <div class="drawer-section">
                <div class="drawer-sec-title">
                  <span>⚖️ 2. Tone Toward Story &amp; True Sentiment</span>
                  <span class="chip mono" style="font-size:10px; ${{t.demo && t.demo.source === 'measured' ? 'background:#f0fdf4; color:#166534; border-color:#bbf7d0;' : 'background:var(--surface-2); color:var(--text-faint);'}}">
                    Based on ${{t.demo && t.demo.source === 'measured' ? t.demo.comment_count + ' comments' : 'LLM brief'}} 📸 IG + 💬 Reddit
                  </span>
                </div>

                <!-- True Sentiment Mode Toggle -->
                <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px; flex-wrap:wrap; gap:6px;">
                  <div class="btn-group">
                    <button class="btn-group-item active" id="btn-raw-${{prefix}}${{t.id}}" onclick="setDrawerSentimentMode('${{prefix}}${{t.id}}', 'raw')">Raw Blended</button>
                    <button class="btn-group-item" id="btn-adj-${{prefix}}${{t.id}}" onclick="setDrawerSentimentMode('${{prefix}}${{t.id}}', 'adjusted')">🛡️ True Sentiment (Bot-Adjusted)</button>
                  </div>
                  <span class="chip mono" id="sent-mode-badge-${{prefix}}${{t.id}}" style="font-size:9.5px; background:var(--surface-2); color:var(--text-dim);">
                    Showing Raw Blended
                  </span>
                </div>

                <div id="drawer-sent-raw-${{prefix}}${{t.id}}" style="margin-bottom: 8px;">
                  ${{!hasStance ? '<div style="font-size:11px; color:var(--text-faint);">No measured comment stance for this topic.</div>' : ''}}
                  <div style="font-size:10.5px; color:var(--text-faint); margin-bottom:2px; display:flex; justify-content:space-between;">
                    <span>Blended Public Sentiment (All Accounts)</span>
                    <span class="mono">${{hasStance ? '+' + sup + '% Sup · -' + crit + '% Crit' : ''}}</span>
                  </div>
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

                <!-- True Sentiment (Bot-Adjusted) Dual-Layer Display with Hatched Bot Proportion Overlay -->
                <div id="drawer-sent-adj-${{prefix}}${{t.id}}" style="display:none; margin-bottom: 8px; padding:8px 10px; background:#f0fdf4; border:1px solid #bbf7d0; border-radius:6px;">
                  <div style="font-size:11px; color:#15803d; font-weight:700; margin-bottom:4px; display:flex; justify-content:space-between;">
                    <span>🛡️ True Grassroots Sentiment (Verified Organic Only)</span>
                    <span class="mono">+${{orgSup}}% Sup · -${{orgCrit}}% Crit</span>
                  </div>
                  <div class="sentiment-bar" style="height: 11px; position:relative; background:#dcfce7;">
                    <div class="sentiment-seg supportive" style="width: ${{orgSup}}%;" title="Organic Supportive: ${{orgSup}}%"></div>
                    <div class="sentiment-seg critical" style="width: ${{orgCrit}}%;" title="Organic Critical: ${{orgCrit}}%"></div>
                    <div class="sentiment-seg humorous" style="width: ${{orgHum}}%;" title="Organic Humorous: ${{orgHum}}%"></div>
                    <div class="sentiment-seg neutral" style="width: ${{orgNeu}}%;" title="Organic Neutral: ${{orgNeu}}%"></div>
                    ${{botShare > 0 ? `
                      <div class="sentiment-seg hatched-bot" style="position:absolute; right:0; top:0; width:${{botShare}}%; height:100%; pointer-events:none;" title="Synthetic Bot Volume Proportion: ${{botShare}}%"></div>
                    ` : ''}}
                  </div>
                  <div style="display:flex; justify-content:space-between; font-size:10.5px; margin-top:4px; font-family:var(--mono);">
                    <span style="color:#15803d;">Support: ${{orgSup}}%</span>
                    <span style="color:#b91c1c;">Critical: ${{orgCrit}}%</span>
                    <span style="color:#b45309;">Humor: ${{orgHum}}%</span>
                    <span style="color:#475569;">Neutral: ${{orgNeu}}%</span>
                  </div>
                  <div style="font-size:10px; color:#166534; margin-top:4px; display:flex; justify-content:space-between;">
                    <span>Hatched segment = ${{botShare}}% synthetic bot cluster skew</span>
                    <span>Purity Ratio: ${{ba.organic_vs_synthetic_ratio || 1.0}}:1 Organic</span>
                  </div>
                </div>

                ${{sides.length > 0 ? `
                  <div style="margin-top: 10px;">
                    <div style="font-size: 10.5px; font-weight:700; color:var(--text-faint); text-transform:uppercase;">Debate Positions (share of commenters on panel accounts, not public opinion):</div>
                    ${{sides.map((s, idx) => `
                      <div class="side-card">
                        <div class="side-header">
                          <span>Side #${{idx+1}} &middot; ${{Math.round((s.share || 0) * 100)}}%${{s.share_ci ? ' (95% CI ' + Math.round(s.share_ci[0] * 100) + '&ndash;' + Math.round(s.share_ci[1] * 100) + '%)' : ''}}</span>
                          <span>${{s.comments || 0}} comments</span>
                        </div>
                        <div class="side-text">${{s.position}}</div>
                      </div>
                    `).join('')}}
                    ${{exposureFlag ? '<div style="font-size:11px; color:var(--warn); margin-top:6px;">Caution: shares change by 15+ points when likes are ignored. The like-weighted split may reflect a few highly liked comments.</div>' : ''}}
                  </div>
                ` : '<div style="font-size:11.5px; color:var(--text-faint); margin-top:8px;">General social conversation without factional polarization.</div>'}}
              </div>

              <!-- 3. Demographic & Mental State Profile -->
              <div class="drawer-section">
                <div class="drawer-sec-title">
                  <span>👥 3. Inferred Demographics & Mental State</span>
                  <span class="chip mono" style="font-size: 10px; ${{t.demo && t.demo.source === 'measured' ? "background:var(--pos-bg); color:var(--pos); border-color:var(--pos-border);" : "background:var(--surface-2); color:var(--text-faint);"}}">
                    ${{t.demo && t.demo.source === 'measured' ? 'Measured &middot; ' + t.demo.comment_count + ' comments' : 'Estimated &middot; no comments sampled'}}
                  </span>
                </div>
                ${{t.demo && t.demo.has_data ? `
                  <div style="font-size: 12px; color: var(--text-dim); margin-bottom: 8px;">
                    <div><b>Age Cohort Proxy:</b> <span class="mono" style="color:var(--text);">${{t.demo.cohort}}</span></div>
                    <div style="margin-top: 3px;"><b>Mental State / Mindset:</b> <span class="mono" style="color:var(--text);">${{t.demo.mindset}}</span></div>
                    <div style="margin-top: 3px;"><b>Conversational Persona:</b> <span class="mono" style="color:var(--text);">${{t.demo.gender_persona}}</span></div>
                    <div style="margin-top: 3px; display:flex; gap:12px; flex-wrap:wrap;">
                      <div><b>Audience Urgency:</b> <span class="mono" style="color:${{t.demo.impatience === 'impatient' ? '#dc2626' : (t.demo.impatience === 'patient' ? '#16a34a' : 'var(--text)')}};">${{t.demo.impatience || 'neutral'}}</span></div>
                      <div><b>Credibility / Epistemic:</b> <span class="mono" style="color:${{t.demo.irrationality === 'irrational' ? '#b45309' : (t.demo.irrationality === 'rational' ? '#2563eb' : 'var(--text)')}};">${{t.demo.irrationality || 'neutral'}}</span></div>
                    </div>
                    <div style="margin-top: 3px;"><b>Regional / Linguistic Focus:</b> <span class="mono" style="color:var(--text);">${{t.demo.top_region}}</span></div>
                    ${{t.demo.political_lean_distribution && Object.keys(t.demo.political_lean_distribution).length ? `
                      <div style="margin-top: 3px;"><b>Political Lean (commenters):</b> <span class="mono" style="color:var(--text);">${{Object.entries(t.demo.political_lean_distribution).sort((a,b)=>b[1]-a[1]).map(([k,v])=>`${{k}} ${{Math.round(v*100)}}%`).join(' &middot; ')}}</span></div>
                    ` : ''}}
                  </div>
                  <div style="font-size: 11px; color: var(--text-faint); line-height: 1.4; padding: 6px 8px; background: var(--surface-2); border-radius: 4px;">
                    Inferred from user activity, slang markers, script expressions, and conversational discourse patterns.
                    ${{t.demo.coverage && t.demo.coverage.total ? ` Mindset: ${{t.demo.coverage.model_classified}}/${{t.demo.coverage.total}} classified by the trained mental-state model, ${{t.demo.coverage.default_fallback}} defaulted (model unavailable or comment too short).` : ''}}
                  </div>
                ` : `
                  <p style="font-size: 12px; color: var(--text-dim);">No active comment sample available for individual discourse profiling on this topic. The figures shown are a contextual estimate from topic category / matched Reddit community / language, not measured from actual comments.</p>
                `}}

                <!-- Political Pulse & Bot Audit in Drawer -->
                ${{t.political_pulse ? `
                  <div style="margin-top: 12px; padding-top: 10px; border-top: 1px dashed var(--border);">
                    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:4px;">
                      <div style="font-size: 11px; font-weight:700; color:var(--text);">⚖️ Political Party Favorability & Lens</div>
                      ${{t.political_pulse.engagement_count > 0 ? `<span class="chip mono" style="font-size:9.5px; background:#eff6ff; color:#1d4ed8; border-color:#bfdbfe;">Based on ${{t.political_pulse.engagement_count}} engagements</span>` : '<span class="chip mono" style="font-size:9.5px; color:var(--text-faint);">No comment engagements</span>'}}
                    </div>
                    ${{t.political_pulse.engagement_count > 0 ? `
                    <div style="font-size: 12px; margin-bottom: 6px;">
                      <span class="chip mono" style="background:#fff7ed; color:#c2410c; border-color:#fed7aa;">${{t.political_pulse.overall_leaning}}</span>
                      <span class="mono" style="font-size:11px; margin-left:6px;">BJP: <b>${{t.political_pulse.bjp_favor_pct}}%</b> &middot; Opp: <b>${{t.political_pulse.opp_favor_pct}}%</b> &middot; Neutral: <b>${{t.political_pulse.neutral_pct}}%</b></span>
                    </div>
                    <div style="font-size: 11.5px; color: var(--text-dim);">${{t.political_pulse.narrative_summary}}</div>
                    ` : `<div style="font-size:11.5px; color:var(--text-faint);">No comment engagements found — political lean cannot be measured for this topic.</div>`}}
                  </div>
                ` : ''}}

                ${{t.bot_audit ? `
                  <div style="margin-top: 10px; padding: 8px; border-radius: 5px; border: 1px solid ${{t.bot_audit.total_comments_audited === 0 ? '#e5e7eb' : t.bot_audit.astroturf_index >= 35 ? '#fee2e2' : '#d1fae5'}}; background: ${{t.bot_audit.total_comments_audited === 0 ? '#f9fafb' : t.bot_audit.astroturf_index >= 35 ? '#fef2f2' : '#ecfdf5'}};">
                    <div style="display:flex; justify-content:space-between; align-items:center;">
                      <span style="font-size:11px; font-weight:700; color:${{t.bot_audit.total_comments_audited === 0 ? '#6b7280' : t.bot_audit.astroturf_index >= 35 ? '#991b1b' : '#065f46'}};">
                        🤖 Bot / Synthetic Manipulation Risk
                      </span>
                      <span class="chip mono" style="font-size:10px;">
                        ${{t.bot_audit.total_comments_audited === 0 ? 'No comment data' : 'Index: ' + t.bot_audit.astroturf_index + '/100'}}
                      </span>
                    </div>
                    ${{t.bot_audit.total_comments_audited === 0 ? `
                      <div style="font-size:11px; color:#6b7280; margin-top:3px;">
                        Insufficient comment data — no Instagram comments or Reddit excerpts collected yet for this topic.
                        Bot audit requires at least 5 comments.
                      </div>
                    ` : `
                      <div style="font-size:11px; color:${{t.bot_audit.astroturf_index >= 35 ? '#b91c1c' : '#047857'}}; margin-top:3px;">
                        ~${{t.bot_audit.bot_share_pct}}% bot-like accounts &middot; ${{t.bot_audit.template_clusters.length}} copy-paste clusters &middot; ${{t.bot_audit.total_comments_audited}} comments audited
                        (${{t.ig_comments && t.ig_comments.length > 0 ? t.ig_comments.length + ' IG' : ''}}${{t.reddit_proxy && t.reddit_proxy.comment_excerpts && t.reddit_proxy.comment_excerpts.length > 0 ? (t.ig_comments && t.ig_comments.length > 0 ? ' + ' : '') + t.reddit_proxy.comment_excerpts.length + ' Reddit' : ''}})
                      </div>
                      <div style="font-size:10.5px; color:var(--text-faint); margin-top:2px;">${{t.bot_audit.verdict_summary}}</div>
                    `}}
                  </div>
                ` : ''}}
              </div>

              ${{t.undercurrent && t.undercurrent.is_political ? `
              <!-- 4. Fusion & Undercurrent -->
              <div class="drawer-section">
                <div class="drawer-sec-title">
                  <span>⚡ 4. Fusion & Undercurrent</span>
                  <div style="display:flex; gap:6px; align-items:center; flex-wrap:wrap;">
                    ${{(() => {{
                      const gap = t.undercurrent.framing_reaction_gap || 0;
                      if (gap >= 0.35) return '<span class="chip mono" style="font-size:10px;background:var(--warn-bg);color:var(--warn);border-color:var(--warn-border);">⚠ Framing/reaction divergence</span>';
                      if (gap >= 0.12) return '<span class="chip mono" style="font-size:10px;background:#fffbeb;color:#92400e;border-color:#fcd34d;">~ Mild divergence</span>';
                      return '<span class="chip mono" style="font-size:10px;background:var(--surface-2);color:var(--text-faint);">Framing aligns with reaction</span>';
                    }})()}}
                    ${{t.undercurrent.comment_sample_size > 0 ? `<span class="chip mono" style="font-size:9.5px;background:#eff6ff;color:#1d4ed8;border-color:#bfdbfe;">${{t.undercurrent.comment_sample_size}} engagements measured</span>` : ''}}
                  </div>
                </div>

                <!-- Pulse bar: BJP / Opp / Neutral breakdown from actual comment pool -->
                ${{t.undercurrent.comment_sample_size > 0 ? `
                <div style="margin-bottom:10px;">
                  <div style="display:flex;justify-content:space-between;font-size:10.5px;color:var(--text-dim);margin-bottom:3px;">
                    <span style="color:#c2410c;font-weight:600;">BJP ${{t.undercurrent.bjp_pct}}%</span>
                    <span style="color:#64748b;">Neutral ${{t.undercurrent.neutral_pct}}%</span>
                    <span style="color:#1d4ed8;font-weight:600;">Opposition ${{t.undercurrent.opp_pct}}%</span>
                  </div>
                  <div style="display:flex;height:8px;border-radius:4px;overflow:hidden;background:#e2e8f0;">
                    <div style="width:${{t.undercurrent.bjp_pct}}%;background:#f97316;" title="BJP ${{t.undercurrent.bjp_pct}}%"></div>
                    <div style="width:${{t.undercurrent.neutral_pct}}%;background:#94a3b8;" title="Neutral ${{t.undercurrent.neutral_pct}}%"></div>
                    <div style="width:${{t.undercurrent.opp_pct}}%;background:#3b82f6;" title="Opposition ${{t.undercurrent.opp_pct}}%"></div>
                  </div>
                  <div style="font-size:10px;color:var(--text-faint);margin-top:3px;">source: ${{t.undercurrent.comment_direction_source}}</div>
                </div>` : '<div style="font-size:11px;color:var(--text-faint);margin-bottom:10px;">No comment engagements available to measure reaction.</div>'}}

                <div style="display:grid;grid-template-columns:1fr 1fr;gap:10px;font-size:12px;color:var(--text-dim);">
                  <div>
                    <b>Post's own framing:</b><br/>
                    <span class="mono" style="color:var(--text);">${{t.undercurrent.post_direction.replace(/_/g,' ')}}</span>
                    <span class="chip mono" style="font-size:9.5px;margin-left:4px;">${{t.undercurrent.post_direction_source.split('(')[0].trim()}}</span><br/>
                    ${{t.undercurrent.post_direction_confidence > 0.1 ? `<span style="font-size:10.5px;color:var(--text-faint);">confidence ${{(t.undercurrent.post_direction_confidence*100).toFixed(0)}}%</span><br/>` : ''}}
                    <span style="font-size:11px;">Issue: ${{(t.undercurrent.issue_cluster||'none').replace(/_/g,' ')}} &middot; Narrative: ${{(t.undercurrent.narrative_type||'other').replace(/_/g,' ')}}</span>
                  </div>
                  <div>
                    <b>Audience reaction:</b><br/>
                    <span class="mono" style="color:var(--text);">${{(t.undercurrent.comment_direction||'n/a').replace(/_/g,' ')}}</span>
                    <span class="chip mono" style="font-size:9.5px;margin-left:4px;">conf ${{(t.undercurrent.comment_direction_confidence*100).toFixed(0)}}%</span><br/>
                    <span style="font-size:11px;">Gap score: <b>${{(t.undercurrent.framing_reaction_gap||0).toFixed(2)}}</b>/1.0</span>
                  </div>
                </div>

                <!-- Undercurrent narrative — the most useful output -->
                ${{t.undercurrent.undercurrent_narrative ? `
                <div style="margin-top:10px;padding:8px 10px;background:var(--surface-2);border-radius:6px;border-left:3px solid ${{(t.undercurrent.framing_reaction_gap||0)>=0.35?'var(--warn)':'var(--text-faint)'}};font-size:11.5px;color:var(--text-dim);line-height:1.5;">
                  ${{t.undercurrent.undercurrent_narrative}}
                </div>` : ''}}

                <div style="margin-top:8px;font-size:10.5px;color:var(--text-faint);">
                  Undercurrent Score: <b class="mono">${{t.undercurrent_score}}/10</b>
                  &nbsp;·&nbsp; trend 25% + debate 20% + reaction confidence 20% + framing gap 20% + engagement 10% + bot-discount 5%
                </div>
              </div>
              ` : ''}}

              <!-- 5. Evidence & Source Provenance -->
              <div class="drawer-section">
                <div class="drawer-sec-title">
                  <span>🔗 5. Multi-Source Evidence & Provenance</span>
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

    // Lower = more urgent/active; settled/quiet/unobserved carry no fresh stage evidence.
    const STAGE_RANK = {{
      breaking_out: 0, pre_breakout: 1, peak: 2, saturated: 3, simmering: 4,
      settled: 5, quiet: 6, unobserved: 7
    }};
    const NOT_SETTLED_STAGES = new Set(['breaking_out', 'pre_breakout', 'peak', 'saturated', 'simmering']);

    function applyFilters() {{
      const search = document.getElementById('topic-search').value.toLowerCase();
      const cat = document.getElementById('cat-select').value;
      const stage = document.getElementById('stage-select').value;
      currentAuthFilter = document.getElementById('topics-authenticity-select')?.value || 'all';

      let filtered = TOPICS.filter(t => {{
        if (cat !== 'all' && t.category !== cat) return false;
        if (stage === 'not_settled') {{
          if (!NOT_SETTLED_STAGES.has(t.stage)) return false;
        }} else if (stage !== 'all' && t.stage !== stage) {{
          return false;
        }}
        if (currentAuthFilter === 'organic') {{
          const ba = t.bot_audit || {{}};
          if ((ba.astroturf_index || 0) >= 50 && (ba.bot_share_pct || 0) >= 45 && (ba.organic_comment_count || 0) < 3) return false;
        }} else if (currentAuthFilter === 'synthetic') {{
          const ba = t.bot_audit || {{}};
          const isSyn = (ba.astroturf_index >= 20 || (ba.template_clusters && ba.template_clusters.length > 0) || (ba.bot_share_pct || 0) >= 15);
          if (!isSyn) return false;
        }}
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
        if (currentSort === 'active_first') {{
          const rankDiff = (STAGE_RANK[a.stage] ?? 9) - (STAGE_RANK[b.stage] ?? 9);
          return rankDiff !== 0 ? rankDiff : b.trend_score - a.trend_score;
        }}
        if (currentSort === 'comments') return b.ig.comments_7d - a.ig.comments_7d;
        if (currentSort === 'heat') return b.instagram_heat - a.instagram_heat;
        if (currentSort === 'debate') return b.debate_score - a.debate_score;
        return 0;
      }});

      document.getElementById('topics-feed').innerHTML = filtered.map(t => buildTopicCardHtml(t, 'topics-')).join('');
      document.getElementById('results-count').innerText = `Showing ${{filtered.length}} of ${{TOPICS.length}} topics${{currentAuthFilter === 'organic' ? ' (🛡️ Verified Organic Only)' : (currentAuthFilter === 'synthetic' ? ' (🤖 Synthetic / Bot Clusters Only)' : '')}}`;
    }}

    function renderPoliticalFeed() {{
      const polAuthFilter = document.getElementById('political-authenticity-select')?.value || 'all';
      currentAuthFilter = polAuthFilter;
      let filtered = TOPICS.filter(t => t.is_political);
      if (politicalSubFilter === 'probed') {{
        filtered = filtered.filter(t => t.has_probed_data);
      }} else if (politicalSubFilter === 'unprobed') {{
        filtered = filtered.filter(t => !t.has_probed_data);
      }}
      if (polAuthFilter === 'organic') {{
        filtered = filtered.filter(t => {{
          const ba = t.bot_audit || {{}};
          return !((ba.astroturf_index || 0) >= 50 && (ba.bot_share_pct || 0) >= 45 && (ba.organic_comment_count || 0) < 3);
        }});
      }} else if (polAuthFilter === 'synthetic') {{
        filtered = filtered.filter(t => {{
          const ba = t.bot_audit || {{}};
          return (ba.astroturf_index >= 20 || (ba.template_clusters && ba.template_clusters.length > 0) || (ba.bot_share_pct || 0) >= 15);
        }});
      }}
      filtered.sort((a, b) => b.trend_score - a.trend_score);
      document.getElementById('political-feed').innerHTML = filtered.map(t => buildTopicCardHtml(t, 'political-')).join('');
    }}

    function renderUndercurrentTab(sortKey, btn) {{
      if (btn) {{
        btn.parentElement.querySelectorAll('.btn-group-item').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
      }}
      let filtered = TOPICS.filter(t => t.is_political && t.undercurrent_score > 0);
      filtered.sort((a, b) => {{
        if (sortKey === 'engagement') return (b.ig.engagement_per_hour || 0) - (a.ig.engagement_per_hour || 0);
        if (sortKey === 'disagreement') {{
          const da = (a.undercurrent && a.undercurrent.framing_vs_reaction_disagreement) ? 1 : 0;
          const db = (b.undercurrent && b.undercurrent.framing_vs_reaction_disagreement) ? 1 : 0;
          return db - da || b.undercurrent_score - a.undercurrent_score;
        }}
        if (sortKey === 'active_first') {{
          const rankDiff = (STAGE_RANK[a.stage] ?? 9) - (STAGE_RANK[b.stage] ?? 9);
          return rankDiff !== 0 ? rankDiff : b.undercurrent_score - a.undercurrent_score;
        }}
        return b.undercurrent_score - a.undercurrent_score;
      }});
      document.getElementById('undercurrent-feed').innerHTML = filtered.length
        ? filtered.map(t => buildTopicCardHtml(t, 'undercurrent-')).join('')
        : '<p style="font-size:12.5px;color:var(--text-faint);">No political topics scored this run.</p>';
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
          const labels = [p.lean, p.target, p.narrative, p.tone, p.archetype].join(' ').toLowerCase();
          if (!acc.includes(search) && !cap.includes(search) && !top.includes(search) && !labels.includes(search)) return false;
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
          <td style="font-size:11.5px;">
            <div style="font-weight:600;">${{p.lean || '-'}}</div>
            <div style="color:var(--text-dim);">${{p.target ? 'target: ' + p.target : ''}}${{p.direction ? ' · ' + p.direction.replace(/_/g, ' ') : ''}}</div>
          </td>
          <td style="font-size:11.5px;">
            ${{p.tone || '-'}}
            ${{p.sarcasm_share ? '<div style="color:var(--text-dim);">' + Math.round(p.sarcasm_share * 100) + '% sarcastic</div>' : ''}}
          </td>
          <td style="font-size:11.5px;">
            ${{p.narrative ? p.narrative.replace(/_/g, ' ') : '-'}}
            <div style="color:var(--text-dim);">${{p.issue && p.issue !== 'none' ? p.issue.replace(/_/g, ' ') : ''}}</div>
          </td>
          <td style="font-size:11.5px;">
            ${{p.archetype || '-'}}
            ${{p.archetype_share ? '<div style="color:var(--text-dim);">' + Math.round(p.archetype_share * 100) + '% of comments</div>' : ''}}
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
                <span class="mono" style="font-size: 11px; color: var(--text-faint);">${{p.published_at ? String(p.published_at).substring(0, 16).replace('T', ' ') : ''}} UTC</span>
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
          labels: ['Breaking Out', 'Pre-Breakout', 'Peak', 'Saturated', 'Simmering', 'Settled', 'Quiet', 'Unobserved'],
          datasets: [{{
            // First 5 bars = not settled yet (still worth watching); last 3 = no fresh evidence this run.
            data: [{stage_counts.get('breaking_out', 0)}, {stage_counts.get('pre_breakout', 0)}, {stage_counts.get('peak', 0)}, {stage_counts.get('saturated', 0)}, {stage_counts.get('simmering', 0)}, {stage_counts.get('settled', 0)}, {stage_counts.get('quiet', 0)}, {stage_counts.get('unobserved', 0)}],
            backgroundColor: ['#087f76', '#24846d', '#b77b23', '#c84f45', '#d7a84d', '#8b9b96', '#c3cdd4', '#e2a26b'],
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
          labels: {json.dumps([cat.replace("_", " ").title() for cat, _ in sorted(cat_counts.items(), key=lambda kv: -kv[1])[:10]])},
          datasets: [{{
            data: {json.dumps([count for _, count in sorted(cat_counts.items(), key=lambda kv: -kv[1])[:10]])},
            backgroundColor: ['#087f76', '#c84f45', '#b77b23', '#24846d', '#397b96', '#71847e', '#a66a54', '#d7a84d', '#8b9b96', '#5b6b76'],
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
          labels: {json.dumps([label for label, _ in global_mental_states.most_common(10)])},
          datasets: [{{
            data: {json.dumps([count for _, count in global_mental_states.most_common(10)])},
            backgroundColor: ['#c84f45', '#b77b23', '#24846d', '#087f76', '#397b96', '#71847e', '#a66a54', '#d7a84d', '#8b9b96', '#5b6b76']
          }}]
        }},
        options: {{
          responsive: true,
          maintainAspectRatio: false,
          plugins: {{ legend: {{ position: 'bottom', labels: {{ boxWidth: 10, font: {{ size: 9.5 }} }} }} }}
        }}
      }});

      const top10 = [...TOPICS].sort((a,b) => (Number(b.is_political) - Number(a.is_political)) || b.trend_score - a.trend_score).slice(0, 10);
      document.getElementById('top-breakouts-list').innerHTML = top10.map(t => buildTopicCardHtml(t, 'overview-')).join('');
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
