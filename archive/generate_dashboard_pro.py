import json
import html

# Load report data
with open('output/runs/20261003T114148Z/report.json') as f:
    report = json.load(f)

topics = report.get('topics', [])
print(f"Total topics: {len(topics)}")

# Prepare categories and counts
cat_counts = {}
stage_counts = {}
stance_counts = {'supportive': 0, 'critical': 0, 'neutral': 0, 'humorous': 0}
political_cats = {'politics_governance', 'crime_law', 'civic_issues'}

for t in topics:
    cat = t.get('category', 'other')
    cat_counts[cat] = cat_counts.get(cat, 0) + 1
    
    stage = (t.get('breakout') or {}).get('stage', 'emerging')
    stage_counts[stage] = stage_counts.get(stage, 0) + 1
    
    brief = t.get('brief') or {}
    st = brief.get('stance') or {}
    if st:
        max_s = max(st.items(), key=lambda x: x[1])[0]
        stance_counts[max_s] = stance_counts.get(max_s, 0) + 1

print("Categories:", cat_counts)
print("Stages:", stage_counts)
print("Stance:", stance_counts)
