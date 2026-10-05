with open('make_full_dashboard.py') as f:
    text = f.read()

# Replace toggleCard definition
text = text.replace(
    'function toggleCard(cardEl) {{',
    'function toggleCard(rowEl) {{\n      const cardEl = rowEl.closest(".topic-card");'
)
# Also in python if it had syntax error:
import re
text = re.sub(r'function toggleCard\(rowEl\)\s*\{\s*const cardEl[^;]+;\s*\{', 'function toggleCard(rowEl) {\n      const cardEl = rowEl.closest(".topic-card");', text)
text = text.replace("onclick=\"toggleCard(document.getElementById('card-${prefix}${t.id}'))\"", "onclick=\"toggleCard(this)\"")

with open('make_full_dashboard.py', 'w') as f:
    f.write(text)

