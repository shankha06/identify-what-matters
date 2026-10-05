with open('make_full_dashboard.py') as f:
    text = f.read()

target = """    function renderPanelTable() {
      const tbody = document.getElementById('panel-table-body');
      tbody.innerHTML = PANEL.map(p => `
        <tr>
          <td style="padding-left: 18px; font-weight: 600;">
            <a href="https://instagram.com/${p.username}" target="_blank" style="color:var(--text); text-decoration:none;">
              @${p.username} &nearr;
            </a>
          </td>
          <td><span class="cat-pill">${p.category || 'media'}</span></td>
          <td class="mono">${p.posts_7d || p.recent_posts || 1}</td>
          <td class="mono">${(p.comments_7d || p.total_comments || 0).toLocaleString()}</td>
          <td class="mono">${(p.likes_7d || p.total_likes || 0).toLocaleString()}</td>
          <td class="mono">${Math.round((p.likes_7d || 0) / Math.max(1, p.posts_7d || 1)).toLocaleString()}</td>
          <td style="padding-right: 18px;">
            <span class="chip" style="color:var(--pos); background:var(--pos-bg); border-color:var(--pos-border);">Active</span>
          </td>
        </tr>
      `).join('');
    }"""

replacement = """    function renderPanelTable() {
      const tbody = document.getElementById('panel-table-body');
      tbody.innerHTML = PANEL.map(p => `
        <tr>
          <td style="padding-left: 18px; font-weight: 600;">
            <a href="${p.url || ('https://instagram.com/' + p.account)}" target="_blank" style="color:var(--accent); text-decoration:none;">
              @${p.account} &nearr;
            </a>
            <div style="font-size:11px; color:var(--text-faint); margin-top:2px; max-width:280px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap;">
              ${p.caption || ''}
            </div>
          </td>
          <td><span class="cat-pill">${p.group ? p.group.replace(/_/g, ' ') : 'news'}</span></td>
          <td class="mono">${p.age_hours}h ago</td>
          <td class="mono"><b>${(p.comments || 0).toLocaleString()}</b></td>
          <td class="mono">${(p.likes || 0).toLocaleString()}</td>
          <td class="mono"><b>${p.heat ? p.heat.toFixed(2) : '-'}</b> heat</td>
          <td style="padding-right: 18px;">
            <span class="chip" style="color:var(--pos); background:var(--pos-bg); border-color:var(--pos-border);">Active Pulse</span>
          </td>
        </tr>
      `).join('');
    }"""

# In the template we used double braces for functions
target_fstring = target.replace('{', '{{').replace('}', '}}')
replacement_fstring = replacement.replace('{', '{{').replace('}', '}}')

if target_fstring in text:
    text = text.replace(target_fstring, replacement_fstring)
    with open('make_full_dashboard.py', 'w') as f:
        f.write(text)
    print("Replaced panel table template successfully")
else:
    print("Target not found, checking manual snippet...")
