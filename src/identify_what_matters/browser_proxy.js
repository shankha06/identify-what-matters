// Thin Instagram fetch proxy for the India trend pipeline.
// 1. Run `identify-what-matters-trends run --instagram browser` (it waits on ws://localhost:8765).
// 2. Paste this file into the DevTools console of a logged-in instagram.com tab.
// Python drives every request and its pacing over the WebSocket (Instagram's CSP allows ws://localhost);
// this page only performs the same-origin fetches and returns slimmed JSON.
(() => {
  const headers = {'X-Requested-With': 'XMLHttpRequest', 'X-IG-App-ID': '936619743392459'};
  const media = node => {
    const found = [];
    const visit = n => {
      if (!n || typeof n !== 'object') return;
      if (n.media && n.media.code) found.push(n.media);
      Object.values(n).forEach(visit);
    };
    visit(node);
    return found;
  };
  const slimMedia = m => ({pk: m.pk, code: m.code, taken_at: m.taken_at, like_count: m.like_count,
    comment_count: m.comment_count, play_count: m.play_count || m.ig_play_count || m.view_count,
    product_type: m.product_type, caption: {text: (m.caption && m.caption.text) || ''},
    timeline_pinned_user_ids: m.timeline_pinned_user_ids || null, is_paid_partnership: !!m.is_paid_partnership,
    like_and_view_counts_disabled: !!m.like_and_view_counts_disabled,
    user: {pk: m.user && m.user.pk, username: m.user && m.user.username, is_verified: m.user && m.user.is_verified}});
  const slimComment = c => ({pk: c.pk, text: c.text, created_at: c.created_at,
    comment_like_count: c.comment_like_count, child_comment_count: c.child_comment_count, user: {pk: c.user && c.user.pk}});
  // Same-origin GraphQL needs the page's anti-CSRF tokens; they are embedded in the logged-in page HTML.
  const token = name => (document.documentElement.innerHTML.match(new RegExp(`"${name}",\\[\\],\\{"token":"([^"]+)"`)) || [])[1];
  const csrf = () => (document.cookie.match(/csrftoken=([^;]+)/) || [])[1] || '';
  const routes = {
    search: async q => {
      const r = await fetch('/api/v1/fbsearch/web/top_serp/?query=' + encodeURIComponent(q), {headers});
      return [r.status, r.ok ? {media_grid: {sections: media(await r.json()).map(m => ({media: slimMedia(m)}))}} : null];
    },
    hashtag: async tag => {
      const r = await fetch('/api/v1/tags/web_info/?tag_name=' + encodeURIComponent(tag), {headers});
      if (!r.ok) return [r.status, null];
      const data = (await r.json()).data || {};
      return [r.status, {media_count: data.media_count, sections: media([data.top, data.recent]).map(m => ({media: slimMedia(m)}))}];
    },
    comments: async id => {
      const r = await fetch(`/api/v1/media/${id}/comments/?can_support_threading=true`, {headers});
      if (!r.ok) return [r.status, null];
      const body = await r.json();
      return [r.status, {comment_count: body.comment_count, comments: (body.comments || []).map(slimComment)}];
    },
    user_feed: async ({username, doc_id}) => {
      const lsd = token('LSD') || '';
      const variables = JSON.stringify({data: {count: 12}, username,
        __relay_internal__pv__PolarisMultiCaptionCarouselEnabledrelayprovider: true,
        __relay_internal__pv__PolarisShortDramaEnabledrelayprovider: false,
        __relay_internal__pv__PolarisReelsRecoDebugOverlayEnabledrelayprovider: false});
      const body = new URLSearchParams({fb_dtsg: token('DTSGInitialData') || '', lsd, doc_id, variables,
        fb_api_req_friendly_name: 'PolarisProfilePostsQuery', server_timestamps: 'true'});
      const r = await fetch('/graphql/query', {method: 'POST', body, headers: {...headers,
        'Content-Type': 'application/x-www-form-urlencoded', 'X-FB-LSD': lsd, 'X-CSRFToken': csrf(),
        'X-FB-Friendly-Name': 'PolarisProfilePostsQuery'}});
      if (!r.ok) return [r.status, null];
      const json = await r.json();
      const connection = json.data && json.data.xdt_api__v1__feed__user_timeline_graphql_connection;
      if (!connection) return [404, null];
      return [200, {items: connection.edges.map(e => slimMedia(e.node))}];
    },
  };
  const port = window.TRENDS_BRIDGE_PORT || 8765;
  const ws = new WebSocket(`ws://localhost:${port}`);
  ws.onopen = () => console.log('[trends] connected to pipeline on port', port);
  ws.onmessage = async event => {
    const {id, kind, arg} = JSON.parse(event.data);
    if (kind === 'done') { console.log('[trends] finished:', arg); ws.close(); return; }
    let status = 0, body = null;
    try { [status, body] = await routes[kind](arg); } catch (error) { body = {error: String(error)}; }
    ws.send(JSON.stringify({id, status, body}));
  };
  ws.onerror = () => console.error(`[trends] cannot reach ws://localhost:${port}; start the pipeline first.`);
})();
