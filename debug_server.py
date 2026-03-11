"""
debug_server.py
Run alongside main.py to get a live debug dashboard in your browser.
Start with: python debug_server.py
Then open: http://localhost:5001
"""

from flask import Flask, jsonify, render_template_string
from collections import deque
import threading

app = Flask(__name__)
_lock = threading.Lock()

# Circular buffer — keeps last 50 events
_events = deque(maxlen=50)
_current_state = {
    "active_agent": None,
    "routing_context": None,
    "brand_fields": {},
    "collected_fields": {},
    "last_user_input": None,
    "last_response": None,
    "onboarding_mode": None,
    "onboarding_section": None,
    "onboarding_missing": None,
    "routing_decision": None,
    "gemini_choice": None,
}


def push_event(event_type: str, data: dict):
    """Call this from any part of the codebase to push a debug event."""
    import time
    with _lock:
        _events.appendleft({
            "type": event_type,
            "data": data,
            "ts": time.strftime("%H:%M:%S")
        })
        # Update current state for relevant event types
        if event_type == "agent_selected":
            _current_state["active_agent"] = data.get("agent")
            _current_state["routing_context"] = data.get("routing_context")
            _current_state["gemini_choice"] = data.get("gemini_choice")
        elif event_type == "routing_decision":
            _current_state["routing_decision"] = data.get("decision")
            _current_state["gemini_choice"] = data.get("gemini_choice")
        elif event_type == "brand_fields":
            _current_state["brand_fields"] = data
        elif event_type == "collected_fields":
            _current_state["collected_fields"] = data
        elif event_type == "user_input":
            _current_state["last_user_input"] = data.get("text")
        elif event_type == "agent_response":
            _current_state["last_response"] = data.get("text")
        elif event_type == "onboarding_state":
            _current_state["onboarding_mode"] = data.get("mode")
            _current_state["onboarding_section"] = data.get("section")
            _current_state["onboarding_missing"] = data.get("missing")
            _current_state["collected_fields"] = data.get("collected", {})


@app.route("/")
def index():
    return render_template_string(HTML)


@app.route("/api/state")
def state():
    with _lock:
        return jsonify({
            "state": _current_state,
            "events": list(_events)
        })


def start():
    """Start debug server in background thread with all logs suppressed."""
    import logging
    # Silence Werkzeug request logs — only show errors
    logging.getLogger('werkzeug').setLevel(logging.ERROR)
    logging.getLogger('werkzeug').disabled = True

    def _run():
        import sys, os
        # Redirect werkzeug output to devnull
        devnull = open(os.devnull, 'w')
        old_stderr = sys.stderr
        sys.stderr = devnull
        try:
            app.run(port=5001, debug=False, use_reloader=False)
        finally:
            sys.stderr = old_stderr

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    print("🖥️  Debug dashboard → http://localhost:5001")


# ── HTML Dashboard ─────────────────────────────────────────────────────────────

HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Creativo · Debug</title>
<link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&family=Syne:wght@400;600;800&display=swap" rel="stylesheet">
<style>
  :root {
    --bg: #0a0a0f;
    --surface: #12121a;
    --surface2: #1a1a26;
    --border: #2a2a3d;
    --accent: #6c63ff;
    --accent2: #ff6584;
    --accent3: #43d9ad;
    --accent4: #ffd166;
    --text: #e8e8f0;
    --muted: #6b6b8a;
    --mono: 'JetBrains Mono', monospace;
    --sans: 'Syne', sans-serif;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    background: var(--bg);
    color: var(--text);
    font-family: var(--mono);
    font-size: 13px;
    height: 100vh;
    overflow: hidden;
    display: grid;
    grid-template-rows: 48px 1fr;
    grid-template-columns: 1fr 1fr 320px;
  }

  /* Header */
  header {
    grid-column: 1 / -1;
    background: var(--surface);
    border-bottom: 1px solid var(--border);
    display: flex;
    align-items: center;
    padding: 0 20px;
    gap: 16px;
  }
  .logo {
    font-family: var(--sans);
    font-weight: 800;
    font-size: 16px;
    color: var(--accent);
    letter-spacing: -0.5px;
  }
  .logo span { color: var(--accent2); }
  .status-dot {
    width: 8px; height: 8px;
    border-radius: 50%;
    background: var(--accent3);
    box-shadow: 0 0 8px var(--accent3);
    animation: pulse 2s infinite;
  }
  @keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.4; }
  }
  .header-info {
    margin-left: auto;
    color: var(--muted);
    font-size: 11px;
    display: flex;
    gap: 20px;
  }
  .header-info span { color: var(--text); }

  /* Panels */
  .panel {
    background: var(--surface);
    border-right: 1px solid var(--border);
    display: flex;
    flex-direction: column;
    overflow: hidden;
  }
  .panel-header {
    padding: 12px 16px;
    border-bottom: 1px solid var(--border);
    font-family: var(--sans);
    font-weight: 600;
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    color: var(--muted);
    display: flex;
    align-items: center;
    gap: 8px;
    flex-shrink: 0;
  }
  .panel-header .dot {
    width: 6px; height: 6px;
    border-radius: 50%;
  }
  .panel-body {
    flex: 1;
    overflow-y: auto;
    padding: 12px;
    scrollbar-width: thin;
    scrollbar-color: var(--border) transparent;
  }

  /* State panel */
  .state-section {
    margin-bottom: 16px;
  }
  .state-label {
    font-size: 10px;
    text-transform: uppercase;
    letter-spacing: 1px;
    color: var(--muted);
    margin-bottom: 6px;
  }
  .state-value {
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: 8px 10px;
    font-size: 12px;
    line-height: 1.5;
    word-break: break-all;
  }
  .agent-badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 12px;
    font-weight: 600;
    font-family: var(--sans);
  }
  .agent-caption    { background: #1a2a3a; color: #60b0ff; border: 1px solid #60b0ff44; }
  .agent-email      { background: #2a1a3a; color: #c060ff; border: 1px solid #c060ff44; }
  .agent-talk       { background: #1a2a2a; color: #43d9ad; border: 1px solid #43d9ad44; }
  .agent-brand_onboarding { background: #2a2a1a; color: #ffd166; border: 1px solid #ffd16644; }
  .agent-brand_update     { background: #2a1a1a; color: #ff6584; border: 1px solid #ff658444; }
  .agent-media_approval   { background: #1a1a2a; color: #6c63ff; border: 1px solid #6c63ff44; }
  .agent-media_search     { background: #1a2a1a; color: #69ff80; border: 1px solid #69ff8044; }
  .agent-docs       { background: #2a2a2a; color: #aaa; border: 1px solid #aaa4; }

  /* Brand fields */
  .field-row {
    display: flex;
    gap: 8px;
    padding: 5px 0;
    border-bottom: 1px solid var(--border);
    align-items: flex-start;
  }
  .field-row:last-child { border-bottom: none; }
  .field-key {
    color: var(--accent);
    min-width: 140px;
    font-size: 11px;
    flex-shrink: 0;
    padding-top: 1px;
  }
  .field-val {
    color: var(--text);
    font-size: 11px;
    line-height: 1.5;
    opacity: 0.85;
  }
  .field-empty { opacity: 0.25; font-style: italic; }

  /* Events feed */
  .event {
    padding: 8px 10px;
    margin-bottom: 6px;
    border-radius: 4px;
    border-left: 3px solid var(--border);
    background: var(--surface2);
    animation: slideIn 0.2s ease;
  }
  @keyframes slideIn {
    from { opacity: 0; transform: translateY(-4px); }
    to   { opacity: 1; transform: translateY(0); }
  }
  .event-agent_selected  { border-color: var(--accent); }
  .event-routing_decision { border-color: var(--accent4); }
  .event-brand_fields    { border-color: var(--accent3); }
  .event-collected_fields { border-color: var(--accent3); }
  .event-user_input      { border-color: #ffffff33; }
  .event-agent_response  { border-color: #6c63ff66; }
  .event-onboarding_state { border-color: var(--accent4); }
  .event-passive_capture { border-color: var(--accent2); }
  .event-finalizer       { border-color: var(--accent3); }
  .event-error           { border-color: var(--accent2); }

  .event-ts {
    font-size: 10px;
    color: var(--muted);
    margin-bottom: 3px;
    display: flex;
    justify-content: space-between;
  }
  .event-type-badge {
    font-size: 10px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    padding: 1px 6px;
    border-radius: 3px;
    background: var(--surface);
  }
  .event-body {
    font-size: 11px;
    line-height: 1.5;
    color: var(--text);
    opacity: 0.8;
  }
  .event-body .key { color: var(--accent); }
  .event-body .val { color: var(--accent3); }

  /* Conversation panel */
  .message {
    margin-bottom: 12px;
    animation: slideIn 0.2s ease;
  }
  .message-role {
    font-size: 10px;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 4px;
  }
  .message-role.user { color: var(--accent2); }
  .message-role.ai   { color: var(--accent); }
  .message-body {
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: 8px 10px;
    font-size: 12px;
    line-height: 1.6;
    white-space: pre-wrap;
  }
  .message-agent {
    font-size: 10px;
    color: var(--muted);
    margin-top: 3px;
  }

  /* Scrollbar */
  ::-webkit-scrollbar { width: 4px; }
  ::-webkit-scrollbar-track { background: transparent; }
  ::-webkit-scrollbar-thumb { background: var(--border); border-radius: 2px; }
</style>
</head>
<body>

<header>
  <div class="status-dot"></div>
  <div class="logo">Creativo <span>Debug</span></div>
  <div class="header-info">
    Active agent: <span id="hdr-agent">—</span>
    &nbsp;·&nbsp;
    Last update: <span id="hdr-ts">—</span>
  </div>
</header>

<!-- Panel 1: Conversation -->
<div class="panel">
  <div class="panel-header">
    <div class="dot" style="background:#6c63ff"></div>
    Conversation
  </div>
  <div class="panel-body" id="conv-panel"></div>
</div>

<!-- Panel 2: Brand + Onboarding state -->
<div class="panel">
  <div class="panel-header">
    <div class="dot" style="background:#ffd166"></div>
    Brand State
  </div>
  <div class="panel-body" id="brand-panel"></div>
</div>

<!-- Panel 3: Event feed -->
<div class="panel" style="border-right:none">
  <div class="panel-header">
    <div class="dot" style="background:#43d9ad"></div>
    Live Events
  </div>
  <div class="panel-body" id="events-panel"></div>
</div>

<script>
let lastEventCount = 0;

function agentBadge(name) {
  if (!name) return '<span style="color:var(--muted)">—</span>';
  return `<span class="agent-badge agent-${name}">${name}</span>`;
}

function renderState(state) {
  document.getElementById('hdr-agent').innerHTML = agentBadge(state.active_agent);

  // Conversation panel
  const conv = document.getElementById('conv-panel');
  let html = '';
  if (state.last_user_input) {
    html += `<div class="message">
      <div class="message-role user">You</div>
      <div class="message-body">${esc(state.last_user_input)}</div>
    </div>`;
  }
  if (state.last_response) {
    html += `<div class="message">
      <div class="message-role ai">AI · ${agentBadge(state.active_agent)}</div>
      <div class="message-body">${esc(state.last_response)}</div>
    </div>`;
  }
  if (state.routing_context) {
    html += `<div class="state-section" style="margin-top:12px">
      <div class="state-label">Routing Context Passed</div>
      <div class="state-value" style="font-size:11px;opacity:0.7">${esc(state.routing_context)}</div>
    </div>`;
  }
  conv.innerHTML = html || '<div style="color:var(--muted);padding:8px">Waiting for conversation...</div>';

  // Brand panel
  const brand = document.getElementById('brand-panel');
  let bhtml = '';

  // Active agent
  bhtml += `<div class="state-section">
    <div class="state-label">Active Agent</div>
    <div class="state-value">${agentBadge(state.active_agent)}</div>
  </div>`;

  // Gemini routing decision
  if (state.gemini_choice) {
    bhtml += `<div class="state-section">
      <div class="state-label">Gemini Routing Decision</div>
      <div class="state-value">${esc(state.gemini_choice)}</div>
    </div>`;
  }

  // Onboarding state
  if (state.onboarding_mode) {
    bhtml += `<div class="state-section">
      <div class="state-label">Onboarding Mode</div>
      <div class="state-value">${esc(state.onboarding_mode)}</div>
    </div>`;
  }
  if (state.onboarding_section) {
    bhtml += `<div class="state-section">
      <div class="state-label">Current Section</div>
      <div class="state-value">${esc(state.onboarding_section)}</div>
    </div>`;
  }
  if (state.onboarding_missing) {
    bhtml += `<div class="state-section">
      <div class="state-label">Missing Fields</div>
      <div class="state-value" style="color:var(--accent2)">${esc(state.onboarding_missing)}</div>
    </div>`;
  }

  // Brand fields from Firestore
  const bf = state.brand_fields || {};
  const cf = state.collected_fields || {};
  const allKeys = [...new Set([...Object.keys(bf), ...Object.keys(cf)])].filter(k => k !== 'metadata');

  // Flatten nested brand fields for display
  function flattenFields(obj, prefix) {
    const result = {};
    prefix = prefix || '';
    for (const [k, v] of Object.entries(obj)) {
      if (k === 'metadata') continue;
      const label = prefix ? prefix + '.' + k : k;
      if (v && typeof v === 'object' && !Array.isArray(v)) {
        Object.assign(result, flattenFields(v, label));
      } else if (Array.isArray(v)) {
        if (v.length > 0) result[label] = v.join(', ');
      } else if (v !== null && v !== undefined && v !== '') {
        result[label] = v;
      }
    }
    return result;
  }

  const flatBF = flattenFields(bf);
  const flatCF = flattenFields(cf);
  const allFlatKeys = [...new Set([...Object.keys(flatBF), ...Object.keys(flatCF)])];
  const filledKeys = allFlatKeys.filter(k => flatBF[k] || flatCF[k]);
  const emptyKeys = allFlatKeys.filter(k => !flatBF[k] && !flatCF[k]);

  if (filledKeys.length > 0) {
    bhtml += `<div class="state-section">
      <div class="state-label">Brand Fields — Filled (${filledKeys.length})</div>
      <div class="state-value" style="padding:4px 8px">`;
    for (const k of filledKeys) {
      const v = flatBF[k] || flatCF[k];
      bhtml += `<div class="field-row">
        <div class="field-key" style="font-size:10px">${esc(k)}</div>
        <div class="field-val">${esc(String(v).slice(0, 100))}</div>
      </div>`;
    }
    bhtml += `</div></div>`;
  }

  if (emptyKeys.length > 0) {
    bhtml += `<div class="state-section">
      <div class="state-label" style="cursor:pointer" onclick="this.nextElementSibling.style.display=this.nextElementSibling.style.display==='none'?'block':'none'">
        Empty Fields (${emptyKeys.length}) — click to expand
      </div>
      <div class="state-value" style="padding:4px 8px;display:none">`;
    for (const k of emptyKeys) {
      bhtml += `<div class="field-row">
        <div class="field-key field-empty" style="font-size:10px">${esc(k)}</div>
        <div class="field-val field-empty">—</div>
      </div>`;
    }
    bhtml += `</div></div>`;
  }

  brand.innerHTML = bhtml;
}

function renderEvents(events) {
  const panel = document.getElementById('events-panel');
  if (events.length === lastEventCount) return;
  lastEventCount = events.length;

  let html = '';
  for (const ev of events) {
    const d = ev.data;
    let body = '';

    if (ev.type === 'agent_selected') {
      body = `<span class="key">agent</span> → <span class="val">${d.agent}</span>`;
      if (d.gemini_choice) body += `<br><span class="key">gemini said</span> → <span class="val">${d.gemini_choice}</span>`;
    } else if (ev.type === 'routing_decision') {
      body = `<span class="key">input</span>: ${esc((d.user_input||'').slice(0,60))}<br><span class="key">→</span> <span class="val">${d.gemini_choice}</span>`;
    } else if (ev.type === 'user_input') {
      body = esc((d.text||'').slice(0, 100));
    } else if (ev.type === 'agent_response') {
      body = esc((d.text||'').slice(0, 100)) + (d.text && d.text.length > 100 ? '...' : '');
    } else if (ev.type === 'onboarding_state') {
      body = `<span class="key">mode</span>: ${d.mode}<br><span class="key">section</span>: ${d.section}<br><span class="key">missing</span>: <span style="color:var(--accent2)">${(d.missing||'').slice(0,80)}</span>`;
    } else if (ev.type === 'brand_fields') {
      const keys = Object.keys(d).filter(k => k !== 'metadata' && d[k]);
      body = `<span class="key">${keys.length} fields loaded</span>: ${keys.slice(0,5).join(', ')}${keys.length > 5 ? '...' : ''}`;
    } else if (ev.type === 'passive_capture') {
      body = `<span class="key">captured</span>: ${esc((d.answer||'').slice(0,80))}`;
    } else if (ev.type === 'finalizer') {
      body = `<span class="val">${esc(d.message||'')}</span>`;
    } else if (ev.type === 'error') {
      body = `<span style="color:var(--accent2)">${esc(d.message||'')}</span>`;
    } else {
      body = esc(JSON.stringify(d).slice(0, 120));
    }

    html += `<div class="event event-${ev.type}">
      <div class="event-ts">
        <span>${ev.ts}</span>
        <span class="event-type-badge">${ev.type}</span>
      </div>
      <div class="event-body">${body}</div>
    </div>`;
  }
  panel.innerHTML = html || '<div style="color:var(--muted);padding:8px">No events yet...</div>';
}

function esc(s) {
  return String(s||'')
    .replace(/&/g,'&amp;')
    .replace(/</g,'&lt;')
    .replace(/>/g,'&gt;');
}

let lastStateHash = '';

function hashState(state) {
  return JSON.stringify({
    agent: state.active_agent,
    input: state.last_user_input,
    response: state.last_response,
    routing: state.routing_context,
    gemini: state.gemini_choice,
    brand: state.brand_fields,
    collected: state.collected_fields,
    mode: state.onboarding_mode,
    section: state.onboarding_section,
    missing: state.onboarding_missing,
  });
}

async function poll() {
  try {
    const r = await fetch('/api/state');
    const json = await r.json();
    document.getElementById('hdr-ts').textContent = new Date().toLocaleTimeString();

    const newHash = hashState(json.state);
    if (newHash !== lastStateHash) {
      lastStateHash = newHash;
      renderState(json.state);
    }

    renderEvents(json.events);
  } catch(e) {}
}

setInterval(poll, 800);
poll();
</script>
</body>
</html>
"""