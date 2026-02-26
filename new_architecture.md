# Creativo AI — New Multi-Agent Architecture Design

## Core Principles
1. Each agent has its own persisted conversation thread in Firestore
2. Routing passes an LLM-generated summary, not raw history
3. Brand knowledge has two retrieval paths: structured (Firestore) vs contextual (Vertex AI)
4. Brand onboarding finalizer triggers after inactivity timeout

---

## New File Structure

```
current_version/          ← snapshot of today's working code (baseline)

new_version/
├── main.py               ← entry point, creates/loads agent thread
├── thread_manager.py     ← NEW: Firestore thread persistence per agent per user
├── agent_base.py         ← UPDATED: per-thread history, summary-based routing
├── agent_router.py       ← UPDATED: uses thread_manager, passes summary on route
├── agents/
│   ├── __init__.py
│   ├── email_agent.py
│   ├── caption_agent.py
│   ├── talk_agent.py
│   ├── media_agent.py
│   ├── brand_update_agent.py
│   ├── docs_agent.py
│   └── brand_onboarding_agent.py  ← NEW
├── brand_finalizer.py    ← NEW: inactivity listener + dual write
├── data_layer.py         ← UPDATED: dual retrieval (Firestore + Vertex AI)
├── llm_client.py         ← same
└── orchestrator.py       ← simplified: just entry routing
```

---

## 1. Thread Manager (`thread_manager.py`)

Firestore collection: `agent_threads`
Document ID: `{user_id}_{agent_name}`

```
{
  user_id: "manar",
  agent_name: "email",
  brand_id: "creativo",
  messages: [...],          # full history for THIS agent
  summary: "...",           # LLM-generated summary, updated on every route-out
  last_active: timestamp,
  created_at: timestamp
}
```

**Functions:**
- `load_thread(user_id, agent_name)` → returns messages list
- `save_thread(user_id, agent_name, messages)` → persists to Firestore
- `get_summary(user_id, agent_name)` → returns latest summary string
- `update_summary(user_id, agent_name, summary)` → saves summary
- `get_last_active(user_id, agent_name)` → returns timestamp

---

## 2. Updated Agent Base (`agent_base.py`)

Each agent:
- Loads its own thread from Firestore on `handle()`
- Saves updated thread after each response
- On route-out: generates a summary of its thread using Gemini, saves it, passes it to next agent

```python
class BaseAgent(ABC):
    def handle(self, user_input, user_id, brand_id):
        # 1. Load own thread from Firestore
        history = thread_manager.load_thread(user_id, self.name)
        
        # 2. Run own logic
        response = self.run(user_input, history, user_id, brand_id)
        
        # 3. Save updated thread
        history.append({"role": "user", "content": user_input})
        history.append({"role": "assistant", "content": response})
        thread_manager.save_thread(user_id, self.name, history)
        
        return response

    def route_to(self, next_agent, user_input, user_id, brand_id):
        # Generate summary of current thread
        summary = self._summarize_thread(user_id)
        thread_manager.update_summary(user_id, self.name, summary)
        
        # Pass summary as context to next agent (not full history)
        return next_agent.handle(
            user_input=user_input,
            user_id=user_id,
            brand_id=brand_id,
            routing_context=summary
        )

    def _summarize_thread(self, user_id):
        history = thread_manager.load_thread(user_id, self.name)
        return gemini_generate(
            prompt=format_history(history),
            system="Summarize this conversation concisely, capturing the user's intent, key decisions, and any relevant context for another agent."
        )
```

---

## 3. Brand Onboarding Agent (`brand_onboarding_agent.py`)

**Goal:** Collect brand info through natural conversation, not a form.

**Flow:**
```
Agent asks questions naturally →
User answers (some map to schema fields, some are free-form) →
Agent tracks what's been collected vs what's missing →
When inactivity detected → finalizer triggers
```

**Structured fields tracked during conversation:**
```python
BRAND_SCHEMA_FIELDS = [
    "brand_name", "industry", "mission", "vision", "values",
    "tone", "brand_voice", "communication_style",
    "target_audience", "primary_goal", "default_email_signature"
]
```

**Agent behavior:**
- Asks about missing fields naturally, one topic at a time
- Also listens for and captures free-form info (competitor insights, 
  brand story, market positioning, etc.)
- Maintains a `collected` dict tracking what's been gathered so far
- Stores `collected` in the thread document in Firestore after each turn

**Firestore thread document extras:**
```
{
  ...thread fields...,
  collected_fields: { "brand_name": "Creativo", "tone": "bold" ... },
  free_form_notes: "User mentioned they compete with X, target Gen Z..."
}
```

---

## 4. Brand Finalizer (`brand_finalizer.py`)

**Trigger:** Inactivity on the onboarding agent thread (configurable, default 5 min)

**Two writes on trigger:**

### Write 1 — Structured fields → Firestore
```python
update_brand_info(brand_id, collected_fields)
```

### Write 2 — Free-form context → Vertex AI (new version)
```python
# Summarize full onboarding conversation into a rich brand context
brand_context_summary = gemini_generate(
    prompt=full_conversation,
    system="Extract a comprehensive brand context summary. Include personality, 
            story, positioning, competitors, audience nuances, and anything 
            relevant that isn't captured in standard fields."
)

# Embed and upsert as new brand context version
embed_and_upsert(
    text=brand_context_summary,
    collection="brand_context",
    id_prefix=f"ctx_{brand_id}_{new_version}"
)
```

**How to detect inactivity:**
- `brand_finalizer.py` runs as a scheduled Cloud Function (every 5 min)
- Checks `last_active` on all onboarding threads
- If `now - last_active > INACTIVITY_THRESHOLD` and `status != "finalized"` → triggers

---

## 5. Dual Brand Retrieval (`data_layer.py`)

Other agents choose retrieval path based on need:

```python
# Path A: Specific field needed (fast, exact)
def get_brand_field(brand_id, field_name):
    return firestore.collection("brands").document(brand_id).get()[field_name]

# Path B: Rich contextual info needed (semantic search)
def get_brand_context(brand_id, query, top_k=3):
    # Vector search scoped to brand_context collection
    # Returns free-form brand context chunks
    return search_vector(query, prefix=f"ctx_{brand_id}", top_k=top_k)
```

**When agents use each path:**
- `write_email` → needs tone, voice, signature → Path A (Firestore)
- `talk_agent` answering "what makes us different?" → Path B (Vertex AI context)
- `write_caption` → needs platform tone + audience → Path A
- `search_docs_agent` → no brand retrieval needed
- `brand_update_agent` → Path A to confirm current value before updating

---

## 6. Routing Flow (updated)

```
User message
    │
    ▼
main.py — determines active agent from last thread activity
    │
    ▼
ActiveAgent.handle(user_input, user_id, brand_id)
    │
    ├── Loads own Firestore thread
    ├── Runs LLM with own history only
    ├── Saves updated thread
    │
    └── If routing needed:
            │
            ├── Generates summary of own thread (Gemini)
            ├── Saves summary to Firestore
            └── Calls next_agent.handle(routing_context=summary)
                    │
                    └── Next agent uses summary as context seed
                        (not full history — clean slate with context)
```

---

## 7. Firestore Collections (new/updated)

| Collection | Purpose |
|---|---|
| `brands` | Structured brand fields (unchanged) |
| `agent_threads` | Per-agent per-user conversation history + summary |
| `brand_context` | Free-form brand context chunks (from onboarding) |
| `images` | Media items (unchanged) |
| `docs` | Documentation chunks (unchanged) |
| `brand_book` | Versioned brand book chunks (unchanged) |
| `brand_updates` | Pending structured updates (unchanged) |

---

## 8. Build Order

1. `thread_manager.py` — foundation everything depends on
2. `agent_base.py` — update to use threads + summary routing
3. `brand_onboarding_agent.py` — new agent
4. `brand_finalizer.py` — inactivity listener + dual write
5. `data_layer.py` — add dual retrieval functions
6. `agent_router.py` — update routing to use summaries
7. `main.py` — update entry point to load active agent thread
8. All existing agents — update to use thread_manager

---

## What Stays the Same
- All agent LLM logic (prompts, model assignments)
- Vertex AI index and embedding model
- Tool schemas in orchestrator
- `sync_brand_book.py` scheduler pattern
- `seed_vertexAI.py`
- `llm_client.py`
