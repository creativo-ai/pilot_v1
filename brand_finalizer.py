"""
brand_finalizer.py
Local background listener that polls every POLL_INTERVAL_SECONDS for stale
onboarding threads and finalizes them — no GCP Cloud Scheduler needed.

Started automatically by main.py via start_listener().
"""

import os
import threading
import time
from dotenv import load_dotenv
from google.cloud import firestore, aiplatform
import vertexai
from vertexai.language_models import TextEmbeddingModel
from llm_client import gemini_generate
import json
import thread_manager

TEMPLATE_PATH = os.path.join(os.path.dirname(__file__), "brand_book_template.json")

# Flat field → nested path mapping so collected flat fields populate the full template
FIELD_MAP = {
    # Metadata
    "brand_name":                   ["metadata", "brand_name"],
    "brand_id":                     ["metadata", "brand_id"],
    "industry":                     ["metadata", "industry"],

    # Brand Foundation
    "mission":                      ["brand_foundation", "mission"],
    "vision":                       ["brand_foundation", "vision"],
    "purpose":                      ["brand_foundation", "purpose"],
    "values":                       ["brand_foundation", "core_values"],
    "core_values":                  ["brand_foundation", "core_values"],
    "brand_promise":                ["brand_foundation", "brand_promise"],
    "brand_story":                  ["brand_foundation", "brand_story", "origin_story"],
    "origin_story":                 ["brand_foundation", "brand_story", "origin_story"],
    "founder_story":                ["brand_foundation", "brand_story", "founder_story"],
    "tagline":                      ["brand_foundation", "tagline"],
    "slogan":                       ["brand_foundation", "slogan"],
    "positioning":                  ["brand_foundation", "positioning_statement"],
    "elevator_pitch":               ["brand_foundation", "value_proposition"],
    "value_proposition":            ["brand_foundation", "value_proposition"],
    "unique_selling_points":        ["brand_foundation", "unique_selling_points"],

    # Brand Personality — use correct nested paths from template
    "tone":                         ["brand_personality", "tone_of_voice", "primary_tone"],
    "brand_voice":                  ["brand_personality", "brand_voice_description"],
    "brand_archetype":              ["brand_personality", "archetype_primary"],
    "personality_traits":           ["brand_personality", "personality_traits"],
    "language_style":               ["brand_personality", "brand_human_character"],
    "communication_style":          ["brand_personality", "brand_human_character"],

    # Target Audience
    "target_audience":              ["target_audience", "primary_audience", "description"],
    "primary_audience_age":         ["target_audience", "primary_audience", "demographics", "age_range"],
    "primary_audience_location":    ["target_audience", "primary_audience", "demographics", "location"],
    "primary_audience_pain_points": ["target_audience", "primary_audience", "psychographics", "pain_points"],

    # Market Positioning
    "competitors":                  ["market_positioning", "competitive_landscape", "direct_competitors"],
    "competitive_advantages":       ["market_positioning", "competitive_landscape", "competitive_advantages"],
    "pricing_strategy":             ["market_positioning", "pricing_strategy"],

    # Brand Messaging
    "core_messages":                ["brand_messaging", "core_messages"],
    "key_benefits":                 ["brand_messaging", "key_benefits"],
    "primary_goal":                 ["brand_messaging", "primary_goal"],
    "elevator_pitch_short":         ["brand_messaging", "elevator_pitch_short"],
    "elevator_pitch_long":          ["brand_messaging", "elevator_pitch_long"],

    # Visual Identity
    "primary_color":                ["visual_identity", "color_palette", "primary_colors"],
    "secondary_color":              ["visual_identity", "color_palette", "secondary_colors"],

    # Content Strategy
    "content_pillars":              ["content_strategy", "content_pillars"],
    "posting_frequency":            ["content_strategy", "posting_frequency"],
    "hashtags":                     ["content_strategy", "hashtag_strategy"],

    # Email Branding
    "email_greeting":               ["email_branding", "email_greeting_style"],
    "email_closing":                ["email_branding", "email_closing_style"],
    "default_email_signature":      ["email_branding", "default_signature"],

    # Product / Service
    "flagship_product":             ["product_or_service", "flagship_product"],
    "offerings":                    ["product_or_service", "offerings"],

    # Future Vision
    "short_term_goals":             ["future_vision", "short_term_goals"],
    "long_term_goals":              ["future_vision", "long_term_goals"],
    "expansion_plans":              ["future_vision", "expansion_plans"],
}

def _set_nested(d: dict, path: list, value):
    """Set a value at a nested path in dict d, creating intermediate dicts."""
    for key in path[:-1]:
        d = d.setdefault(key, {})
    d[path[-1]] = value

def _load_template() -> dict:
    with open(TEMPLATE_PATH, "r") as f:
        return json.load(f)

def _load_existing_brand(brand_id: str) -> dict:
    """
    Load existing brand doc from Firestore into full template structure.
    Always starts from the full template so no fields are ever missing.
    Then overlays any existing saved values on top.
    """
    template = _load_template()
    doc = db.collection("brands").document(brand_id).get()
    if not doc.exists:
        return template
    existing = doc.to_dict() or {}
    if "brand_foundation" in existing:
        # Already nested — deep merge existing into template
        _deep_merge_into(template, existing)
    else:
        # Flat doc — map flat fields into template paths
        for flat_key, path in FIELD_MAP.items():
            if flat_key in existing and existing[flat_key]:
                _set_nested(template, path, existing[flat_key])
    return template


def _deep_merge_into(base: dict, overlay: dict):
    """Merge overlay into base in place, only overwriting non-empty values."""
    for k, v in overlay.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_merge_into(base[k], v)
        elif v not in (None, "", [], {}):
            base[k] = v

load_dotenv()
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "creativo-bf5c8-fc5f772a16e4.json"

PROJECT_ID = "creativo-bf5c8"
REGION = "us-central1"
INACTIVITY_MINUTES = 5
POLL_INTERVAL_SECONDS = 60  # check every minute, finalize if inactive > 5 min

db = firestore.Client(project=PROJECT_ID, database="agent-orchestration")
vertexai.init(project=PROJECT_ID, location=REGION)
aiplatform.init(project=PROJECT_ID, location=REGION)
embedding_model = TextEmbeddingModel.from_pretrained("text-embedding-005")

CONTEXT_SUMMARY_SYSTEM = """
You are a brand strategist. Given a brand onboarding conversation, write a rich,
comprehensive brand context summary. Include:
- Brand personality and voice
- Origin story or founding context if mentioned
- Market positioning and differentiators
- Competitor landscape if mentioned
- Target audience nuances beyond demographics
- Any other brand insights that don't fit standard fields

Write in flowing prose, 200-400 words. This will be used as semantic context
for AI agents answering questions about this brand.
"""


# ── Public API ────────────────────────────────────────────────────────────────

def start_listener():
    """
    Start the background finalizer as a daemon thread.
    Call once from main.py on startup. Stops automatically when main exits.
    """
    t = threading.Thread(target=_poll_loop, daemon=True, name="BrandFinalizer")
    t.start()
    print(f"🔁 Brand finalizer started (polls every {POLL_INTERVAL_SECONDS}s, "
          f"finalizes after {INACTIVITY_MINUTES}min inactivity)")
    return t


# ── Internal loop ─────────────────────────────────────────────────────────────

def _poll_loop():
    while True:
        try:
            _run_once()
        except Exception as e:
            print(f"⚠️  Finalizer error: {e}")
        time.sleep(POLL_INTERVAL_SECONDS)


def _run_once():
    stale = thread_manager.get_stale_onboarding_threads(INACTIVITY_MINUTES)
    if not stale:
        return
    for thread in stale:
        user_id = thread["user_id"]
        brand_id = thread.get("brand_id", "creativo")
        try:
            _finalize_thread(thread, user_id, brand_id)
            thread_manager.mark_finalized(user_id, "brand_onboarding")
            from debug_hooks import log_finalizer
            log_finalizer(f"Brand book finalized for {brand_id}")
            import sys
            sys.stdout.write(f"\n\n[Brand book saved for {brand_id}]\n\n")
            sys.stdout.flush()
        except Exception as e:
            from debug_hooks import log_error
            log_error(f"Finalizer failed for {brand_id}: {e}")


# ── Finalization logic ────────────────────────────────────────────────────────

def _finalize_thread(thread: dict, user_id: str, brand_id: str):
    collected_fields = thread.get("collected_fields", {})
    messages = thread.get("messages", [])
    import datetime

    # Write 1: Load existing brand book (or template), merge all collected fields
    # into the full nested structure, then write the complete document to Firestore.
    brand_book = _load_existing_brand(brand_id)

    # Map each flat collected field into the correct nested path
    for flat_key, value in collected_fields.items():
        if not value or value in ({}, [], ""):
            continue
        if flat_key in FIELD_MAP:
            _set_nested(brand_book, FIELD_MAP[flat_key], value)
        else:
            # Unknown field — store it in a catch-all section
            brand_book.setdefault("extra_fields", {})[flat_key] = value

    # Update metadata
    brand_book.setdefault("metadata", {})
    brand_book["metadata"]["brand_id"] = brand_id
    brand_book["metadata"]["brand_name"] = (
        collected_fields.get("brand_name")
        or brand_book["metadata"].get("brand_name", "")
    )
    brand_book["metadata"]["updated_at"] = str(datetime.date.today())
    brand_book["metadata"]["version"] = _get_next_brand_version(brand_id)
    brand_book["metadata"]["status"] = "active"

    # Write full nested brand book to Firestore
    db.collection("brands").document(brand_id).set(brand_book)

    # Write 2: Summarize the full conversation to extract rich free-form brand context.
    # Free-form info is never stored in variables — it lives only in conversation history.
    conversation_text = "\n".join(
        f"{m['role'].upper()}: {m['content']}" for m in messages
    )
    source_text = conversation_text.strip()

    if not source_text:
        pass  # silent
        return

    brand_context_summary = gemini_generate(
        prompt=source_text,
        system=CONTEXT_SUMMARY_SYSTEM
    )

    new_version = _get_next_context_version(brand_id)
    chunk_id = f"ctx_{brand_id}_{new_version}"

    db.collection("brand_context").document(chunk_id).set({
        "chunk_id": chunk_id,
        "brand_id": brand_id,
        "version": new_version,
        "content": brand_context_summary,
        "source": "onboarding",
        "user_id": user_id,
    })

    vector = embedding_model.get_embeddings([brand_context_summary])[0].values
    _upsert_to_vertex(chunk_id, vector)
    pass  # silent




def _get_next_brand_version(brand_id: str) -> str:
    """Increment the brand document version on each finalized session."""
    doc = db.collection("brands").document(brand_id).get()
    if doc.exists:
        d = doc.to_dict() or {}
        # Check nested metadata first, then flat version field
        current = (
            d.get("metadata", {}).get("version")
            or d.get("version", "1.0.0")
        )
        try:
            parts = str(current).split(".")
            parts[-1] = str(int(parts[-1]) + 1)
            return ".".join(parts)
        except Exception:
            return "1.0.1"
    return "1.0.0"
def _get_next_context_version(brand_id: str) -> str:
    docs = list(
        db.collection("brand_context")
        .where("brand_id", "==", brand_id)
        .stream()
    )
    versions = []
    for doc in docs:
        v = doc.to_dict().get("version", "v1")
        try:
            versions.append(int(v.replace("v", "")))
        except Exception:
            versions.append(1)
    return f"v{max(versions) + 1}" if versions else "v1"


def _upsert_to_vertex(chunk_id: str, vector: list):
    endpoint = aiplatform.MatchingEngineIndexEndpoint(
        index_endpoint_name=os.getenv("VERTEX_INDEX_ENDPOINT_ID")
    )
    deployed = endpoint.deployed_indexes
    if not deployed:
        raise RuntimeError("No deployed indexes found on endpoint")
    index = aiplatform.MatchingEngineIndex(deployed[0].index)
    index.upsert_datapoints(datapoints=[{
        "datapoint_id": chunk_id,
        "feature_vector": vector
    }])