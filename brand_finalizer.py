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
import thread_manager

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
        print(f"\n🔔 Finalizer: stale onboarding detected — user={user_id}, brand={brand_id}")
        try:
            _finalize_thread(thread, user_id, brand_id)
            thread_manager.mark_finalized(user_id, "brand_onboarding")
            print(f"✅ Onboarding finalized for {user_id}")
        except Exception as e:
            print(f"❌ Finalization failed for {user_id}: {e}")


# ── Finalization logic ────────────────────────────────────────────────────────

def _finalize_thread(thread: dict, user_id: str, brand_id: str):
    structured_fields = thread.get("collected_fields", {})
    free_form_notes = thread.get("free_form_notes", "")
    messages = thread.get("messages", [])

    # Write 1: Structured fields → Firestore brands doc
    if structured_fields:
        db.collection("brands").document(brand_id).set(structured_fields, merge=True)
        print(f"  📝 Wrote {len(structured_fields)} structured field(s) to Firestore")

    # Write 2: Free-form brand context → Vertex AI
    conversation_text = "\n".join(
        f"{m['role'].upper()}: {m['content']}" for m in messages
    )
    source_text = f"{conversation_text}\n\nAdditional notes:\n{free_form_notes}".strip()

    if not source_text:
        print("  ⚠️  No context to embed, skipping Vertex AI write")
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
    print(f"  🧠 Brand context {chunk_id} → Firestore + Vertex AI")


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