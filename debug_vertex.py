"""
Deep diagnostic: checks Vertex AI index type, endpoint config, and whether
a recently upserted vector can be fetched back.
Run: python debug_vertex.py
"""
import os
from dotenv import load_dotenv
from google.cloud import firestore, aiplatform
import vertexai
from vertexai.language_models import TextEmbeddingModel

load_dotenv()
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "creativo-bf5c8-fc5f772a16e4.json"

PROJECT_ID = "creativo-bf5c8"
REGION = "us-central1"
INDEX_ENDPOINT_ID = os.getenv("VERTEX_INDEX_ENDPOINT_ID")
DEPLOYED_INDEX_ID = os.getenv("VERTEX_DEPLOYED_INDEX_ID")
INDEX_ID = os.getenv("VERTEX_INDEX_ID")

aiplatform.init(project=PROJECT_ID, location=REGION)
vertexai.init(project=PROJECT_ID, location=REGION)

# ── 1. Print env vars ──────────────────────────────────────────────
print("=" * 55)
print("1. ENV VARS")
print("=" * 55)
print(f"  VERTEX_INDEX_ID:           {INDEX_ID}")
print(f"  VERTEX_INDEX_ENDPOINT_ID:  {INDEX_ENDPOINT_ID}")
print(f"  VERTEX_DEPLOYED_INDEX_ID:  {DEPLOYED_INDEX_ID}")

# ── 2. Inspect the index ───────────────────────────────────────────
print("\n" + "=" * 55)
print("2. INDEX INFO")
print("=" * 55)
try:
    index = aiplatform.MatchingEngineIndex(INDEX_ID)
    print(f"  Display name:  {index.display_name}")
    print(f"  Resource name: {index.resource_name}")
    print(f"  Update method: {index.to_dict().get('indexUpdateMethod', 'NOT SET — likely BATCH')}")
except Exception as e:
    print(f"  ❌ Could not fetch index: {e}")

# ── 3. Inspect the endpoint and its deployed indexes ──────────────
print("\n" + "=" * 55)
print("3. ENDPOINT & DEPLOYED INDEXES")
print("=" * 55)
try:
    endpoint = aiplatform.MatchingEngineIndexEndpoint(INDEX_ENDPOINT_ID)
    print(f"  Endpoint display name: {endpoint.display_name}")
    deployed = endpoint.deployed_indexes
    if deployed:
        for d in deployed:
            print(f"  Deployed index id:     {d.id}")
            print(f"  Deployed index ref:    {d.index}")
            match = "✅ matches DEPLOYED_INDEX_ID" if d.id == DEPLOYED_INDEX_ID else "❌ MISMATCH with DEPLOYED_INDEX_ID"
            print(f"  Match check:           {match}")
    else:
        print("  ❌ No deployed indexes found on this endpoint.")
except Exception as e:
    print(f"  ❌ Could not fetch endpoint: {e}")

# ── 4. Try querying with a known Firestore doc ID ─────────────────
print("\n" + "=" * 55)
print("4. TEST QUERY — known doc chunk")
print("=" * 55)
try:
    db = firestore.Client(project=PROJECT_ID, database="agent-orchestration")
    sample_docs = list(db.collection("docs").limit(1).stream())
    if not sample_docs:
        print("  ❌ No docs in Firestore to test with.")
    else:
        sample_id = sample_docs[0].id
        sample_content = sample_docs[0].to_dict().get("content", "")[:100]
        print(f"  Testing with Firestore doc: {sample_id}")
        print(f"  Content preview: {sample_content}...")

        embedding_model = TextEmbeddingModel.from_pretrained("text-embedding-005")
        vector = embedding_model.get_embeddings([sample_content])[0].values

        endpoint = aiplatform.MatchingEngineIndexEndpoint(INDEX_ENDPOINT_ID)
        response = endpoint.find_neighbors(
            deployed_index_id=DEPLOYED_INDEX_ID,
            queries=[vector],
            num_neighbors=5,
            return_full_datapoint=False
        )
        matches = response[0] if response else []
        if matches:
            print(f"  ✅ Got {len(matches)} result(s):")
            for m in matches:
                print(f"     - {m.id} (distance: {m.distance:.4f})")
        else:
            print("  ❌ Still no results — index update method is likely BATCH.")
            print("     Batch indexes require a full rebuild job, not just upsert.")
            print("     → Solution: rebuild the index or switch to a streaming index.")
except Exception as e:
    print(f"  ❌ Query failed: {e}")
