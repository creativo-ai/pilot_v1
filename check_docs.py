"""
Diagnostic: Check if documentation exists in Firestore and Vertex AI.
Run: python check_docs.py
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

db = firestore.Client(project=PROJECT_ID, database="agent-orchestration")
vertexai.init(project=PROJECT_ID, location=REGION)
aiplatform.init(project=PROJECT_ID, location=REGION)

# ── 1. Check Firestore ──────────────────────────────────────────────
print("=" * 50)
print("1. FIRESTORE — docs collection")
print("=" * 50)

docs = list(db.collection("docs").limit(5).stream())
if docs:
    print(f"✅ Found {len(docs)} doc chunk(s) (showing up to 5):")
    for d in docs:
        data = d.to_dict()
        print(f"  - {d.id} | file: {data.get('filename', 'N/A')} | chunk {data.get('chunk_index', '?')}/{data.get('total_chunks', '?')}")
else:
    print("❌ No documents found in Firestore 'docs' collection — index is empty.")

# Count total
total = len(list(db.collection("docs").stream()))
print(f"\nTotal doc chunks in Firestore: {total}")

# ── 2. Check Vertex AI with a test query ───────────────────────────
print("\n" + "=" * 50)
print("2. VERTEX AI — test semantic search")
print("=" * 50)

INDEX_ENDPOINT_ID = os.getenv("VERTEX_INDEX_ENDPOINT_ID")
DEPLOYED_INDEX_ID = os.getenv("VERTEX_DEPLOYED_INDEX_ID")

if not INDEX_ENDPOINT_ID or not DEPLOYED_INDEX_ID:
    print("❌ VERTEX_INDEX_ENDPOINT_ID or VERTEX_DEPLOYED_INDEX_ID not set in .env")
else:
    try:
        embedding_model = TextEmbeddingModel.from_pretrained("text-embedding-005")
        endpoint = aiplatform.MatchingEngineIndexEndpoint(index_endpoint_name=INDEX_ENDPOINT_ID)

        test_query = "how to cancel subscription"
        vector = embedding_model.get_embeddings([test_query])[0].values

        response = endpoint.find_neighbors(
            deployed_index_id=DEPLOYED_INDEX_ID,
            queries=[vector],
            num_neighbors=5,
            return_full_datapoint=False
        )

        matches = response[0] if response else []
        doc_matches = [m for m in matches if not m.id.startswith("brand_")]

        if doc_matches:
            print(f"✅ Vertex AI returned {len(doc_matches)} doc match(es) for '{test_query}':")
            for m in doc_matches:
                print(f"  - ID: {m.id} | distance: {m.distance:.4f}")
                # Fetch content from Firestore
                doc = db.collection("docs").document(m.id).get()
                if doc.exists:
                    content = doc.to_dict().get("content", "")[:150]
                    print(f"    Preview: {content}...")
                else:
                    print(f"    ⚠️  ID found in Vertex AI but NOT in Firestore (stale index)")
        else:
            print(f"❌ Vertex AI returned no doc matches for '{test_query}'")
            print("   → Docs may not be indexed, or the index is empty/stale.")

    except Exception as e:
        print(f"❌ Vertex AI query failed: {e}")

print("\n" + "=" * 50)
print("SUMMARY")
print("=" * 50)
if total == 0:
    print("→ Firestore 'docs' collection is empty. Re-run seed_vertexAI.py to re-index.")
else:
    print(f"→ {total} chunks in Firestore.")
    print("→ If Vertex AI returned no matches, the index may need to be re-seeded.")
    print("   Re-run: python seed_vertexAI.py")
