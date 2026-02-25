"""
Check if specific doc chunk IDs from Firestore exist in Vertex AI.
Run: python check_doc_vectors.py
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

aiplatform.init(project=PROJECT_ID, location=REGION)
vertexai.init(project=PROJECT_ID, location=REGION)
db = firestore.Client(project=PROJECT_ID, database="agent-orchestration")

embedding_model = TextEmbeddingModel.from_pretrained("text-embedding-005")
endpoint = aiplatform.MatchingEngineIndexEndpoint(INDEX_ENDPOINT_ID)

# Get all doc IDs from Firestore
print("Fetching doc IDs from Firestore...")
all_docs = list(db.collection("docs").stream())
print(f"Found {len(all_docs)} doc chunks in Firestore\n")

# Query Vertex AI using actual content from each doc and see if its own ID comes back
print("Testing if doc vectors exist in Vertex AI...")
print("(Querying with each doc's own content — its ID should appear in top results if indexed)\n")

found = 0
missing = 0

for doc in all_docs[:10]:  # Test first 10
    data = doc.to_dict()
    content = data.get("content", "").strip()[:200]
    if not content:
        continue

    vector = embedding_model.get_embeddings([content])[0].values
    response = endpoint.find_neighbors(
        deployed_index_id=DEPLOYED_INDEX_ID,
        queries=[vector],
        num_neighbors=10,
        return_full_datapoint=False
    )

    matches = response[0] if response else []
    match_ids = [m.id for m in matches]
    doc_matches = [mid for mid in match_ids if mid.startswith("doc_")]

    if doc.id in match_ids:
        print(f"  ✅ {doc.id} — found in Vertex AI")
        found += 1
    elif doc_matches:
        print(f"  ⚠️  {doc.id} — NOT found, but other doc_ IDs returned: {doc_matches[:2]}")
        found += 1
    else:
        print(f"  ❌ {doc.id} — NO doc_ vectors returned at all (only brand_ chunks)")
        missing += 1

print(f"\nResult: {found} docs reachable, {missing} docs missing from Vertex AI index")
if missing > 0:
    print("\n→ Doc vectors are missing. The upsert in seed_vertexAI.py likely used")
    print("  a DIFFERENT index than the one deployed on the endpoint.")
    print("  Check: does VERTEX_INDEX_ID match the index deployed on the endpoint?")
