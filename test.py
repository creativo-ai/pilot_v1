from dotenv import load_dotenv
load_dotenv()
import os
from google.cloud import firestore, aiplatform

db = firestore.Client(project="creativo-bf5c8", database="agent-orchestration")
aiplatform.init(project="creativo-bf5c8", location="us-central1")

# 1. Delete ALL agent threads
for doc in db.collection("agent_threads").stream():
    doc.reference.delete()
    print(f"Deleted thread: {doc.id}")

# 2. Delete brand doc
db.collection("brands").document("marketing").delete()
print("Deleted brand doc")

# 3. Delete ALL brand_context chunks
for doc in db.collection("brand_context").stream():
    doc.reference.delete()
    print(f"Deleted chunk: {doc.id}")

# 4. Remove ALL datapoints from Vertex AI index
INDEX_ID = os.getenv("VERTEX_INDEX_ID")
index = aiplatform.MatchingEngineIndex(index_name=INDEX_ID)
ids = [f"ctx_marketing_{i}" for i in range(100)] + \
      [f"ctx_creativo_{i}" for i in range(100)] + \
      [f"ctx_zerox_{i}" for i in range(100)]
try:
    index.remove_datapoints(datapoint_ids=ids)
    print("Vertex AI cleared")
except Exception as e:
    print(f"Vertex: {e}")

print("\n✅ Full reset complete.")