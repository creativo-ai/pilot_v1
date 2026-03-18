from dotenv import load_dotenv
load_dotenv()
from google.cloud import firestore

db = firestore.Client(project="creativo-bf5c8", database="agent-orchestration")

# 1. Delete all agent threads
threads = db.collection("agent_threads").stream()
count = 0
for doc in threads:
    doc.reference.delete()
    count += 1
print(f"✅ Deleted {count} agent threads")

# 2. Delete brand doc
db.collection("brands").document("marketing").delete()
print("✅ Brand doc deleted")

# 3. Delete all brand_context chunks
chunks = db.collection("brand_context").stream()
count = 0
for doc in chunks:
    doc.reference.delete()
    count += 1
print(f"✅ Deleted {count} brand_context chunks")

print("\nDone — fresh start. Media untouched.")