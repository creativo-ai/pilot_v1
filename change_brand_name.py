from dotenv import load_dotenv
load_dotenv()

from google.cloud import firestore

db = firestore.Client(project="creativo-bf5c8", database="agent-orchestration")

# Copy brand doc to new ID
old_ref = db.collection("brands").document("creativo")
new_ref = db.collection("brands").document("marketing")

data = old_ref.get().to_dict()
if data:
    new_ref.set(data)
    old_ref.delete()
    print("Brand renamed: creativo → marketing")

# Copy all agent_threads for this brand
threads = db.collection("agent_threads").stream()
for thread in threads:
    d = thread.to_dict()
    if d.get("brand_id") == "creativo":
        d["brand_id"] = "marketing"
        db.collection("agent_threads").document(thread.id).set(d)
        print(f"Updated thread: {thread.id}")

# Copy brand_context chunks
chunks = db.collection("brand_context").stream()
for chunk in chunks:
    d = chunk.to_dict()
    if d.get("brand_id") == "creativo":
        old_chunk_ref = db.collection("brand_context").document(chunk.id)
        new_id = chunk.id.replace("creativo", "marketing")
        d["brand_id"] = "marketing"
        db.collection("brand_context").document(new_id).set(d)
        old_chunk_ref.delete()
        print(f"Updated chunk: {chunk.id} → {new_id}")

print("Done.")