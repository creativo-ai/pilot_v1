import os
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "creativo-bf5c8-fc5f772a16e4.json"
from google.cloud import firestore

db = firestore.Client(project="creativo-bf5c8", database="agent-orchestration")

docs = list(db.collection("images").stream())
docs_sorted = sorted(docs, key=lambda d: d.id)

print(f"{'ID':<12} {'type':<6} {'status':<10} {'user':<8} {'brand':<12} {'description'}")
print("-" * 80)
for doc in docs_sorted:
    d = doc.to_dict()
    print(f"{doc.id:<12} {d.get('type','image'):<6} {d.get('status','?'):<10} "
          f"{d.get('user_id','?'):<8} {d.get('brand_id','?'):<12} "
          f"{d.get('description','')[:40]}")
