import os
from google.cloud import firestore

# ← Make sure this path is correct for your service account JSON
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "creativo-bf5c8-fc5f772a16e4.json"

db = firestore.Client(project="creativo-bf5c8", database="agent-orchestration")

USER_ID = "marmar"
BRAND_ID = "test11"

# Fetch the document
doc_ref = db.collection("brands").document(BRAND_ID)
doc = doc_ref.get()

if doc.exists:
    data = doc.to_dict()
    print(f"Brand Info for {BRAND_ID}:")
    for k, v in data.items():
        print(f"\n{k}:")
        print(v)
else:
    print(f"No document found for BRAND_ID = {BRAND_ID}")