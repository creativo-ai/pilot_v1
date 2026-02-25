import os
from google.cloud import firestore

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "creativo-bf5c8-fc5f772a16e4.json"

db = firestore.Client(project="creativo-bf5c8" , database= "agent-orchestration")

# List all collections
collections = db.collections()
for col in collections:
    print(f"\nCollection: {col.id}")
    # Get 2 sample docs from each
    docs = col.limit(2).stream()
    for doc in docs:
        print(f"  Doc ID: {doc.id}")
        print(f"  Data: {doc.to_dict()}")