import os
from google.cloud import firestore

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "creativo-bf5c8-fc5f772a16e4.json"
db = firestore.Client(project="creativo-bf5c8", database="agent-orchestration")

db.collection("images").document("image_001").update({"status": "pending"})
print("Done")