import os
from dotenv import load_dotenv
from google.cloud import firestore

load_dotenv()
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "creativo-bf5c8-fc5f772a16e4.json"

db = firestore.Client(project="creativo-bf5c8", database="agent-orchestration")

resets = {
    "image_001": "pending",
    "image_002": "approved",
    "image_003": "pending",
    "image_004": "rejected",
    "image_005": "pending",
    "image_006": "approved",
    "video_001": "pending",
    "video_002": "approved",
    "video_003": "pending",
    "video_004": "rejected",
    "video_005": "pending",
    "video_006": "approved",
}

for doc_id, status in resets.items():
    db.collection("images").document(doc_id).update({"status": status})
    print(f"✅ {doc_id} → {status}")