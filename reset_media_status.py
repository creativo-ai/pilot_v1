import os
from dotenv import load_dotenv
from google.cloud import firestore

load_dotenv()
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "creativo-bf5c8-fc5f772a16e4.json"

db = firestore.Client(project="creativo-bf5c8", database="agent-orchestration")

# Must match the brand_id stored IN the media documents (not necessarily main.py BRAND_ID)
# Run check_media.py first to see what brand_id your documents actually have
BRAND_ID = "mario"
USER_ID  = "manar"

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

# Verify these belong to the right brand before resetting
docs = {d.id: d.to_dict() for d in db.collection("images").stream()}
for doc_id, status in resets.items():
    doc = docs.get(doc_id, {})
    if doc.get("brand_id") != BRAND_ID or doc.get("user_id") != USER_ID:
        print(f"⚠️  Skipping {doc_id} — belongs to brand={doc.get('brand_id')}, user={doc.get('user_id')}")
        continue
    db.collection("images").document(doc_id).update({"status": status})
    print(f"✅ {doc_id} → {status}")