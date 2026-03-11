from google.cloud import firestore
import os
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "creativo-bf5c8-fc5f772a16e4.json"

db = firestore.Client(project="creativo-bf5c8", database="agent-orchestration")

USER_ID = "manar"

threads = db.collection("agent_threads").where("user_id", "==", USER_ID).stream()
for doc in threads:
    doc.reference.delete()
    print(f"Deleted thread: {doc.id}")

print("Conversation threads cleared — brand data preserved")