from google.cloud import firestore
import os, json

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'creativo-bf5c8-fc5f772a16e4.json'
db = firestore.Client(project='creativo-bf5c8', database='agent-orchestration')

#brand_name_to_search = "Ellena"

# Firestore query
#query = db.collection('brand_books').where("metadata.brand_name", "==", brand_name_to_search)
#docs = query.stream()

#found = False
#for doc in docs:
#    print(f"Document ID: {doc.id}")
#    print(json.dumps(doc.to_dict(), indent=2, default=str))
#    found = True

#if not found:
#    print("No brand book found with that name")


docs = db.collection('brand_book').stream()

for doc in docs:
    print(f"Document ID: {doc.id}")
    print(json.dumps(doc.to_dict(), indent=2, default=str))
    print("-"*50)


import thread_manager
user_id = "marmar"
thread = thread_manager.load_thread(user_id, "brand_onboarding")
print(thread)