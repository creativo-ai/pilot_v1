from dotenv import load_dotenv; load_dotenv()
from google.cloud import firestore
import os, json
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'creativo-bf5c8-fc5f772a16e4.json'
db = firestore.Client(project='creativo-bf5c8', database='agent-orchestration')

brand_id = "test2"  
doc = db.collection('brands').document(brand_id).get()
if doc.exists:
    print(json.dumps(doc.to_dict(), indent=2, default=str))
else:
    print('No brand doc found')