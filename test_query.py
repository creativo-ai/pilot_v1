import os
from pinecone import Pinecone

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("pilot-main-index")

results = index.query(
    vector=[0.0]*1536,
    filter={
        "type": "image",
        "status": "pending"
    },
    top_k=10,
    include_metadata=True
)
print(results)