import os
from pinecone import Pinecone, ServerlessSpec

# Initialize client
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

index_name = "pilot-main-index"

# Check existing indexes
existing_indexes = [index["name"] for index in pc.list_indexes()]

if index_name in existing_indexes:
    print(f"Index '{index_name}' already exists.")
else:
    pc.create_index(
        name=index_name,
        dimension=1536, 
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",        
            region="us-east-1"  
        )
    )
    print(f"Index '{index_name}' created successfully.")