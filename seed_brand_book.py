import os
import uuid
import re
from dotenv import load_dotenv
from google.cloud import firestore, aiplatform
import vertexai
from vertexai.language_models import TextEmbeddingModel

load_dotenv()

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "creativo-bf5c8-fc5f772a16e4.json"

PROJECT_ID = "creativo-bf5c8"
REGION = "us-central1"
BRAND_ID = "creativo"
BRAND_BOOK_PATH = "./brand_book.md"
VERSION = "v1"

db = firestore.Client(project=PROJECT_ID, database="agent-orchestration")
vertexai.init(project=PROJECT_ID, location=REGION)
aiplatform.init(project=PROJECT_ID, location=REGION)

embedding_model = TextEmbeddingModel.from_pretrained("text-embedding-005")
index = aiplatform.MatchingEngineIndex(os.getenv("VERTEX_INDEX_ID"))


def embed_text(text: str) -> list:
    return embedding_model.get_embeddings([text])[0].values


def chunk_by_section(text: str) -> list:
    """
    Split markdown by ## headings so each chunk = one brand book section.
    This gives much better retrieval precision than paragraph chunking.
    """
    sections = re.split(r'\n(?=## )', text)
    chunks = []
    for section in sections:
        section = section.strip()
        if section:
            # Extract heading as title
            lines = section.split('\n')
            title = lines[0].replace('#', '').strip()
            chunks.append({"title": title, "content": section})
    return chunks


def seed_brand_book(filepath: str):
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    chunks = chunk_by_section(content)
    print(f"Found {len(chunks)} sections\n")

    vectors_to_upsert = []

    for i, chunk in enumerate(chunks):
        chunk_id = f"brand_{BRAND_ID}_{VERSION}_chunk_{i:03d}"

        # Store in Firestore
        db.collection("brand_book").document(chunk_id).set({
            "chunk_id": chunk_id,
            "brand_id": BRAND_ID,
            "version": VERSION,
            "chunk_index": i,
            "total_chunks": len(chunks),
            "title": chunk["title"],
            "content": chunk["content"]
        })

        # Embed section title + content for better semantic match
        vector = embed_text(f"{chunk['title']}\n\n{chunk['content']}")
        vectors_to_upsert.append({
            "id": chunk_id,
            "embedding": vector
        })

        print(f"✅ Section {i+1}/{len(chunks)}: {chunk['title']}")

    # Upsert all vectors
    print(f"\nUpserting {len(vectors_to_upsert)} vectors to Vertex AI...")
    index.upsert_datapoints(
        datapoints=[
            {"datapoint_id": v["id"], "feature_vector": v["embedding"]}
            for v in vectors_to_upsert
        ]
    )

    print(f"\n✅ Brand book seeding complete. {len(vectors_to_upsert)} sections indexed.")


if __name__ == "__main__":
    seed_brand_book(BRAND_BOOK_PATH)
