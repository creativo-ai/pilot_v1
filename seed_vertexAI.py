import os
import re
import uuid
import glob
from google.cloud import firestore
import vertexai
from vertexai.language_models import TextEmbeddingModel
from google.cloud import aiplatform
from dotenv import load_dotenv


load_dotenv()
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "creativo-bf5c8-fc5f772a16e4.json"

PROJECT_ID = "creativo-bf5c8"
REGION = "us-central1"
INDEX_ENDPOINT_ID = os.getenv("VERTEX_INDEX_ENDPOINT_ID")
DEPLOYED_INDEX_ID = os.getenv("VERTEX_DEPLOYED_INDEX_ID")
DOCS_DIR = "/Users/manar.ayman/Developer/Documentation/docs/Creativo-Help/Brand Management/1.0 General Info/"

db = firestore.Client(project=PROJECT_ID, database="agent-orchestration")
vertexai.init(project=PROJECT_ID, location=REGION)
aiplatform.init(project=PROJECT_ID, location=REGION)

embedding_model = TextEmbeddingModel.from_pretrained("text-embedding-005")
endpoint = aiplatform.MatchingEngineIndexEndpoint(index_endpoint_name=INDEX_ENDPOINT_ID)


def get_deployed_index() -> aiplatform.MatchingEngineIndex:
    deployed = endpoint.deployed_indexes
    if not deployed:
        raise RuntimeError("No deployed indexes found on the endpoint.")
    index_resource_name = deployed[0].index
    print(f"Resolved deployed index: {index_resource_name}")
    return aiplatform.MatchingEngineIndex(index_resource_name)


def clean_double_underscores(text: str) -> str:
    """Convert __url__ syntax to standard markdown URLs."""
    # ![alt](__url__) → ![alt](url)
    text = re.sub(r'!\[([^\]]*)\]\(__([^_].*?)__\)', r'![\1](\2)', text)
    # [text](__url__) → [text](url)
    text = re.sub(r'\[([^\]]*)\]\(__([^_].*?)__\)', r'[\1](\2)', text)
    # bare __url__ → url
    text = re.sub(r'__([^_].*?)__', r'\1', text)
    return text


def extract_media_urls(text: str) -> list:
    """Extract all image and video URLs from markdown content (after cleaning)."""
    urls = []
    # Markdown images: ![alt](url)
    urls += re.findall(r'!\[[^\]]*\]\((https?://[^)\s]+)\)', text)
    # YouTube embed URLs
    urls += re.findall(r'(https?://(?:www\.)?youtube\.com/embed/[^\s"\')\]]+)', text)
    # Bare media URLs (images/videos by extension)
    urls += re.findall(r'(https?://\S+\.(?:jpg|jpeg|png|gif|webp|mp4|mov|webm))', text)
    # Deduplicate preserving order
    seen = set()
    return [u for u in urls if not (u in seen or seen.add(u))]


def embed_text(text: str) -> list:
    embeddings = embedding_model.get_embeddings([text])
    return embeddings[0].values


def chunk_text(text: str, max_chars: int = 1500) -> list:
    paragraphs = text.split("\n\n")
    chunks = []
    current = ""
    for para in paragraphs:
        if len(current) + len(para) < max_chars:
            current += para + "\n\n"
        else:
            if current.strip():
                chunks.append(current.strip())
            current = para + "\n\n"
    if current.strip():
        chunks.append(current.strip())
    return chunks


def get_relative_path(filepath: str, base_dir: str) -> str:
    return os.path.relpath(filepath, base_dir)


def clear_existing_docs():
    print("Clearing existing doc chunks from Firestore...")
    docs = list(db.collection("docs").stream())
    for doc in docs:
        doc.reference.delete()
    print(f"  Deleted {len(docs)} old chunk(s)\n")


def seed_docs(docs_dir: str):
    md_files = glob.glob(os.path.join(docs_dir, "**/*.md"), recursive=True)

    if not md_files:
        print(f"No markdown files found in {docs_dir}")
        return

    clear_existing_docs()
    print(f"Found {len(md_files)} markdown file(s)\n")

    index = get_deployed_index()
    vectors_to_upsert = []

    for filepath in md_files:
        relative_path = get_relative_path(filepath, docs_dir)
        filename = os.path.basename(filepath)
        print(f"Processing: {relative_path}")

        with open(filepath, "r", encoding="utf-8") as f:
            raw_content = f.read()

        # Clean __url__ syntax before chunking
        clean_content = clean_double_underscores(raw_content)
        chunks = chunk_text(clean_content)
        print(f"  → {len(chunks)} chunk(s)")

        for i, chunk in enumerate(chunks):
            chunk_id = f"doc_{uuid.uuid4().hex}"
            media_urls = extract_media_urls(chunk)

            db.collection("docs").document(chunk_id).set({
                "chunk_id": chunk_id,
                "filename": filename,
                "filepath": relative_path,
                "chunk_index": i,
                "total_chunks": len(chunks),
                "content": chunk,
                "media_urls": media_urls,  # videos & images stored explicitly
                "brand_id": "creativo"
            })

            vector = embed_text(chunk)
            vectors_to_upsert.append({"id": chunk_id, "embedding": vector})

            media_note = f" | {len(media_urls)} media URL(s)" if media_urls else ""
            print(f"    ✅ Chunk {i+1}/{len(chunks)} stored (ID: {chunk_id}){media_note}")

    print(f"\nUpserting {len(vectors_to_upsert)} vectors to Vertex AI...")
    index.upsert_datapoints(
        datapoints=[
            {"datapoint_id": v["id"], "feature_vector": v["embedding"]}
            for v in vectors_to_upsert
        ]
    )
    print(f"\n✅ Seeding complete. {len(vectors_to_upsert)} chunks indexed.")


if __name__ == "__main__":
    seed_docs(DOCS_DIR)