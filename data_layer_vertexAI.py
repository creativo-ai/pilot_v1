import os
from dotenv import load_dotenv
from google.cloud import firestore, aiplatform
from google.cloud.firestore_v1.base_query import FieldFilter
import vertexai
from vertexai.language_models import TextEmbeddingModel

load_dotenv()

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "creativo-bf5c8-fc5f772a16e4.json"
os.environ["GOOGLE_CLOUD_AIPLATFORM_MATCHING_ENGINE_USE_GRPC"] = "true"

PROJECT_ID = "creativo-bf5c8"
REGION = "us-central1"
INDEX_ENDPOINT_ID = os.getenv("VERTEX_INDEX_ENDPOINT_ID")
DEPLOYED_INDEX_ID = os.getenv("VERTEX_DEPLOYED_INDEX_ID")
VECTOR_DIMENSION = 768

# ---- Init clients ----
db = firestore.Client(project=PROJECT_ID, database="agent-orchestration")

vertexai.init(project=PROJECT_ID, location=REGION)
aiplatform.init(project=PROJECT_ID, location=REGION)

embedding_model = TextEmbeddingModel.from_pretrained("text-embedding-005")
endpoint = aiplatform.MatchingEngineIndexEndpoint(
    index_endpoint_name=INDEX_ENDPOINT_ID
)


# -------------------------------------------------
# BRAND INFO
# -------------------------------------------------

def get_brand_info(brand_id: str):
    doc_ref = db.collection("brands").document(brand_id)
    doc = doc_ref.get()
    return doc.to_dict() if doc.exists else {}


def update_brand_info(brand_id: str, updates: dict):
    db.collection("brands").document(brand_id).update(updates)


# -------------------------------------------------
# IMAGE HISTORY
# -------------------------------------------------

def get_media_history(user_id: str, brand_id: str):
    images = (
        db.collection("images")
        .where(filter=FieldFilter("user_id", "==", user_id))
        .stream()
    )
    return [img.to_dict() for img in images]


def update_media_status(image_id: str, new_status: str) -> bool:
    try:
        doc_ref = db.collection("images").document(image_id)
        doc = doc_ref.get()
        if not doc.exists:
            return False
        doc_ref.update({"status": new_status})
        return True
    except Exception as e:
        print(f"[update_media_status] Failed for {image_id}: {e}")
        return False


# -------------------------------------------------
# VERTEX AI VECTOR SEARCH
# -------------------------------------------------

def embed_text(text: str) -> list:
    if not text or not text.strip():
        return [0.0] * VECTOR_DIMENSION
    embeddings = embedding_model.get_embeddings([text.strip()])
    return embeddings[0].values


def search_documentation(query: str, top_k: int = 3) -> list:
    if not query or not query.strip():
        return []

    query_embedding = embed_text(query.strip())

    # Check if embedding is zero vector (empty input fallback)
    if all(v == 0.0 for v in query_embedding[:5]):
        return []

    response = endpoint.find_neighbors(
        deployed_index_id=DEPLOYED_INDEX_ID,
        queries=[query_embedding],
        num_neighbors=top_k * 2,  # fetch more to filter out brand book chunks
        return_full_datapoint=False
    )

    if not response or not response[0]:
        return []

    results = []
    for match in response[0]:
        # Skip brand book chunks
        if match.id.startswith("brand_"):
            continue
        doc = db.collection("docs").document(match.id).get()
        if doc.exists:
            data = doc.to_dict()
            # Normalize field names — support both 'content' and 'text'
            if "text" in data and "content" not in data:
                data["content"] = data["text"]
            if "filepath" in data and "filename" not in data:
                data["filename"] = data["filepath"].split("/")[-1]
            results.append(data)
        if len(results) >= top_k:
            break

    return results


# -------------------------------------------------
# BRAND BOOK SEARCH
# -------------------------------------------------


# -------------------------------------------------
# BRAND BOOK SEARCH (Firestore structured data)
# -------------------------------------------------

def search_brand_book(brand_id: str, query: str = None, top_k: int = 3) -> list:
    """
    Fetch brand info from the structured Firestore brands doc.
    Returns a list of dicts (section_name, content) so agents can use it
    the same way they used vector search results.
    query is accepted for API compatibility but filtering is done structurally.
    """
    doc = db.collection("brands").document(brand_id).get()
    if not doc.exists:
        return []

    brand_data = doc.to_dict()
    results = []

    for section_key, section_value in brand_data.items():
        if not section_value:
            continue
        # Flatten section to readable text
        if isinstance(section_value, dict):
            content = "\n".join(
                f"{k.replace('_', ' ').title()}: {v}"
                for k, v in section_value.items()
                if v not in (None, "", [], {})
            )
        elif isinstance(section_value, list):
            content = ", ".join(str(i) for i in section_value if i)
        else:
            content = str(section_value)

        if content.strip():
            results.append({
                "title": section_key.replace("_", " ").title(),
                "content": content,
                "source": "firestore"
            })

    return results[:top_k] if top_k else results


# -------------------------------------------------
# MEDIA SEARCH
# -------------------------------------------------

def search_media(user_id: str, brand_id: str, status: str = None, media_type: str = None, platform: str = None) -> list:
    """
    Search media (images/videos) with optional filters.
    """
    query = db.collection("images").where(filter=FieldFilter("user_id", "==", user_id))

    if status:
        query = query.where(filter=FieldFilter("status", "==", status.lower()))
    if media_type:
        query = query.where(filter=FieldFilter("type", "==", media_type.lower()))
    if platform:
        query = query.where(filter=FieldFilter("platform", "==", platform))

    return [doc.to_dict() for doc in query.stream()]

# Backward compatibility aliases
get_image_history = get_media_history
update_image_status = update_media_status

# =============================================================================
# DUAL BRAND RETRIEVAL (new)
# =============================================================================

def get_brand_field(brand_id: str, field_name: str):
    """
    Path A: Fetch a specific structured field directly from Firestore.
    Fast and exact. Use when you need tone, mission, colors, etc.
    """
    doc = db.collection("brands").document(brand_id).get()
    return doc.to_dict().get(field_name) if doc.exists else None


def get_brand_context(brand_id: str, query: str, top_k: int = 3) -> list:
    """
    Path B: Semantic search over free-form brand context in Vertex AI.
    Use when you need rich contextual info: positioning, story, nuances.
    Returns list of context chunk dicts.
    """
    try:
        query_embedding = embed_text(query)

        response = endpoint.find_neighbors(
            deployed_index_id=DEPLOYED_INDEX_ID,
            queries=[query_embedding],
            num_neighbors=top_k * 3,
            return_full_datapoint=False
        )

        if not response or not response[0]:
            return []

        results = []
        for match in response[0]:
            # Only brand context chunks (prefix: ctx_)
            if not match.id.startswith(f"ctx_{brand_id}"):
                continue
            doc = db.collection("brand_context").document(match.id).get()
            if doc.exists:
                results.append(doc.to_dict())
            if len(results) >= top_k:
                break

        return results

    except Exception as e:
        print(f"⚠️ get_brand_context failed: {e}")
        return []


def get_latest_brand_context(brand_id: str) -> str:
    """
    Returns the most recent free-form brand context summary as plain text.
    Useful for agents that need the full context without a specific query.
    """
    docs = list(
        db.collection("brand_context")
        .where("brand_id", "==", brand_id)
        .stream()
    )
    if not docs:
        return ""

    # Sort by version descending, return latest content
    def version_num(d):
        v = d.to_dict().get("version", "v1")
        try:
            return int(v.replace("v", ""))
        except Exception:
            return 1

    latest = max(docs, key=version_num)
    return latest.to_dict().get("content", "")