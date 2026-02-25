from pinecone import Pinecone
import os

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = "pilot-main-index"
VECTOR_DIMENSION = 1536  

pc = Pinecone(api_key=PINECONE_API_KEY)


def get_index():
    # print(pc.describe_index(INDEX_NAME))
    return pc.Index(INDEX_NAME)


# -------------------------------------------------
# BRAND INFO
# -------------------------------------------------

def get_brand_info(brand_id: str):
    """
    Retrieve brand info using metadata filtering.
    """
    index = get_index()

    dummy_vector = [0.0] * VECTOR_DIMENSION

    res = index.query(
        vector=dummy_vector,
        filter={
            "type": "brand_info",
            "brand_id": brand_id
        },
        top_k=1,
        include_metadata=True
    )

    if res.matches:
        return res.matches[0].metadata

    return None


# -------------------------------------------------
# IMAGE HISTORY
# -------------------------------------------------

def get_image_history(user_id: str , brand_id: str):
    """
    Retrieve all image vectors for a user.
    """
    index = get_index()

    dummy_vector = [0.0] * VECTOR_DIMENSION

    res = index.query(
        vector=dummy_vector,
        filter={
            "type": "image",
            "user_id": user_id
        },
        top_k=100,
        include_metadata=True
    )

    return res.matches if res.matches else []


def update_image_status(image_id: str, new_status: str):
    """
    Update ONLY metadata while keeping vector intact.
    """
    index = get_index()

    # Fetch the existing vector
    existing = index.fetch(ids=[image_id])

    if image_id not in existing.vectors:
        return False

    old_vector = existing.vectors[image_id].values
    old_metadata = existing.vectors[image_id].metadata or {}

    # Safely update the status in metadata
    old_metadata["status"] = new_status

    # Re-upsert with same vector + updated metadata
    index.upsert(
        vectors=[
            {
                "id": image_id,
                "values": old_vector,
                "metadata": old_metadata
            }
        ]
    )

    return True