import os
from pinecone import Pinecone
import random

# ---- Connect to Pinecone ----
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("pilot-main-index")

# ---- Helper for fake embeddings ----
def fake_embedding(dim=1536):
    return [random.random() for _ in range(dim)]

# -----------------------------
# BRAND INFO
# -----------------------------
brand_metadata = {
    "type": "brand_info",
    "brand_id": "creativo",
    "section": "profile",

    "brand_name": "Creativo",
    "industry": "Marketing Agency",
    "target_audience": "Small and medium-sized businesses",
    "tone": "Professional, creative, and friendly",
    "brand_voice": "Strategic yet approachable",
    "communication_style": "Clear, persuasive, benefit-driven",

    "mission": "Empower brands to tell their story creatively and effectively.",
    "vision": "To be the leading creative agency recognized for impactful branding campaigns.",
    "values": "Creativity, Collaboration, Excellence, Customer Focus",

    "default_email_signature": "Best regards,\nCreativo Team",
    "primary_goal": "Deliver high-impact branding campaigns",
    "updated_at": "2026-02-13"
}

index.upsert(
    vectors=[
        {
            "id": "brand_creativo_profile",
            "values": fake_embedding(),
            "metadata": brand_metadata
        }
    ]
)

# -----------------------------
# IMAGE RECORDS
# -----------------------------
images = [
    {
        "id": "image_001",
        "user_id": "manar",
        "status": "pending",
        "description": "Instagram winter campaign banner",
        "campaign_name": "Winter Instagram Campaign",
        "platform": "Instagram",
        "file_url": "https://cdn.creativo.com/winter-banner.jpg",
        "priority": "high",
        "assigned_to": "manar",
        "created_at": "2025-01-10",
        "last_updated": "2025-01-11"
    },
    {
        "id": "image_002",
        "user_id": "manar",
        "status": "approved",
        "description": "Facebook spring campaign banner",
        "campaign_name": "Spring Facebook Campaign",
        "platform": "Facebook",
        "file_url": "https://cdn.creativo.com/spring-banner.jpg",
        "priority": "medium",
        "assigned_to": "manar",
        "created_at": "2025-02-01",
        "last_updated": "2025-02-05"
    },
    {
        "id": "image_003",
        "user_id": "ahmed",
        "status": "pending",
        "description": "Website homepage hero image",
        "campaign_name": "Website Redesign",
        "platform": "Website",
        "file_url": "https://cdn.creativo.com/homepage-hero.jpg",
        "priority": "high",
        "assigned_to": "ahmed",
        "created_at": "2025-03-10",
        "last_updated": "2025-03-12"
    }
]

# Insert images into Pinecone
for img in images:
    metadata = {
        "type": "image",
        "brand_id": "creativo",
        **img
    }
    index.upsert(
        vectors=[
            {
                "id": img["id"],
                "values": fake_embedding(),
                "metadata": metadata
            }
        ]
    )

print("Seed data for brand info and images inserted successfully.")