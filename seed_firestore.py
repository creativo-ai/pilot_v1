import os
from google.cloud import firestore

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "creativo-bf5c8-fc5f772a16e4.json"

db = firestore.Client(project="creativo-bf5c8" , database= "agent-orchestration")

# -----------------------------
# BRAND INFO
# -----------------------------
brand_data = {
    "brand_id": "creativo",
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

db.collection("brands").document("creativo").set(brand_data)
print("✅ Brand inserted")

# -----------------------------
# IMAGES
# -----------------------------
images = [
    {
        "id": "image_001",
        "user_id": "manar",
        "brand_id": "creativo",
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
        "brand_id": "creativo",
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
        "brand_id": "creativo",
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

for img in images:
    db.collection("images").document(img["id"]).set(img)
    print(f"✅ Image {img['id']} inserted")

print("\nFirestore seed complete.")