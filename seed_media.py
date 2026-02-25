import os
from dotenv import load_dotenv
from google.cloud import firestore

load_dotenv()
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "creativo-bf5c8-fc5f772a16e4.json"

db = firestore.Client(project="creativo-bf5c8", database="agent-orchestration")

media = [
    # ---- IMAGES ----
    {
        "id": "image_001",
        "user_id": "manar",
        "brand_id": "creativo",
        "type": "image",
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
        "type": "image",
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
        "user_id": "manar",
        "brand_id": "creativo",
        "type": "image",
        "status": "pending",
        "description": "LinkedIn product launch graphic",
        "campaign_name": "Product Launch Q2",
        "platform": "LinkedIn",
        "file_url": "https://cdn.creativo.com/product-launch.jpg",
        "priority": "high",
        "assigned_to": "manar",
        "created_at": "2025-03-05",
        "last_updated": "2025-03-06"
    },
    {
        "id": "image_004",
        "user_id": "manar",
        "brand_id": "creativo",
        "type": "image",
        "status": "rejected",
        "description": "Twitter/X promotional banner",
        "campaign_name": "Summer Promo",
        "platform": "Twitter",
        "file_url": "https://cdn.creativo.com/summer-promo.jpg",
        "priority": "low",
        "assigned_to": "manar",
        "created_at": "2025-03-15",
        "last_updated": "2025-03-16"
    },
    {
        "id": "image_005",
        "user_id": "manar",
        "brand_id": "creativo",
        "type": "image",
        "status": "pending",
        "description": "TikTok campaign thumbnail",
        "campaign_name": "TikTok Awareness Campaign",
        "platform": "TikTok",
        "file_url": "https://cdn.creativo.com/tiktok-thumb.jpg",
        "priority": "medium",
        "assigned_to": "manar",
        "created_at": "2025-04-01",
        "last_updated": "2025-04-01"
    },
    {
        "id": "image_006",
        "user_id": "manar",
        "brand_id": "creativo",
        "type": "image",
        "status": "approved",
        "description": "Website hero banner redesign",
        "campaign_name": "Website Redesign",
        "platform": "Website",
        "file_url": "https://cdn.creativo.com/hero-banner.jpg",
        "priority": "high",
        "assigned_to": "manar",
        "created_at": "2025-04-10",
        "last_updated": "2025-04-12"
    },
    # ---- VIDEOS ----
    {
        "id": "video_001",
        "user_id": "manar",
        "brand_id": "creativo",
        "type": "video",
        "status": "pending",
        "description": "Brand intro reel for Instagram",
        "campaign_name": "Brand Awareness Q1",
        "platform": "Instagram",
        "file_url": "https://cdn.creativo.com/brand-intro-reel.mp4",
        "duration_seconds": 30,
        "priority": "high",
        "assigned_to": "manar",
        "created_at": "2025-01-20",
        "last_updated": "2025-01-21"
    },
    {
        "id": "video_002",
        "user_id": "manar",
        "brand_id": "creativo",
        "type": "video",
        "status": "approved",
        "description": "Product demo video for YouTube",
        "campaign_name": "Product Launch Q2",
        "platform": "YouTube",
        "file_url": "https://cdn.creativo.com/product-demo.mp4",
        "duration_seconds": 120,
        "priority": "high",
        "assigned_to": "manar",
        "created_at": "2025-02-15",
        "last_updated": "2025-02-18"
    },
    {
        "id": "video_003",
        "user_id": "manar",
        "brand_id": "creativo",
        "type": "video",
        "status": "pending",
        "description": "TikTok behind-the-scenes clip",
        "campaign_name": "TikTok Awareness Campaign",
        "platform": "TikTok",
        "file_url": "https://cdn.creativo.com/bts-clip.mp4",
        "duration_seconds": 60,
        "priority": "medium",
        "assigned_to": "manar",
        "created_at": "2025-03-22",
        "last_updated": "2025-03-22"
    },
    {
        "id": "video_004",
        "user_id": "manar",
        "brand_id": "creativo",
        "type": "video",
        "status": "rejected",
        "description": "Facebook testimonial video",
        "campaign_name": "Customer Stories",
        "platform": "Facebook",
        "file_url": "https://cdn.creativo.com/testimonial.mp4",
        "duration_seconds": 90,
        "priority": "medium",
        "assigned_to": "manar",
        "created_at": "2025-04-05",
        "last_updated": "2025-04-06"
    },
    {
        "id": "video_005",
        "user_id": "manar",
        "brand_id": "creativo",
        "type": "video",
        "status": "pending",
        "description": "LinkedIn thought leadership video",
        "campaign_name": "Brand Awareness Q2",
        "platform": "LinkedIn",
        "file_url": "https://cdn.creativo.com/thought-leadership.mp4",
        "duration_seconds": 180,
        "priority": "low",
        "assigned_to": "manar",
        "created_at": "2025-04-18",
        "last_updated": "2025-04-18"
    },
    # ---- AHMED's MEDIA ----
    {
        "id": "image_007",
        "user_id": "ahmed",
        "brand_id": "creativo",
        "type": "image",
        "status": "pending",
        "description": "Website homepage hero image",
        "campaign_name": "Website Redesign",
        "platform": "Website",
        "file_url": "https://cdn.creativo.com/homepage-hero.jpg",
        "priority": "high",
        "assigned_to": "ahmed",
        "created_at": "2025-03-10",
        "last_updated": "2025-03-12"
    },
    {
        "id": "video_006",
        "user_id": "ahmed",
        "brand_id": "creativo",
        "type": "video",
        "status": "approved",
        "description": "Instagram story ad",
        "campaign_name": "Summer Promo",
        "platform": "Instagram",
        "file_url": "https://cdn.creativo.com/story-ad.mp4",
        "duration_seconds": 15,
        "priority": "medium",
        "assigned_to": "ahmed",
        "created_at": "2025-04-20",
        "last_updated": "2025-04-21"
    }
]

for item in media:
    db.collection("images").document(item["id"]).set(item)
    print(f"✅ {item['type'].capitalize()} {item['id']} inserted ({item['status']})")

print(f"\nDone. {len(media)} media items seeded.")