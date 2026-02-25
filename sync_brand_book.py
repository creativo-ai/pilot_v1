"""
Brand Book Sync Scheduler
Run this every 5 minutes via cron or Cloud Scheduler.

Cron example:
*/5 * * * * /path/to/.venv/bin/python /path/to/sync_brand_book.py

Cloud Scheduler: point to this script or wrap it in a Cloud Function.
"""

import os
from dotenv import load_dotenv
from data_layer_vertexAI import sync_brand_book_to_vertex, db
from google.cloud.firestore_v1.base_query import FieldFilter


load_dotenv()
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "creativo-bf5c8-fc5f772a16e4.json"

def run_sync():
    # Find all brands that have pending unsynced updates
    pending = (
        db.collection("brand_updates")
        .where(filter=FieldFilter("synced", "==", False))
        .stream()
    )

    brand_ids = set()
    for doc in pending:
        brand_id = doc.to_dict().get("brand_id")
        if brand_id:
            brand_ids.add(brand_id)

    if not brand_ids:
        print("No pending brand updates to sync.")
        return

    for brand_id in brand_ids:
        print(f"Syncing brand: {brand_id}")
        result = sync_brand_book_to_vertex(brand_id)
        if result:
            print(f"✅ {brand_id} synced successfully.")
        else:
            print(f"⚠️ {brand_id} sync skipped or failed.")


if __name__ == "__main__":
    run_sync()