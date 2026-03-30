# test_save.py
import os, json
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "creativo-bf5c8-fc5f772a16e4.json"

from google.cloud import firestore
from brand_finalizer import _finalize_thread, FIELD_MAP
import thread_manager

db = firestore.Client(project="creativo-bf5c8", database="agent-orchestration")

# ── Simulate a complete collected_fields dict ─────────────────────────────────
TEST_USER_ID  = "test_user"
TEST_BRAND_ID = "test_brand_save"

mock_thread = {
    "user_id": TEST_USER_ID,
    "brand_id": TEST_BRAND_ID,
    "agent_name": "brand_onboarding",
    "messages": [
        {"role": "user",      "content": "Our brand is Ellena, a bakery."},
        {"role": "assistant", "content": "Love it! Tell me about your mission."},
        {"role": "user",      "content": "We bring people together through baked goods."},
    ],
    "collected_fields": {
        # Metadata
        "brand_name":           "Ellena",
        # Brand Core
        "vision":               "To be the most beloved bakery in every neighborhood",
        "mission":              "Bring people together through beautifully crafted baked goods",
        "emotion":              "Warmth, comfort, belonging",
        "competitor":           ["Magnolia Bakery", "Crumbl Cookies"],
        "domain":               "ellena.com",
        "usp_statement":        "Homemade quality at scale across 15 locations",
        "market_positioning":   "Premium neighborhood bakery",
        "core_values":          ["Quality", "Warmth", "Community"],
        # Brand Voice
        "tone_of_voice":        "Warm, approachable, genuine",
        "words_to_avoid":       ["cheap", "fast food", "industrial"],
        "key_messages":         ["Made with love", "Quality you can taste"],
        "taglines_and_slogans": ["Baked with heart"],
        # Target Audience
        "primary_audience":     "People who value quality and care in their food",
        "demographics_age":     "25-45",
        "demographics_location":"United States",
        "demographics_gender":  "All genders",
        "demographics_interests": ["food", "home baking", "celebrations"],
        "demographics_pain_points_behaviors": "Want homemade quality without making it themselves",
        "demographics_communication_style":   "Warm and conversational",
        # Visual Identity
        "primary_colors":       ["buttery yellow", "chocolate brown"],
        "secondary_colors":     ["cream white", "soft beige"],
        "font_for_headings":    "Playfair Display",
        "font_for_body_text":   "Lato",
    },
    "status": "active",
}

# ── Step 1: verify every collected field has a mapping ───────────────────────
print("=" * 55)
print("STEP 1 — Field map coverage check")
print("=" * 55)
missing_mappings = []
for field in mock_thread["collected_fields"]:
    if field not in FIELD_MAP:
        missing_mappings.append(field)

if missing_mappings:
    print(f"WARNING — these fields have no FIELD_MAP entry: {missing_mappings}")
else:
    print("All fields have a FIELD_MAP entry")

# ── Step 2: run the finalizer ─────────────────────────────────────────────────
print("\n" + "=" * 55)
print("STEP 2 — Running finalizer")
print("=" * 55)
_finalize_thread(mock_thread, TEST_USER_ID, TEST_BRAND_ID)
print("Finalizer completed")

# ── Step 3: read back from Firestore and verify ───────────────────────────────
print("\n" + "=" * 55)
print("STEP 3 — Verifying Firestore write")
print("=" * 55)
doc = db.collection("brands").document(TEST_BRAND_ID).get()

if not doc.exists:
    print("FAIL — document was not written to Firestore")
else:
    saved = doc.to_dict()
    print("Document found. Checking each section:\n")

    sections = ["metadata", "brand_core", "brand_voice", "target_audience", "visual_identity"]
    all_good = True

    for section in sections:
        section_data = saved.get(section, {})
        empty = [k for k, v in section_data.items() if v in (None, "", [], {})] if isinstance(section_data, dict) else []
        filled = [k for k, v in section_data.items() if v not in (None, "", [], {})] if isinstance(section_data, dict) else []
        status = "OK" if filled else "EMPTY"
        print(f"  [{status}] {section}")
        if filled:
            for k in filled:
                print(f"         {k}: {section_data[k]}")
        if empty:
            print(f"         (empty: {', '.join(empty)})")
        if not filled:
            all_good = False

    print("\n" + "=" * 55)
    if all_good:
        print("ALL SECTIONS SAVED CORRECTLY")
    else:
        print("SOME SECTIONS ARE EMPTY — check FIELD_MAP alignment")
    print("=" * 55)

# ── Step 4: clean up test document ───────────────────────────────────────────
db.collection("brands").document(TEST_BRAND_ID).delete()
print(f"\nTest document '{TEST_BRAND_ID}' cleaned up.")





