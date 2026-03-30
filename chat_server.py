"""
chat_server.py
Flask server that exposes the brand onboarding agent as an HTTP API
for the chat_ui.html frontend.

Run with:  python chat_server.py
Serves on: http://localhost:8001
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from main import _flush_on_exit
import os

load_dotenv()
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "creativo-bf5c8-fc5f772a16e4.json"
app = Flask(__name__)
CORS(app)

# 1. Add these global variables to track the active session
active_user_id = None
active_brand_id = None

# 2. Define the dynamic flush function
import atexit

def server_flush_on_exit():
    print("\n🛑 Server shutting down... looking for active brand books to save.")
    try:
        from brand_finalizer import _finalize_thread
        import thread_manager
        
        # Fetch ALL active onboarding threads by passing 0 for inactivity minutes
        active_threads = thread_manager.get_stale_onboarding_threads(inactivity_minutes=0)
        
        if not active_threads:
            print("No active brand books needed saving.")
            return

        for thread in active_threads:
            u_id = thread.get("user_id")
            b_id = thread.get("brand_id")
            
            if not b_id:               
                print(f"[Flush] Skipping thread for {u_id} — no brand_id")
                continue
            # Only save if we actually collected fields or had a conversation
            if thread.get("collected_fields") or thread.get("messages"):
                print(f"Flushing thread for User: {u_id}, Brand: {b_id}...")
                _finalize_thread(thread, u_id, b_id)
                thread_manager.mark_finalized(u_id, "brand_onboarding")
                print(f"✅ Brand book successfully saved for {b_id}!")
                
    except Exception as e:
        print(f"⚠️ Flush error: {e}")

atexit.register(server_flush_on_exit)



@app.route("/chat", methods=["POST"])
def chat():
    global active_user_id, active_brand_id
    data = request.json or {}

    user_id      = data.get("user_id", "marmar")
    brand_id     = data.get("brand_id", "")   # no fallback — must be explicit
    message      = data.get("message", "").strip()
    history      = data.get("history", [])
    file_content = data.get("file_content")
    file_name    = data.get("file_name")
    file_bytes_b64 = data.get("file_bytes")
    file_media_type = data.get("file_media_type", "application/pdf")
    visual_identity = data.get("visual_identity")  # from UI panel

    if not brand_id:
        return jsonify({"error": "brand_id is required"}), 400

    # Update the trackers
    active_user_id = user_id
    active_brand_id = brand_id

    if not message and not file_content and not file_bytes_b64 and not visual_identity:
        return jsonify({"error": "No message or file provided"}), 400

    try:
        from brand_onboarding_agent import BrandOnboardingAgent, upload_file_to_claude, delete_file_from_claude
        from data_layer_vertexAI import update_brand_info
        import thread_manager

        # ── Visual identity update (from UI panel — no chat turn needed) ─────
        if message == "__visual_identity_update__" and visual_identity:
            update_brand_info(brand_id, {
                k: v for k, v in visual_identity.items() if v not in (None, "", [], {})
            })
            return jsonify({"response": "visual_identity_saved", "completion_pct": 0})

        agent = BrandOnboardingAgent()
        thread = thread_manager.load_thread(user_id, "brand_onboarding")

        if thread.get("brand_id") and thread.get("brand_id") != brand_id:
            print(f"[Session] Brand changed from {thread['brand_id']} to {brand_id}. Clearing stale thread.")
            thread["messages"] = []
            thread["collected_fields"] = {}
            thread["summary"] = ""
            
        # Always update brand_id to what the current request says
        thread["brand_id"] = brand_id

        collected_before = dict(thread.get("collected_fields", {}))

        file_id = None
        if file_bytes_b64 and file_name:
            import base64
            file_bytes = base64.b64decode(file_bytes_b64)
            file_id = upload_file_to_claude(file_bytes, file_name, file_media_type)

        response = agent.run(
            user_input=message or f"I uploaded a file: {file_name}",
            history=history,
            user_id=user_id,
            brand_id=brand_id,
            thread=thread,
            file_id=file_id,
            file_media_type=file_media_type if file_id else None,
            file_content=file_content,
            file_name=file_name,
        )

        if file_id:
            delete_file_from_claude(file_id)

        thread_manager.append_messages(thread, message or f"[file: {file_name}]", response)
        thread_manager.save_thread(user_id, "brand_onboarding", thread, brand_id)

        collected_after = thread.get("collected_fields", {})
        completion_pct = agent._completion_percentage(collected_after)
        brand_name = collected_after.get("brand_name") or brand_id

        from brand_onboarding_agent import ONBOARDING_SECTIONS
        current_key, _, _ = agent._get_current_section(collected_after)
        section_keys = [k for k, _, _ in ONBOARDING_SECTIONS]
        active_idx = section_keys.index(current_key) if current_key in section_keys else len(section_keys)
        done_indices = [
            i for i, (k, _, _) in enumerate(ONBOARDING_SECTIONS)
            if agent._is_section_complete(k, collected_after)
        ]

        extracted_fields = [k for k in collected_after if k not in collected_before]

        return jsonify({
            "response": response,
            "completion_pct": completion_pct,
            "brand_name": brand_name,
            "active_section": active_idx,
            "done_sections": done_indices,
            "extracted_fields": extracted_fields,
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    print("🚀 Chat server starting on http://localhost:8001")
    app.run(port=8001, debug=False)
