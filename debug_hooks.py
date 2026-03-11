"""
debug_hooks.py
Import this in any file to push debug events to the dashboard.
If the debug server is not running, all calls are silent no-ops.
"""


def _push(event_type: str, data: dict):
    try:
        from debug_server import push_event
        push_event(event_type, data)
    except Exception:
        pass  # silent if debug server not started


def log_user_input(text: str):
    _push("user_input", {"text": text})


def log_routing_decision(user_input: str, gemini_choice: str):
    _push("routing_decision", {
        "user_input": user_input,
        "gemini_choice": gemini_choice
    })


def log_agent_selected(agent_name: str, routing_context: str = None, gemini_choice: str = None):
    _push("agent_selected", {
        "agent": agent_name,
        "routing_context": routing_context,
        "gemini_choice": gemini_choice
    })


def log_agent_response(agent_name: str, text: str):
    _push("agent_response", {"agent": agent_name, "text": text})


def log_brand_fields(fields: dict):
    _push("brand_fields", fields)


def log_onboarding_state(mode: str, section: str, missing: str, collected: dict):
    _push("onboarding_state", {
        "mode": mode,
        "section": section,
        "missing": missing,
        "collected": collected
    })


def log_passive_capture(question: str, answer: str, brand_id: str):
    _push("passive_capture", {
        "question": question,
        "answer": answer,
        "brand_id": brand_id
    })


def log_finalizer(message: str):
    _push("finalizer", {"message": message})


def log_error(message: str, context: str = None):
    _push("error", {"message": message, "context": context})
