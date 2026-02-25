from anthropic import Anthropic
from google import genai
from google.genai import types
import os
import json


# ── Claude ────────────────────────────────────────────────────────────────────

def get_claude_client():
    return Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))


# ── Gemini ────────────────────────────────────────────────────────────────────

def get_gemini_client() -> genai.Client:
    return genai.Client(api_key=os.getenv("GEMINI_API_KEY"))


def gemini_generate(prompt: str, system: str = None) -> str:
    """
    Single-turn Gemini 2.5 Flash call.
    Accepts the same (prompt, system) pattern as Claude calls.
    Always returns a plain string — no content blocks, no formatting surprises.
    Strips markdown code fences Gemini sometimes adds around JSON.
    """
    client = get_gemini_client()

    config = types.GenerateContentConfig(
        system_instruction=system,
        temperature=0.3,
    ) if system else types.GenerateContentConfig(temperature=0.3)

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config=config,
    )

    text = (response.text or "").strip()

    # Strip markdown code fences Gemini sometimes wraps JSON in
    if text.startswith("```"):
        parts = text.split("```")
        text = parts[1] if len(parts) > 1 else text
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()

    return text


# ── Legacy classify_intent (used by actions_registry path) ───────────────────

def classify_intent(client, user_input: str, conversation_history: list,
                    available_actions: dict, user_id: str = "", brand_id: str = ""):

    actions_text = ""
    for name, meta in available_actions.items():
        actions_text += f"""
            Action: {name}
            Description: {meta.get("description")}
            Arguments:
            {json.dumps(meta.get("arguments", {}), indent=2)}
            """

    system_prompt = f"""
        You are a strict AI router.

        Current session context:
        - user_id: {user_id}
        - brand_id: {brand_id}

        Your ONLY job:
        - Select exactly ONE action from the available actions below.
        - Extract required arguments.
        - NEVER default to "talk" if another action clearly matches the user's intent.
        - NEVER ask questions. NEVER explain. NEVER respond conversationally.

        Available actions:
        {actions_text}

        You MUST return ONLY valid JSON. No markdown. No text. No explanation.

        Format exactly:
        {{
        "action": "<action_name>",
        "arguments": {{ ... }}
        }}
        """

    messages = [{"role": m["role"], "content": m["content"]} for m in conversation_history]
    messages.append({"role": "user", "content": user_input})

    response = client.messages.create(
        model="claude-opus-4-6",
        system=system_prompt,
        messages=messages,
        max_tokens=400
    )

    raw_text = response.content[0].text.strip()
    print("\nRaw Text from the classifier: ", raw_text)

    try:
        cleaned = raw_text.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("```")[1]
            if cleaned.startswith("json"):
                cleaned = cleaned[4:]
            cleaned = cleaned.strip()
        return json.loads(cleaned)
    except Exception:
        return {"action": "talk", "arguments": {}}