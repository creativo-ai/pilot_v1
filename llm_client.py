from anthropic import Anthropic
import os
import json


def get_claude_client():
    return Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))


def classify_intent(client, user_input: str, conversation_history: list, available_actions: dict, user_id: str = "", brand_id: str = ""):

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

        Use these values to fill in arguments whenever user_id or brand_id are required.

        Your ONLY job:
        - Select exactly ONE action from the available actions below.
        - Extract required arguments. Use the session context above for user_id and brand_id.
        - NEVER default to "talk" if another action clearly matches the user's intent.
        - NEVER ask questions.
        - NEVER explain.
        - NEVER respond conversationally.

        Available actions:
        {actions_text}

        You MUST return ONLY valid JSON. No markdown. No text. No explanation.

        Format exactly:
        {{
        "action": "<action_name>",
        "arguments": {{ ... }}
        }}
        """

    messages = []
    for msg in conversation_history:
        messages.append({"role": msg["role"], "content": msg["content"]})
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