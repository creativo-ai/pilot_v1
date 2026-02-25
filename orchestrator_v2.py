from llm_client import get_claude_client
from data_layer_vertexAI import get_brand_info
from agents import (
    write_email,
    talk_agent,
    list_pending_images,
    approve_images,
    reject_images,
    search_docs_agent
)
import json

TOOLS = [
    {
        "name": "write_email",
        "description": "Write a brand-aligned email to a specific audience.",
        "input_schema": {
            "type": "object",
            "properties": {
                "email_goal": {"type": "string", "description": "The purpose of the email"},
                "audience": {"type": "string", "description": "Target audience of the email"}
            },
            "required": ["email_goal"]
        }
    },
    {
        "name": "list_pending_images",
        "description": "List images that are pending approval for the user.",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": []
        }
    },
    {
        "name": "approve_images",
        "description": "Approve one or more images by ID.",
        "input_schema": {
            "type": "object",
            "properties": {
                "image_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of image IDs to approve"
                }
            },
            "required": ["image_ids"]
        }
    },
    {
        "name": "reject_images",
        "description": "Reject one or more images by ID.",
        "input_schema": {
            "type": "object",
            "properties": {
                "image_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of image IDs to reject"
                }
            },
            "required": ["image_ids"]
        }
    },
    {
        "name": "search_docs",
        "description": "Search documentation to answer questions about how to use the platform, sign up, sign in, subscription plans, or any feature-related how-to questions.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The user's question to search in the documentation"}
            },
            "required": ["query"]
        }
    }
]

AVAILABLE_AGENTS = {
    "write_email": write_email,
    "list_pending_images": list_pending_images,
    "approve_images": approve_images,
    "reject_images": reject_images,
    "search_docs": search_docs_agent
}


def build_system_prompt(brand_info: dict) -> str:
    return f"""
        You are a helpful AI assistant for a marketing agency.

        You already know the following brand information — answer brand-related questions directly from this data without using any tool:

            {json.dumps(brand_info, indent=2)}

        """


def orchestrate(
    user_input: str,
    conversation_history: list,
    user_id: str,
    brand_id: str,
    **kwargs
):
    client = get_claude_client()

    # Fetch brand info and inject into system prompt
    brand_info = get_brand_info(brand_id) or {}
    system_prompt = build_system_prompt(brand_info)

    messages = []
    for msg in conversation_history:
        messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": user_input})

    response = client.messages.create(
        model="claude-opus-4-6",
        system=system_prompt,
        messages=messages,
        tools=TOOLS,
        max_tokens=1000
    )

    print("\nStop reason:", response.stop_reason)

    if response.stop_reason == "tool_use":
        for block in response.content:
            if block.type == "tool_use":
                tool_name = block.name
                tool_args = block.input

                print(f"\nTool called: {tool_name}")
                print(f"Tool args: {tool_args}")

                agent_fn = AVAILABLE_AGENTS.get(tool_name)
                if agent_fn:
                    return agent_fn(
                        user_input=user_input,
                        conversation_history=conversation_history,
                        user_id=user_id,
                        brand_id=brand_id,
                        client=client,
                        **tool_args
                    )

    # Claude responded directly
    for block in response.content:
        if getattr(block, "type", None) == "text":
            return block.text.strip()

    return talk_agent(
        user_input=user_input,
        conversation_history=conversation_history,
        user_id=user_id,
        brand_id=brand_id,
        client=client
    )