import json
from llm_client import get_claude_client
from data_layer_vertexAI import get_brand_info, search_brand_book
from agents import (
    write_email,
    talk_agent,
    list_pending_media,
    approve_media,
    reject_media,
    search_docs_agent,
    search_media_agent,
    brand_update_agent,
    write_caption_agent
)

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
        "name": "list_pending_media",
        "description": "List media that are pending approval for the user.",
        "input_schema": {"type": "object", "properties": {}, "required": []}
    },
    {
        "name": "approve_media",
        "description": "Approve one or more media items by their resolved IDs. Pass all resolved IDs — the tool validates status internally and only actions pending items.",
        "input_schema": {
            "type": "object",
            "properties": {
                "media_ids": {"type": "array", "items": {"type": "string"}, "description": "List of resolved media IDs to approve"}
            },
            "required": ["media_ids"]
        }
    },
    {
        "name": "reject_media",
        "description": "Reject one or more media items by their resolved IDs. Pass all resolved IDs — the tool validates status internally and only actions pending items.",
        "input_schema": {
            "type": "object",
            "properties": {
                "media_ids": {"type": "array", "items": {"type": "string"}, "description": "List of resolved media IDs to reject"}
            },
            "required": ["media_ids"]
        }
    },
    {
        "name": "search_media",
        "description": "Search and retrieve all media (images and videos) for the user. Use when the user asks about their media history, current status of media, or wants to filter by status (pending/approved/rejected), type (image/video), or platform.",
        "input_schema": {
            "type": "object",
            "properties": {
                "status": {"type": "string", "description": "Filter by status: pending, approved, or rejected."},
                "media_type": {"type": "string", "description": "Filter by type: must be exactly 'image' or 'video'. Never use 'media'."},
                "platform": {"type": "string", "description": "Filter by platform: Instagram, LinkedIn, TikTok, Facebook, YouTube, Website, Twitter."}
            },
            "required": []
        }
    },
    {
        "name": "write_caption",
        "description": "Write a social media post or caption for a specific platform.",
        "input_schema": {
            "type": "object",
            "properties": {
                "platform": {"type": "string", "description": "Target platform: Instagram, LinkedIn, Facebook, TikTok, Twitter, YouTube"},
                "topic": {"type": "string", "description": "The topic or subject of the post"}
            },
            "required": ["platform", "topic"]
        }
    },
    {
        "name": "update_brand",
        "description": "Update or add brand information such as name, mission, vision, values, tone, voice, target audience, or any other brand identity fields.",
        "input_schema": {
            "type": "object",
            "properties": {
                "update_request": {"type": "string", "description": "The user's brand update request in their own words"}
            },
            "required": ["update_request"]
        }
    },
    {
        "name": "search_docs",
        "description": "Search documentation to answer how-to or platform usage questions.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The user's question"}
            },
            "required": ["query"]
        }
    }
]

AVAILABLE_AGENTS = {
    "write_email": write_email,
    "list_pending_media": list_pending_media,
    "approve_media": approve_media,
    "reject_media": reject_media,
    "search_media": search_media_agent,
    "write_caption": write_caption_agent,
    "update_brand": brand_update_agent,
    "search_docs": search_docs_agent
}


def _nested_get(d: dict, *paths):
    """Try nested path first, then flat key fallback."""
    for path in paths:
        if isinstance(path, str):
            path = [path]
        node = d
        for key in path:
            if isinstance(node, dict):
                node = node.get(key, {})
            else:
                node = {}
                break
        if isinstance(node, str) and node.strip():
            return node.strip()
        if isinstance(node, list) and node:
            return ", ".join(str(v) for v in node if v)
    return ""


def build_system_prompt(brand_info: dict, brand_context_chunks: list) -> str:
    brand_identity = json.dumps({
        "brand_name": _nested_get(brand_info, ["metadata", "brand_name"], ["brand_name"]),
        "tone":       _nested_get(brand_info, ["brand_personality", "tone_of_voice", "primary_tone"], ["tone"]),
        "mission":    _nested_get(brand_info, ["brand_foundation", "mission"], ["mission"]),
        "vision":     _nested_get(brand_info, ["brand_foundation", "vision"], ["vision"]),
        "values":     _nested_get(brand_info, ["brand_foundation", "core_values"], ["values"]),
        "target_audience": _nested_get(brand_info, ["target_audience", "primary_audience", "description"], ["target_audience"]),
    }, indent=2)

    brand_book_context = "\n\n---\n\n".join(
        [f"## {c.get('title', '')}\n{c.get('content', '')}" for c in brand_context_chunks]
    ) if brand_context_chunks else "No additional brand book context retrieved."

    return f"""
        You are an internal media management tool for Creativo, a marketing agency.
        You help Creativo's team members review, approve, reject, and search media assets belonging to their clients.
        The active client brand is shown in the brand identity below — all media belongs to that client.
        Be direct and efficient — you are an internal tool used by agency staff, not a customer-facing assistant.

        ## Brand Identity (always available)
        {brand_identity}

        ## Relevant Brand Book Sections
        {brand_book_context}

        ## Media Rules

        WORKFLOW — always follow this exact order:

        STEP 1 — Call search_media to resolve references.
        Always call search_media first to get the current media list from Firestore.
        Use the results to resolve positional references (first, last, third, etc.).

        STEP 2 — Resolve positions from search_media results.
        Map "first image" → first image in results, "last video" → last video in results.
        When user says "last 2 videos" → find the last 2 videos in the results list.
        Use ONLY the search_media results for this — never conversation history.

        STEP 3 — Call approve_media or reject_media with the resolved IDs.
        Pass ALL resolved IDs directly to the tool — do NOT pre-filter by status.
        The tool functions validate status internally and only act on pending items.
        They return exactly what happened: which items were actioned, which were skipped.

        STEP 4 — Report the tool result to the user.
        Report exactly what the tool returned. Do not add your own status judgements.
        The tool result is ground truth — trust it completely.

        POSITIONAL RESOLUTION RULES:
        - "first image" → index 0 of images in search_media results
        - "last video" → last video in search_media results
        - "last 2 videos" → last 2 videos in search_media results
        - "third image" → index 2 of images in search_media results
        - When user requests action on multiple items, resolve ALL positions first,
          then call the appropriate tool(s) with all resolved IDs at once.

        NEVER pre-check status before calling tools — the tools do this internally.
        NEVER skip calling a tool because you think the item is not pending — let the tool decide.
        NEVER report a status based on conversation history or session context.

        Always include item title AND platform in your approval/rejection summary.
        Never mention Firestore, search_media, internal tools, or internal steps to the user.
        Respond only with the outcome — not how you got there.

        - Use search_docs only for platform how-to questions.
        """


def orchestrate(
    user_input: str,
    conversation_history: list,
    user_id: str,
    brand_id: str,
    routing_context: str = None,
    last_turn: dict = None,
    **kwargs
):
    client = get_claude_client()

    # Guard against empty input crashing embed_text
    if not user_input or not user_input.strip():
        return "I didn't catch that — could you say that again?"

    brand_info = get_brand_info(brand_id) or {}
    brand_chunks = search_brand_book(brand_id=brand_id, query=user_input.strip(), top_k=3)
    system_prompt = build_system_prompt(brand_info, brand_chunks)

    if routing_context:
        system_prompt += (
            "\n\n## Recent Conversation Context\n"
            + routing_context
        )

    messages = []
    for msg in conversation_history:
        messages.append({"role": msg["role"], "content": msg["content"]})

    # Inject last_turn only from media agents for item name/position resolution.
    # Strip ALL status indicators — search_media results are the only status source.
    if last_turn and last_turn.get("response") and last_turn.get("agent") in ("media_search", "media_approval"):
        import re as _re
        print(f"[Orchestrator] last_turn agent={last_turn.get('agent','?')}, response_len={len(last_turn['response'])}")
        sanitised = last_turn["response"]
        sanitised = _re.sub(r"[✅❌⏳]", "", sanitised)
        sanitised = _re.sub(r"Status:\s*\w+", "", sanitised, flags=_re.IGNORECASE)
        sanitised = _re.sub(r"\b(Approved|Rejected|Pending)\b", "", sanitised, flags=_re.IGNORECASE)
        messages.append({"role": "user", "content": last_turn.get("user", "")})
        messages.append({"role": "assistant", "content": sanitised})

    messages.append({"role": "user", "content": user_input})

    MAX_STEPS = 5
    step = 0
    last_text_response = None
    actioned_approved = []   # IDs actually approved this turn — ground truth from tool calls
    actioned_rejected = []   # IDs actually rejected this turn — ground truth from tool calls

    # Force a tool call on step 0 for any media-related request
    # so Claude always hits Firestore — never reads status from conversation history
    _media_keywords = ("approve", "reject", "accept", "decline", "last", "first",
                       "pending", "image", "video", "media", "status")
    _needs_tool = any(k in user_input.lower() for k in _media_keywords)

    while step < MAX_STEPS:
        # Force any tool use on first step for media requests
        tool_choice = {"type": "any"} if (_needs_tool and step == 0) else {"type": "auto"}
        response = client.messages.create(
            model="claude-sonnet-4-6",
            system=system_prompt,
            messages=messages,
            tools=TOOLS,
            tool_choice=tool_choice,
            max_tokens=1000
        )

        print(f"\nStep {step + 1} — Stop reason: {response.stop_reason}")
        step += 1

        # Claude finished — stream or return final text
        if response.stop_reason == "end_turn":
            for block in response.content:
                if getattr(block, "type", None) == "text":
                    return block.text.strip()
            return last_text_response or "Done."

        # Claude wants to use a tool
        if response.stop_reason == "tool_use":
            messages.append({"role": "assistant", "content": response.content})

            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    tool_name = block.name
                    tool_args = block.input
                    print(f"  Tool called: {tool_name} | Args: {tool_args}")

                    agent_fn = AVAILABLE_AGENTS.get(tool_name)
                    if agent_fn:
                        result = agent_fn(
                            user_input=user_input,
                            conversation_history=conversation_history,
                            user_id=user_id,
                            brand_id=brand_id,
                            client=client,
                            brand_chunks=brand_chunks,
                            **tool_args
                        )
                        last_text_response = result
                        # Track actually actioned IDs from agent result — skips non-pending items
                        # Result format: "Approved: id1, id2 | Skipped: id3 | ..."
                        if tool_name in ("approve_media", "reject_media") and isinstance(result, str):
                            import re as _re2
                            if tool_name == "approve_media":
                                _ids = _re2.findall(r'Approved:\s*((?:image_\d+|video_\d+)(?:,\s*(?:image_\d+|video_\d+))*)', result)
                                for group in _ids:
                                    actioned_approved.extend([i.strip() for i in group.split(",")])
                            elif tool_name == "reject_media":
                                _ids = _re2.findall(r'Rejected:\s*((?:image_\d+|video_\d+)(?:,\s*(?:image_\d+|video_\d+))*)', result)
                                for group in _ids:
                                    actioned_rejected.extend([i.strip() for i in group.split(",")])
                    else:
                        result = f"Tool '{tool_name}' not found."

                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": str(result)
                    })

            messages.append({"role": "user", "content": tool_results})
            continue

    # Return actioned IDs alongside response so main.py can persist exact session state
    if actioned_approved or actioned_rejected:
        return {
            "response": last_text_response or "Done.",
            "approved": actioned_approved,
            "rejected": actioned_rejected
        }
    return last_text_response or "Could not complete the request."
