from agents import (
    write_email,
    talk_agent,
    list_pending_images,
    approve_images,
    reject_images,
    search_docs_agent
)


ACTION_SCHEMAS = {
    "write_email": {
        "description": "Write a brand-aligned email to a specific audience.",
        "arguments": {
            "email_goal": "The purpose of the email",
            "audience": "Target audience of the email"
        }
    },
    "talk": {
        "description": "General conversation, brand info questions, or unclear requests.",
        "arguments": {"user_message": "the message of the user"}
    },
    "list_pending_images": {
        "description": "List images that are pending approval for the user.",
        "arguments": {
            "brand_id": "the exact brand id that we are working with",
            "user_id": "the user ID that we are working with now"
        }
    },
    "approve_images": {
        "description": "Approve one or more images by ID.",
        "arguments": {
            "image_ids": "List of image IDs to approve"
        }
    },
    "reject_images": {
        "description": "Reject one or more images by ID.",
        "arguments": {
            "image_ids": "List of image IDs to reject"
        }
    },
    "search_docs": {
        "description": "Search the documentation to answer questions about how to use the platform, sign up, sign in, subscription plans, or any feature-related questions.",
        "arguments": {
            "query": "The user's question to search in the documentation"
        }
    }
}

AVAILABLE_ACTIONS = {
    "write_email": write_email,
    "talk": talk_agent,
    "list_pending_images": list_pending_images,
    "approve_images": approve_images,
    "reject_images": reject_images,
    "search_docs": search_docs_agent
}