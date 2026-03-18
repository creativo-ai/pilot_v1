from anthropic import Anthropic
from google import genai
from google.genai import types
import os
import json


def get_claude_client() -> Anthropic:
    return Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))


def get_gemini_client() -> genai.Client:
    return genai.Client(api_key=os.getenv("GEMINI_API_KEY"))


def gemini_generate(prompt: str, system: str = None) -> str:
    """Single-turn Gemini 2.5 Flash call. Strips markdown fences automatically."""
    client = get_gemini_client()
    config = types.GenerateContentConfig(
        system_instruction=system, temperature=0.3
    ) if system else types.GenerateContentConfig(temperature=0.3)

    response = client.models.generate_content(
        model="gemini-2.5-flash", contents=prompt, config=config
    )
    text = (response.text or "").strip()
    if text.startswith("```"):
        parts = text.split("```")
        text = parts[1] if len(parts) > 1 else text
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()
    return text
