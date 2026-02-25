import os
import anthropic

# Make sure your Claude API key is set in the environment
CLAUDE_API_KEY = os.getenv("CLAUDE_API_KEY")  # or replace with your key as string
client = anthropic.Client(api_key=CLAUDE_API_KEY)

models = client.models.list()
print("Available models:", models)