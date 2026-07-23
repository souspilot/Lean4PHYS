#!/usr/bin/env python3
import os
from dotenv import load_dotenv, find_dotenv
from openai import OpenAI

# 1. Load the environment variables (automatically finds .env in parent directories)
load_dotenv(find_dotenv())
api_key = os.environ.get("OPENAI_API_KEY")

if not api_key:
    print("❌ Error: OPENAI_API_KEY not found. Please check your .env file.")
    exit(1)

# 2. Initialize the client using Google's OpenAI-compatible endpoint
client = OpenAI(
    api_key=api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

# 3. Send a basic test prompt
print("Sending test request to Gemini 2.5 Pro...")
try:
    response = client.chat.completions.create(
        model="gemini-2.5-pro",
        messages=[
            {"role": "system", "content": "You are a helpful API test assistant."},
            {"role": "user", "content": "Hello! Please reply with a short, single-sentence greeting confirming the API connection is working."}
        ],
        max_tokens=50
    )
    
    # 4. Print the output
    print("\n✅ Success! Received response:")
    print("-" * 40)
    print(response.choices[0].message.content.strip())
    print("-" * 40)
    
except Exception as e:
    print(f"\n❌ Request failed: {e}")
