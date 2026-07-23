#!/usr/bin/env python3
import os
from dotenv import load_dotenv, find_dotenv
from openai import OpenAI

# 1. Load the environment variables (automatically finds .env)
load_dotenv(find_dotenv())
api_key = os.environ.get("OPENROUTER_API_KEY")

if not api_key:
    print("❌ Error: OPENROUTER_API_KEY not found. Please check your .env file.")
    exit(1)

# 2. Initialize the client using OpenRouter's endpoint
client = OpenAI(
    api_key=api_key,
    base_url="https://openrouter.ai/api/v1"
)

# 3. Send a basic test prompt
print("Sending test request to OpenRouter (nvidia/nemotron-3-ultra-550b-a55b:free)...")
try:
    response = client.chat.completions.create(
        model="nvidia/nemotron-3-ultra-550b-a55b:free",
        messages=[
            {"role": "system", "content": "You are a helpful API test assistant."},
            {"role": "user", "content": "Hello! Please reply with a short, single-sentence greeting confirming the OpenRouter API connection is working."}
        ],
        # OpenRouter recommends providing these headers to identify your app
        extra_headers={
            "HTTP-Referer": "https://localhost", 
            "X-Title": "Lean4Phys API Test"
        },
        max_tokens=50
    )
    
    # 4. Print the output
    print("\n✅ Success! Received response:")
    print("-" * 40)
    print(response.choices[0].message.content.strip())
    print("-" * 40)
    
except Exception as e:
    print(f"\n❌ Request failed: {e}")
