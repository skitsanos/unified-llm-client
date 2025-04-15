"""
Basic example of using the Unified LLM Client

@author: skitsanos
"""

import asyncio
import os
from dotenv import load_dotenv

from llm import AsyncLLMClient

# Load environment variables from .env file
load_dotenv()


async def main():
    # Initialize the client
    client = AsyncLLMClient()
    
    # Get a response from OpenAI's GPT model
    print("Asking OpenAI GPT-4...")
    openai_response = await client.response(
        "Explain the difference between REST and GraphQL in simple terms",
        model="gpt-4o",
        temperature=0.7
    )
    
    print(f"\nOpenAI Response:\n{openai_response['text']}")
    print(f"Tokens used: {openai_response['input_tokens']} input, {openai_response['output_tokens']} output")
    
    # Get a response from Anthropic's Claude model
    print("\nAsking Anthropic Claude...")
    claude_response = await client.response(
        "Explain the difference between REST and GraphQL in simple terms",
        model="claude-3-opus-20240229",
        temperature=0.7
    )
    
    print(f"\nClaude Response:\n{claude_response['text']}")
    print(f"Tokens used: {claude_response['input_tokens']} input, {claude_response['output_tokens']} output")


if __name__ == "__main__":
    asyncio.run(main())
