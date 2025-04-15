"""
Basic example of using the Unified LLM Client with both OpenAI and Anthropic

This example shows simple text generation with both providers.

@author: skitsanos
"""

import os
import asyncio
import sys
from dotenv import load_dotenv

# Add the project root to PYTHONPATH for running the example
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm import AsyncLLMClient

# Load environment variables from .env file
load_dotenv()


async def test_openai():
    """Test basic text generation with OpenAI"""
    print("\n--- Testing OpenAI Basic Text Generation ---")
    
    # Initialize client
    client = AsyncLLMClient()
    
    try:
        # Using GPT-4o mini model
        response = await client.response(
            "Explain what makes Python a popular programming language in 2-3 sentences.",
            model="gpt-4o-mini",
            temperature=0.0,
            max_tokens=100
        )
        
        print(f"OpenAI Response: {response['text']}")
        print(f"Tokens used: {response['input_tokens']} input, {response['output_tokens']} output")
        return True
    except Exception as e:
        print(f"Error: {str(e)}")
        return False


async def test_anthropic():
    """Test basic text generation with Anthropic Claude"""
    print("\n--- Testing Anthropic Claude Basic Text Generation ---")
    
    # Initialize client
    client = AsyncLLMClient()
    
    try:
        # Using Claude 3.5 Haiku model
        response = await client.response(
            "Explain what makes Python a popular programming language in 2-3 sentences.",
            model="claude-3-5-haiku-latest",
            temperature=0.0,
            max_tokens=100
        )
        
        print(f"Claude Response: {response['text']}")
        print(f"Tokens used: {response['input_tokens']} input, {response['output_tokens']} output")
        return True
    except Exception as e:
        print(f"Error: {str(e)}")
        return False


async def main():
    """Run all tests"""
    openai_success = await test_openai()
    print(f"OpenAI test {'PASSED' if openai_success else 'FAILED'}\n")
    
    anthropic_success = await test_anthropic()
    print(f"Anthropic test {'PASSED' if anthropic_success else 'FAILED'}\n")


if __name__ == "__main__":
    asyncio.run(main())
