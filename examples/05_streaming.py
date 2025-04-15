"""
Example of using streaming with Unified LLM Client

This example demonstrates how to use streaming with both OpenAI and Anthropic models.
Streaming allows for receiving partial responses as they are generated, which improves
the user experience for longer responses.

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


async def handle_chunk(chunk):
    """Process each chunk from the stream"""
    # In a real application, you might append to a UI or log
    # Just print for this example
    print(chunk, end="", flush=True)


async def test_openai_streaming():
    """Test streaming with OpenAI"""
    print("\n--- Testing OpenAI Streaming ---")
    
    # Initialize client
    client = AsyncLLMClient()
    
    try:
        # Using the latest GPT model with streaming
        prompt = "Write a short poem about artificial intelligence, one line at a time."
        
        print(f"\nPrompt: {prompt}")
        print("\nResponse (streaming):")
        
        # Make streaming request
        response = await client.stream(
            prompt,
            model="gpt-4o-mini",
            temperature=0.7,
            max_tokens=200,
            stream_handler=handle_chunk
        )
        
        print(f"\n\nTokens used: {response['input_tokens']} input, {response['output_tokens']} output")
        return True
    except Exception as e:
        print(f"Error: {str(e)}")
        if hasattr(e, '__cause__') and e.__cause__:
            print(f"Cause: {e.__cause__}")
        return False


async def test_anthropic_streaming():
    """Test streaming with Anthropic Claude"""
    print("\n--- Testing Anthropic Claude Streaming ---")
    
    # Initialize client
    client = AsyncLLMClient()
    
    try:
        # Using Claude with streaming
        prompt = "Explain the concept of neural networks in simple terms, step by step."
        
        print(f"\nPrompt: {prompt}")
        print("\nResponse (streaming):")
        
        # Make streaming request
        response = await client.stream(
            prompt,
            model="claude-3-5-haiku-latest",
            temperature=0.7,
            max_tokens=300,
            stream_handler=handle_chunk
        )
        
        print(f"\n\nTokens used: {response['input_tokens']} input, {response['output_tokens']} output")
        return True
    except Exception as e:
        print(f"Error: {str(e)}")
        if hasattr(e, '__cause__') and e.__cause__:
            print(f"Cause: {e.__cause__}")
        return False


async def test_streaming_with_tools():
    """Test streaming with tool usage (Anthropic)"""
    print("\n--- Testing Streaming with Tool Usage ---")
    
    # Initialize client with tool (no need for custom tools in this example)
    client = AsyncLLMClient()
    
    try:
        # Using Claude with streaming and suggesting a tool usage scenario
        prompt = """Please explain step by step how you would solve this problem:
        
        A merchant sells coffee for $18.99 per pound and tea for $12.50 per pound.
        A customer buys 2.5 pounds of coffee and 1.75 pounds of tea.
        Calculate the total cost of the purchase."""
        
        print(f"\nPrompt: {prompt}")
        print("\nResponse (streaming):")
        
        # Make streaming request
        response = await client.stream(
            prompt,
            model="claude-3-5-haiku-latest",
            temperature=0.3,
            max_tokens=500,
            stream_handler=handle_chunk
        )
        
        print(f"\n\nTokens used: {response['input_tokens']} input, {response['output_tokens']} output")
        return True
    except Exception as e:
        print(f"Error: {str(e)}")
        if hasattr(e, '__cause__') and e.__cause__:
            print(f"Cause: {e.__cause__}")
        return False


async def test_openai_stream_with_system_prompt():
    """Test OpenAI streaming with a system prompt"""
    print("\n--- Testing OpenAI Streaming with System Prompt ---")
    
    # Initialize client
    client = AsyncLLMClient()
    
    try:
        # Create a system prompt for a specific persona
        system_prompt = """You are a helpful tech educator who explains concepts clearly and concisely.
        Use simple analogies and break down complex ideas into manageable parts.
        Avoid technical jargon when possible, and when you must use it, explain its meaning."""
        
        user_prompt = "Explain how public key cryptography works in about 5 sentences."
        
        print(f"\nSystem: {system_prompt}")
        print(f"User: {user_prompt}")
        print("\nResponse (streaming):")
        
        # Make streaming request with system prompt
        response = await client.stream(
            user_prompt,
            model="gpt-4o-mini",
            instructions=system_prompt,  # Pass system prompt as instructions
            temperature=0.5,
            max_tokens=200,
            stream_handler=handle_chunk
        )
        
        print(f"\n\nTokens used: {response['input_tokens']} input, {response['output_tokens']} output")
        return True
    except Exception as e:
        print(f"Error: {str(e)}")
        if hasattr(e, '__cause__') and e.__cause__:
            print(f"Cause: {e.__cause__}")
        return False


async def test_stream_chat_history():
    """Test streaming with a conversation history"""
    print("\n--- Testing Streaming with Conversation History ---")
    
    # Initialize client
    client = AsyncLLMClient()
    
    # Create a conversation history
    conversation = [
        {"role": "user", "content": "What is machine learning?"},
        {"role": "assistant", "content": "Machine learning is a subset of artificial intelligence that enables computers to learn from data and improve their performance on specific tasks without being explicitly programmed. Instead of following hard-coded rules, machine learning systems identify patterns in data and make decisions based on what they've learned."},
        {"role": "user", "content": "Can you give me an example of supervised learning?"}
    ]
    
    try:
        print("\nConversation history:")
        for msg in conversation:
            print(f"{msg['role'].title()}: {msg['content'][:50]}..." if len(msg['content']) > 50 else f"{msg['role'].title()}: {msg['content']}")
        
        print("\nResponse (streaming):")
        
        # Make streaming request with conversation history
        response = await client.stream(
            conversation,  # Pass the conversation history
            model="claude-3-5-haiku-latest",
            temperature=0.7,
            max_tokens=400,
            stream_handler=handle_chunk
        )
        
        print(f"\n\nTokens used: {response['input_tokens']} input, {response['output_tokens']} output")
        return True
    except Exception as e:
        print(f"Error: {str(e)}")
        if hasattr(e, '__cause__') and e.__cause__:
            print(f"Cause: {e.__cause__}")
        return False


async def main():
    """Run all streaming tests"""
    print("=== Streaming Examples with Unified LLM Client ===")
    print("\nThis example demonstrates how to use streaming capabilities.")
    print("Streaming provides partial responses as they are generated,")
    print("which improves perceived latency and user experience.")
    
    # Test OpenAI streaming
    openai_success = await test_openai_streaming()
    print(f"OpenAI streaming test: {'PASSED' if openai_success else 'FAILED'}\n")
    
    # Test Anthropic streaming
    anthropic_success = await test_anthropic_streaming()
    print(f"Anthropic streaming test: {'PASSED' if anthropic_success else 'FAILED'}\n")
    
    # Test streaming with tools
    tools_success = await test_streaming_with_tools()
    print(f"Streaming with tools test: {'PASSED' if tools_success else 'FAILED'}\n")
    
    # Test OpenAI streaming with system prompt
    system_success = await test_openai_stream_with_system_prompt()
    print(f"Streaming with system prompt test: {'PASSED' if system_success else 'FAILED'}\n")
    
    # Test streaming with conversation history
    history_success = await test_stream_chat_history()
    print(f"Streaming with conversation history test: {'PASSED' if history_success else 'FAILED'}\n")
    
    print("Benefits of streaming:")
    print("✓ Improved perceived latency - first words appear immediately")
    print("✓ Better user experience for longer responses")
    print("✓ Can handle and display tool calls during generation")
    print("✓ Works with both OpenAI and Anthropic models")
    print("✓ Compatible with conversation history and system prompts")


if __name__ == "__main__":
    asyncio.run(main())
