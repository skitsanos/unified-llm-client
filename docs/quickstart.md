# Quick Start Guide

This guide will help you get started with the Unified LLM Client library.

## Installation

Install the library using pip:

```bash
pip install unified-llm-client
```

## Basic Usage

```python
import asyncio
from llm import AsyncLLMClient

async def main():
    # Initialize the client
    client = AsyncLLMClient()
    
    # Get a response from OpenAI's GPT-4o
    response = await client.response(
        "Explain quantum computing in simple terms",
        model="gpt-4o"
    )
    
    print(response["text"])
    
    # Get a response from Anthropic's Claude
    claude_response = await client.response(
        "Explain quantum computing in simple terms",
        model="claude-3-opus-20240229"
    )
    
    print(claude_response["text"])

if __name__ == "__main__":
    asyncio.run(main())
```

## Using System Instructions

```python
import asyncio
from llm import AsyncLLMClient

async def main():
    client = AsyncLLMClient()
    
    response = await client.response(
        "Create a short poem about technology",
        model="gpt-4o",
        instructions="You are a poetic assistant that specializes in creating rhyming poems."
    )
    
    print(response["text"])

if __name__ == "__main__":
    asyncio.run(main())
```

## Working with Conversations

```python
import asyncio
from llm import AsyncLLMClient

async def main():
    client = AsyncLLMClient()
    
    # Create a conversation with multiple messages
    messages = [
        {"role": "system", "content": "You are a helpful coding assistant."},
        {"role": "user", "content": "How do I read a file in Python?"},
        {"role": "assistant", "content": "You can use the built-in `open()` function..."},
        {"role": "user", "content": "Can you show me how to read a JSON file specifically?"}
    ]
    
    response = await client.response(
        messages,
        model="gpt-4o"
    )
    
    print(response["text"])

if __name__ == "__main__":
    asyncio.run(main())
```

## Streaming Responses

Streaming responses can improve the perceived latency and user experience for longer responses:

```python
import asyncio
from llm import AsyncLLMClient

# Define a handler function for stream chunks
async def handle_chunk(chunk: str):
    print(chunk, end="", flush=True)

async def main():
    client = AsyncLLMClient()
    
    # Stream a response from OpenAI
    response = await client.stream(
        "Explain the theory of relativity in simple terms",
        model="gpt-4o",
        stream_handler=handle_chunk
    )
    
    print(f"\n\nTotal tokens: {response['input_tokens']} input, {response['output_tokens']} output")
    
    # Stream a response from Anthropic
    response = await client.stream(
        "Write a short story about a robot discovering emotions",
        model="claude-3-5-haiku-latest",
        stream_handler=handle_chunk
    )
    
    print(f"\n\nTotal tokens: {response['input_tokens']} input, {response['output_tokens']} output")

if __name__ == "__main__":
    asyncio.run(main())
```

## Next Steps

- Learn about [Tool/Function Calling](tools.md)
- Explore [Claude Tool Calling](claude_tools.md)
- Learn about [Using Ollama](ollama.md) for local models
- Check out more [Examples](examples.md)
- View the [API Reference](api_reference.md)
