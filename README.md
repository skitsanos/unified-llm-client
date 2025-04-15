# Unified LLM Client

A unified async client library for interacting with multiple LLM providers (OpenAI, Anthropic, and more) using a consistent API.

## Features

- Unified interface for multiple LLM providers (OpenAI, Anthropic, Ollama)
- Async-first design for high-performance applications
- Tool/function calling support with a consistent interface
- Streaming support for improved user experience with long responses
- Rich error handling and logging
- Support for both OpenAI Completions and Responses APIs
- Support for local models via Ollama integration
- Type hints throughout the codebase

## Installation

```bash
pip install unified-llm-client
```

## Quick Start

```python
import asyncio
from llm import AsyncLLMClient, ToolRegistry, llm_tool

# Define a tool function using the decorator
@llm_tool
async def get_weather(location: str, unit: str = "celsius"):
    """Get the current weather for a location."""
    # Mock implementation
    return f"The weather in {location} is sunny and 22 degrees {unit}"

async def main():
    # Create a tool registry and register your tools
    tools = ToolRegistry()
    tools.register("get_weather", get_weather)
    
    # Initialize the client with the tool registry
    client = AsyncLLMClient(tool_registry=tools)
    
    # Use the client with different models
    response = await client.response(
        "What's the weather like in Paris?",
        model="gpt-4o-mini",  # Default model is gpt-4o-mini
        instructions="You are a helpful assistant that provides weather information."
    )
    
    print(response["text"])
    
    # Use with Anthropic's Claude
    claude_response = await client.response(
        "What's the weather like in Tokyo?",
        model="claude-3-opus-20240229",
        instructions="You are a helpful assistant that provides weather information."
    )
    
    print(claude_response["text"])
    
    # Use with Ollama (local models)
    # Requires Ollama to be installed: https://ollama.com
    ollama_response = await client.response(
        "What's the weather like in London?",
        model="llama3",  # Or any model you've pulled with Ollama
        instructions="You are a helpful assistant that provides weather information.",
        base_url="http://localhost:11434/v1",
        api_key="ollama",
        use_responses_api=False
    )
    
    print(ollama_response["text"])

if __name__ == "__main__":
    asyncio.run(main())
```

## Advanced Features

### Using Conversations

```python
import asyncio
from llm import AsyncLLMClient, Message

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

### Tool/Function Calling

The library supports a consistent interface for tool calling across providers:

```python
import asyncio
import json
from llm import AsyncLLMClient, ToolRegistry, llm_tool

@llm_tool
async def search_database(query: str, limit: int = 5):
    """Search a database for information."""
    # Mock implementation
    results = [{"id": i, "title": f"Result {i} for {query}"} for i in range(limit)]
    return results

async def main():
    tools = ToolRegistry()
    tools.register("search_database", search_database)
    
    client = AsyncLLMClient(tool_registry=tools)
    
    response = await client.response(
        "Find information about machine learning frameworks",
        model="gpt-4o",
        instructions="You are a helpful assistant with access to a database."
    )
    
    print(response["text"])

if __name__ == "__main__":
    asyncio.run(main())
```

### Streaming Responses

The library supports streaming responses for both OpenAI and Anthropic models, which improves perceived latency and user
experience for longer responses:

```python
import asyncio
from llm import AsyncLLMClient, StreamHandler

# Define a custom handler function for the stream chunks
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

Streaming also works with conversation history, system instructions, and tool usage.

## License

MIT

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

### Development Installation

To set up the development environment:

```bash
git clone https://github.com/skitsanos/unified-llm-client.git
cd unified-llm-client
pip install -e .
pip install -r requirements-dev.txt
```

### Testing

Run tests with pytest:

```bash
pytest
```

### Publishing to PyPI

See [Publishing Documentation](docs/publishing.md) for detailed instructions on publishing releases to PyPI.
