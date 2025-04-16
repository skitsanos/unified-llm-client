# Using Ollama with Unified LLM Client

Ollama is an open-source project that allows you to run large language models locally on your machine. The Unified LLM Client supports Ollama, making it easy to work with local models using the same API as cloud-based providers.

## Prerequisites

Before using Ollama with the Unified LLM Client, you need to:

1. Install Ollama on your machine by following the instructions at [ollama.com](https://ollama.com)
2. Start the Ollama service
3. Pull the models you want to use (e.g., `ollama pull qwen2.5` or `ollama pull llama3`)

## Basic Usage

```python
import asyncio
from llm import AsyncLLMClient

async def main():
    # Initialize the client with Ollama settings
    client = AsyncLLMClient(
        base_url="http://localhost:11434/v1",
        api_key="ollama"  # Ollama doesn't require a real API key
    )
    
    # Use with Ollama models
    response = await client.response(
        "Explain quantum computing in simple terms",
        model="qwen2.5",  # Use any model you've pulled with Ollama
        temperature=0.7,
        use_responses_api=False  # Important: Ollama works with the chat completions API
    )
    
    print(response["text"])

if __name__ == "__main__":
    asyncio.run(main())
```

## Available Models

Ollama supports a wide range of models, including:

- `llama3` - Meta's Llama 3 models
- `qwen2.5` - Alibaba's Qwen 2.5 models
- `mistral` - Mistral AI's models
- `gemma` - Google's Gemma models
- `phi3` - Microsoft's Phi-3 models
- And many more

You can see a complete list by running `ollama list` in your terminal or visiting the [Ollama model library](https://ollama.com/library).

## Tool Calling with Ollama

Some Ollama models support basic tool calling. The client automatically formats tool calls to work with these models:

```python
import asyncio
from llm import AsyncLLMClient, ToolRegistry, llm_tool

@llm_tool
def calculator(expression: str) -> float:
    """Evaluate a mathematical expression."""
    try:
        return float(eval(expression, {"__builtins__": {}}, {}))
    except Exception as e:
        return f"Error: {str(e)}"

async def main():
    # Create tool registry and register the calculator tool
    tools = ToolRegistry()
    tools.register("calculator", calculator)
    
    # Initialize client with tool registry
    client = AsyncLLMClient(
        base_url="http://localhost:11434/v1",
        api_key="ollama",
        tool_registry=tools
    )
    
    # Use with tool calling
    response = await client.response(
        "What is 123 × 456?",
        model="qwen2.5",  # Qwen models have good tool-calling support
        instructions="You have access to a calculator tool",
        temperature=0.7,
        use_responses_api=False
    )
    
    print(response["text"])

if __name__ == "__main__":
    asyncio.run(main())
```

## Streaming with Ollama

Ollama supports streaming responses, which can improve the perceived latency:

```python
import asyncio
from llm import AsyncLLMClient

# Define a handler function for stream chunks
async def handle_chunk(chunk: str):
    print(chunk, end="", flush=True)

async def main():
    client = AsyncLLMClient(
        base_url="http://localhost:11434/v1",
        api_key="ollama"
    )
    
    # Stream a response from Ollama
    response = await client.stream(
        "Write a short story about space exploration",
        model="llama3",
        stream_handler=handle_chunk,
        use_responses_api=False
    )
    
    print(f"\n\nTotal tokens: {response['input_tokens']} input, {response['output_tokens']} output")

if __name__ == "__main__":
    asyncio.run(main())
```

## Performance Considerations

When using Ollama:

- Performance depends on your hardware (CPU/GPU/RAM)
- First inference with a model may be slower while it loads into memory
- Response times are typically slower than cloud APIs but have no network latency
- There are no token limits or usage costs
- Streaming can significantly improve perceived performance

## Advanced Configuration

For more advanced Ollama setups:

```python
client = AsyncLLMClient(
    base_url="http://localhost:11434/v1",  # Change if running on a different machine
    api_key="ollama"
)

# You can also specify model parameters
response = await client.response(
    "Explain quantum computing",
    model="qwen2.5",
    temperature=0.7,
    max_tokens=2000,
    top_p=0.9,
    frequency_penalty=0.0,
    presence_penalty=0.0,
    use_responses_api=False
)
```

Refer to the [Ollama API documentation](https://github.com/ollama/ollama/blob/main/docs/api.md) for more details on supported parameters.
