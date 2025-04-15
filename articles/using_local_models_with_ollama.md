# Running Large Language Models Locally with Ollama and Unified LLM Client

In the rapidly evolving landscape of artificial intelligence, Large Language Models (LLMs) have become powerful tools for developers. However, using these models typically requires sending data to third-party APIs, which can raise concerns about privacy, latency, and usage costs. This is where Ollama and the Unified LLM Client come in, offering a solution to run state-of-the-art models locally on your own hardware.

## What is Ollama?

Ollama is an open-source project that simplifies running LLMs locally. It packages models in an easy-to-use format and provides a compatible API interface, allowing you to run models like Llama 3, Mistral, Qwen, and many others directly on your machine without sending data to external services.

## Benefits of Running Models Locally

1. **Privacy**: Your data never leaves your machine
2. **No API Costs**: No usage fees or token limits
3. **No Internet Dependency**: Works offline
4. **Reduced Latency**: No network transmission delays
5. **Full Control**: Customize model parameters and fine-tuning

## Getting Started with Ollama and Unified LLM Client

The Unified LLM Client provides a consistent interface for working with various LLM providers, including Ollama. Here's how to get started:

### Step 1: Install Ollama

First, you need to install Ollama on your machine:

1. Visit [ollama.com](https://ollama.com) and download the appropriate version for your operating system
2. Follow the installation instructions
3. Start the Ollama service

### Step 2: Pull the Models You Want to Use

Once Ollama is installed, you can pull the models you want to use:

```bash
# Pull the Llama 3 model
ollama pull llama3

# Pull the Qwen 2.5 model
ollama pull qwen2.5

# Pull the Mistral model
ollama pull mistral
```

### Step 3: Use the Unified LLM Client

With Ollama set up, you can now use the Unified LLM Client to interact with these local models:

```python
import asyncio
from llm import AsyncLLMClient

async def main():
    # Initialize the client for Ollama
    client = AsyncLLMClient(
        base_url="http://localhost:11434/v1",
        api_key="ollama"  # Ollama doesn't need a real API key
    )
    
    # Get a response from a local model
    response = await client.response(
        "Explain how nuclear fusion works in simple terms",
        model="llama3",  # Use any model you've pulled
        temperature=0.7,
        use_responses_api=False  # Important for Ollama
    )
    
    print(response["text"])

if __name__ == "__main__":
    asyncio.run(main())
```

## Advanced Features: Tool Calling with Local Models

One of the powerful features of modern LLMs is their ability to use tools or functions. The Unified LLM Client makes this easy with local models too:

```python
import asyncio
from llm import AsyncLLMClient, ToolRegistry, llm_tool

@llm_tool
def search_database(query: str, limit: int = 5):
    """Search a database for information."""
    # Mock implementation
    results = [{"id": i, "title": f"Result {i} for {query}"} for i in range(limit)]
    return results

async def main():
    # Set up tool registry
    tools = ToolRegistry()
    tools.register("search_database", search_database)
    
    # Initialize client with tools
    client = AsyncLLMClient(
        base_url="http://localhost:11434/v1",
        api_key="ollama",
        tool_registry=tools
    )
    
    # Use with tool calling (works best with Qwen models)
    response = await client.response(
        "Find information about quantum computing",
        model="qwen2.5",
        instructions="You have access to a database search tool",
        use_responses_api=False
    )
    
    print(response["text"])

if __name__ == "__main__":
    asyncio.run(main())
```

Note: Tool calling capabilities vary by model. Qwen models generally have the best support for tool calling among the Ollama-supported models.

## Performance Considerations

When running models locally, performance depends on your hardware:

- **CPU vs. GPU**: Models run much faster with GPU acceleration
- **Memory Requirements**: Most models need 8-16GB of RAM, with larger models requiring more
- **Disk Space**: Models can be 3-8GB each
- **First-Run Latency**: The first request to a model has higher latency as it loads into memory

## Switching Between Local and Cloud Models

One of the advantages of using the Unified LLM Client is the ability to easily switch between local and cloud models:

```python
import asyncio
from llm import AsyncLLMClient

async def main():
    # Initialize the client
    client = AsyncLLMClient()
    
    # Use OpenAI's GPT models (cloud)
    openai_response = await client.response(
        "Explain quantum computing",
        model="gpt-4o-mini"
    )
    
    # Initialize for Ollama (local)
    ollama_client = AsyncLLMClient(
        base_url="http://localhost:11434/v1",
        api_key="ollama"
    )
    
    # Use a local model
    ollama_response = await ollama_client.response(
        "Explain quantum computing",
        model="llama3",
        use_responses_api=False
    )
    
    # Compare the responses
    print("OpenAI Response:", openai_response["text"])
    print("Ollama Response:", ollama_response["text"])

if __name__ == "__main__":
    asyncio.run(main())
```

## Conclusion

The combination of Ollama and the Unified LLM Client provides a powerful and flexible way to work with LLMs locally. Whether you're concerned about privacy, want to reduce costs, or need offline capabilities, running models locally is now more accessible than ever.

The Unified LLM Client's consistent API means you can easily switch between local and cloud models as needed, giving you the best of both worlds. Start exploring the possibilities of local LLMs today with Ollama and the Unified LLM Client!
