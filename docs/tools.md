# Tool/Function Calling

The Unified LLM Client provides a consistent interface for tool calling across different LLM providers. In version
0.2.0, the tool calling support has been significantly enhanced, particularly for Anthropic Claude models.

## Defining Tools

You can define tools using the `llm_tool` decorator:

```python
from llm import llm_tool

@llm_tool
def get_weather(location: str, unit: str = "celsius"):
    """Get the current weather for a location."""
    # Implementation
    return f"The weather in {location} is sunny and 22 degrees {unit}"

@llm_tool
async def search_database(query: str, limit: int = 5):
    """Search a database for information."""
    # Async implementation
    results = [{"id": i, "title": f"Result {i} for {query}"} for i in range(limit)]
    return results
```

## Registering Tools

Tools need to be registered with a `ToolRegistry`:

```python
from llm import ToolRegistry

# Create a tool registry
tools = ToolRegistry()

# Register your tools
tools.register("get_weather", get_weather)
tools.register("search_database", search_database)
```

## Using Tools with LLM Client

```python
from llm import AsyncLLMClient

# Initialize the client with the tool registry
client = AsyncLLMClient(tool_registry=tools)

# The client will automatically make the tools available to the LLM
response = await client.response(
    "What's the weather like in Paris?",
    model="gpt-4o",
    instructions="You are a helpful assistant that can access weather information."
)

print(response["text"])
```

## Tool Type Annotations

For better tool schema generation, you can add type annotations to your tool functions:

```python
from typing import List, Dict, Any

@llm_tool
def search_products(
    query: str,
    category: str = "all",
    max_results: int = 5
) -> List[Dict[str, Any]]:
    """
    Search for products in a catalog.
    
    Args:
        query: The search query
        category: Product category to filter by
        max_results: Maximum number of results to return
        
    Returns:
        List of product records
    """
    # Implementation
    return [{"id": i, "name": f"Product {i}", "category": category} for i in range(max_results)]
```

## Provider-Specific Tool Behavior

The library automatically adapts tools to the format required by each provider:

- For OpenAI, tools are formatted as OpenAI function tools
- For Anthropic, tools are formatted in Anthropic's tool format
- For Ollama, tools are formatted in OpenAI-compatible format

You don't need to worry about the differences in API formats.

## Streaming with Tools

In version 0.2.0, you can now use streaming along with tool calls:

```python
async def handle_chunk(chunk: str):
    print(chunk, end="", flush=True)

response = await client.stream(
    "What's the weather like in Paris?",
    model="claude-3-5-haiku-latest",
    instructions="You are a helpful assistant with access to tools.",
    stream_handler=handle_chunk
)
```

## Advanced Tool Calling

For advanced tool calling scenarios, particularly with Anthropic Claude models, see the following guides:

- [Claude Tools Guide](claude_tools.md) - Covers best practices for Claude tools
- [Anthropic Tool Calling](anthropic/tool_calling.md) - Detailed implementation guide

These guides provide important information about Claude's tool format, schema requirements, and debugging tools.
