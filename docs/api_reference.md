# API Reference

## AsyncLLMClient

The main client for interacting with LLM providers.

```python
class AsyncLLMClient:
    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        tool_registry: Optional[ToolRegistry] = None,
        max_tool_call_depth: int = 3
    ):
        """
        Initialize async LLM clients.

        Args:
            base_url: Optional base URL for the OpenAI API
            api_key: Optional API key (falls back to environment variables)
            tool_registry: Optional tool registry for function calling
            max_tool_call_depth: Maximum depth for recursive tool calls
        """
        # ...

    async def response(
        self,
        user_input: Union[str, List[Message]],
        model: str,
        instructions: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        use_responses_api: bool = True,
        previous_response_id: Optional[str] = None,
        **kwargs
    ) -> LLMResponse:
        """
        Get a response from an LLM asynchronously.

        Args:
            user_input: Either a string or a list of message objects with role and content
            model: Model identifier (e.g., "claude-3-opus-20240229", "gpt-4o")
            instructions: System instructions for the model
            tools: Tool definitions for function calling (deprecated, use tool_registry instead)
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            use_responses_api: Whether to use OpenAI's responses API (vs. chat completions)
            previous_response_id: ID of the previous response (only for OpenAI Responses API)
            **kwargs: Additional parameters to pass to the API

        Returns:
            LLMResponse object containing:
                - text: The LLM's response text
                - input_tokens: Number of input tokens used
                - output_tokens: Number of output tokens used
                - response_id: ID of the response (only for OpenAI Responses API, otherwise None)
                - sources: List of sources (if available)

        Raises:
            ValueError: If the model name is not recognized
            Exception: If the API call fails
        """
        # ...
        
    async def stream(
        self,
        user_input: Union[str, List[Message]],
        model: str = "gpt-4o-mini",
        instructions: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        use_responses_api: bool = True,
        stream_handler: Optional[StreamHandler] = None,
        **kwargs
    ) -> LLMResponse:
        """
        Stream a response from an LLM asynchronously.

        Args:
            user_input: Either a string or a list of message objects with role and content
            model: Model identifier (e.g., "claude-3-opus-20240229", "gpt-4o-mini")
            instructions: System instructions for the model
            tools: Tool definitions for function calling (deprecated, use tool_registry instead)
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            use_responses_api: Whether to use OpenAI's responses API (vs. chat completions)
            stream_handler: Optional callback function to handle each chunk of the stream
            **kwargs: Additional parameters to pass to the API

        Returns:
            LLMResponse object containing:
                - text: The LLM's complete response text
                - input_tokens: Number of input tokens used
                - output_tokens: Number of output tokens used
                - response_id: ID of the response (only for OpenAI Responses API, otherwise None)
                - sources: List of sources (if available)

        Raises:
            ValueError: If the model name is not recognized
            Exception: If the API call fails
        """
        # ...
```

## ToolRegistry

Registry for managing tools/functions that can be called by LLM models.

```python
class ToolRegistry:
    def __init__(self, internal_tool_types: Set[str] = None):
        """
        Initialize the tool registry.

        Args:
            internal_tool_types: Set of tool types handled internally by LLM providers
        """
        # ...

    def register(self, name, func):
        """
        Register a tool function with its schema.

        Args:
            name: Name of the tool
            func: Tool function (should have the llm_tool decorator)

        Returns:
            Self for chaining
        """
        # ...

    def unregister(self, name):
        """
        Unregister a tool function.

        Args:
            name: Name of the tool to unregister

        Returns:
            Self for chaining
        """
        # ...

    async def execute_tool(self, name, args):
        """
        Execute a tool by name with the given arguments.
        Supports both synchronous and asynchronous tool functions.

        Args:
            name: Tool name
            args: Arguments to pass to the tool function

        Returns:
            Result from the tool function

        Raises:
            KeyError: If tool is not registered
            ValueError: If tool is an internal tool that should be handled by the LLM provider
            Exception: If tool execution fails
        """
        # ...
```

## Types

```python
class Message(TypedDict):
    """Structure for a single message in a conversation."""
    role: Literal["user", "assistant", "system", "developer"]
    content: str


class LLMResponse(TypedDict):
    """Type definition for the LLM response object."""
    text: str
    input_tokens: int
    output_tokens: int
    response_id: Optional[str]
    sources: Optional[list[str]]
    
# Stream handler type
StreamHandler = Callable[[str], Awaitable[None]]
"""Type definition for a stream handler function that processes chunks of streamed responses."""
```

## llm_tool Decorator

```python
def llm_tool(func):
    """
    Generic decorator to register a function as a tool for multiple LLM providers.

    This decorator inspects the function signature and creates schema definitions
    for OpenAI, Anthropic, and potentially other LLM providers.

    Args:
        func: The function to register as a tool

    Returns:
        The decorated function with LLM-specific tool schemas attached
    """
    # ...
```
