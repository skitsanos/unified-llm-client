import json
import logging
import os
from typing import List, Optional, Dict, Any, Union, overload

from anthropic import AsyncAnthropic
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageToolCall
from openai.types.responses import ResponseFunctionToolCall

from llm.anthropic import handle_anthropic_api, stream_anthropic_api
from llm.chat_completions import handle_chat_completions_api, prepare_messages, stream_chat_completions_api
from llm.responses_api import handle_responses_api
from llm.streaming_responses import stream_responses_api
from llm.tooling import ToolRegistry
from llm.types import Message, LLMResponse, ModelProvider, ToolCallResponse, StreamHandler

logger = logging.getLogger(__name__)


class AsyncLLMClient:
    """Async client for interacting with various LLM providers including OpenAI, Anthropic, and Ollama."""

    def __init__(
            self,
            base_url: Optional[str] = None,
            api_key: Optional[str] = None,
            tool_registry: Optional[ToolRegistry] = None,
            max_tool_call_depth: int = 3
    ) -> None:
        """
        Initialize async LLM clients.

        Args:
            base_url: Optional base URL for the OpenAI API
            api_key: Optional API key (falls back to environment variables)
            tool_registry: Optional tool registry for function calling
            max_tool_call_depth: Maximum depth for recursive tool calls
        """
        self.openai_client = AsyncOpenAI(
            base_url=base_url,
            api_key=api_key or os.getenv("OPENAI_API_KEY")
        )
        self.anthropic_client = AsyncAnthropic(
            api_key=api_key or os.getenv("ANTHROPIC_API_KEY")
        )

        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY") or os.getenv("OPENAI_API_KEY")
        self.tool_registry = tool_registry or ToolRegistry()
        self.max_tool_call_depth = max_tool_call_depth

    async def handle_tool_calls(
            self,
            tool_calls: Union[List[ResponseFunctionToolCall], List[ChatCompletionMessageToolCall]]
    ) -> List[ToolCallResponse]:
        """
        Process tool calls from either OpenAI Responses API or Chat Completions API.

        Args:
            tool_calls: List of tool calls from either API format

        Returns:
            List of tool responses with tool_call_id and output
        """
        tools_responses: List[ToolCallResponse] = []

        for tool_call in tool_calls:
            try:
                if isinstance(tool_call, ResponseFunctionToolCall):
                    # Handle OpenAI Responses API tool calls
                    function_name = tool_call.name
                    arguments_json = tool_call.arguments
                    tool_call_id = tool_call.id
                else:
                    # Handle Chat Completions API tool calls
                    function_name = tool_call.function.name
                    arguments_json = tool_call.function.arguments
                    tool_call_id = tool_call.id

                logger.info(f"Handling tool call: {function_name}")

                if not self.tool_registry.has_tool(function_name):
                    error_message = f"Error: Tool '{function_name}' not found in registry"
                    logger.error(error_message)
                    tools_responses.append({
                        "tool_call_id": tool_call_id,
                        "output": error_message
                    })
                    continue

                try:
                    # Parse arguments JSON
                    args = json.loads(arguments_json)

                    # Execute the tool with the arguments
                    result = await self.tool_registry.execute_tool(function_name, args)

                    # Format the result
                    if isinstance(result, dict):
                        formatted_result = json.dumps(result)
                    else:
                        formatted_result = str(result)

                    tools_responses.append({
                        "tool_call_id": tool_call_id,
                        "output": formatted_result
                    })
                    logger.info(f"Tool processed: {function_name}")
                except json.JSONDecodeError as e:
                    error_message = f"Invalid JSON arguments for tool {function_name}: {e}"
                    logger.error(error_message)
                    tools_responses.append({
                        "tool_call_id": tool_call_id,
                        "output": error_message
                    })
                except Exception as e:
                    error_message = f"Error executing tool {function_name}: {str(e)}"
                    logger.error(f"{error_message}\nArguments: {arguments_json}")
                    tools_responses.append({
                        "tool_call_id": tool_call_id,
                        "output": error_message
                    })
            except Exception as e:
                # Catch-all for any unexpected errors in tool call processing
                error_message = f"Unexpected error processing tool call: {str(e)}"
                logger.error(error_message)
                try:
                    # Try to get tool_call_id if possible
                    tool_call_id = getattr(tool_call, 'id', 'unknown_id')
                    tools_responses.append({
                        "tool_call_id": tool_call_id,
                        "output": error_message
                    })
                except:
                    # Last resort if we can't even get the ID
                    tools_responses.append({
                        "tool_call_id": "error_processing_id",
                        "output": error_message
                    })

        return tools_responses

    @overload
    async def stream(
            self,
            user_input: str,
            model: str = "gpt-4o-mini",
            instructions: Optional[str] = None,
            tools: Optional[List[Dict[str, Any]]] = None,
            temperature: float = 0.0,
            max_tokens: int = 4096,
            use_responses_api: bool = True,
            stream_handler: Optional[StreamHandler] = None,
            **kwargs
    ) -> LLMResponse:
        ...

    @overload
    async def stream(
            self,
            user_input: List[Message],
            model: str = "gpt-4o-mini",
            instructions: Optional[str] = None,
            tools: Optional[List[Dict[str, Any]]] = None,
            temperature: float = 0.0,
            max_tokens: int = 4096,
            use_responses_api: bool = True,
            stream_handler: Optional[StreamHandler] = None,
            **kwargs
    ) -> LLMResponse:
        ...

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
        instructions = instructions or None

        # Get model provider to use appropriate client
        provider: ModelProvider = self._detect_provider(model)

        # Get the appropriate tools for the model (but only if tools not explicitly provided)
        provider_tools = None
        if not tools:
            if provider == "anthropic":
                provider_tools = self.tool_registry.get_schemas("anthropic") if self.tool_registry else None
            else:  # openai or ollama
                provider_tools = self.tool_registry.get_schemas("openai") if self.tool_registry else None
        else:
            # Use explicitly provided tools (for backward compatibility)
            provider_tools = tools

        # Log tools summary for debugging
        tools_count = len(provider_tools) if provider_tools else 0
        logger.info(f"Using {tools_count} tools from {'parameter' if tools else 'registry'}")

        # Set default stream handler if none provided
        if not stream_handler:
            stream_handler = lambda chunk: None

        try:
            if provider == "anthropic":
                return await stream_anthropic_api(
                    client=self.anthropic_client,
                    user_input=user_input,
                    model=model,
                    instructions=instructions,
                    tools=provider_tools,
                    tool_registry=self.tool_registry,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    stream_handler=stream_handler,
                    **kwargs
                )
            elif provider in ("openai", "ollama"):
                if provider == "ollama":
                    # Configure client to use Ollama API endpoint if not already set
                    if not hasattr(self, '_using_ollama') or not self._using_ollama:
                        base_url = kwargs.get('base_url') or "http://localhost:11434/v1"
                        self.openai_client = AsyncOpenAI(
                            base_url=base_url,
                            api_key="ollama"  # Ollama doesn't require a real API key
                        )
                        self._using_ollama = True

                if use_responses_api:
                    return await stream_responses_api(
                        client=self.openai_client,
                        user_input=user_input,
                        model=model,
                        instructions=instructions,
                        tools=provider_tools,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        stream_handler=stream_handler,
                        **kwargs
                    )
                else:
                    return await stream_chat_completions_api(
                        client=self.openai_client,
                        user_input=user_input,
                        model=model,
                        instructions=instructions,
                        tools=provider_tools,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        stream_handler=stream_handler,
                        **kwargs
                    )
            else:
                raise ValueError(f"Unsupported model: {model}")
        except Exception as e:
            # Re-raise with more context
            raise Exception(f"Error streaming response from {model}: {str(e)}") from e

    @overload
    async def response(
            self,
            user_input: str,
            model: str = "gpt-4o-mini",
            instructions: Optional[str] = None,
            tools: Optional[List[Dict[str, Any]]] = None,
            temperature: float = 0.0,
            max_tokens: int = 4096,
            use_responses_api: bool = True,
            previous_response_id: Optional[str] = None,
            **kwargs
    ) -> LLMResponse: ...

    @overload
    async def response(
            self,
            user_input: List[Message],
            model: str = "gpt-4o-mini",
            instructions: Optional[str] = None,
            tools: Optional[List[Dict[str, Any]]] = None,
            temperature: float = 0.0,
            max_tokens: int = 4096,
            use_responses_api: bool = True,
            previous_response_id: Optional[str] = None,
            **kwargs
    ) -> LLMResponse: ...

    async def response(
            self,
            user_input: Union[str, List[Message]],
            model: str = "gpt-4o-mini",
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
            model: Model identifier (e.g., "claude-3-opus-20240229", "gpt-4o-mini")
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
        instructions = instructions or None

        # Get model provider to use appropriate client
        provider: ModelProvider = self._detect_provider(model)

        # Get the appropriate tools for the model (but only if tools not explicitly provided)
        provider_tools = None
        if not tools:
            if provider == "anthropic":
                provider_tools = self.tool_registry.get_schemas("anthropic") if self.tool_registry else None
            else:  # openai or ollama
                provider_tools = self.tool_registry.get_schemas("openai") if self.tool_registry else None
        else:
            # Use explicitly provided tools (for backward compatibility)
            provider_tools = tools

        # Log tools summary for debugging
        tools_count = len(provider_tools) if provider_tools else 0
        logger.info(f"Using {tools_count} tools from {'parameter' if tools else 'registry'}")

        try:
            if provider == "anthropic":
                return await handle_anthropic_api(
                    client=self.anthropic_client,
                    user_input=user_input,
                    model=model,
                    instructions=instructions,
                    tools=provider_tools,
                    tool_registry=self.tool_registry,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    anthropic_tool_debug=kwargs.pop('anthropic_tool_debug', False),
                    **kwargs
                )
            elif provider in ("openai", "ollama"):
                if provider == "ollama":
                    # Configure client to use Ollama API endpoint if not already set
                    if not hasattr(self, '_using_ollama') or not self._using_ollama:
                        base_url = kwargs.get('base_url') or "http://localhost:11434/v1"
                        self.openai_client = AsyncOpenAI(
                            base_url=base_url,
                            api_key="ollama"  # Ollama doesn't require a real API key
                        )
                        self._using_ollama = True
                
                if use_responses_api:
                    return await handle_responses_api(
                        client=self.openai_client,
                        tool_registry=self.tool_registry,
                        user_input=user_input,
                        model=model,
                        instructions=instructions,
                        tools=provider_tools,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        previous_response_id=previous_response_id,
                        current_tool_call_depth=0,
                        max_tool_call_depth=self.max_tool_call_depth,
                        **kwargs
                    )
                else:
                    return await handle_chat_completions_api(
                        client=self.openai_client,
                        tool_registry=self.tool_registry,
                        user_input=user_input,
                        model=model,
                        instructions=instructions,
                        tools=provider_tools,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        current_tool_call_depth=0,
                        max_tool_call_depth=self.max_tool_call_depth,
                        **kwargs
                    )
            else:
                raise ValueError(f"Unsupported model: {model}")
        except Exception as e:
            # Re-raise with more context
            raise Exception(f"Error getting response from {model}: {str(e)}") from e

    def _prepare_messages(self, user_input: Union[str, List[Message]], instructions: Optional[str] = None) -> List[Dict[str, str]]:
        """Prepare messages based on user input type."""
        return prepare_messages(user_input, instructions)
        
    def _detect_provider(self, model: str) -> ModelProvider:
        """
        Detect the provider based on the model name.
        
        Args:
            model: The model name/identifier
            
        Returns:
            The detected provider (anthropic, openai, or ollama)
        """
        if model.startswith(("claude", "anthropic")):
            return "anthropic"
        elif any(model.startswith(prefix) for prefix in ["gpt", "o1", "o3", "text-", "dall-e"]):
            return "openai"
        elif any([
            model.startswith(("llama", "qwen", "mistral", "phi", "gemma", "mixtral")),
            self.api_key == "ollama"
        ]):
            return "ollama"
        else:
            # Default to OpenAI if we can't determine
            logger.warning(f"Could not determine provider for model {model}, defaulting to OpenAI")
            return "openai"
