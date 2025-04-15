import json
import logging
from typing import List, Dict, Any, Optional, Union

from anthropic import AsyncAnthropic

from llm.tool_handling import prepare_tools_for_api
from llm.tooling import ToolRegistry
from llm.types import Message, LLMResponse, StreamHandler

logger = logging.getLogger(__name__)


async def handle_anthropic_api(
        client: AsyncAnthropic,
        user_input: Union[str, List[Message]],
        model: str,
        instructions: Optional[str],
        tools: Optional[List[Dict[str, Any]]],
        tool_registry: ToolRegistry,
        temperature: float,
        max_tokens: int,
        anthropic_tool_debug: bool = False,
        **kwargs
) -> LLMResponse:
    """
    Handle interactions with Anthropic's API including tool calling.

    Args:
        client: The AsyncAnthropic client
        user_input: Either a string or a list of message objects
        model: Model identifier (e.g., "claude-3-opus-20240229")
        instructions: System instructions for the model
        tools: Tool definitions for function calling
        tool_registry: Registry of available tools
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate
        anthropic_tool_debug: Whether to print tool debugging information
        **kwargs: Additional parameters to pass to Anthropic's API

    Returns:
        LLMResponse containing the model's response and token usage
    """
    # Convert user input to Anthropic's expected format
    if isinstance(user_input, str):
        # Simple text query
        system = instructions or ""
        messages = [{"role": "user", "content": user_input}]
        # Log user prompt for debugging
        print(f"User prompt: {user_input}")
    else:
        # Convert conversation history
        # Handle system message separately
        system = None
        messages = []
        
        for msg in user_input:
            if msg.get("role") == "system" or msg.get("role") == "developer":
                # If we find a system message, use it as the system prompt
                # If multiple system messages exist, the last one will be used
                system = msg.get("content", "")
            elif msg.get("role") == "user":
                messages.append({"role": "user", "content": msg.get("content", "")})
                # Log user prompt for debugging
                print(f"User prompt: {msg.get('content', '')}")
            elif msg.get("role") == "assistant":
                messages.append({"role": "assistant", "content": msg.get("content", "")})
            elif msg.get("role") == "tool":
                # Tool responses need to be handled specially
                # They should be attached to the last assistant message
                tool_call_id = msg.get("tool_call_id", "unknown")
                content = msg.get("content", "")

                # Find the most recent assistant message
                for i in range(len(messages) - 1, -1, -1):
                    if messages[i].get("role") == "assistant":
                        # Attach tool response to this message
                        # Note: This assumes Claude will properly handle this format
                        # May need updates as Claude's API evolves
                        if "tool_responses" not in messages[i]:
                            messages[i]["tool_responses"] = []

                        messages[i]["tool_responses"].append({
                            "tool_call_id": tool_call_id,
                            "content": content
                        })
                        break

        # If no system message was found, use the provided instructions
        if system is None:
            system = instructions or ""

    # Prepare properly formatted tools for Anthropic API
    api_tools = prepare_tools_for_api(tools, 'anthropic') if tools else None

    # Debug output for tool schema conversion
    if anthropic_tool_debug and api_tools:
        logger.info(f"Converted tool schemas for Anthropic:")
        for tool in api_tools:
            logger.info(f"Tool: {tool['name']}")
            logger.info(f"Description: {tool.get('description', '')}")
            logger.info(f"Input Schema: {json.dumps(tool['input_schema'], indent=2)}")

    # Prepare request parameters
    request_params = {
        "model": model,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "system": system,
        "messages": messages,
        **{k: v for k, v in kwargs.items() if k != 'tools'}  # Remove any tools from kwargs
    }

    # Add tools if available
    if api_tools:
        request_params["tools"] = api_tools

    # Make the API call
    logger.info(
        f"Making Anthropic API call with {len(messages)} messages and {len(api_tools) if api_tools else 0} tools")
    response = await client.messages.create(**request_params)

    # Extract tool calls if any and process them
    response_text = response.content[0].text

    # Process tool calls if they exist
    if hasattr(response, 'tool_calls') and response.tool_calls:
        logger.info(f"Processing {len(response.tool_calls)} tool calls from Claude")

        tool_inputs = {}

        # First, process all tool calls
        for tool_call in response.tool_calls:
            try:
                # Print tool call info for debugging
                if anthropic_tool_debug:
                    logger.info(f"Tool call ID: {tool_call.id}")
                    logger.info(f"Tool: {tool_call.name}")
                    logger.info(f"Input: {json.dumps(tool_call.input, indent=2)}")

                tool_name = tool_call.name
                tool_input = tool_call.input

                # Log the tool input
                print(f"Tool input for {tool_name}: {tool_input}")

                # Execute the tool if it exists
                if tool_registry.has_tool(tool_name):
                    result = await tool_registry.execute_tool(tool_name, tool_input)

                    # Store the result for this tool call
                    tool_inputs[tool_call.id] = {
                        "tool_call_id": tool_call.id,
                        "output": result
                    }

                    if anthropic_tool_debug:
                        logger.info(f"Tool result: {result}")
                else:
                    error = f"Tool '{tool_name}' not found in registry"
                    logger.error(error)
                    tool_inputs[tool_call.id] = {
                        "tool_call_id": tool_call.id,
                        "output": {"error": error}
                    }
            except Exception as e:
                logger.error(f"Error executing tool {tool_call.name}: {e}")
                tool_inputs[tool_call.id] = {
                    "tool_call_id": tool_call.id,
                    "output": {"error": str(e)}
                }

        # If we have tool results, make a follow-up request with them
        if tool_inputs:
            # Prepare tool outputs for follow-up request
            tool_results = [
                {
                    "tool_call_id": tool_data["tool_call_id"],
                    "output": json.dumps(tool_data["output"]) if isinstance(tool_data["output"], dict) else str(
                        tool_data["output"])
                }
                for tool_data in tool_inputs.values()
            ]

            # Make follow-up request with tool results
            follow_up_params = {
                "model": model,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "system": system,
                "messages": messages + [{"role": "assistant", "content": response_text}],
                "tool_results": tool_results,
                **{k: v for k, v in kwargs.items() if k != 'tools'}  # Remove any tools from kwargs
            }

            # Add tools if they should be available for follow-up
            if api_tools:
                follow_up_params["tools"] = api_tools

            logger.info(f"Making follow-up Anthropic API call with {len(tool_results)} tool results")
            follow_up_response = await client.messages.create(**follow_up_params)

            # Return the follow-up response
            return {
                "text": follow_up_response.content[0].text,
                "input_tokens": follow_up_response.usage.input_tokens,
                "output_tokens": follow_up_response.usage.output_tokens,
                "response_id": follow_up_response.id,
                "sources": None
            }

    # Return the response if no tools were called
    return {
        "text": response_text,
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens,
        "response_id": response.id,
        "sources": None
    }


async def stream_anthropic_api(
        client: AsyncAnthropic,
        user_input: Union[str, List[Message]],
        model: str,
        instructions: Optional[str],
        tools: Optional[List[Dict[str, Any]]],
        tool_registry: ToolRegistry,
        temperature: float,
        max_tokens: int,
        stream_handler: StreamHandler,
        **kwargs
) -> LLMResponse:
    """
    Stream responses from Anthropic's API.

    Args:
        client: The AsyncAnthropic client
        user_input: Either a string or a list of message objects
        model: Model identifier (e.g., "claude-3-opus-20240229")
        instructions: System instructions for the model
        tools: Tool definitions for function calling
        tool_registry: Registry of available tools
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate
        stream_handler: Callback function to handle each chunk of the stream
        **kwargs: Additional parameters to pass to Anthropic's API

    Returns:
        LLMResponse containing the model's complete response and token usage
    """
    # Convert user input to Anthropic's expected format
    if isinstance(user_input, str):
        # Simple text query
        system = instructions or ""
        messages = [{"role": "user", "content": user_input}]
    else:
        # Convert conversation history
        # Handle system message separately
        system = None
        messages = []

        for msg in user_input:
            if msg.get("role") == "system" or msg.get("role") == "developer":
                # If we find a system message, use it as the system prompt
                # If multiple system messages exist, the last one will be used
                system = msg.get("content", "")
            elif msg.get("role") == "user":
                messages.append({"role": "user", "content": msg.get("content", "")})
            elif msg.get("role") == "assistant":
                messages.append({"role": "assistant", "content": msg.get("content", "")})
            elif msg.get("role") == "tool":
                # Tool responses need to be handled specially
                # They should be attached to the last assistant message
                tool_call_id = msg.get("tool_call_id", "unknown")
                content = msg.get("content", "")

                # Find the most recent assistant message
                for i in range(len(messages) - 1, -1, -1):
                    if messages[i].get("role") == "assistant":
                        # Attach tool response to this message
                        if "tool_responses" not in messages[i]:
                            messages[i]["tool_responses"] = []

                        messages[i]["tool_responses"].append({
                            "tool_call_id": tool_call_id,
                            "content": content
                        })
                        break

        # If no system message was found, use the provided instructions
        if system is None:
            system = instructions or ""

    # Prepare properly formatted tools for Anthropic API
    api_tools = prepare_tools_for_api(tools, 'anthropic') if tools else None

    # Prepare request parameters
    request_params = {
        "model": model,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "system": system,
        "messages": messages,
        "stream": True,  # Enable streaming
        **{k: v for k, v in kwargs.items() if k != 'tools'}  # Remove any tools from kwargs
    }

    # Add tools if available
    if api_tools:
        request_params["tools"] = api_tools

    # Make the API call with streaming
    logger.info(f"Making streaming Anthropic API call")
    with_stream = await client.messages.create(**request_params)

    # Process the stream
    full_text = ""
    input_tokens = 0
    output_tokens = 0
    response_id = None

    async for chunk in with_stream:
        if hasattr(chunk, 'type') and chunk.type == 'content_block_delta':
            if hasattr(chunk, 'delta') and hasattr(chunk.delta, 'text'):
                content = chunk.delta.text
                full_text += content
                await stream_handler(content)

        # Get token usage from the chunk if available
        if hasattr(chunk, 'usage'):
            if hasattr(chunk.usage, 'input_tokens'):
                input_tokens = chunk.usage.input_tokens
            if hasattr(chunk.usage, 'output_tokens'):
                output_tokens = chunk.usage.output_tokens

        # Get response ID if available
        if hasattr(chunk, 'message') and hasattr(chunk.message, 'id'):
            response_id = chunk.message.id

    # If we didn't get token counts from streaming chunks, make a non-streaming call to get them
    if input_tokens == 0 or output_tokens == 0:
        try:
            # Make a quick non-streaming call with the same parameters
            request_params["stream"] = False
            request_params["max_tokens"] = 1  # Minimize token usage
            non_stream_resp = await client.messages.create(**request_params)
            input_tokens = non_stream_resp.usage.input_tokens
            # Estimate output tokens based on text length
            output_tokens = len(full_text) // 4  # Rough estimate: ~4 chars per token
        except Exception as e:
            logger.warning(f"Failed to estimate token counts: {e}")
            # Fallback to very rough estimates
            input_tokens = sum(len(m.get("content", "")) for m in messages) // 4
            output_tokens = len(full_text) // 4

    return {
        "text": full_text,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "response_id": response_id,
        "sources": None
    }
