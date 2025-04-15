import json
import logging
from typing import List, Dict, Any, Union, Optional

from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionChunk

from llm.tool_handling import prepare_tools_for_api
from llm.tooling import ToolRegistry
from llm.types import Message, LLMResponse, StreamHandler

logger = logging.getLogger(__name__)


def prepare_messages(user_input, instructions=None):
    """
    Prepare messages based on user input type.

    Args:
        user_input: Either a string or list of Message objects
        instructions: Optional system instructions

    Returns:
        List of message dictionaries formatted for API calls
    """
    messages = []

    # Add system message if instructions are provided
    if instructions:
        messages.append({"role": "system", "content": instructions})

    # Process based on input type
    if isinstance(user_input, str):
        messages.append({"role": "user", "content": user_input})
    else:
        # Convert Message objects to the format expected by the APIs
        for msg in user_input:
            # Skip system messages if we already added instructions
            if msg.get("role") == "system" and any(m.get("role") == "system" for m in messages):
                continue

            # Map developer role to system for compatibility
            if msg.get("role") == "developer":
                messages.append({"role": "system", "content": msg.get("content", "")})
            else:
                messages.append(msg)

    return messages


async def handle_chat_completions_api(
        client: AsyncOpenAI,
        tool_registry: ToolRegistry,
        user_input: Union[str, List[Message]],
        model: str,
        instructions: Optional[str],
        tools: Optional[List[Dict[str, Any]]],
        temperature: float,
        max_tokens: int,
        current_tool_call_depth: int = 0,
        max_tool_call_depth: int = 3,
        **kwargs
) -> LLMResponse:
    """Handle interactions with OpenAI's Chat Completions API including tool calling."""
    # Check if we've reached the maximum tool call depth
    if current_tool_call_depth > max_tool_call_depth:
        logger.warning(f"Maximum tool call depth ({max_tool_call_depth}) reached. Stopping recursion.")
        return {
            "text": "I've reached the maximum number of tool calls I can make for this request. Please provide more specific instructions if needed.",
            "input_tokens": 0,
            "output_tokens": 0,
            "response_id": None
        }

    # Prepare messages for Chat Completions API
    messages = prepare_messages(user_input, instructions)

    # Prepare properly formatted tools for Chat Completions API
    api_tools = prepare_tools_for_api(tools, 'completions')

    # Prepare the parameters for the Chat Completions API
    completion_params = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        **{k: v for k, v in kwargs.items() if k != 'tools'}  # Remove any tools from kwargs
    }

    # Only add tools if we have properly formatted function tools
    if api_tools:
        completion_params["tools"] = api_tools

    # Make the API call
    logger.info(f"Making Chat Completions API call with tools: {bool(api_tools)}")
    completion = await client.chat.completions.create(**completion_params)

    # Check for tool calls in the response
    choice = completion.choices[0]
    message = choice.message

    if message.tool_calls and current_tool_call_depth < max_tool_call_depth:
        logger.info(f"Processing tool calls at depth {current_tool_call_depth + 1}/{max_tool_call_depth}")

        # First, add the assistant message with tool_calls to the messages
        # This MUST be done before processing tool responses to ensure IDs match
        assistant_message = {
            "role": "assistant",
            "content": message.content,
            "tool_calls": message.tool_calls
        }
        messages.append(assistant_message)

        # Create a dictionary of tool call IDs for easy lookup
        tool_call_ids = {tc.id: tc for tc in message.tool_calls}
        logger.info(f"Tool call IDs in assistant message: {list(tool_call_ids.keys())}")

        # Process tool calls
        tool_responses = []
        for tool_call in message.tool_calls:
            try:
                function_name = tool_call.function.name
                arguments_json = tool_call.function.arguments
                tool_call_id = tool_call.id

                logger.info(f"Processing tool call: {function_name} with ID {tool_call_id}")

                if not tool_registry.has_tool(function_name):
                    error_message = f"Error: Tool '{function_name}' not found in registry"
                    logger.error(error_message)
                    tool_responses.append({
                        "tool_call_id": tool_call_id,
                        "output": error_message
                    })
                    continue

                try:
                    # Parse arguments JSON
                    args = json.loads(arguments_json)

                    # Execute the tool
                    result = await tool_registry.execute_tool(function_name, args)

                    # Format the result
                    if isinstance(result, dict):
                        formatted_result = json.dumps(result)
                    else:
                        formatted_result = str(result)

                    tool_responses.append({
                        "tool_call_id": tool_call_id,
                        "output": formatted_result
                    })
                    logger.info(f"Tool executed successfully: {function_name}")
                except json.JSONDecodeError as e:
                    error_message = f"Invalid JSON arguments for tool {function_name}: {e}"
                    logger.error(error_message)
                    tool_responses.append({
                        "tool_call_id": tool_call_id,
                        "output": error_message
                    })
                except Exception as e:
                    error_message = f"Error executing tool {function_name}: {str(e)}"
                    logger.error(error_message)
                    tool_responses.append({
                        "tool_call_id": tool_call_id,
                        "output": error_message
                    })
            except Exception as e:
                error_message = f"Unexpected error processing tool call: {str(e)}"
                logger.error(error_message)
                tool_call_id = getattr(tool_call, 'id', 'unknown_id')
                tool_responses.append({
                    "tool_call_id": tool_call_id,
                    "output": error_message
                })

        # Add tool responses to messages
        for tool_response in tool_responses:
            # Verify that the tool_call_id exists in the assistant's tool_calls
            if tool_response["tool_call_id"] not in tool_call_ids:
                logger.warning(
                    f"Tool call ID {tool_response['tool_call_id']} not found in assistant message tool_calls")
                continue

            messages.append({
                "role": "tool",
                "content": tool_response["output"],
                "tool_call_id": tool_response["tool_call_id"]
            })

        # Make a recursive call with the updated messages and incremented depth
        return await handle_chat_completions_api(
            client=client,
            tool_registry=tool_registry,
            user_input=messages,
            model=model,
            instructions=None,  # Already included in messages
            tools=tools,
            temperature=temperature,
            max_tokens=max_tokens,
            current_tool_call_depth=current_tool_call_depth + 1,
            max_tool_call_depth=max_tool_call_depth,
            **{k: v for k, v in kwargs.items() if k != 'tools'}
        )

    # If no tool calls or max depth reached, return the response as is
    return {
        "text": message.content,
        "input_tokens": completion.usage.prompt_tokens,
        "output_tokens": completion.usage.completion_tokens,
        "response_id": None  # Chat Completions API doesn't provide a response ID
    }


async def stream_chat_completions_api(
        client: AsyncOpenAI,
        tool_registry: ToolRegistry,
        user_input: Union[str, List[Message]],
        model: str,
        instructions: Optional[str],
        tools: Optional[List[Dict[str, Any]]],
        temperature: float,
        max_tokens: int,
        stream_handler: StreamHandler,
        **kwargs
) -> LLMResponse:
    """Stream responses from OpenAI's Chat Completions API."""
    # Prepare messages for Chat Completions API
    messages = prepare_messages(user_input, instructions)

    # Prepare properly formatted tools for Chat Completions API
    api_tools = prepare_tools_for_api(tools, 'completions')

    # Prepare the parameters for the Chat Completions API
    completion_params = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": True,  # Enable streaming
        **{k: v for k, v in kwargs.items() if k not in ['tools']}  # Remove any tools from kwargs
    }

    # Only add tools if we have properly formatted function tools
    if api_tools:
        completion_params["tools"] = api_tools

    # Make the API call with streaming
    logger.info(f"Making streaming Chat Completions API call with tools: {bool(api_tools)}")
    stream = await client.chat.completions.create(**completion_params)

    # Process the stream
    full_text = ""
    input_tokens = 0
    output_tokens = 0

    async for chunk in stream:
        chunk: ChatCompletionChunk
        if chunk.choices and len(chunk.choices) > 0:
            if chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                full_text += content
                await stream_handler(content)

            # Update token counts if available
            if hasattr(chunk, 'usage') and chunk.usage:
                if hasattr(chunk.usage, 'prompt_tokens') and chunk.usage.prompt_tokens:
                    input_tokens = chunk.usage.prompt_tokens
                if hasattr(chunk.usage, 'completion_tokens') and chunk.usage.completion_tokens:
                    output_tokens = chunk.usage.completion_tokens

    # If we didn't get token counts from streaming chunks, estimate them
    if input_tokens == 0 or output_tokens == 0:
        # Make a non-streaming call to get token counts
        # This is a fallback and isn't ideal, but OpenAI doesn't always
        # include usage info in streaming responses
        try:
            completion_params["stream"] = False
            completion_params["max_tokens"] = 1  # Minimize token usage for this call
            non_stream_resp = await client.chat.completions.create(**completion_params)
            input_tokens = non_stream_resp.usage.prompt_tokens
            # We can't get an accurate output token count this way, so estimate based on text length
            # This is very rough but better than nothing
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
        "response_id": None  # Chat Completions API doesn't provide a response ID
    }
