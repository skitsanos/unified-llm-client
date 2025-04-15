import logging
from typing import List, Dict, Any, Union, Optional

from anthropic import AsyncAnthropic

from llm.tool_handling import prepare_tools_for_api
from llm.tooling import ToolRegistry
from llm.types import Message, LLMResponse

logger = logging.getLogger(__name__)


async def handle_anthropic_api(
        client: AsyncAnthropic,
        tool_registry: ToolRegistry,
        user_input: Union[str, List[Message]],
        model: str,
        instructions: Optional[str],
        tools: Optional[List[Dict[str, Any]]],
        temperature: float,
        max_tokens: int,
        **kwargs
) -> LLMResponse:
    """Handle interactions with Anthropic's Claude models with advanced tool handling."""
    # Extract system prompt from messages or use provided instructions
    system_prompt = instructions or "You are a helpful assistant."

    # Prepare messages for Anthropic API (excluding system messages)
    anthropic_messages = []

    # Handle string input
    if isinstance(user_input, str):
        anthropic_messages = [
            {"role": "user", "content": user_input}
        ]
    else:
        # Handle message list - extract system message if present
        # Filter out tool messages since Anthropic doesn't support them
        for msg in user_input:
            role = msg.get("role")
            content = msg.get("content", "")

            # Extract system message if present
            if role == "system" or role == "developer":
                if content and not instructions:  # Only override if instructions not provided
                    system_prompt = content
            elif role == "user" or role == "assistant":
                # Include only user and assistant messages
                anthropic_messages.append({"role": role, "content": content})
            # Skip tool messages completely

    # Prepare parameters for Anthropic API
    anthropic_params = {
        "model": model,
        "messages": anthropic_messages,
        "system": system_prompt,  # Use top-level system parameter
        "temperature": temperature,
        "max_tokens": max_tokens,
        **{k: v for k, v in kwargs.items() if k not in ['tools', 'tool_choice']}  # Remove tools and tool_choice
    }

    # Only add tools if we have properly formatted ones
    if tools:
        # Prepare tools with Anthropic format
        formatted_tools = prepare_tools_for_api(tools, 'anthropic')
        if formatted_tools:
            anthropic_params["tools"] = formatted_tools
            logger.info(f"Sending {len(formatted_tools)} tools to Anthropic API")

    # Log parameters for debugging
    logger.debug(f"Anthropic API parameters: {anthropic_params}")

    # Call Anthropic API
    try:
        message = await client.messages.create(**anthropic_params)

        # Process tool use in the response if present
        final_text = ""
        tool_use_blocks = []

        for content_block in message.content:
            if content_block.type == "text":
                final_text += content_block.text
            elif content_block.type == "tool_use":
                # Collect tool use blocks to process
                tool_use_blocks.append(content_block)

        # If no tool use blocks, return the response as is
        if not tool_use_blocks:
            return {
                "text": final_text,
                "input_tokens": message.usage.input_tokens,
                "output_tokens": message.usage.output_tokens,
                "response_id": None
            }

        # Process all tool use blocks
        tool_results_text = "I called the following tools:\n\n"

        for block in tool_use_blocks:
            try:
                tool_name = block.name
                tool_id = block.id
                tool_input = block.input or {}

                # Log the tool input for debugging
                logger.debug(f"Tool input for {tool_name}: {tool_input}")

                # Validate tool exists in registry
                if not tool_registry.has_tool(tool_name):
                    logger.warning(f"Tool {tool_name} not found in registry")
                    tool_results_text += f"- {tool_name}: Tool not found in registry\n"
                    continue

                # Execute the tool
                result = await tool_registry.execute_tool(tool_name, tool_input)

                # Format the result
                if isinstance(result, dict):
                    formatted_result = str(result)  # Convert dict to string
                else:
                    formatted_result = str(result)

                # Add to results text
                tool_results_text += f"- {tool_name}: {formatted_result}\n"
                logger.info(f"Tool processed: {tool_name}")
            except Exception as e:
                logger.error(f"Error processing tool use block: {e}")
                tool_results_text += f"- {tool_name}: Error - {str(e)}\n"

        # Make a follow-up call with the tool results as a user message
        follow_up_messages = anthropic_messages.copy()

        # Add the assistant's response with tool calls
        follow_up_messages.append({"role": "assistant", "content": final_text})

        # Add a user message with tool results
        follow_up_messages.append({"role": "user", "content": tool_results_text})

        # Create the follow-up request
        follow_up_params = {
            "model": model,
            "messages": follow_up_messages,
            "system": system_prompt,
            "temperature": temperature,
            "max_tokens": max_tokens,
            **{k: v for k, v in kwargs.items() if k not in ['tools', 'tool_choice']}
        }

        # Keep the tools from the original request
        if 'tools' in anthropic_params:
            follow_up_params['tools'] = anthropic_params['tools']

        try:
            follow_up_message = await client.messages.create(**follow_up_params)

            # Extract final text
            follow_up_text = ""
            for content_block in follow_up_message.content:
                if content_block.type == "text":
                    follow_up_text += content_block.text

            return {
                "text": follow_up_text,
                "input_tokens": follow_up_message.usage.input_tokens,
                "output_tokens": follow_up_message.usage.output_tokens,
                "response_id": None
            }

        except Exception as e:
            logger.error(f"Error in follow-up call: {e}")
            # Fall back to returning the initial message
            return {
                "text": final_text + "\n\n[Tool results could not be processed: " + str(e) + "]",
                "input_tokens": message.usage.input_tokens,
                "output_tokens": message.usage.output_tokens,
                "response_id": None
            }

    except Exception as e:
        logger.error(f"Anthropic API error: {str(e)}")
        if "Extra inputs are not permitted" in str(e) or "tool_choice" in str(e) or "Unexpected role" in str(e):
            # If we get tool-related errors, try without tools
            logger.warning("Retrying Anthropic API call without tools")
            anthropic_params.pop("tools", None)
            anthropic_params.pop("tool_choice", None)

            # Ensure all messages have simple string content
            for i, msg in enumerate(anthropic_params["messages"]):
                if isinstance(msg.get("content"), list):
                    # Convert structured content back to string
                    text_content = ""
                    for item in msg["content"]:
                        if item.get("type") == "text":
                            text_content += item.get("text", "")
                    anthropic_params["messages"][i]["content"] = text_content

            message = await client.messages.create(**anthropic_params)

            final_text = ""
            if message.content:
                for content_block in message.content:
                    if content_block.type == "text":
                        final_text += content_block.text

            return {
                "text": final_text,
                "input_tokens": message.usage.input_tokens,
                "output_tokens": message.usage.output_tokens,
                "response_id": None
            }
        else:
            # Re-raise other errors
            raise
