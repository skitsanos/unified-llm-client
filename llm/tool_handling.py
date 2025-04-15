import json
import logging
import uuid
from typing import Dict, Any, List, Optional, Tuple, Union, cast

from llm.types import ToolCallResponse

logger = logging.getLogger(__name__)


def extract_tool_info(tool_call: Any) -> Dict[str, Any]:
    """
    Extract consistent tool information from different tool call formats

    Args:
        tool_call: A tool call object from any provider

    Returns:
        Dictionary with standardized tool information
    """
    tool_info: Dict[str, Optional[str]] = {
        'id': None,
        'name': None,
        'type': None,
        'arguments': None,
    }

    # Try to extract the tool ID
    if hasattr(tool_call, 'call_id'):
        tool_info['id'] = tool_call.call_id
    elif hasattr(tool_call, 'id'):
        tool_info['id'] = tool_call.id

    # Try to extract tool type
    if hasattr(tool_call, 'type'):
        tool_info['type'] = tool_call.type

    # Try to extract tool name and arguments
    if hasattr(tool_call, 'name'):
        tool_info['name'] = tool_call.name
    if hasattr(tool_call, 'arguments'):
        tool_info['arguments'] = tool_call.arguments

    # Handle nested function structure (OpenAI Chat Completions API)
    if hasattr(tool_call, 'function'):
        if not tool_info['name'] and hasattr(tool_call.function, 'name'):
            tool_info['name'] = tool_call.function.name
        if not tool_info['arguments'] and hasattr(tool_call.function, 'arguments'):
            tool_info['arguments'] = tool_call.function.arguments

    return tool_info


def prepare_tools_for_api(tools_list: Optional[List[Dict[str, Any]]], api_type: str) -> Optional[List[Dict[str, Any]]]:
    """
    Format tools differently based on API type

    Args:
        tools_list: List of tool definitions
        api_type: Either 'responses', 'completions', or 'anthropic'

    Returns:
        Properly formatted tools list for the specified API
    """
    if not tools_list:
        return None

    if api_type == 'responses':
        # Responses API accepts both function and internal tools
        return tools_list
    elif api_type == 'completions':
        # Chat Completions API only accepts function tools with specific format
        formatted_tools: List[Dict[str, Any]] = []
        for tool in tools_list:
            # Skip internal tools for Chat Completions API
            if tool.get('type') in ['function', None]:
                # Ensure function tools have name in both places
                if 'function' in tool:
                    tool["function"]["name"] = tool.get('name', tool["function"].get('name'))
                formatted_tools.append(tool)

        return formatted_tools if formatted_tools else None
    elif api_type == 'anthropic':
        # Anthropic API requires a different format
        formatted_tools: List[Dict[str, Any]] = []
        for tool in tools_list:
            # Skip internal tools for Anthropic API
            if tool.get('type') in ['function', None]:
                anthropic_tool = {
                    "name": tool.get('name'),
                    "description": tool.get("function", {}).get('description', ''),
                    "input_schema": tool.get("function", {}).get('parameters', {}),
                }
                formatted_tools.append(anthropic_tool)

        return formatted_tools if formatted_tools else None

    return None


def create_shortened_tool_ids(tool_calls: List[Any]) -> Dict[str, str]:
    """
    Create shortened IDs for tool calls that are compatible with Chat Completions API
    
    Args:
        tool_calls: List of tool call objects
        
    Returns:
        Mapping of original IDs to shortened IDs
    """
    id_mapping: Dict[str, str] = {}

    for tool_call in tool_calls:
        # Generate a new UUID that's exactly 36 characters
        short_id = f"call_{str(uuid.uuid4())[:35 - 5]}"  # Keep under 40 chars
        original_id = getattr(tool_call, 'call_id',
                              getattr(tool_call, 'id', 'unknown'))
        id_mapping[original_id] = short_id
        logger.info(f"Mapped original ID {original_id} to shorter ID {short_id}")

    return id_mapping


async def process_function_calls(
    tool_registry: Any, 
    function_calls: List[Any], 
    id_mapping: Optional[Dict[str, str]] = None
) -> List[ToolCallResponse]:
    """
    Process function calls using the tool registry with optional ID mapping for shortened IDs

    Args:
        tool_registry: Registry containing tool implementations
        function_calls: List of function call objects to process
        id_mapping: Optional mapping of original IDs to shortened IDs

    Returns:
        List of tool responses with tool_call_id and output
    """
    if id_mapping is None:
        id_mapping = create_shortened_tool_ids(function_calls)

    tool_responses: List[ToolCallResponse] = []

    for tool_call in function_calls:
        try:
            # Extract tool info consistently across different formats
            tool_info = extract_tool_info(tool_call)

            # Skip internal tools
            if tool_info['type'] and tool_registry.is_internal_tool(tool_info['type']):
                logger.info(f"Skipping internal tool of type {tool_info['type']}")
                continue

            function_name = tool_info['name']
            arguments_json = tool_info['arguments']
            original_id = tool_info['id']

            # Use the shortened ID if available
            tool_call_id = id_mapping.get(original_id, original_id) if original_id else "unknown_id"

            logger.info(f"Handling function call: {function_name} with ID {tool_call_id}")

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
                args: Dict[str, Any] = json.loads(arguments_json) if arguments_json else {}

                # Execute the tool with the arguments
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
                logger.info(f"Tool processed: {function_name}")
            except json.JSONDecodeError as e:
                error_message = f"Invalid JSON arguments for tool {function_name}: {e}"
                logger.error(error_message)
                tool_responses.append({
                    "tool_call_id": tool_call_id,
                    "output": error_message
                })
            except Exception as e:
                error_message = f"Error executing tool {function_name}: {str(e)}"
                logger.error(f"{error_message}\nArguments: {arguments_json}")
                tool_responses.append({
                    "tool_call_id": tool_call_id,
                    "output": error_message
                })
        except Exception as e:
            # Catch-all for any unexpected errors in tool call processing
            error_message = f"Unexpected error processing tool call: {str(e)}"
            logger.error(error_message)
            # Try to get tool_call_id if possible
            tool_call_id = 'error_id'
            try:
                original_id = getattr(tool_call, 'call_id',
                                      getattr(tool_call, 'id', 'unknown'))
                tool_call_id = id_mapping.get(original_id, 'error_id') if original_id else 'error_id'
            except:
                pass

            tool_responses.append({
                "tool_call_id": tool_call_id,
                "output": error_message
            })

    return tool_responses


def prepare_assistant_message_with_tool_calls(
    content: str, 
    function_calls: List[Any], 
    id_mapping: Dict[str, str]
) -> Dict[str, Any]:
    """
    Create an assistant message with properly formatted tool calls for Chat Completions API

    Args:
        content: Text content of the assistant message
        function_calls: List of function call objects
        id_mapping: Mapping of original IDs to shortened IDs

    Returns:
        Dict containing the assistant message with tool_calls field
    """
    assistant_message: Dict[str, Any] = {
        "role": "assistant",
        "content": content,
        "tool_calls": []
    }

    for tool_call in function_calls:
        # Extract tool info consistently
        tool_info = extract_tool_info(tool_call)

        # Get original ID
        original_id = tool_info['id']

        # Use shortened ID
        short_id = id_mapping.get(original_id, original_id) if original_id else str(uuid.uuid4())

        # Get function name and arguments
        function_name = tool_info['name']
        arguments_json = tool_info['arguments']

        assistant_message["tool_calls"].append({
            "id": short_id,
            "type": "function",
            "function": {
                "name": function_name,
                "arguments": arguments_json
            }
        })

    return assistant_message
