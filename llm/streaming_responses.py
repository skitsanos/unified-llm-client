"""
Functions for handling streaming with OpenAI Responses API.
"""

import logging
from typing import Dict, Any, Optional, Union, List

from openai import AsyncOpenAI

from llm.tool_handling import prepare_tools_for_api
from llm.types import Message, LLMResponse, StreamHandler
from llm.responses_api import extract_cited_files

logger = logging.getLogger(__name__)


async def stream_responses_api(
        client: AsyncOpenAI,
        user_input: Union[str, List[Message]],
        model: str,
        instructions: Optional[str],
        tools: Optional[List[Dict[str, Any]]],
        temperature: float,
        max_tokens: int,
        stream_handler: StreamHandler,
        **kwargs
) -> LLMResponse:
    """Stream responses from OpenAI's Responses API."""
    # Prepare request parameters for Responses API
    if isinstance(user_input, str):
        response_params = {
            "model": model,
            "input": user_input,
            "temperature": temperature,
            "max_output_tokens": max_tokens,
            "stream": True,  # Enable streaming
            **{k: v for k, v in kwargs.items() if k not in ['tool_outputs']}
        }

        # Add tools if provided, properly formatted for Responses API
        if tools:
            response_params["tools"] = prepare_tools_for_api(tools, 'responses')

        # Add instructions if provided
        if instructions:
            response_params["instructions"] = instructions

    else:
        response_params = {
            "model": model,
            "input": user_input,
            "temperature": temperature,
            "max_output_tokens": max_tokens,
            "stream": True,  # Enable streaming
            **{k: v for k, v in kwargs.items() if k not in ['tool_outputs']}
        }

        # Add tools if provided
        if tools:
            response_params["tools"] = tools

        # Add instructions if provided and not in messages
        if instructions and not any(isinstance(msg, dict) and msg.get("role") == "system" for msg in user_input):
            response_params["instructions"] = instructions

    # Make the API call with streaming
    logger.info(f"Making streaming Responses API call")
    stream = await client.responses.create(**response_params)
    
    # Process the stream
    full_text = ""
    input_tokens = 0
    output_tokens = 0
    response_id = None
    sources = []
    
    async for chunk in stream:
        # Get text chunks
        if hasattr(chunk, 'choices') and chunk.choices:
            for choice in chunk.choices:
                if hasattr(choice, 'delta') and hasattr(choice.delta, 'text'):
                    content = choice.delta.text
                    if content:
                        full_text += content
                        await stream_handler(content)
        
        # Get usage information
        if hasattr(chunk, 'usage'):
            if hasattr(chunk.usage, 'input_tokens'):
                input_tokens = chunk.usage.input_tokens
            if hasattr(chunk.usage, 'output_tokens'):
                output_tokens = chunk.usage.output_tokens
        
        # Get response ID
        if hasattr(chunk, 'id') and chunk.id:
            response_id = chunk.id
        
        # Collect file citations
        if hasattr(chunk, 'output'):
            file_citations = extract_cited_files(chunk.output)
            if file_citations:
                for citation in file_citations:
                    if citation not in sources:
                        sources.append(citation)
    
    # If we didn't get token counts from streaming chunks, estimate them
    if input_tokens == 0 or output_tokens == 0:
        try:
            # Make a non-streaming call with minimal tokens to get usage info
            response_params["stream"] = False
            # Use a more meaningful minimum token value - OpenAI's minimum is 16
            response_params["max_output_tokens"] = 16
            non_stream_resp = await client.responses.create(**response_params)
            input_tokens = non_stream_resp.usage.input_tokens
            # Estimate output tokens based on text length
            output_tokens = len(full_text) // 4  # Rough estimate: ~4 chars per token
        except Exception as e:
            logger.warning(f"Failed to estimate token counts: {e}")
            # Fallback to very rough estimates
            if isinstance(user_input, str):
                input_tokens = len(user_input) // 4
            else:
                input_tokens = sum(len(m.get("content", "")) for m in user_input if isinstance(m, dict)) // 4
            output_tokens = len(full_text) // 4
    
    return {
        "text": full_text,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "response_id": response_id,
        "sources": sources,
    }
