import logging
from typing import List, Dict, Any, Optional, Union

from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionChunk

from llm.types import Message, LLMResponse, StreamHandler


logger = logging.getLogger(__name__)


async def stream_chat_completions_api(
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
    """Stream responses from OpenAI's Chat Completions API."""
    # Prepare messages for Chat Completions API
    from llm.chat_completions import prepare_messages, prepare_tools_for_api
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
    logger.info(f"Making streaming Chat Completions API call")
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
                
            # Update token counts
            if hasattr(chunk, 'usage') and chunk.usage:
                if chunk.usage.prompt_tokens:
                    input_tokens = chunk.usage.prompt_tokens
                if chunk.usage.completion_tokens:
                    output_tokens = chunk.usage.completion_tokens
    
    # If we didn't get token counts from streaming chunks, estimate them
    if input_tokens == 0 or output_tokens == 0:
        # Make a non-streaming call to get token counts
        # This is a fallback and isn't ideal, but OpenAI doesn't always
        # include usage info in streaming responses
        try:
            completion_params["stream"] = False
            # Use a more meaningful minimum token value - OpenAI's minimum is 16
            completion_params["max_tokens"] = 16  # Minimize token usage for this call
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
