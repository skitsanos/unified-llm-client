"""
Tests for the AsyncLLMClient

@author: skitsanos
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from llm import AsyncLLMClient, ToolRegistry, llm_tool


@pytest.fixture
def mock_openai_client():
    """Create a mock OpenAI client."""
    mock_client = AsyncMock()
    mock_completion = AsyncMock()
    mock_choice = MagicMock()
    mock_message = MagicMock()

    mock_message.content = "This is a test response"
    mock_message.tool_calls = []

    mock_choice.message = mock_message

    mock_completion.choices = [mock_choice]
    mock_completion.usage.prompt_tokens = 10
    mock_completion.usage.completion_tokens = 20

    mock_client.chat.completions.create = AsyncMock(return_value=mock_completion)

    return mock_client


@pytest.fixture
def mock_anthropic_client():
    """Create a mock Anthropic client."""
    mock_client = AsyncMock()
    mock_message = AsyncMock()
    mock_content = MagicMock()

    mock_content.type = "text"
    mock_content.text = "This is a test response from Claude"

    mock_message.content = [mock_content]
    mock_message.usage.input_tokens = 15
    mock_message.usage.output_tokens = 25

    mock_client.messages.create = AsyncMock(return_value=mock_message)

    return mock_client


@pytest.fixture
def llm_client(mock_openai_client, mock_anthropic_client):
    """Create an LLMClient with mocked API clients."""
    with patch("llm.client.AsyncOpenAI", return_value=mock_openai_client), patch(
            "llm.client.AsyncAnthropic", return_value=mock_anthropic_client
    ):
        client = AsyncLLMClient(api_key="test-key")
        yield client


# Define a test tool function
@llm_tool
def example_tool(param: str) -> str:
    """A test tool."""
    return f"Tool response: {param}"


@pytest.mark.asyncio
async def test_openai_response(llm_client, mock_openai_client):
    """Test getting a response from OpenAI."""
    response = await llm_client.response(
        "This is a test",
        model="gpt-4o-mini",
        use_responses_api=False,  # Use chat completions API for this test
    )

    # Verify the client was called with correct parameters
    mock_openai_client.chat.completions.create.assert_called_once()
    call_args = mock_openai_client.chat.completions.create.call_args[1]
    assert call_args["model"] == "gpt-4o-mini"
    assert call_args["messages"][0]["role"] == "user"
    assert call_args["messages"][0]["content"] == "This is a test"

    # Verify the response format
    assert isinstance(response, dict)
    assert "text" in response
    assert "input_tokens" in response
    assert "output_tokens" in response
    assert response["text"] == "This is a test response"
    assert response["input_tokens"] == 10
    assert response["output_tokens"] == 20


@pytest.mark.asyncio
async def test_anthropic_response(llm_client, mock_anthropic_client):
    """Test getting a response from Anthropic."""
    response = await llm_client.response(
        "This is a test", model="claude-3-opus-20240229"
    )

    # Verify the client was called with correct parameters
    mock_anthropic_client.messages.create.assert_called_once()
    call_args = mock_anthropic_client.messages.create.call_args[1]
    assert call_args["model"] == "claude-3-opus-20240229"
    assert call_args["messages"][0]["role"] == "user"
    assert call_args["messages"][0]["content"] == "This is a test"

    # Verify the response format
    assert isinstance(response, dict)
    assert "text" in response
    assert "input_tokens" in response
    assert "output_tokens" in response
    assert response["text"] == "This is a test response from Claude"
    assert response["input_tokens"] == 15
    assert response["output_tokens"] == 25


@pytest.mark.asyncio
async def test_invalid_model(llm_client):
    """Test response with an invalid model."""
    # Our updated code now defaults to OpenAI for unrecognized models with a warning
    # Instead of checking for an exception, we'll check if the code executes
    # successfully with a mock OpenAI client
    response = await llm_client.response("This is a test", model="invalid-model")

    # Since we're using a mock, we should still get a valid response
    assert "text" in response


@pytest.mark.asyncio
async def test_with_instructions(llm_client, mock_openai_client):
    """Test response with system instructions."""
    await llm_client.response(
        "This is a test",
        model="gpt-4o-mini",
        instructions="You are a helpful assistant",
        use_responses_api=False,
    )

    # Verify the system message was included
    call_args = mock_openai_client.chat.completions.create.call_args[1]
    messages = call_args["messages"]
    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert messages[0]["content"] == "You are a helpful assistant"
    assert messages[1]["role"] == "user"


@pytest.mark.asyncio
async def test_with_message_list(llm_client, mock_openai_client):
    """Test response with a list of messages."""
    messages = [
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there!"},
        {"role": "user", "content": "How are you?"},
    ]

    await llm_client.response(messages, model="gpt-4o-mini", use_responses_api=False)

    # Verify all messages were included
    call_args = mock_openai_client.chat.completions.create.call_args[1]
    passed_messages = call_args["messages"]
    assert len(passed_messages) == 4
    assert passed_messages[0]["role"] == "system"
    assert passed_messages[1]["role"] == "user"
    assert passed_messages[2]["role"] == "assistant"
    assert passed_messages[3]["role"] == "user"


@pytest.mark.asyncio
async def test_with_tools(llm_client, mock_openai_client):
    """Test response with tools."""
    # Create a registry with a test tool
    registry = ToolRegistry()
    registry.register("example_tool", example_tool)
    llm_client.tool_registry = registry

    # Setup mock for tool calls
    mock_tool_call = MagicMock()
    mock_tool_call.id = "call_123"
    mock_tool_call.function.name = "example_tool"
    mock_tool_call.function.arguments = json.dumps({"param": "test"})

    # Configure the mock to return different responses for each call
    first_response = MagicMock()
    first_response.choices = [MagicMock()]
    first_response.choices[0].message.content = "I'll use a tool"
    first_response.choices[0].message.tool_calls = [mock_tool_call]
    first_response.usage.prompt_tokens = 10
    first_response.usage.completion_tokens = 20

    second_response = MagicMock()
    second_response.choices = [MagicMock()]
    second_response.choices[0].message.content = (
        "Here's the result: Tool response: test"
    )
    second_response.choices[0].message.tool_calls = []
    second_response.usage.prompt_tokens = 30
    second_response.usage.completion_tokens = 40

    mock_openai_client.chat.completions.create.side_effect = [
        first_response,
        second_response,
    ]

    # Call with tools
    response = await llm_client.response(
        "Use the example_tool", model="gpt-4o-mini", use_responses_api=False
    )

    # Verify the response
    assert "Here's the result" in response["text"]
    assert response["input_tokens"] == 30
    assert response["output_tokens"] == 40

    # Verify the client was called twice (with tool calls and with tool results)
    assert mock_openai_client.chat.completions.create.call_count == 2
