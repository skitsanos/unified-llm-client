"""
Tests for the tool registry and llm_tool decorator

@author: skitsanos
"""

from typing import Any, Dict

import pytest

from llm.tooling import ToolRegistry, llm_tool


# Define a test tool function
@llm_tool
def example_tool(param1: str, param2: int = 42) -> Dict[str, Any]:
    """
    A test tool function.

    Args:
        param1: First parameter
        param2: Second parameter with default
    """
    return {"param1": param1, "param2": param2}


@llm_tool
async def example_async_tool(param1: str, param2: int = 42) -> Dict[str, Any]:
    """
    A test async tool function.

    Args:
        param1: First parameter
        param2: Second parameter with default
    """
    return {"param1": param1, "param2": param2, "async": True}


def test_tool_registry_init():
    """Test ToolRegistry initialization."""
    registry = ToolRegistry()
    assert registry is not None
    assert registry.get_names() == []


def test_tool_registry_register():
    """Test registering tools in the registry."""
    registry = ToolRegistry()

    # Register a tool
    registry.register("example_tool", example_tool)
    assert "example_tool" in registry.get_names()
    assert registry.has_tool("example_tool")

    # Register another tool
    registry.register("example_async_tool", example_async_tool)
    assert "example_async_tool" in registry.get_names()

    # Verify we have 2 tools
    assert len(registry.get_names()) == 2


def test_tool_registry_unregister():
    """Test unregistering tools from the registry."""
    registry = ToolRegistry()

    # Register and then unregister a tool
    registry.register("example_tool", example_tool)
    assert registry.has_tool("example_tool")

    registry.unregister("example_tool")
    assert not registry.has_tool("example_tool")


def test_llm_tool_decorator():
    """Test that the llm_tool decorator properly decorates functions."""
    # Check that the decorator added the expected attributes
    assert hasattr(example_tool, "openai_tool")
    assert hasattr(example_tool, "anthropic_tool")
    assert hasattr(example_tool, "tool")

    # Check the content of the tool definitions
    assert example_tool.openai_tool["name"] == "example_tool"
    assert "description" in example_tool.openai_tool["function"]
    assert "parameters" in example_tool.openai_tool["function"]


def test_get_schemas():
    """Test retrieving tool schemas from the registry."""
    registry = ToolRegistry()
    registry.register("example_tool", example_tool)

    # Get OpenAI schemas
    openai_schemas = registry.get_schemas("openai")
    assert len(openai_schemas) == 1
    assert openai_schemas[0]["name"] == "example_tool"
    assert openai_schemas[0]["type"] == "function"

    # Get Anthropic schemas
    anthropic_schemas = registry.get_schemas("anthropic")
    assert len(anthropic_schemas) == 1
    assert anthropic_schemas[0]["name"] == "example_tool"
    assert "input_schema" in anthropic_schemas[0]


@pytest.mark.asyncio
async def test_execute_tool():
    """Test executing a tool from the registry."""
    registry = ToolRegistry()
    registry.register("example_tool", example_tool)
    registry.register("example_async_tool", example_async_tool)

    # Execute a synchronous tool
    result = await registry.execute_tool("example_tool", {"param1": "hello"})
    assert result["param1"] == "hello"
    assert result["param2"] == 42

    # Execute an asynchronous tool
    async_result = await registry.execute_tool(
        "example_async_tool", {"param1": "hello", "param2": 100}
    )
    assert async_result["param1"] == "hello"
    assert async_result["param2"] == 100
    assert async_result["async"] is True

    # Test with missing required parameter should raise an error
    with pytest.raises(Exception):
        await registry.execute_tool("example_tool", {})

    # Test with non-existent tool should raise KeyError
    with pytest.raises(KeyError):
        await registry.execute_tool("non_existent_tool", {})
