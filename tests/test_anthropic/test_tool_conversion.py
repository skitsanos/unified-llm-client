"""
Tests for Anthropic tool format conversion functions.

@author: skitsanos
"""

import pytest
from llm.tool_handling import _format_tools_for_anthropic


def test_anthropic_already_in_correct_format():
    """Test tools already in Anthropic format."""
    tool = {
        "name": "get_weather",
        "description": "Get weather information",
        "input_schema": {
            "type": "object",
            "properties": {
                "location": {"type": "string"}
            },
            "required": ["location"]
        }
    }
    
    formatted_tools = _format_tools_for_anthropic([tool])
    assert len(formatted_tools) == 1
    assert formatted_tools[0] == tool


def test_anthropic_format_from_openai_format():
    """Test converting from OpenAI format to Anthropic format."""
    tool = {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get weather information",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string"}
                },
                "required": ["location"]
            }
        }
    }
    
    formatted_tools = _format_tools_for_anthropic([tool])
    assert len(formatted_tools) == 1
    
    expected = {
        "name": "get_weather",
        "description": "Get weather information",
        "input_schema": {
            "type": "object",
            "properties": {
                "location": {"type": "string"}
            },
            "required": ["location"]
        }
    }
    
    assert formatted_tools[0] == expected


def test_anthropic_format_from_custom_format():
    """Test converting from custom format to Anthropic format."""
    tool = {
        "type": "custom",
        "custom": {
            "name": "get_weather",
            "description": "Get weather information",
            "input_schema": {
                "type": "object",
                "properties": {
                    "location": {"type": "string"}
                },
                "required": ["location"]
            }
        }
    }
    
    formatted_tools = _format_tools_for_anthropic([tool])
    assert len(formatted_tools) == 1
    
    expected = {
        "name": "get_weather",
        "description": "Get weather information",
        "input_schema": {
            "type": "object",
            "properties": {
                "location": {"type": "string"}
            },
            "required": ["location"]
        }
    }
    
    assert formatted_tools[0] == expected


def test_anthropic_format_from_simplified_format():
    """Test converting from simplified format to Anthropic format."""
    tool = {
        "name": "get_weather",
        "description": "Get weather information",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string"}
            },
            "required": ["location"]
        }
    }
    
    formatted_tools = _format_tools_for_anthropic([tool])
    assert len(formatted_tools) == 1
    
    expected = {
        "name": "get_weather",
        "description": "Get weather information",
        "input_schema": {
            "type": "object",
            "properties": {
                "location": {"type": "string"}
            },
            "required": ["location"]
        }
    }
    
    assert formatted_tools[0] == expected


def test_anthropic_format_mixed_tools():
    """Test converting a mix of tool formats."""
    tools = [
        # Already in correct format
        {
            "name": "get_weather",
            "description": "Get weather information",
            "input_schema": {
                "type": "object",
                "properties": {
                    "location": {"type": "string"}
                },
                "required": ["location"]
            }
        },
        # OpenAI format
        {
            "type": "function",
            "function": {
                "name": "get_news",
                "description": "Get news articles",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "topic": {"type": "string"}
                    },
                    "required": ["topic"]
                }
            }
        },
        # Simplified format
        {
            "name": "search",
            "description": "Search for information",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"}
                },
                "required": ["query"]
            }
        }
    ]
    
    formatted_tools = _format_tools_for_anthropic(tools)
    assert len(formatted_tools) == 3
    
    # Check each tool was converted correctly
    assert formatted_tools[0]["name"] == "get_weather"
    assert formatted_tools[1]["name"] == "get_news"
    assert formatted_tools[2]["name"] == "search"
    
    # Check input_schema in each
    assert "input_schema" in formatted_tools[0]
    assert "input_schema" in formatted_tools[1]
    assert "input_schema" in formatted_tools[2]


if __name__ == "__main__":
    pytest.main()
