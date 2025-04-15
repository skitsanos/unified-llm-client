"""
Basic example of using tools with OpenAI via Unified LLM Client

This example demonstrates different ways to use tools with OpenAI:
1. Custom tools with Chat Completions API 
2. Custom tools with Responses API
3. Built-in tools (web_search_preview) with Responses API
4. Location-aware web search with Responses API
5. Different search context sizes with web_search_preview

Key concepts:
- Chat Completions API: Use with custom tools via tool_registry
- Responses API: Can use both custom tools and OpenAI's built-in tools
- Web Search: Provides real-time information from the internet
- Search Context Size: Controls the amount of web content used (low/medium/high)
  - Higher context = more comprehensive answers but higher cost
  - Search context tokens don't count against model's context window

@author: skitsanos
"""

import os
import asyncio
import sys
from dotenv import load_dotenv

# Add the project root to PYTHONPATH for running the example
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm import AsyncLLMClient, ToolRegistry, llm_tool

# Load environment variables from .env file
load_dotenv()


# Define a weather tool
@llm_tool
def get_weather(location: str, unit: str = "celsius"):
    """
    Get the current weather for a location.
    
    Args:
        location: City or region name (e.g., Paris, London, New York)
        unit: Temperature unit (celsius or fahrenheit)
    """
    # This is a mock implementation
    weather_data = {
        "location": location,
        "temperature": 22 if unit == "celsius" else 72,
        "unit": unit,
        "condition": "sunny",
        "humidity": 45,
        "wind_speed": 10
    }
    
    return weather_data


async def test_openai_chat_completions():
    """Test tool usage with OpenAI using Chat Completions API"""
    print("\n--- Testing OpenAI with Custom Tools (Chat Completions API) ---")
    
    # Create a tool registry and register the weather tool
    tools = ToolRegistry()
    tools.register("get_weather", get_weather)
    
    # Initialize client with tool registry
    client = AsyncLLMClient(tool_registry=tools)
    
    try:
        # Use Chat Completions API explicitly 
        response = await client.response(
            "What's the weather like in Paris?",
            model="gpt-4o-mini",
            temperature=0.0,
            max_tokens=100,
            use_responses_api=False,  # Explicitly use Chat Completions API
            tool_choice="auto"  # String format is correct for OpenAI Chat Completions
        )
        
        print(f"OpenAI Response: {response['text']}")
        print(f"Tokens used: {response['input_tokens']} input, {response['output_tokens']} output")
        return True
    except Exception as e:
        print(f"Error: {str(e)}")
        if hasattr(e, '__cause__') and e.__cause__:
            print(f"Cause: {e.__cause__}")
        return False


async def test_openai_responses_custom_tools():
    """Test usage of custom tools with OpenAI's Responses API"""
    print("\n--- Testing OpenAI with Custom Tools (Responses API) ---")
    
    # Create a tool registry and register the weather tool
    tools = ToolRegistry()
    tools.register("get_weather", get_weather)
    
    # Initialize client with tool registry
    client = AsyncLLMClient(tool_registry=tools)
    
    try:
        # Use Responses API explicitly
        response = await client.response(
            "What's the weather like in Paris?",
            model="gpt-4o-mini", 
            temperature=0.0,
            max_tokens=100,
            use_responses_api=True  # Explicitly use Responses API
            # Note: Don't specify tool_choice for custom tools in Responses API
        )
        
        print(f"OpenAI Response: {response['text']}")
        print(f"Tokens used: {response['input_tokens']} input, {response['output_tokens']} output")
        return True
    except Exception as e:
        print(f"Error: {str(e)}")
        if hasattr(e, '__cause__') and e.__cause__:
            print(f"Cause: {e.__cause__}")
        return False


async def test_openai_responses_web_search():
    """Test usage of OpenAI's built-in web_search_preview tool with Responses API"""
    print("\n--- Testing OpenAI with Built-in Web Search (Responses API) ---")
    
    # Initialize client without tool registry since we're using OpenAI's built-in tool
    client = AsyncLLMClient()
    
    try:
        # Use Responses API with web_search_preview built-in tool
        response = await client.response(
            "What are the latest developments in quantum computing?",
            model="gpt-4o-mini", 
            temperature=0.0,
            max_tokens=250,
            use_responses_api=True,  # Use Responses API
            tools=[{
                "type": "web_search_preview",
                "search_context_size": "medium"  # Control amount of context (low/medium/high)
            }],  
            tool_choice={"type": "web_search_preview"}  # Instruct model to use web search
        )
        
        print(f"OpenAI Response (truncated): {response['text'][:150]}...")
        print(f"Tokens used: {response['input_tokens']} input, {response['output_tokens']} output")
        
        # If there are sources cited, show them
        if response.get('sources'):
            print(f"Sources cited: {len(response['sources'])}")
            
        # Note: Web search tool tokens don't count against model's context window
        print("Note: Search context tokens are not counted in the model's token usage")
        
        return True
    except Exception as e:
        print(f"Error: {str(e)}")
        if hasattr(e, '__cause__') and e.__cause__:
            print(f"Cause: {e.__cause__}")
        return False


async def test_openai_responses_web_search_with_location():
    """Test usage of OpenAI's web_search_preview with location information"""
    print("\n--- Testing OpenAI Web Search with Location (Responses API) ---")
    
    # Initialize client without tool registry
    client = AsyncLLMClient()
    
    try:
        # Use Responses API with web_search_preview and location information
        response = await client.response(
            "What are some popular attractions nearby?",
            model="gpt-4o-mini", 
            temperature=0.0,
            max_tokens=250,
            use_responses_api=True,
            tools=[{
                "type": "web_search_preview",
                "user_location": {
                    "type": "approximate",
                    "country": "US",
                    "city": "San Francisco",
                    "region": "California"
                }
            }],
            tool_choice={"type": "web_search_preview"}
        )
        
        print(f"OpenAI Response (truncated): {response['text'][:150]}...")
        print(f"Tokens used: {response['input_tokens']} input, {response['output_tokens']} output")
        
        # If there are sources cited, show them
        if response.get('sources'):
            print(f"Sources cited: {len(response['sources'])}")
        
        return True
    except Exception as e:
        print(f"Error: {str(e)}")
        if hasattr(e, '__cause__') and e.__cause__:
            print(f"Cause: {e.__cause__}")
        return False


async def test_openai_responses_search_context_size():
    """Test different search context sizes with web_search_preview tool"""
    print("\n--- Testing Web Search with Different Context Sizes ---")
    
    # Initialize client without tool registry
    client = AsyncLLMClient()
    
    # Define search query
    query = "Explain the James Webb Space Telescope's recent discoveries"
    
    # Test with low context size
    try:
        print("Using LOW search context size:")
        response_low = await client.response(
            query,
            model="gpt-4o-mini", 
            temperature=0.0,
            max_tokens=150,
            use_responses_api=True,
            tools=[{
                "type": "web_search_preview",
                "search_context_size": "low"  # Less context, faster, cheaper
            }],
            tool_choice={"type": "web_search_preview"}
        )
        
        print(f"Response (truncated): {response_low['text'][:100]}...")
        print(f"Tokens used: {response_low['input_tokens']} input, {response_low['output_tokens']} output")
        if response_low.get('sources'):
            print(f"Sources cited: {len(response_low['sources'])}")
        print()
        
        # Test with high context size
        print("Using HIGH search context size:")
        response_high = await client.response(
            query,
            model="gpt-4o-mini", 
            temperature=0.0,
            max_tokens=150,
            use_responses_api=True,
            tools=[{
                "type": "web_search_preview",
                "search_context_size": "high"  # More context, slower, more expensive
            }],
            tool_choice={"type": "web_search_preview"}
        )
        
        print(f"Response (truncated): {response_high['text'][:100]}...")
        print(f"Tokens used: {response_high['input_tokens']} input, {response_high['output_tokens']} output")
        if response_high.get('sources'):
            print(f"Sources cited: {len(response_high['sources'])}")
        
        print("\nNote: Higher search context can provide more comprehensive results")
        print("      but does not impact the main model's token usage.")
        
        return True
    except Exception as e:
        print(f"Error: {str(e)}")
        if hasattr(e, '__cause__') and e.__cause__:
            print(f"Cause: {e.__cause__}")
        return False


async def main():
    """Run all the OpenAI tests"""
    # Test with Chat Completions API and custom tools
    openai_chat_success = await test_openai_chat_completions()
    print(f"OpenAI Chat Completions test: {'PASSED' if openai_chat_success else 'FAILED'}\n")
    
    # Test with Responses API and custom tools
    openai_responses_success = await test_openai_responses_custom_tools()
    print(f"OpenAI Responses API with custom tools test: {'PASSED' if openai_responses_success else 'FAILED'}\n")
    
    # Test with Responses API and web_search_preview
    openai_web_search_success = await test_openai_responses_web_search()
    print(f"OpenAI Responses API with web search test: {'PASSED' if openai_web_search_success else 'FAILED'}\n")
    
    # Test with Responses API and web_search_preview with location
    openai_web_search_location_success = await test_openai_responses_web_search_with_location()
    print(f"OpenAI Responses API with web search and location test: {'PASSED' if openai_web_search_location_success else 'FAILED'}\n")
    
    # Test different search context sizes
    context_size_success = await test_openai_responses_search_context_size()
    print(f"OpenAI Responses API with different search context sizes test: {'PASSED' if context_size_success else 'FAILED'}\n")


if __name__ == "__main__":
    asyncio.run(main())
