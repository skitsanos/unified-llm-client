"""
Example of using the Unified LLM Client with tools/functions

@author: skitsanos
"""

import asyncio
import os
import json
from dotenv import load_dotenv

from llm import AsyncLLMClient, ToolRegistry, llm_tool

# Load environment variables from .env file
load_dotenv()


# Define a tool function using the decorator
@llm_tool
async def search_database(query: str, limit: int = 5):
    """
    Search a fictional database for information.
    
    Args:
        query: The search query
        limit: Maximum number of results to return
    """
    # Mock implementation - in a real application, this would query a database
    results = [
        {"id": 1, "title": "Introduction to Python", "category": "Programming"},
        {"id": 2, "title": "Advanced Python Techniques", "category": "Programming"},
        {"id": 3, "title": "Web Development with Django", "category": "Web"},
        {"id": 4, "title": "Machine Learning Basics", "category": "AI"},
        {"id": 5, "title": "Deep Learning with PyTorch", "category": "AI"},
    ]
    
    # Filter results based on the query (simple case-insensitive match)
    filtered_results = [
        r for r in results 
        if query.lower() in r["title"].lower() or query.lower() in r["category"].lower()
    ]
    
    # Apply limit
    limited_results = filtered_results[:limit]
    
    return limited_results


# Define another tool function
@llm_tool
def get_weather(location: str, unit: str = "celsius"):
    """
    Get the current weather for a location.
    
    Args:
        location: City or region name
        unit: Temperature unit (celsius or fahrenheit)
    """
    # Mock implementation - in a real application, this would call a weather API
    weather_data = {
        "location": location,
        "temperature": 22 if unit == "celsius" else 72,
        "unit": unit,
        "condition": "sunny",
        "humidity": 45,
        "wind_speed": 10
    }
    
    return weather_data


async def main():
    # Create a tool registry and register the tools
    tools = ToolRegistry()
    tools.register("search_database", search_database)
    tools.register("get_weather", get_weather)
    
    # Initialize the client with the tool registry
    client = AsyncLLMClient(tool_registry=tools)
    
    # Prepare a prompt that will require tool use
    prompt = "I need information about Python programming and also tell me the weather in New York."
    
    # Get a response from OpenAI with tools
    print("Asking OpenAI GPT-4 with tools...")
    openai_response = await client.response(
        prompt,
        model="gpt-4o",
        instructions="You are a helpful assistant that can search for information and check the weather.",
        temperature=0.7
    )
    
    print(f"\nOpenAI Response:\n{openai_response['text']}")
    print(f"Tokens used: {openai_response['input_tokens']} input, {openai_response['output_tokens']} output")
    
    # Get a response from Anthropic with tools
    print("\nAsking Anthropic Claude with tools...")
    claude_response = await client.response(
        prompt,
        model="claude-3-opus-20240229",
        instructions="You are a helpful assistant that can search for information and check the weather.",
        temperature=0.7
    )
    
    print(f"\nClaude Response:\n{claude_response['text']}")
    print(f"Tokens used: {claude_response['input_tokens']} input, {claude_response['output_tokens']} output")


if __name__ == "__main__":
    asyncio.run(main())
