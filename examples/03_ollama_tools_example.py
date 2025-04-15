"""
Example of using tools with Ollama's qwen2.5 model

This example demonstrates how to use function calling with a local model
through Ollama, specifically with Qwen 2.5 which has excellent tool calling support.

Prerequisites:
1. Install Ollama from https://ollama.com
2. Pull the Qwen 2.5 model with: `ollama pull qwen2.5`
3. Make sure Ollama service is running

@author: skitsanos
"""

import os
import asyncio
import sys
import json
from dotenv import load_dotenv

# Add the project root to PYTHONPATH for running the example
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm import AsyncLLMClient, ToolRegistry, llm_tool

# Load environment variables from .env file (not required for Ollama)
load_dotenv()


# Define a simple calculator tool
@llm_tool
def calculator(expression: str) -> float:
    """
    Evaluate a mathematical expression.
    
    Args:
        expression: A string containing a mathematical expression like "123 * 456"
    """
    # Create a safe evaluation environment with only math operations
    safe_dict = {
        'abs': abs, 'round': round,
        'min': min, 'max': max,
        'sum': sum, 'pow': pow
    }
    
    # Replace common math operators with Python syntax
    expression = expression.replace('×', '*').replace('÷', '/')
    
    try:
        # Safely evaluate the expression
        result = eval(expression, {"__builtins__": {}}, safe_dict)
        return float(result)
    except Exception as e:
        return f"Error: {str(e)}"


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


# Define a product information tool
@llm_tool
def get_product_info(product_id: str):
    """
    Get information about a product by its ID.
    
    Args:
        product_id: The unique identifier for the product
    """
    # Mock product database
    products = {
        "P12345": {
            "name": "Premium Coffee Maker",
            "price": 129.99,
            "category": "Kitchen Appliances",
            "stock": 25,
            "description": "Programmable coffee maker with 12-cup capacity and built-in grinder"
        },
        "P67890": {
            "name": "Wireless Headphones",
            "price": 89.99,
            "category": "Electronics",
            "stock": 42,
            "description": "Bluetooth headphones with noise cancellation and 20-hour battery life"
        },
        "P54321": {
            "name": "Yoga Mat",
            "price": 29.99,
            "category": "Fitness",
            "stock": 15,
            "description": "Non-slip exercise mat with carrying strap, ideal for yoga and pilates"
        }
    }
    
    # Return product info if found, otherwise return error message
    if product_id in products:
        return products[product_id]
    else:
        return {"error": f"Product with ID '{product_id}' not found"}


async def test_ollama_calculator():
    """Test tool calling with Ollama using the calculator tool"""
    print("\n--- Testing Ollama with Calculator Tool ---")
    
    # Create a tool registry and register the calculator tool
    tools = ToolRegistry()
    tools.register("calculator", calculator)
    
    # Initialize client with tool registry and Ollama settings
    client = AsyncLLMClient(
        base_url="http://localhost:11434/v1",
        api_key="ollama",
        tool_registry=tools
    )
    
    try:
        # Ask a question that requires calculation
        response = await client.response(
            "What is 123 multiplied by 456? Use the calculator tool to find the exact answer.",
            model="qwen2.5",  # Qwen 2.5 supports tool calling
            temperature=0.0,
            max_tokens=500,
            use_responses_api=False,  # Ollama uses the Chat Completions format
            instructions="You have access to a calculator tool to perform mathematical calculations. Always use the tool for precise calculations."
        )
        
        print(f"Ollama Response: {response['text']}")
        print(f"Tokens used: {response['input_tokens']} input, {response['output_tokens']} output")
        return True
    except Exception as e:
        print(f"Error: {str(e)}")
        if hasattr(e, '__cause__') and e.__cause__:
            print(f"Cause: {e.__cause__}")
        return False


async def test_ollama_multi_tools():
    """Test tool calling with Ollama using multiple tools"""
    print("\n--- Testing Ollama with Multiple Tools ---")
    
    # Create a tool registry and register multiple tools
    tools = ToolRegistry()
    tools.register("calculator", calculator)
    tools.register("get_weather", get_weather)
    tools.register("get_product_info", get_product_info)
    
    # Initialize client with tool registry and Ollama settings
    client = AsyncLLMClient(
        base_url="http://localhost:11434/v1",
        api_key="ollama",
        tool_registry=tools
    )
    
    try:
        # Ask a complex question that might require multiple tools
        response = await client.response(
            "I need three pieces of information: 1) The weather in Tokyo, 2) The price of product P67890, and 3) What is 15% of 89.99?",
            model="qwen2.5",
            temperature=0.0,
            max_tokens=800,
            use_responses_api=False,
            instructions="""You have access to several tools:
1. A calculator for mathematical calculations
2. A weather service to check weather conditions
3. A product database to retrieve product information
Use these tools to provide precise answers."""
        )
        
        print(f"Ollama Response (truncated): {response['text'][:200]}...")
        print(f"Tokens used: {response['input_tokens']} input, {response['output_tokens']} output")
        return True
    except Exception as e:
        print(f"Error: {str(e)}")
        if hasattr(e, '__cause__') and e.__cause__:
            print(f"Cause: {e.__cause__}")
        return False


async def test_ollama_complex_reasoning():
    """Test qwen2.5's ability to handle complex reasoning with tools"""
    print("\n--- Testing Ollama with Complex Reasoning ---")
    
    # Create a tool registry and register tools
    tools = ToolRegistry()
    tools.register("calculator", calculator)
    tools.register("get_product_info", get_product_info)
    
    # Initialize client with tool registry and Ollama settings
    client = AsyncLLMClient(
        base_url="http://localhost:11434/v1",
        api_key="ollama",
        tool_registry=tools
    )
    
    try:
        # Ask a question requiring both tool usage and reasoning
        response = await client.response(
            """
            A customer is buying 3 units of product P12345 and 2 units of product P67890.
            1. What is the total cost before tax?
            2. If sales tax is 8.5%, what is the final amount?
            3. If the customer pays with $600 cash, how much change should they receive?
            """,
            model="qwen2.5",
            temperature=0.0,
            max_tokens=1000,
            use_responses_api=False,
            instructions="""You have access to a calculator and product database.
1. Use get_product_info to retrieve product details including prices
2. Use calculator to perform precise calculations
3. Provide step-by-step calculations so the customer understands the breakdown"""
        )
        
        print(f"Ollama Response (truncated): {response['text'][:200]}...")
        print(f"Tokens used: {response['input_tokens']} input, {response['output_tokens']} output")
        return True
    except Exception as e:
        print(f"Error: {str(e)}")
        if hasattr(e, '__cause__') and e.__cause__:
            print(f"Cause: {e.__cause__}")
        return False


async def main():
    """Run all tests"""
    print("=== Ollama Tool Calling Example with qwen2.5 ===")
    print("\nPrerequisites:")
    print("1. Make sure Ollama is running (download from https://ollama.com if not installed)")
    print("2. Pull the Qwen 2.5 model with: ollama pull qwen2.5")
    print("\nNotes on Ollama tool calling with qwen2.5:")
    print("- qwen2.5 has excellent function calling support")
    print("- All processing happens locally on your machine")
    print("- No internet connection required for the model")
    print("- Performance depends on your CPU/GPU")
    print("\nRunning tests with qwen2.5...")
    
    # Test with calculator tool
    calc_success = await test_ollama_calculator()
    print(f"Ollama calculator tool test: {'PASSED' if calc_success else 'FAILED'}\n")
    
    # Test with multiple tools
    multi_tools_success = await test_ollama_multi_tools() 
    print(f"Ollama multiple tools test: {'PASSED' if multi_tools_success else 'FAILED'}\n")
    
    # Test complex reasoning
    complex_success = await test_ollama_complex_reasoning()
    print(f"Ollama complex reasoning test: {'PASSED' if complex_success else 'FAILED'}\n")
    
    print("Benefits of using qwen2.5 with Ollama:")
    print("✓ Privacy - all processing happens locally")
    print("✓ Cost savings - no API usage fees")
    print("✓ Offline capability - works without internet")
    print("✓ Good tool calling capability - especially with qwen2.5")


if __name__ == "__main__":
    asyncio.run(main())
