"""
Example of using tools with Anthropic Claude via Unified LLM Client

This example demonstrates how to use function calling with Anthropic's Claude models,
specifically focusing on claude-3-5-haiku-latest which has excellent tool calling support.

Note: You'll need an Anthropic API key set in your environment variables (ANTHROPIC_API_KEY).

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


async def test_claude_basic_tool_calling():
    """Test basic tool calling with Claude"""
    print("\n--- Testing Claude with Calculator Tool ---")
    
    # Create a tool registry and register the calculator tool
    tools = ToolRegistry()
    tools.register("calculator", calculator)
    
    # Initialize client with tool registry
    client = AsyncLLMClient(tool_registry=tools)
    
    try:
        # Ask a question that requires calculation
        response = await client.response(
            "What is 123 multiplied by 456? Use the calculator tool to find the exact answer.",
            model="claude-3-5-haiku-latest",
            temperature=0.0,
            max_tokens=500,
            instructions="You have access to a calculator tool to perform mathematical calculations. Always use the tool for precise calculations."
        )
        
        print(f"Claude Response: {response['text']}")
        print(f"Tokens used: {response['input_tokens']} input, {response['output_tokens']} output")
        return True
    except Exception as e:
        print(f"Error: {str(e)}")
        if hasattr(e, '__cause__') and e.__cause__:
            print(f"Cause: {e.__cause__}")
        return False


async def test_claude_multi_tools():
    """Test Claude with multiple tools"""
    print("\n--- Testing Claude with Multiple Tools ---")
    
    # Create a tool registry and register multiple tools
    tools = ToolRegistry()
    tools.register("calculator", calculator)
    tools.register("get_weather", get_weather)
    tools.register("get_product_info", get_product_info)
    
    # Initialize client with tool registry
    client = AsyncLLMClient(tool_registry=tools)
    
    try:
        # Ask a complex question that requires multiple tools
        response = await client.response(
            "I need three pieces of information: 1) The weather in Tokyo, 2) The price of product P67890, and 3) What is 15% of 89.99?",
            model="claude-3-5-haiku-latest",
            temperature=0.0,
            max_tokens=800,
            instructions="""You have access to several tools:
1. A calculator for mathematical calculations
2. A weather service to check weather conditions
3. A product database to retrieve product information
Use these tools to provide precise answers."""
        )
        
        print(f"Claude Response (truncated): {response['text'][:200]}...")
        print(f"Tokens used: {response['input_tokens']} input, {response['output_tokens']} output")
        return True
    except Exception as e:
        print(f"Error: {str(e)}")
        if hasattr(e, '__cause__') and e.__cause__:
            print(f"Cause: {e.__cause__}")
        return False


async def test_claude_complex_reasoning():
    """Test Claude's ability to handle complex reasoning with tools"""
    print("\n--- Testing Claude with Complex Reasoning ---")
    
    # Create a tool registry and register tools
    tools = ToolRegistry()
    tools.register("calculator", calculator)
    tools.register("get_product_info", get_product_info)
    
    # Initialize client with tool registry
    client = AsyncLLMClient(tool_registry=tools)
    
    try:
        # Ask a question requiring both tool usage and reasoning
        response = await client.response(
            """
            A customer is buying 3 units of product P12345 and 2 units of product P67890.
            1. What is the total cost before tax?
            2. If sales tax is 8.5%, what is the final amount?
            3. If the customer pays with $600 cash, how much change should they receive?
            """,
            model="claude-3-5-haiku-latest",
            temperature=0.0,
            max_tokens=1000,
            instructions="""You have access to a calculator and product database.
1. Use get_product_info to retrieve product details including prices
2. Use calculator to perform precise calculations
3. Provide step-by-step calculations so the customer understands the breakdown"""
        )
        
        print(f"Claude Response (truncated): {response['text'][:200]}...")
        print(f"Tokens used: {response['input_tokens']} input, {response['output_tokens']} output")
        return True
    except Exception as e:
        print(f"Error: {str(e)}")
        if hasattr(e, '__cause__') and e.__cause__:
            print(f"Cause: {e.__cause__}")
        return False


async def test_claude_format_handling():
    """Test Claude's ability to format outputs nicely"""
    print("\n--- Testing Claude's Format Handling ---")
    
    # Create a tool registry and register the product info tool
    tools = ToolRegistry()
    tools.register("get_product_info", get_product_info)
    
    # Initialize client with tool registry
    client = AsyncLLMClient(tool_registry=tools)
    
    try:
        # Ask for a nicely formatted product comparison
        response = await client.response(
            """
            Compare products P12345, P67890, and P54321. 
            Create a comparison table with the following information:
            1. Product Name
            2. Price
            3. Category
            4. Stock availability
            5. Key features (from description)
            
            After the table, recommend which product offers the best value.
            """,
            model="claude-3-5-haiku-latest",
            temperature=0.0,
            max_tokens=1000,
            instructions="""You have access to a product database.
1. Use get_product_info to retrieve details for each product
2. Format the information in a clear, readable markdown table
3. Make a recommendation based on features and price"""
        )
        
        print(f"Claude Response (truncated): {response['text'][:300]}...")
        print(f"Tokens used: {response['input_tokens']} input, {response['output_tokens']} output")
        return True
    except Exception as e:
        print(f"Error: {str(e)}")
        if hasattr(e, '__cause__') and e.__cause__:
            print(f"Cause: {e.__cause__}")
        return False


async def main():
    """Run all tests"""
    print("=== Anthropic Claude Tool Calling Example ===")
    print("\nModel: claude-3-5-haiku-latest")
    print("\nPrerequisites:")
    print("1. Make sure you have an Anthropic API key set in your environment (ANTHROPIC_API_KEY)")
    print("2. The Unified LLM Client should handle all Claude-specific tool formatting")
    print("\nNotes on Claude's tool calling:")
    print("- Claude has excellent tool calling capabilities")
    print("- Tool schemas are automatically converted to Anthropic's format")
    print("- Claude can gracefully handle multiple tools and complex reasoning")
    print("- The claude-3-5-haiku model offers good performance and lower latency")
    print("\nRunning tests with Claude...")
    
    # Test with calculator tool
    calc_success = await test_claude_basic_tool_calling()
    print(f"Claude calculator tool test: {'PASSED' if calc_success else 'FAILED'}\n")
    
    # Test with multiple tools
    multi_tools_success = await test_claude_multi_tools() 
    print(f"Claude multiple tools test: {'PASSED' if multi_tools_success else 'FAILED'}\n")
    
    # Test complex reasoning
    complex_success = await test_claude_complex_reasoning()
    print(f"Claude complex reasoning test: {'PASSED' if complex_success else 'FAILED'}\n")
    
    # Test formatting capabilities
    format_success = await test_claude_format_handling()
    print(f"Claude format handling test: {'PASSED' if format_success else 'FAILED'}\n")
    
    print("Benefits of Claude's tool calling:")
    print("✓ Excellent reasoning ability with tools")
    print("✓ Clear, detailed responses")
    print("✓ Good at formatting data")
    print("✓ Reliable tool calling implementation")


if __name__ == "__main__":
    asyncio.run(main())
