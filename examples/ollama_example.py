"""
Example of using the Unified LLM Client with Ollama models

@author: skitsanos
"""

import asyncio
import os
from dotenv import load_dotenv

from llm import AsyncLLMClient, ToolRegistry, llm_tool

# Load environment variables from .env file
load_dotenv()


@llm_tool
def calculator(expression: str) -> float:
    """
    Evaluate a mathematical expression.
    
    Args:
        expression: A mathematical expression to evaluate (e.g., "2 + 2 * 3")
    """
    # Use eval with care in production applications
    try:
        return float(eval(expression, {"__builtins__": {}}, {}))
    except Exception as e:
        return f"Error evaluating expression: {str(e)}"


async def main():
    # Create a tool registry and register the calculator tool
    tools = ToolRegistry()
    tools.register("calculator", calculator)
    
    # Initialize the client with the tool registry
    # For Ollama, we'll use a different base URL
    client = AsyncLLMClient(
        base_url="http://localhost:11434/v1",
        api_key="ollama",  # Ollama doesn't require a real API key
        tool_registry=tools
    )
    
    # List of models to test
    ollama_models = [
        "qwen2.5",
        "llama3", 
        "mistral:latest"
    ]
    
    for model in ollama_models:
        print(f"\n-------- Testing {model} --------")
        try:
            # Simple question
            prompt = "What is the capital of France?"
            print(f"\nPrompt: {prompt}")
            
            # Call the model
            response = await client.response(
                prompt,
                model=model,
                temperature=0.7,
                max_tokens=500,
                use_responses_api=False  # Ollama works with the chat completions API format
            )
            
            print(f"\nResponse from {model}:\n{response['text']}")
            
            # Now try with tool calling
            prompt = "What is the result of 25 * 13 + 7?"
            print(f"\nPrompt with tool calling: {prompt}")
            
            # Call the model with tool access
            response = await client.response(
                prompt,
                model=model,
                instructions="You are a helpful assistant. You have access to a calculator tool that can evaluate mathematical expressions.",
                temperature=0.7,
                max_tokens=500,
                use_responses_api=False
            )
            
            print(f"\nResponse with tool from {model}:\n{response['text']}")
            
        except Exception as e:
            print(f"Error with {model}: {str(e)}")


if __name__ == "__main__":
    print("Note: This example requires Ollama to be installed and running locally.")
    print("If you don't have Ollama installed, visit: https://ollama.com\n")
    
    asyncio.run(main())
