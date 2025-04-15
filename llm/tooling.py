import inspect
import logging
from typing import Set, Dict, Any, Callable, List, Optional, TypeVar, Union, Awaitable, get_type_hints, get_origin, \
    get_args

from llm.types import OpenAIToolSchema, AnthropicToolSchema, Tool

logger = logging.getLogger(__name__)

# Types of tools that are handled internally by OpenAI
INTERNAL_TOOL_TYPES = {"file_search", "web_search", "web_search_preview", "code_interpreter", "retrieval"}

# Mapping Python types to JSON schema types
TYPE_MAP = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
    None: "null",
    type(None): "null"
}

# Type variable for tool function return type
T = TypeVar('T')
ToolFunction = Callable[..., Union[T, Awaitable[T]]]
DecoratedToolFunction = ToolFunction[T]


def llm_tool(func: ToolFunction[T]) -> DecoratedToolFunction[T]:
    """
    Generic decorator to register a function as a tool for multiple LLM providers.

    This decorator inspects the function signature and creates schema definitions
    for OpenAI, Anthropic, and potentially other LLM providers.

    Args:
        func: The function to register as a tool

    Returns:
        The decorated function with LLM-specific tool schemas attached
    """
    signature = inspect.signature(func)
    type_hints = get_type_hints(func)

    # Common properties for all LLM schemas
    func_name = func.__name__
    func_description = func.__doc__ or f"Function {func_name}"

    # Build parameters schema
    parameters: Dict[str, Any] = {
        "type": "object",
        "properties": {},
        "required": []
    }

    for name, param in signature.parameters.items():
        # Get the type annotation for this parameter
        param_type_hint = type_hints.get(name, Any)
        
        # Get the JSON schema type
        json_type = "string"  # Default to string
        
        # Handle Union types
        if get_origin(param_type_hint) is Union:
            # Get the Union arguments
            union_args = get_args(param_type_hint)
            # If one of the union types is None, it means this is an Optional parameter
            if type(None) in union_args or None in union_args:
                # Use the first non-None type
                non_none_types = [t for t in union_args if t is not type(None) and t is not None]
                if non_none_types:
                    param_type_hint = non_none_types[0]
                    json_type = TYPE_MAP.get(param_type_hint, "string")
            else:
                # Just take the first type in the Union
                json_type = TYPE_MAP.get(union_args[0], "string")
        else:
            # Handle regular types
            json_type = TYPE_MAP.get(param_type_hint, "string")
        
        # Get parameter description from docstring if available
        description = ""
        if func.__doc__:
            # Try to parse docstring to find parameter descriptions
            # This is a simple implementation - more sophisticated docstring parsing could be used
            doc_lines = func.__doc__.split("\n")
            for line in doc_lines:
                line = line.strip()
                if line.startswith(f"{name}:"):
                    description = line[len(name) + 1:].strip()
                elif line.startswith(f"    {name}: "):
                    description = line[len(name) + 6:].strip()
        
        # Add parameter to schema
        param_schema: Dict[str, Any] = {"type": json_type}
        if description:
            param_schema["description"] = description
            
        parameters["properties"][name] = param_schema
        
        # Add to required if no default value
        if param.default is param.empty:
            parameters["required"].append(name)

    # Create OpenAI tool schema
    openai_tool: OpenAIToolSchema = {
        "type": "function",
        "name": func_name,
        "function": {
            "description": func_description,
            "parameters": parameters
        }
    }
    setattr(func, 'openai_tool', openai_tool)

    # Create Anthropic tool schema
    # Make a deep copy to avoid reference problems
    import copy
    anthropic_tool: AnthropicToolSchema = {
        "name": func_name,
        "description": func_description,
        "input_schema": {
            "type": "object",  # This is required by Anthropic
            "properties": copy.deepcopy(parameters["properties"]),
            "required": copy.deepcopy(parameters["required"])
        }
    }
    # Log the schema with proper level
    logger.debug(f"Created Anthropic tool schema for {func_name}")
    setattr(func, 'anthropic_tool', anthropic_tool)

    # Generic tool schema (used by the registry for execution)
    tool: Tool = {
        "type": "function",
        "function": {
            "name": func_name,
            "description": func_description,
            "parameters": parameters
        }
    }
    setattr(func, 'tool', tool)

    return func


class ToolRegistry:
    """Registry for tool functions that can be called by LLM models."""
    
    def __init__(self, internal_tool_types: Optional[Set[str]] = None) -> None:
        """
        Initialize the tool registry.
        
        Args:
            internal_tool_types: Set of tool types that are handled internally by the LLM provider
        """
        self._tools: Dict[str, ToolFunction] = {}
        self._schema_cache: Dict[str, List[Any]] = {}  # Cache for schemas by provider
        self._cache_valid: Dict[str, bool] = {}  # Track if cache is valid for each provider
        self._internal_tool_types: Set[str] = internal_tool_types or INTERNAL_TOOL_TYPES.copy()

    def register(self, name: str, func: ToolFunction) -> 'ToolRegistry':
        """
        Register a tool function with its schema.
        
        Args:
            name: Name of the tool
            func: Tool function (should have the llm_tool decorator)
            
        Returns:
            Self for chaining
            
        Raises:
            ValueError: If the function does not have a tool attribute
        """
        if not hasattr(func, 'tool'):
            raise ValueError(f"Function {name} does not have a 'tool' attribute")

        self._tools[name] = func

        # Invalidate all schema caches when a new tool is registered
        for provider in self._cache_valid:
            self._cache_valid[provider] = False

        return self

    def unregister(self, name: str) -> 'ToolRegistry':
        """
        Unregister a tool function.
        
        Args:
            name: Name of the tool to unregister
            
        Returns:
            Self for chaining
        """
        if name in self._tools:
            del self._tools[name]

            # Invalidate all schema caches when a tool is removed
            for provider in self._cache_valid:
                self._cache_valid[provider] = False

        return self

    def is_internal_tool(self, tool_type: str) -> bool:
        """
        Check if a tool type is handled internally by the LLM provider (not by our registry).

        Args:
            tool_type: The type of the tool to check

        Returns:
            True if the tool type is handled internally, False otherwise
        """
        return tool_type in self._internal_tool_types

    def add_internal_tool_type(self, tool_type: str) -> None:
        """
        Register a tool type as being handled internally by the LLM provider.

        Args:
            tool_type: The type of internal tool
        """
        self._internal_tool_types.add(tool_type)

    def get_internal_tool_types(self) -> Set[str]:
        """
        Get all registered internal tool types.

        Returns:
            Set of internal tool types
        """
        return self._internal_tool_types.copy()

    def get(self, name: str) -> ToolFunction:
        """
        Get a tool function by name.
        
        Args:
            name: Name of the tool
            
        Returns:
            The tool function
            
        Raises:
            KeyError: If the tool is not registered
        """
        if name not in self._tools:
            raise KeyError(f"Tool '{name}' not registered")
        return self._tools[name]

    def get_schemas(self, provider: str = "openai") -> List[Any]:
        """
        Get all tool schemas for a specific provider.
        Uses cached schemas if available and valid.
        
        Args:
            provider: The provider to get schemas for (openai, anthropic, etc.)
            
        Returns:
            List of tool schemas formatted for the specified provider
        """
        # Initialize cache for provider if not exists
        if provider not in self._schema_cache:
            self._schema_cache[provider] = []
            self._cache_valid[provider] = False

        # Build schema cache if invalid
        if not self._cache_valid.get(provider, False):
            self._build_schema_cache(provider)
            self._cache_valid[provider] = True

        return self._schema_cache[provider]

    def _build_schema_cache(self, provider: str) -> None:
        """
        Build the schema cache for a specific provider.
        
        Args:
            provider: The provider to build the cache for
        """
        schemas = []

        for name, func in self._tools.items():
            if provider == "openai" and hasattr(func, 'openai_tool'):
                schemas.append(func.openai_tool)
            elif provider == "anthropic" and hasattr(func, 'anthropic_tool'):
                schemas.append(func.anthropic_tool)
            else:
                # Always add the generic tool schema if specific provider schema not available
                schemas.append(func.tool)

        self._schema_cache[provider] = schemas
        logger.debug(f"Built schema cache for provider {provider} with {len(schemas)} tools")

    def get_all(self) -> Dict[str, ToolFunction]:
        """
        Get all registered tools.
        
        Returns:
            Dictionary of tool names to tool functions
        """
        return self._tools

    def get_names(self) -> List[str]:
        """
        Get all registered tool names.
        
        Returns:
            List of tool names
        """
        return list(self._tools.keys())

    def has_tool(self, name: str) -> bool:
        """
        Check if a tool is registered.
        
        Args:
            name: Name of the tool
            
        Returns:
            True if the tool is registered, False otherwise
        """
        return name in self._tools

    def clear_cache(self, provider: Optional[str] = None) -> None:
        """
        Clear schema cache for a specific provider or all providers.

        Args:
            provider: Optional provider name, if None clears all caches
        """
        if provider:
            if provider in self._schema_cache:
                self._schema_cache[provider] = []
                self._cache_valid[provider] = False
        else:
            self._schema_cache = {}
            self._cache_valid = {}

        logger.debug(f"Cleared schema cache for {provider if provider else 'all providers'}")

    async def execute_tool(self, name: str, args: Dict[str, Any]) -> Any:
        """
        Execute a tool by name with the given arguments.
        Supports both synchronous and asynchronous tool functions.

        Args:
            name: Tool name
            args: Arguments to pass to the tool function

        Returns:
            Result from the tool function

        Raises:
            KeyError: If tool is not registered
            ValueError: If tool is an internal tool that should be handled by the LLM provider
            Exception: If tool execution fails
        """
        if not self.has_tool(name):
            raise KeyError(f"Tool '{name}' not registered")

        func = self.get(name)
        logger.info(f"Executing tool {name} with args: {args}")

        try:
            # Call the function with the provided arguments
            result = func(**args)

            # If the result is a coroutine (async function), await it
            if inspect.iscoroutine(result):
                result = await result

            return result
        except Exception as e:
            logger.error(f"Error executing tool {name}: {str(e)}")
            raise
