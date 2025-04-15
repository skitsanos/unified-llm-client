from typing import Optional, TypedDict, Literal, List, Dict, Any, Callable

# Type for stream handler callback function
StreamHandler = Callable[[str], Any]


class Message(TypedDict):
    """Structure for a single message in a conversation."""
    role: Literal["user", "assistant", "system", "developer", "tool"]
    content: str
    tool_call_id: Optional[str]  # Only present for tool messages


class ToolCallResponse(TypedDict):
    """Structure for a tool call response."""
    tool_call_id: str
    output: str


class LLMResponse(TypedDict):
    """Type definition for the LLM response object."""
    text: str
    input_tokens: int
    output_tokens: int
    response_id: Optional[str]
    sources: Optional[List[str]]


# Model provider types
ModelProvider = Literal["openai", "anthropic", "ollama"]


# Tool definition types
class ParameterProperty(TypedDict, total=False):
    type: str
    description: Optional[str]
    enum: Optional[List[str]]
    default: Any
    items: Optional[Dict[str, Any]]


class Parameters(TypedDict, total=False):
    type: str
    properties: Dict[str, ParameterProperty]
    required: List[str]


class ToolFunction(TypedDict, total=False):
    name: str
    description: str
    parameters: Parameters


class Tool(TypedDict, total=False):
    type: Literal["function"]
    function: ToolFunction
    name: Optional[str]


class AnthropicToolSchema(TypedDict):
    """Schema for Anthropic tool definition."""
    name: str
    description: str
    input_schema: Dict[str, Any]


class OpenAIToolSchema(TypedDict):
    """Schema for OpenAI tool definition."""
    type: Literal["function"]
    name: str
    function: Dict[str, Any]
