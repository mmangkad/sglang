# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Pydantic models for Anthropic Messages API protocol"""

import time
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field, field_validator


class AnthropicError(BaseModel):
    """Error structure for Anthropic API"""

    type: str
    message: str


class AnthropicErrorResponse(BaseModel):
    """Error response structure for Anthropic API"""

    type: Literal["error"] = "error"
    error: AnthropicError


class AnthropicUsage(BaseModel):
    """Token usage information"""

    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_input_tokens: Optional[int] = None
    cache_read_input_tokens: Optional[int] = None


class AnthropicTextBlock(BaseModel):
    """Text content block"""

    type: Literal["text"] = "text"
    text: str


class AnthropicImageSource(BaseModel):
    """Image source for image content blocks"""

    type: Literal["base64", "url"] = "base64"
    media_type: Optional[Literal["image/jpeg", "image/png", "image/gif", "image/webp"]] = None
    data: Optional[str] = None
    url: Optional[str] = None


class AnthropicImageBlock(BaseModel):
    """Image content block"""

    type: Literal["image"] = "image"
    source: AnthropicImageSource


class AnthropicToolUseBlock(BaseModel):
    """Tool use content block in response"""

    type: Literal["tool_use"] = "tool_use"
    id: str
    name: str
    input: Dict[str, Any]


class AnthropicToolResultContent(BaseModel):
    """Content within a tool result - can be text or image"""

    type: Literal["text", "image"]
    text: Optional[str] = None
    source: Optional[AnthropicImageSource] = None


class AnthropicToolResultBlock(BaseModel):
    """Tool result content block in request"""

    type: Literal["tool_result"] = "tool_result"
    tool_use_id: str
    content: Optional[Union[str, List[AnthropicToolResultContent]]] = None
    is_error: Optional[bool] = None


class AnthropicThinkingBlock(BaseModel):
    """Thinking content block"""

    type: Literal["thinking"] = "thinking"
    thinking: str
    signature: Optional[str] = None


class AnthropicRedactedThinkingBlock(BaseModel):
    """Redacted thinking content block"""

    type: Literal["redacted_thinking"] = "redacted_thinking"
    data: str


# Union type for all content block types
AnthropicContentBlock = Union[
    AnthropicTextBlock,
    AnthropicImageBlock,
    AnthropicToolUseBlock,
    AnthropicToolResultBlock,
    AnthropicThinkingBlock,
    AnthropicRedactedThinkingBlock,
]


class AnthropicContentBlockParam(BaseModel):
    """Generic content block for request messages - handles parsing"""

    type: Literal["text", "image", "tool_use", "tool_result", "thinking", "redacted_thinking"]
    # Text block fields
    text: Optional[str] = None
    # Image block fields
    source: Optional[Dict[str, Any]] = None
    # Tool use block fields
    id: Optional[str] = None
    name: Optional[str] = None
    input: Optional[Dict[str, Any]] = None
    # Tool result block fields
    tool_use_id: Optional[str] = None
    content: Optional[Union[str, List[Dict[str, Any]]]] = None
    is_error: Optional[bool] = None
    # Thinking block fields
    thinking: Optional[str] = None
    signature: Optional[str] = None
    # Redacted thinking block fields
    data: Optional[str] = None
    # Cache control
    cache_control: Optional[Dict[str, Any]] = None


class AnthropicMessage(BaseModel):
    """Message structure for request"""

    role: Literal["user", "assistant"]
    content: Union[str, List[AnthropicContentBlockParam]]


class AnthropicToolInputSchema(BaseModel):
    """Input schema for tool definition"""

    type: str = "object"
    properties: Optional[Dict[str, Any]] = None
    required: Optional[List[str]] = None

    class Config:
        extra = "allow"


class AnthropicTool(BaseModel):
    """Tool definition"""

    name: str
    description: Optional[str] = None
    input_schema: Dict[str, Any]

    @field_validator("input_schema")
    @classmethod
    def validate_input_schema(cls, v):
        if not isinstance(v, dict):
            raise ValueError("input_schema must be a dictionary")
        if "type" not in v:
            v["type"] = "object"  # Default to object type
        return v


class AnthropicToolChoice(BaseModel):
    """Tool choice definition"""

    type: Literal["auto", "any", "tool", "none"] = "auto"
    name: Optional[str] = None
    disable_parallel_tool_use: Optional[bool] = None


class AnthropicThinkingConfig(BaseModel):
    """Extended thinking configuration"""

    type: Literal["enabled", "disabled"] = "disabled"
    budget_tokens: Optional[int] = None


class AnthropicMetadata(BaseModel):
    """Request metadata"""

    user_id: Optional[str] = None


class AnthropicSystemContent(BaseModel):
    """System content block"""

    type: Literal["text"] = "text"
    text: str
    cache_control: Optional[Dict[str, Any]] = None


class AnthropicMessagesRequest(BaseModel):
    """Anthropic Messages API request"""

    model: str
    messages: List[AnthropicMessage]
    max_tokens: int
    metadata: Optional[AnthropicMetadata] = None
    stop_sequences: Optional[List[str]] = None
    stream: Optional[bool] = False
    system: Optional[Union[str, List[AnthropicSystemContent]]] = None
    temperature: Optional[float] = None
    thinking: Optional[AnthropicThinkingConfig] = None
    tool_choice: Optional[Union[AnthropicToolChoice, Dict[str, Any]]] = None
    tools: Optional[List[AnthropicTool]] = None
    top_k: Optional[int] = None
    top_p: Optional[float] = None

    @field_validator("model")
    @classmethod
    def validate_model(cls, v):
        if not v:
            raise ValueError("Model is required")
        return v

    @field_validator("max_tokens")
    @classmethod
    def validate_max_tokens(cls, v):
        if v <= 0:
            raise ValueError("max_tokens must be positive")
        return v


# Response content block for output
class AnthropicResponseContentBlock(BaseModel):
    """Content block in response"""

    type: Literal["text", "tool_use", "thinking", "redacted_thinking"]
    # Text block
    text: Optional[str] = None
    # Tool use block
    id: Optional[str] = None
    name: Optional[str] = None
    input: Optional[Dict[str, Any]] = None
    # Thinking block
    thinking: Optional[str] = None
    signature: Optional[str] = None
    # Redacted thinking
    data: Optional[str] = None


class AnthropicMessagesResponse(BaseModel):
    """Anthropic Messages API response"""

    id: str = Field(default_factory=lambda: f"msg_{int(time.time() * 1000)}")
    type: Literal["message"] = "message"
    role: Literal["assistant"] = "assistant"
    content: List[AnthropicResponseContentBlock]
    model: str
    stop_reason: Optional[Literal["end_turn", "max_tokens", "stop_sequence", "tool_use"]] = None
    stop_sequence: Optional[str] = None
    usage: Optional[AnthropicUsage] = None


# Streaming event types
class AnthropicDelta(BaseModel):
    """Delta for streaming responses"""

    type: Optional[Literal["text_delta", "input_json_delta", "thinking_delta", "signature_delta"]] = None
    text: Optional[str] = None
    partial_json: Optional[str] = None
    thinking: Optional[str] = None
    signature: Optional[str] = None

    # Message delta fields
    stop_reason: Optional[Literal["end_turn", "max_tokens", "stop_sequence", "tool_use"]] = None
    stop_sequence: Optional[str] = None


class AnthropicContentBlockStart(BaseModel):
    """Content block for content_block_start event - same structure as AnthropicContentBlock"""

    type: Literal["text", "tool_use", "thinking"]
    # Text block
    text: Optional[str] = None
    # Tool use block
    id: Optional[str] = None
    name: Optional[str] = None
    input: Optional[Dict[str, Any]] = None
    # Thinking block
    thinking: Optional[str] = None


class AnthropicStreamEvent(BaseModel):
    """Streaming event"""

    type: Literal[
        "message_start",
        "message_delta",
        "message_stop",
        "content_block_start",
        "content_block_delta",
        "content_block_stop",
        "ping",
        "error",
    ]
    message: Optional[AnthropicMessagesResponse] = None
    delta: Optional[AnthropicDelta] = None
    content_block: Optional[AnthropicContentBlockStart] = None
    index: Optional[int] = None
    error: Optional[AnthropicError] = None
    usage: Optional[AnthropicUsage] = None
