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
"""Anthropic Messages API serving handler for SGLang"""

import json
import logging
import time
import uuid
from typing import TYPE_CHECKING, Any, AsyncGenerator, Dict, List, Optional, Union

from fastapi import Request
from fastapi.responses import ORJSONResponse, StreamingResponse

from sglang.srt.entrypoints.anthropic.protocol import (
    AnthropicContentBlockStart,
    AnthropicDelta,
    AnthropicError,
    AnthropicMessagesRequest,
    AnthropicMessagesResponse,
    AnthropicResponseContentBlock,
    AnthropicStreamEvent,
    AnthropicUsage,
)
from sglang.srt.entrypoints.openai.protocol import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionStreamResponse,
    ErrorResponse,
    StreamOptions,
    Tool,
    ToolChoice,
    Function,
)
from sglang.srt.entrypoints.openai.serving_chat import OpenAIServingChat

if TYPE_CHECKING:
    from sglang.srt.managers.template_manager import TemplateManager
    from sglang.srt.managers.tokenizer_manager import TokenizerManager

logger = logging.getLogger(__name__)


def wrap_sse_event(data: str, event: str) -> str:
    """Wrap data with SSE event format"""
    return f"event: {event}\ndata: {data}\n\n"


class AnthropicServingMessages:
    """Handler for Anthropic Messages API requests"""

    def __init__(
        self,
        tokenizer_manager: "TokenizerManager",
        template_manager: "TemplateManager",
    ):
        # Create an internal OpenAI serving chat handler to reuse its logic
        self.openai_chat = OpenAIServingChat(tokenizer_manager, template_manager)
        self.tokenizer_manager = tokenizer_manager
        self.template_manager = template_manager

        # Mapping from OpenAI finish reasons to Anthropic stop reasons
        self.stop_reason_map = {
            "stop": "end_turn",
            "length": "max_tokens",
            "tool_calls": "tool_use",
        }

    def _convert_anthropic_to_openai_request(
        self, anthropic_request: AnthropicMessagesRequest
    ) -> ChatCompletionRequest:
        """Convert Anthropic message format to OpenAI format"""
        openai_messages = []

        # Add system message if provided
        if anthropic_request.system:
            if isinstance(anthropic_request.system, str):
                openai_messages.append(
                    {"role": "system", "content": anthropic_request.system}
                )
            else:
                # List of system content blocks
                system_prompt = ""
                for block in anthropic_request.system:
                    if hasattr(block, "text"):
                        system_prompt += block.text
                    elif isinstance(block, dict) and "text" in block:
                        system_prompt += block["text"]
                openai_messages.append({"role": "system", "content": system_prompt})

        # Convert messages
        for msg in anthropic_request.messages:
            openai_msg: Dict[str, Any] = {"role": msg.role}

            if isinstance(msg.content, str):
                openai_msg["content"] = msg.content
            else:
                # Handle complex content blocks
                content_parts: List[Dict[str, Any]] = []
                tool_calls: List[Dict[str, Any]] = []

                for block in msg.content:
                    if block.type == "text" and block.text:
                        content_parts.append({"type": "text", "text": block.text})

                    elif block.type == "image" and block.source:
                        # Convert Anthropic image format to OpenAI format
                        source = block.source
                        if isinstance(source, dict):
                            if source.get("type") == "base64":
                                media_type = source.get("media_type", "image/png")
                                data = source.get("data", "")
                                url = f"data:{media_type};base64,{data}"
                            else:
                                url = source.get("url", "")
                        else:
                            if source.type == "base64":
                                media_type = source.media_type or "image/png"
                                data = source.data or ""
                                url = f"data:{media_type};base64,{data}"
                            else:
                                url = source.url or ""

                        content_parts.append({
                            "type": "image_url",
                            "image_url": {"url": url}
                        })

                    elif block.type == "tool_use":
                        # Convert tool use to function call format (assistant message)
                        tool_call = {
                            "id": block.id or f"call_{uuid.uuid4().hex[:24]}",
                            "type": "function",
                            "function": {
                                "name": block.name or "",
                                "arguments": json.dumps(block.input or {}),
                            },
                        }
                        tool_calls.append(tool_call)

                    elif block.type == "tool_result":
                        # Tool results in user messages become separate tool messages
                        if msg.role == "user":
                            tool_content = ""
                            if isinstance(block.content, str):
                                tool_content = block.content
                            elif isinstance(block.content, list):
                                # Extract text from content list
                                texts = []
                                for item in block.content:
                                    if isinstance(item, dict) and item.get("type") == "text":
                                        texts.append(item.get("text", ""))
                                tool_content = "\n".join(texts)
                            
                            openai_messages.append({
                                "role": "tool",
                                "tool_call_id": block.tool_use_id or "",
                                "content": tool_content,
                            })
                        else:
                            # Shouldn't happen normally, but handle gracefully
                            tool_result_text = str(block.content) if block.content else ""
                            content_parts.append({
                                "type": "text",
                                "text": f"Tool result: {tool_result_text}",
                            })

                    elif block.type == "thinking" and block.thinking:
                        # Include thinking content as text for now
                        content_parts.append({
                            "type": "text",
                            "text": block.thinking,
                        })

                # Add tool calls to the message if any
                if tool_calls:
                    openai_msg["tool_calls"] = tool_calls

                # Add content parts if any
                if content_parts:
                    if len(content_parts) == 1 and content_parts[0]["type"] == "text":
                        openai_msg["content"] = content_parts[0]["text"]
                    else:
                        openai_msg["content"] = content_parts
                elif not tool_calls:
                    # Skip empty messages
                    continue

            openai_messages.append(openai_msg)

        # Build the OpenAI request
        request_dict = {
            "model": anthropic_request.model,
            "messages": openai_messages,
            "max_completion_tokens": anthropic_request.max_tokens,
            "stream": anthropic_request.stream or False,
        }

        # Optional parameters
        if anthropic_request.stop_sequences:
            request_dict["stop"] = anthropic_request.stop_sequences
        if anthropic_request.temperature is not None:
            request_dict["temperature"] = anthropic_request.temperature
        if anthropic_request.top_p is not None:
            request_dict["top_p"] = anthropic_request.top_p
        if anthropic_request.top_k is not None:
            request_dict["top_k"] = anthropic_request.top_k

        # Handle streaming options
        if anthropic_request.stream:
            request_dict["stream_options"] = {
                "include_usage": True,
                "continuous_usage_stats": True,
            }

        # Handle tool choice
        if anthropic_request.tool_choice is not None:
            tool_choice = anthropic_request.tool_choice
            if isinstance(tool_choice, dict):
                choice_type = tool_choice.get("type", "auto")
            else:
                choice_type = tool_choice.type

            if choice_type == "auto":
                request_dict["tool_choice"] = "auto"
            elif choice_type == "any":
                request_dict["tool_choice"] = "required"
            elif choice_type == "none":
                request_dict["tool_choice"] = "none"
            elif choice_type == "tool":
                tool_name = tool_choice.get("name") if isinstance(tool_choice, dict) else tool_choice.name
                request_dict["tool_choice"] = {
                    "type": "function",
                    "function": {"name": tool_name},
                }

        # Handle tools
        if anthropic_request.tools:
            tools = []
            for tool in anthropic_request.tools:
                tools.append({
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.input_schema,
                    },
                })
            request_dict["tools"] = tools
            # Set default tool choice if not specified
            if "tool_choice" not in request_dict:
                request_dict["tool_choice"] = "auto"

        # Handle thinking/reasoning
        if anthropic_request.thinking:
            if anthropic_request.thinking.type == "enabled":
                request_dict["chat_template_kwargs"] = {"thinking": True}

        return ChatCompletionRequest(**request_dict)

    def _convert_openai_to_anthropic_response(
        self, openai_response: ChatCompletionResponse
    ) -> AnthropicMessagesResponse:
        """Convert OpenAI response to Anthropic format"""
        content: List[AnthropicResponseContentBlock] = []

        # Get the first choice (Anthropic only returns one)
        choice = openai_response.choices[0] if openai_response.choices else None

        if choice:
            # Add text content
            if choice.message.content:
                content.append(
                    AnthropicResponseContentBlock(
                        type="text",
                        text=choice.message.content,
                    )
                )

            # Add tool calls
            if choice.message.tool_calls:
                for tool_call in choice.message.tool_calls:
                    try:
                        tool_input = json.loads(tool_call.function.arguments or "{}")
                    except json.JSONDecodeError:
                        tool_input = {}

                    content.append(
                        AnthropicResponseContentBlock(
                            type="tool_use",
                            id=tool_call.id or f"toolu_{uuid.uuid4().hex[:24]}",
                            name=tool_call.function.name,
                            input=tool_input,
                        )
                    )

        # If no content was added, add an empty text block
        if not content:
            content.append(AnthropicResponseContentBlock(type="text", text=""))

        # Map finish reason
        stop_reason = None
        if choice and choice.finish_reason:
            stop_reason = self.stop_reason_map.get(choice.finish_reason, "end_turn")

        return AnthropicMessagesResponse(
            id=f"msg_{openai_response.id}" if not openai_response.id.startswith("msg_") else openai_response.id,
            content=content,
            model=openai_response.model,
            stop_reason=stop_reason,
            usage=AnthropicUsage(
                input_tokens=openai_response.usage.prompt_tokens if openai_response.usage else 0,
                output_tokens=openai_response.usage.completion_tokens if openai_response.usage else 0,
            ),
        )

    async def create_messages(
        self,
        request: AnthropicMessagesRequest,
        raw_request: Request,
    ) -> Union[AnthropicMessagesResponse, StreamingResponse, ORJSONResponse]:
        """
        Handle Anthropic Messages API request.
        
        See https://docs.anthropic.com/en/api/messages for the API specification.
        """
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("Received Anthropic messages request: %s", request.model_dump_json())

        try:
            # Convert to OpenAI format
            openai_request = self._convert_anthropic_to_openai_request(request)

            if logger.isEnabledFor(logging.DEBUG):
                logger.debug("Converted to OpenAI request: %s", openai_request.model_dump_json())

            # Use the OpenAI handler
            response = await self.openai_chat.handle_request(openai_request, raw_request)

            # Handle error responses
            if isinstance(response, ORJSONResponse):
                # Check if it's an error response
                return response

            # Handle streaming
            if request.stream:
                if isinstance(response, StreamingResponse):
                    # Convert the streaming response
                    return StreamingResponse(
                        self._convert_stream(response.body_iterator),
                        media_type="text/event-stream",
                    )

            # Handle non-streaming response
            if isinstance(response, ChatCompletionResponse):
                anthropic_response = self._convert_openai_to_anthropic_response(response)
                return ORJSONResponse(anthropic_response.model_dump(exclude_none=True))

            # If response is already ORJSONResponse, try to extract and convert
            return response

        except Exception as e:
            logger.exception("Error processing Anthropic messages request")
            error_response = {
                "type": "error",
                "error": {
                    "type": "internal_error",
                    "message": str(e),
                },
            }
            return ORJSONResponse(error_response, status_code=500)

    async def _convert_stream(
        self,
        openai_stream: AsyncGenerator[bytes, None],
    ) -> AsyncGenerator[str, None]:
        """Convert OpenAI streaming response to Anthropic format"""
        first_chunk = True
        content_block_started = False
        content_block_index = 0
        finish_reason = None
        message_id = None
        model_name = None
        input_tokens = 0
        output_tokens = 0
        current_tool_call_id = None
        tool_call_in_progress = False

        try:
            async for chunk_bytes in openai_stream:
                chunk_str = chunk_bytes.decode("utf-8") if isinstance(chunk_bytes, bytes) else chunk_bytes

                # Process each line in the chunk
                for line in chunk_str.strip().split("\n"):
                    if not line.startswith("data:"):
                        continue

                    data_str = line[5:].strip()
                    if data_str == "[DONE]":
                        # Send message_stop event
                        stop_event = AnthropicStreamEvent(type="message_stop")
                        yield wrap_sse_event(
                            stop_event.model_dump_json(exclude_unset=True, exclude_none=True),
                            "message_stop"
                        )
                        return

                    try:
                        openai_chunk = ChatCompletionStreamResponse.model_validate_json(data_str)
                    except Exception as e:
                        logger.debug(f"Failed to parse OpenAI chunk: {e}")
                        continue

                    # Track message ID and model
                    if openai_chunk.id:
                        message_id = f"msg_{openai_chunk.id}" if not openai_chunk.id.startswith("msg_") else openai_chunk.id
                    if openai_chunk.model:
                        model_name = openai_chunk.model

                    # Send message_start on first chunk
                    if first_chunk:
                        if openai_chunk.usage:
                            input_tokens = openai_chunk.usage.prompt_tokens or 0

                        start_event = AnthropicStreamEvent(
                            type="message_start",
                            message=AnthropicMessagesResponse(
                                id=message_id or f"msg_{int(time.time() * 1000)}",
                                content=[],
                                model=model_name or "unknown",
                                usage=AnthropicUsage(
                                    input_tokens=input_tokens,
                                    output_tokens=0,
                                ),
                            ),
                        )
                        first_chunk = False
                        yield wrap_sse_event(
                            start_event.model_dump_json(exclude_unset=True, exclude_none=True),
                            "message_start"
                        )
                        continue

                    # Handle empty choices (usage info chunk)
                    if not openai_chunk.choices:
                        if openai_chunk.usage:
                            input_tokens = openai_chunk.usage.prompt_tokens or input_tokens
                            output_tokens = openai_chunk.usage.completion_tokens or output_tokens

                        # Close current content block if open
                        if content_block_started:
                            stop_block_event = AnthropicStreamEvent(
                                type="content_block_stop",
                                index=content_block_index,
                            )
                            yield wrap_sse_event(
                                stop_block_event.model_dump_json(exclude_unset=True, exclude_none=True),
                                "content_block_stop"
                            )
                            content_block_started = False

                        # Send message_delta with stop reason and final usage
                        stop_reason = self.stop_reason_map.get(finish_reason or "stop", "end_turn")
                        delta_event = AnthropicStreamEvent(
                            type="message_delta",
                            delta=AnthropicDelta(stop_reason=stop_reason),
                            usage=AnthropicUsage(
                                input_tokens=input_tokens,
                                output_tokens=output_tokens,
                            ),
                        )
                        yield wrap_sse_event(
                            delta_event.model_dump_json(exclude_unset=True, exclude_none=True),
                            "message_delta"
                        )
                        continue

                    choice = openai_chunk.choices[0]

                    # Track finish reason
                    if choice.finish_reason:
                        finish_reason = choice.finish_reason
                        continue

                    # Handle text content
                    if choice.delta and choice.delta.content is not None:
                        # Start content block if not started
                        if not content_block_started:
                            start_block_event = AnthropicStreamEvent(
                                type="content_block_start",
                                index=content_block_index,
                                content_block=AnthropicContentBlockStart(
                                    type="text",
                                    text="",
                                ),
                            )
                            yield wrap_sse_event(
                                start_block_event.model_dump_json(exclude_unset=True, exclude_none=True),
                                "content_block_start"
                            )
                            content_block_started = True
                            tool_call_in_progress = False

                        # Send content delta if non-empty
                        if choice.delta.content:
                            delta_event = AnthropicStreamEvent(
                                type="content_block_delta",
                                index=content_block_index,
                                delta=AnthropicDelta(
                                    type="text_delta",
                                    text=choice.delta.content,
                                ),
                            )
                            yield wrap_sse_event(
                                delta_event.model_dump_json(exclude_unset=True, exclude_none=True),
                                "content_block_delta"
                            )

                    # Handle tool calls
                    elif choice.delta and choice.delta.tool_calls:
                        for tool_call in choice.delta.tool_calls:
                            # New tool call starting
                            if tool_call.id is not None:
                                # Close previous content block if open
                                if content_block_started:
                                    stop_block_event = AnthropicStreamEvent(
                                        type="content_block_stop",
                                        index=content_block_index,
                                    )
                                    yield wrap_sse_event(
                                        stop_block_event.model_dump_json(exclude_unset=True, exclude_none=True),
                                        "content_block_stop"
                                    )
                                    content_block_started = False
                                    content_block_index += 1

                                # Start new tool use block
                                current_tool_call_id = tool_call.id
                                start_block_event = AnthropicStreamEvent(
                                    type="content_block_start",
                                    index=content_block_index,
                                    content_block=AnthropicContentBlockStart(
                                        type="tool_use",
                                        id=tool_call.id,
                                        name=tool_call.function.name if tool_call.function else "",
                                        input={},
                                    ),
                                )
                                yield wrap_sse_event(
                                    start_block_event.model_dump_json(exclude_unset=True, exclude_none=True),
                                    "content_block_start"
                                )
                                content_block_started = True
                                tool_call_in_progress = True

                            # Tool call arguments delta
                            elif tool_call.function and tool_call.function.arguments:
                                delta_event = AnthropicStreamEvent(
                                    type="content_block_delta",
                                    index=content_block_index,
                                    delta=AnthropicDelta(
                                        type="input_json_delta",
                                        partial_json=tool_call.function.arguments,
                                    ),
                                )
                                yield wrap_sse_event(
                                    delta_event.model_dump_json(exclude_unset=True, exclude_none=True),
                                    "content_block_delta"
                                )

        except Exception as e:
            logger.exception("Error in stream converter")
            error_event = AnthropicStreamEvent(
                type="error",
                error=AnthropicError(
                    type="internal_error",
                    message=str(e),
                ),
            )
            yield wrap_sse_event(
                error_event.model_dump_json(exclude_unset=True, exclude_none=True),
                "error"
            )
