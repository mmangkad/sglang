"""
Unit-tests for AnthropicServingMessages.

Run with:
    python -m pytest test/registered/anthropic_server/test_anthropic_serving_messages.py -v
"""

import json
import unittest
import uuid
from typing import Optional
from unittest.mock import AsyncMock, Mock, patch

from fastapi import Request

from sglang.srt.entrypoints.anthropic.protocol import (
    AnthropicContentBlockParam,
    AnthropicMessage,
    AnthropicMessagesRequest,
    AnthropicMessagesResponse,
    AnthropicResponseContentBlock,
    AnthropicTool,
    AnthropicToolChoice,
    AnthropicUsage,
)
from sglang.srt.entrypoints.anthropic.serving_messages import AnthropicServingMessages
from sglang.srt.entrypoints.openai.protocol import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionResponseChoice,
    ChatMessage,
    ToolCall,
    FunctionResponse,
    UsageInfo,
)
from sglang.srt.utils import get_or_create_event_loop
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

register_cuda_ci(est_time=10, suite="stage-b-test-small-1-gpu")
register_amd_ci(est_time=10, suite="stage-b-test-small-1-gpu-amd")


class _MockTokenizerManager:
    """Minimal mock that satisfies AnthropicServingMessages."""

    def __init__(self):
        self.model_config = Mock(is_multimodal=False)
        self.server_args = Mock(
            enable_cache_report=False,
            tool_call_parser="hermes",
            reasoning_parser=None,
            enable_lora=False,
        )
        # Mock hf_config for _use_dpsk_v32_encoding check
        mock_hf_config = Mock()
        mock_hf_config.architectures = ["LlamaForCausalLM"]
        self.model_config.hf_config = mock_hf_config

        self.chat_template_name: Optional[str] = "llama-3"

        # tokenizer stub
        self.tokenizer = Mock()
        self.tokenizer.encode.return_value = [1, 2, 3, 4, 5]
        self.tokenizer.decode.return_value = "Test response"
        self.tokenizer.chat_template = None
        self.tokenizer.bos_token_id = 1

        # async generator stub for generate_request
        async def _mock_generate():
            yield {
                "text": "Test response",
                "meta_info": {
                    "id": f"chatcmpl-{uuid.uuid4()}",
                    "prompt_tokens": 10,
                    "completion_tokens": 5,
                    "cached_tokens": 0,
                    "finish_reason": {"type": "stop", "matched": None},
                    "output_token_logprobs": [(0.1, 1, "Test"), (0.2, 2, "response")],
                    "output_top_logprobs": None,
                    "weight_version": 1,
                },
                "index": 0,
            }

        self.generate_request = Mock(return_value=_mock_generate())
        self.create_abort_task = Mock()


class _MockTemplateManager:
    """Minimal mock for TemplateManager."""

    def __init__(self):
        self.chat_template_name: Optional[str] = "llama-3"
        self.jinja_template_content_format: Optional[str] = None
        self.completion_template_name: Optional[str] = None
        self.force_reasoning = False


class AnthropicServingMessagesTestCase(unittest.TestCase):
    """Test cases for AnthropicServingMessages."""

    def setUp(self):
        self.tm = _MockTokenizerManager()
        self.template_manager = _MockTemplateManager()
        self.handler = AnthropicServingMessages(self.tm, self.template_manager)

        self.fastapi_request = Mock(spec=Request)
        self.fastapi_request.headers = {}

    # ------------- Protocol Model Tests -------------
    def test_anthropic_messages_request_basic(self):
        """Test basic AnthropicMessagesRequest creation."""
        request = AnthropicMessagesRequest(
            model="claude-3-sonnet",
            max_tokens=100,
            messages=[
                AnthropicMessage(role="user", content="Hello!")
            ],
        )
        self.assertEqual(request.model, "claude-3-sonnet")
        self.assertEqual(request.max_tokens, 100)
        self.assertEqual(len(request.messages), 1)
        self.assertEqual(request.stream, False)

    def test_anthropic_messages_request_with_system(self):
        """Test AnthropicMessagesRequest with system prompt."""
        request = AnthropicMessagesRequest(
            model="claude-3-sonnet",
            max_tokens=100,
            system="You are a helpful assistant.",
            messages=[
                AnthropicMessage(role="user", content="Hello!")
            ],
        )
        self.assertEqual(request.system, "You are a helpful assistant.")

    def test_anthropic_messages_request_with_tools(self):
        """Test AnthropicMessagesRequest with tools."""
        tool = AnthropicTool(
            name="calculator",
            description="A calculator",
            input_schema={
                "type": "object",
                "properties": {
                    "a": {"type": "number"},
                    "b": {"type": "number"},
                },
                "required": ["a", "b"],
            },
        )
        request = AnthropicMessagesRequest(
            model="claude-3-sonnet",
            max_tokens=100,
            tools=[tool],
            messages=[
                AnthropicMessage(role="user", content="What is 2+2?")
            ],
        )
        self.assertEqual(len(request.tools), 1)
        self.assertEqual(request.tools[0].name, "calculator")

    def test_anthropic_messages_request_validation(self):
        """Test validation of max_tokens."""
        with self.assertRaises(ValueError):
            AnthropicMessagesRequest(
                model="claude-3-sonnet",
                max_tokens=0,  # Invalid
                messages=[
                    AnthropicMessage(role="user", content="Hello!")
                ],
            )

    # ------------- Conversion Tests -------------
    def test_convert_basic_request(self):
        """Test conversion of basic request to OpenAI format."""
        anthropic_req = AnthropicMessagesRequest(
            model="claude-3-sonnet",
            max_tokens=100,
            messages=[
                AnthropicMessage(role="user", content="Hello!")
            ],
        )

        openai_req = self.handler._convert_anthropic_to_openai_request(anthropic_req)

        self.assertIsInstance(openai_req, ChatCompletionRequest)
        self.assertEqual(openai_req.model, "claude-3-sonnet")
        self.assertEqual(openai_req.max_completion_tokens, 100)
        self.assertEqual(len(openai_req.messages), 1)
        self.assertEqual(openai_req.messages[0]["role"], "user")
        self.assertEqual(openai_req.messages[0]["content"], "Hello!")

    def test_convert_request_with_system(self):
        """Test conversion with system prompt."""
        anthropic_req = AnthropicMessagesRequest(
            model="claude-3-sonnet",
            max_tokens=100,
            system="Be helpful.",
            messages=[
                AnthropicMessage(role="user", content="Hello!")
            ],
        )

        openai_req = self.handler._convert_anthropic_to_openai_request(anthropic_req)

        self.assertEqual(len(openai_req.messages), 2)
        self.assertEqual(openai_req.messages[0]["role"], "system")
        self.assertEqual(openai_req.messages[0]["content"], "Be helpful.")

    def test_convert_request_with_temperature(self):
        """Test conversion with temperature."""
        anthropic_req = AnthropicMessagesRequest(
            model="claude-3-sonnet",
            max_tokens=100,
            temperature=0.5,
            messages=[
                AnthropicMessage(role="user", content="Hello!")
            ],
        )

        openai_req = self.handler._convert_anthropic_to_openai_request(anthropic_req)

        self.assertEqual(openai_req.temperature, 0.5)

    def test_convert_request_with_tools(self):
        """Test conversion with tools."""
        tool = AnthropicTool(
            name="calculator",
            description="A calculator",
            input_schema={
                "type": "object",
                "properties": {"a": {"type": "number"}},
            },
        )
        anthropic_req = AnthropicMessagesRequest(
            model="claude-3-sonnet",
            max_tokens=100,
            tools=[tool],
            tool_choice=AnthropicToolChoice(type="auto"),
            messages=[
                AnthropicMessage(role="user", content="Calculate!")
            ],
        )

        openai_req = self.handler._convert_anthropic_to_openai_request(anthropic_req)

        self.assertEqual(len(openai_req.tools), 1)
        self.assertEqual(openai_req.tools[0]["function"]["name"], "calculator")
        self.assertEqual(openai_req.tool_choice, "auto")

    def test_convert_tool_choice_any(self):
        """Test conversion of tool_choice 'any' to 'required'."""
        tool = AnthropicTool(
            name="calc",
            description="A calc",
            input_schema={"type": "object"},
        )
        anthropic_req = AnthropicMessagesRequest(
            model="claude-3-sonnet",
            max_tokens=100,
            tools=[tool],
            tool_choice=AnthropicToolChoice(type="any"),
            messages=[
                AnthropicMessage(role="user", content="Calculate!")
            ],
        )

        openai_req = self.handler._convert_anthropic_to_openai_request(anthropic_req)

        self.assertEqual(openai_req.tool_choice, "required")

    def test_convert_multi_turn_conversation(self):
        """Test conversion of multi-turn conversation."""
        anthropic_req = AnthropicMessagesRequest(
            model="claude-3-sonnet",
            max_tokens=100,
            messages=[
                AnthropicMessage(role="user", content="Hi"),
                AnthropicMessage(role="assistant", content="Hello!"),
                AnthropicMessage(role="user", content="How are you?"),
            ],
        )

        openai_req = self.handler._convert_anthropic_to_openai_request(anthropic_req)

        self.assertEqual(len(openai_req.messages), 3)
        self.assertEqual(openai_req.messages[0]["content"], "Hi")
        self.assertEqual(openai_req.messages[1]["content"], "Hello!")
        self.assertEqual(openai_req.messages[2]["content"], "How are you?")

    # ------------- Response Conversion Tests -------------
    def test_convert_openai_to_anthropic_response(self):
        """Test conversion of OpenAI response to Anthropic format."""
        openai_response = ChatCompletionResponse(
            id="chatcmpl-123",
            model="claude-3-sonnet",
            created=1234567890,
            choices=[
                ChatCompletionResponseChoice(
                    index=0,
                    message=ChatMessage(
                        role="assistant",
                        content="Hello! How can I help you today?",
                    ),
                    finish_reason="stop",
                )
            ],
            usage=UsageInfo(
                prompt_tokens=10,
                completion_tokens=8,
                total_tokens=18,
            ),
        )

        anthropic_response = self.handler._convert_openai_to_anthropic_response(openai_response)

        self.assertIsInstance(anthropic_response, AnthropicMessagesResponse)
        self.assertTrue(anthropic_response.id.startswith("msg_"))
        self.assertEqual(anthropic_response.model, "claude-3-sonnet")
        self.assertEqual(anthropic_response.role, "assistant")
        self.assertEqual(anthropic_response.stop_reason, "end_turn")
        self.assertEqual(len(anthropic_response.content), 1)
        self.assertEqual(anthropic_response.content[0].type, "text")
        self.assertEqual(anthropic_response.content[0].text, "Hello! How can I help you today?")
        self.assertEqual(anthropic_response.usage.input_tokens, 10)
        self.assertEqual(anthropic_response.usage.output_tokens, 8)

    def test_convert_response_with_tool_calls(self):
        """Test conversion of response with tool calls."""
        openai_response = ChatCompletionResponse(
            id="chatcmpl-123",
            model="claude-3-sonnet",
            created=1234567890,
            choices=[
                ChatCompletionResponseChoice(
                    index=0,
                    message=ChatMessage(
                        role="assistant",
                        content="Let me calculate that.",
                        tool_calls=[
                            ToolCall(
                                id="call_123",
                                type="function",
                                function=FunctionResponse(
                                    name="calculator",
                                    arguments='{"a": 2, "b": 2}',
                                ),
                            )
                        ],
                    ),
                    finish_reason="tool_calls",
                )
            ],
            usage=UsageInfo(
                prompt_tokens=10,
                completion_tokens=15,
                total_tokens=25,
            ),
        )

        anthropic_response = self.handler._convert_openai_to_anthropic_response(openai_response)

        self.assertEqual(anthropic_response.stop_reason, "tool_use")
        self.assertEqual(len(anthropic_response.content), 2)
        self.assertEqual(anthropic_response.content[0].type, "text")
        self.assertEqual(anthropic_response.content[1].type, "tool_use")
        self.assertEqual(anthropic_response.content[1].name, "calculator")
        self.assertEqual(anthropic_response.content[1].input, {"a": 2, "b": 2})

    def test_stop_reason_mapping(self):
        """Test mapping of finish reasons."""
        self.assertEqual(self.handler.stop_reason_map["stop"], "end_turn")
        self.assertEqual(self.handler.stop_reason_map["length"], "max_tokens")
        self.assertEqual(self.handler.stop_reason_map["tool_calls"], "tool_use")


class AnthropicProtocolTestCase(unittest.TestCase):
    """Test cases for Anthropic protocol models."""

    def test_content_block_text(self):
        """Test text content block."""
        block = AnthropicContentBlockParam(type="text", text="Hello world")
        self.assertEqual(block.type, "text")
        self.assertEqual(block.text, "Hello world")

    def test_content_block_tool_use(self):
        """Test tool use content block."""
        block = AnthropicContentBlockParam(
            type="tool_use",
            id="toolu_123",
            name="calculator",
            input={"a": 1, "b": 2},
        )
        self.assertEqual(block.type, "tool_use")
        self.assertEqual(block.id, "toolu_123")
        self.assertEqual(block.name, "calculator")
        self.assertEqual(block.input, {"a": 1, "b": 2})

    def test_content_block_tool_result(self):
        """Test tool result content block."""
        block = AnthropicContentBlockParam(
            type="tool_result",
            tool_use_id="toolu_123",
            content="The result is 3",
        )
        self.assertEqual(block.type, "tool_result")
        self.assertEqual(block.tool_use_id, "toolu_123")
        self.assertEqual(block.content, "The result is 3")

    def test_anthropic_response_content_block(self):
        """Test AnthropicResponseContentBlock."""
        text_block = AnthropicResponseContentBlock(
            type="text",
            text="Hello!",
        )
        self.assertEqual(text_block.type, "text")
        self.assertEqual(text_block.text, "Hello!")

        tool_block = AnthropicResponseContentBlock(
            type="tool_use",
            id="toolu_123",
            name="calculator",
            input={"a": 1},
        )
        self.assertEqual(tool_block.type, "tool_use")
        self.assertEqual(tool_block.id, "toolu_123")

    def test_anthropic_usage(self):
        """Test AnthropicUsage model."""
        usage = AnthropicUsage(
            input_tokens=100,
            output_tokens=50,
            cache_creation_input_tokens=10,
            cache_read_input_tokens=5,
        )
        self.assertEqual(usage.input_tokens, 100)
        self.assertEqual(usage.output_tokens, 50)
        self.assertEqual(usage.cache_creation_input_tokens, 10)
        self.assertEqual(usage.cache_read_input_tokens, 5)

    def test_anthropic_tool(self):
        """Test AnthropicTool model."""
        tool = AnthropicTool(
            name="get_weather",
            description="Get the weather for a location",
            input_schema={
                "type": "object",
                "properties": {
                    "location": {"type": "string"},
                },
                "required": ["location"],
            },
        )
        self.assertEqual(tool.name, "get_weather")
        self.assertEqual(tool.description, "Get the weather for a location")
        self.assertEqual(tool.input_schema["type"], "object")


if __name__ == "__main__":
    unittest.main()
