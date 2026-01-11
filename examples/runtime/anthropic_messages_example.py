"""
Example demonstrating how to use SGLang with Anthropic-compatible API.

Usage:
1. Start SGLang server:
   python -m sglang.launch_server --model-path meta-llama/Llama-3.2-1B-Instruct --port 8000

2. Run this script:
   python anthropic_messages_example.py

You can also use the official Anthropic Python SDK by setting:
   ANTHROPIC_BASE_URL=http://localhost:8000
   ANTHROPIC_API_KEY=sk-anything  # Any value works

Example with official SDK:
   from anthropic import Anthropic
   client = Anthropic(base_url="http://localhost:8000", api_key="sk-anything")
   response = client.messages.create(
       model="meta-llama/Llama-3.2-1B-Instruct",
       max_tokens=100,
       messages=[{"role": "user", "content": "Hello!"}]
   )
"""

import json
import requests

# SGLang server URL
BASE_URL = "http://localhost:8000"


def example_basic_message():
    """Basic message example"""
    print("=" * 50)
    print("Example 1: Basic Message")
    print("=" * 50)

    response = requests.post(
        f"{BASE_URL}/v1/messages",
        headers={"Content-Type": "application/json"},
        json={
            "model": "default",  # Use whatever model is served
            "max_tokens": 100,
            "messages": [{"role": "user", "content": "What is 2 + 2? Answer briefly."}],
        },
    )

    if response.status_code == 200:
        result = response.json()
        print(f"Response ID: {result['id']}")
        print(f"Model: {result['model']}")
        print(f"Stop reason: {result['stop_reason']}")
        print(f"Content: {result['content'][0]['text']}")
        print(f"Usage: {result['usage']}")
    else:
        print(f"Error: {response.status_code}")
        print(response.text)


def example_with_system_prompt():
    """Message with system prompt"""
    print("\n" + "=" * 50)
    print("Example 2: With System Prompt")
    print("=" * 50)

    response = requests.post(
        f"{BASE_URL}/v1/messages",
        headers={"Content-Type": "application/json"},
        json={
            "model": "default",
            "max_tokens": 150,
            "system": "You are a helpful assistant that speaks like a pirate.",
            "messages": [{"role": "user", "content": "Tell me about the weather today."}],
        },
    )

    if response.status_code == 200:
        result = response.json()
        print(f"Content: {result['content'][0]['text']}")
    else:
        print(f"Error: {response.status_code}")
        print(response.text)


def example_multi_turn():
    """Multi-turn conversation"""
    print("\n" + "=" * 50)
    print("Example 3: Multi-turn Conversation")
    print("=" * 50)

    response = requests.post(
        f"{BASE_URL}/v1/messages",
        headers={"Content-Type": "application/json"},
        json={
            "model": "default",
            "max_tokens": 150,
            "messages": [
                {"role": "user", "content": "My name is Alice."},
                {"role": "assistant", "content": "Hello Alice! Nice to meet you."},
                {"role": "user", "content": "What is my name?"},
            ],
        },
    )

    if response.status_code == 200:
        result = response.json()
        print(f"Content: {result['content'][0]['text']}")
    else:
        print(f"Error: {response.status_code}")
        print(response.text)


def example_streaming():
    """Streaming response example"""
    print("\n" + "=" * 50)
    print("Example 4: Streaming")
    print("=" * 50)

    response = requests.post(
        f"{BASE_URL}/v1/messages",
        headers={"Content-Type": "application/json"},
        json={
            "model": "default",
            "max_tokens": 200,
            "stream": True,
            "messages": [{"role": "user", "content": "Count from 1 to 10 slowly."}],
        },
        stream=True,
    )

    if response.status_code == 200:
        print("Streaming response:")
        for line in response.iter_lines():
            if line:
                line = line.decode("utf-8")
                if line.startswith("event:"):
                    event_type = line.split(": ", 1)[1] if ": " in line else ""
                    print(f"  Event: {event_type}")
                elif line.startswith("data:"):
                    data = line[5:].strip()
                    if data and data != "[DONE]":
                        try:
                            parsed = json.loads(data)
                            if parsed.get("type") == "content_block_delta":
                                delta = parsed.get("delta", {})
                                if delta.get("type") == "text_delta":
                                    print(f"  Text: {delta.get('text', '')}", end="")
                        except json.JSONDecodeError:
                            pass
        print()  # Final newline
    else:
        print(f"Error: {response.status_code}")
        print(response.text)


def example_with_tools():
    """Tool use example"""
    print("\n" + "=" * 50)
    print("Example 5: Tool Use")
    print("=" * 50)

    # Define a simple calculator tool
    tools = [
        {
            "name": "calculator",
            "description": "A simple calculator that can add, subtract, multiply, or divide two numbers.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "enum": ["add", "subtract", "multiply", "divide"],
                        "description": "The operation to perform",
                    },
                    "a": {"type": "number", "description": "First number"},
                    "b": {"type": "number", "description": "Second number"},
                },
                "required": ["operation", "a", "b"],
            },
        }
    ]

    response = requests.post(
        f"{BASE_URL}/v1/messages",
        headers={"Content-Type": "application/json"},
        json={
            "model": "default",
            "max_tokens": 500,
            "tools": tools,
            "tool_choice": {"type": "auto"},
            "messages": [{"role": "user", "content": "What is 25 * 4?"}],
        },
    )

    if response.status_code == 200:
        result = response.json()
        print(f"Stop reason: {result['stop_reason']}")
        for block in result["content"]:
            if block["type"] == "text":
                print(f"Text: {block['text']}")
            elif block["type"] == "tool_use":
                print(f"Tool use: {block['name']}")
                print(f"  ID: {block['id']}")
                print(f"  Input: {block['input']}")
    else:
        print(f"Error: {response.status_code}")
        print(response.text)


def example_with_temperature():
    """Example with temperature setting"""
    print("\n" + "=" * 50)
    print("Example 6: With Temperature")
    print("=" * 50)

    response = requests.post(
        f"{BASE_URL}/v1/messages",
        headers={"Content-Type": "application/json"},
        json={
            "model": "default",
            "max_tokens": 100,
            "temperature": 0.0,  # Deterministic
            "messages": [{"role": "user", "content": "What is the capital of France?"}],
        },
    )

    if response.status_code == 200:
        result = response.json()
        print(f"Content: {result['content'][0]['text']}")
    else:
        print(f"Error: {response.status_code}")
        print(response.text)


if __name__ == "__main__":
    print("SGLang Anthropic Messages API Examples")
    print("Make sure SGLang server is running on http://localhost:8000")
    print()

    try:
        example_basic_message()
        example_with_system_prompt()
        example_multi_turn()
        example_streaming()
        example_with_tools()
        example_with_temperature()
    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to SGLang server.")
        print("Please start the server first with:")
        print("  python -m sglang.launch_server --model-path <your-model> --port 8000")
