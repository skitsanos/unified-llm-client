# Unified LLM Client Documentation

Welcome to the Unified LLM Client documentation. This library provides a consistent interface for interacting with multiple Large Language Model (LLM) providers including OpenAI, Anthropic, and Ollama.

## Overview

The Unified LLM Client simplifies working with different LLM providers by providing a consistent API interface. It handles the complexities of different API formats, authentication, and response handling, allowing you to focus on your application logic.

## Key Features

- **Unified Interface**: Interact with multiple LLM providers using a consistent API
- **Async-First**: Built for high-performance applications with async/await support
- **Tool/Function Calling**: Consistent interface for tool calling across providers, with enhanced support for Claude
  models
- **Streaming Support**: Stream responses from OpenAI and Anthropic models for better UX with long responses
- **Error Handling**: Rich error handling and logging
- **Type Hinting**: Comprehensive type hints for better IDE support
- **Local Models**: Support for Ollama to run models locally

## Getting Started

Check out the [Quick Start Guide](quickstart.md) to begin using the library.

## Providers

- [OpenAI](quickstart.md) - GPT models like gpt-4o, gpt-4o-mini
- [Anthropic](quickstart.md) - Claude models like claude-3-opus, claude-3-sonnet
- [Ollama](ollama.md) - Run models locally like llama3, qwen2.5, mistral

## API Reference

For detailed information about the library's classes and methods, see the [API Reference](api_reference.md).

## Examples

The [Examples](examples.md) section provides practical code samples for common use cases, including streaming responses
and tool calling with different providers.

## Tool/Function Calling

Learn how to use [Tool/Function Calling](tools.md) with different LLM providers.

For Anthropic Claude-specific tool calling implementation, check the [Claude Tools](claude_tools.md) guide and the
detailed [Anthropic Tool Calling](anthropic/tool_calling.md) documentation.

## Changelog

See the [Changelog](changelog.md) for details on version 0.2.0 and earlier releases.
