# Naori AI

A provider-agnostic Rust library for interacting with AI services. Switch between Ollama, Anthropic, OpenAI, and any OpenAI-compatible API with identical code.

[![Crates.io](https://img.shields.io/crates/v/naori-ai.svg)](https://crates.io/crates/naori-ai)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Features

- Single Interface: Same API across all AI providers
- Streaming Support: Real-time response streaming
- Vision Capabilities: Image analysis and multimodal conversations  
- Tool Calling: Function execution with automatic fallback for unsupported models
- Tool Macros: Automatically converts function doc comments into AI tool descriptions
- Model Management: List, pull, and inspect available models
- Async/Await: Full async support with proper error handling

## Supported Providers

Ollama, Anthropic, and OpenAI all support chat, streaming, vision, tools, and model management through the same interface. Additionally, any OpenAI-compatible API can be used via custom base URL configuration.

## Quick Start

Add library:
```bash
cargo add naori-ai naori-ai-macros
```

Add dependencies:

```bash
cargo add tokio --features full 
cargo add futures-util serde_json
```

See the `examples/` directory for complete working examples that demonstrate the library's capabilities.

## Environment Variables

Set API keys via environment variables for the providers you want to use

```bash
export OPENAI_API_KEY="your-openai-key"
export ANTHROPIC_API_KEY="your-anthropic-key"
```

## Examples

The `examples/` directory contains three comprehensive examples demonstrating all library features, and outside of the constructor, all the code stays the same no matter the model

### Chat

Interactive chat application with provider selection menu (Ollama, Anthropic, OpenAI) and automatic model discovery. Implements streaming chat responses, tool calling with custom functions (weather lookup, password generation), conversation history management, and error handling.

### Vision Chat

Multimodal chat application for image analysis. Takes image file path as command line argument, performs initial analysis, then enables interactive conversation about the image. Handles base64 encoding, message formatting, conversation context preservation, streaming responses, and tool calls across all vision-capable models & providers.

### Ollama Management

Model management utility for Ollama instances. pulling models from registry with progress tracking, model inspection (templates, parameters), and lifecycle management.

Run examples:

```bash
cd examples/chat && cargo run
cd examples/chat-vision && cargo run path/to/image.jpg
cd examples/ollama-management && cargo run
```

## API Reference

### Creating Clients

Besides constructing the client, the rest of the code is completely provider agnostic.
```rust
// Local Ollama instance
let client = NaoriAI::ollama("http://localhost:11434".to_string(), "qwen3-coder:30b".to_string());

// Cloud providers
let client = NaoriAI::openai(api_key, "gpt-5".to_string());
let client = NaoriAI::anthropic(api_key, "claude-sonnet-4.5".to_string());
let client = NaoriAI::openrouter(api_key, "anthropic/claude-sonnet-4.5".to_string());

// OpenAI-compatible APIs
let client = NaoriAI::openai_custom(api_key, "grok-code-fast-1".to_string(), "https://api.x.ai/v1".to_string());
```

### Core

#### Chat
- `send_chat_request(&messages)` - Streaming chat
- `send_chat_request_no_stream(&messages)` - Complete response
- `generate(prompt)` - Simple completion
- `generate_stream(prompt)` - Streaming completion

#### Vision  
- `send_chat_request_with_images(&messages, image_paths)` - Chat with images from files
- `send_chat_request_with_image_data(&messages, image_data)` - Chat with image bytes
- `encode_image_file(path)` - Encode image file to base64
- `encode_image_data(bytes)` - Encode image bytes to base64

#### Tool
- `add_tool(tool)` - Add function tool
- `handle_tool_calls(tool_calls)` - Execute tools and format responses
- `supports_tool_calls()` - Check native tool support
- `is_fallback_mode()` - Check if using XML fallback
- `process_fallback_response(content)` - Parse fallback tool calls

#### Model
- `get_available_models()` - List available models (works with all providers)

#### Usage Tracking
- Token usage automatically tracked in streaming responses via `ChatStreamItem.usage` (prompt tokens, completion tokens, total tokens)

#### Ollama Management
- `show_model_info(model)` - Get model details (Ollama only)  
- `pull_model(model)` - Download model (Ollama only)
- `pull_model_stream(model)` - Download with progress (Ollama only)

### Tool Definition

Use the `#[tool]` macro to define tool functions

```rust
use naori_ai_macros::tool;

/// The AI will see this doc comment
/// Describe what your tool does and its purpose here
/// The macro automatically provides parameter names, types, and marks all as required
/// You should explain what the function returns and provide usage guidance
#[tool]
fn my_function(param1: String, param2: i32) -> String {
    format!("Got {} and {}", param1, param2)
}

// Add to client
client.add_tool(my_function_tool()).await?;
```

## Advanced Features

### Token Usage Tracking

All providers support automatic token usage tracking in streaming responses:

```rust
let mut stream = client.send_chat_request(&messages).await?;
while let Some(item) = stream.next().await {
    let item = item?;
    
    // Token usage is available in the final stream item
    if let Some(usage) = item.usage {
        println!("Usage: {} input + {} output = {} total tokens", 
            usage.prompt_tokens.unwrap_or(0),
            usage.completion_tokens.unwrap_or(0),
            usage.total_tokens.unwrap_or(0)
        );
    }
}
```

Provider-specific usage details:
- **OpenAI & Compatible APIs**: Includes usage in final chunk with `stream_options: {include_usage: true}`
- **Anthropic**: Usage provided via `MessageDelta` events in streaming
- **Ollama**: Usage from `prompt_eval_count` and `eval_count` fields

### Fallback Tool Calling

Models without native tool support automatically use XML-based fallbacks, if you want to know if it's using it or not, feel free to use the is_fallback_mode function

```rust
if client.is_fallback_mode().await {
    println!("Using XML fallback for tools");
}

// Enable debug mode to see raw XML
client.set_debug_mode(true);
```

## OpenAI-Compatible APIs

Any OpenAI-compatible API can be used with the `openai_custom()` constructor. Examples:

```rust
// Groq
let client = NaoriAI::openai_custom(groq_api_key, "mixtral-8x7b-32768".to_string(), 
    "https://api.groq.com/openai/v1".to_string());
```

All OpenAI-compatible APIs work seamlessly with the same chat, streaming, vision, and tool-calling features.

## License

MIT License

## Contributing

Contributions welcome! Feel free to submit issues and pull requests.