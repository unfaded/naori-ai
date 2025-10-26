use futures_util::{Stream, StreamExt};
use reqwest::Client;
use std::error::Error;
use std::pin::Pin;
use std::collections::HashMap;
use bytes::Bytes;

use crate::core::{Message, ToolCall, ChatStreamItem, Tool, TokenUsage};
use super::types::*;

pub struct AnthropicClient {
    client: Client,
    api_key: String,
    pub model: String,
    tools: Vec<Tool>,
}

impl AnthropicClient {
    pub fn new(api_key: String, model: String) -> Self {
        Self {
            client: Client::new(),
            api_key,
            model,
            tools: Vec::new(),
        }
    }

    pub async fn add_tool(&mut self, tool: Tool) -> Result<(), Box<dyn Error>> {
        self.tools.push(tool);
        Ok(())
    }

    pub async fn is_fallback_mode(&self) -> bool {
        false // Anthropic has native tool support
    }

    pub fn set_debug_mode(&mut self, _debug: bool) {
        // Anthropic debug mode not yet implemented or planned
    }

    pub fn debug_mode(&self) -> bool {
        false
    }

    pub async fn supports_tool_calls(&self) -> Result<bool, Box<dyn Error>> {
        Ok(true) // Anthropic Claude models support native tool calling
    }

    pub async fn get_available_models(&self) -> Result<Vec<AnthropicModel>, Box<dyn Error>> {
        let response = self
            .client
            .get("https://api.anthropic.com/v1/models")
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", "2023-06-01")
            .send()
            .await?;

        if !response.status().is_success() {
            let error_text = response.text().await?;
            return Err(format!("Anthropic API error: {}", error_text).into());
        }

        let models_response: AnthropicModelsResponse = response.json().await?;
        Ok(models_response.data)
    }

    fn convert_to_anthropic_message(&self, message: &Message) -> AnthropicMessage {
        // Check if this is a tool result message
        if message.role == "user" && message.content.starts_with("TOOL_RESULT:") {
            // Parse the encoded tool result: "TOOL_RESULT:tool_id:result_content"
            let parts: Vec<&str> = message.content.splitn(3, ':').collect();
            if parts.len() == 3 {
                let tool_use_id = parts[1];
                let result_content = parts[2];

                let content_blocks = vec![ContentBlock::ToolResult {
                    tool_use_id: tool_use_id.to_string(),
                    content: result_content.to_string(),
                }];

                return AnthropicMessage {
                    role: message.role.clone(),
                    content: content_blocks,
                };
            }
        }

        let mut content_blocks = vec![ContentBlock::Text {
            text: message.content.clone(),
        }];

        // Add images if present
        if let Some(images) = &message.images {
            for image_data in images {
                content_blocks.insert(0, ContentBlock::Image {
                    source: ImageSource {
                        source_type: "base64".to_string(),
                        media_type: "image/jpeg".to_string(), 
                        data: image_data.clone(),
                    },
                });
            }
        }

        // Add tool calls if present
        if let Some(tool_calls) = &message.tool_calls {
            for tool_call in tool_calls {
                let tool_id = tool_call.id.clone().unwrap_or_else(|| format!("call_{}", "generated_id"));
                content_blocks.push(ContentBlock::ToolUse {
                    id: tool_id,
                    name: tool_call.function.name.clone(),
                    input: tool_call.function.arguments.clone(),
                });
            }
        }

        AnthropicMessage {
            role: message.role.clone(),
            content: content_blocks,
        }
    }

    fn convert_tools_to_anthropic(&self) -> Vec<AnthropicTool> {
        self.tools
            .iter()
            .map(|tool| AnthropicTool {
                name: tool.name.clone(),
                description: tool.description.clone(),
                input_schema: tool.parameters.clone(),
            })
            .collect()
    }

    pub async fn send_chat_request(
        &self,
        messages: &[Message],
    ) -> Result<Pin<Box<dyn Stream<Item = Result<ChatStreamItem, String>> + Send>>, Box<dyn Error>> {
        let anthropic_messages: Vec<AnthropicMessage> = messages
            .iter()
            .map(|msg| self.convert_to_anthropic_message(msg))
            .collect();


        let request = AnthropicRequest {
            model: self.model.clone(),
            max_tokens: 4096,
            messages: anthropic_messages,
            system: None,
            temperature: None,
            tools: if self.tools.is_empty() {
                None
            } else {
                Some(self.convert_tools_to_anthropic())
            },
            stream: Some(true),
        };

        let response = self
            .client
            .post("https://api.anthropic.com/v1/messages")
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", "2023-06-01")
            .header("content-type", "application/json")
            .json(&request)
            .send()
            .await?;

        if !response.status().is_success() {
            let error_text = response.text().await?;
            return Err(format!("Anthropic API error: {}", error_text).into());
        }

        let stream = response.bytes_stream();
        
        // Create a stateful stream processor
        Ok(Box::pin(AnthropicStreamProcessor::new(stream)))
    }

    pub async fn send_chat_request_no_stream(
        &self,
        messages: &[Message],
    ) -> Result<(String, Option<Vec<ToolCall>>), Box<dyn Error>> {
        let mut full_response = String::new();
        let mut tool_calls: Option<Vec<ToolCall>> = None;
        let mut stream = self.send_chat_request(messages).await?;

        while let Some(item) = stream.next().await {
            let item = item.map_err(|e| format!("Stream error: {}", e))?;
            if !item.content.is_empty() {
                full_response.push_str(&item.content);
            }
            if let Some(tc) = item.tool_calls {
                tool_calls = Some(tc);
            }
            if item.done {
                return Ok((full_response, tool_calls));
            }
        }
        Ok((full_response, tool_calls))
    }

    pub async fn handle_tool_calls(&self, tool_calls: Vec<ToolCall>) -> Vec<Message> {
        let mut tool_responses = Vec::new();
        for tool_call in tool_calls {
            if let Some(tool) = self
                .tools
                .iter()
                .find(|t| t.name == tool_call.function.name)
            {
                let result = (tool.function)(tool_call.function.arguments.clone());
                
                // Use the tool call ID if available, otherwise use "unknown"
                let tool_id = tool_call.id.unwrap_or_else(|| "unknown".to_string());
                
                // Create a message that can be identified as a tool result
                // Use the encoded format: TOOL_RESULT:tool_id:result_content
                tool_responses.push(Message {
                    role: "user".to_string(),
                    content: format!("TOOL_RESULT:{}:{}", tool_id, result),
                    images: None,
                    tool_calls: None,
                });
            }
        }
        tool_responses
    }

    pub async fn process_fallback_response(&self, content: &str) -> (String, Option<Vec<ToolCall>>) {
        // Anthropic doesn't need fallback processing
        (content.to_string(), None)
    }
}

// Custom stream processor to handle stateful tool call accumulation
struct AnthropicStreamProcessor {
    inner: Pin<Box<dyn Stream<Item = Result<Bytes, reqwest::Error>> + Send>>,
    // Track tool calls being accumulated: tool_id -> (name, accumulated_json)
    accumulating_tools: HashMap<String, (String, String)>,
    pending_results: std::collections::VecDeque<Result<ChatStreamItem, String>>,
    usage: Option<TokenUsage>,
}

impl AnthropicStreamProcessor {
    fn new(stream: impl Stream<Item = Result<Bytes, reqwest::Error>> + Send + 'static) -> Self {
        Self {
            inner: Box::pin(stream),
            accumulating_tools: HashMap::new(),
            pending_results: std::collections::VecDeque::new(),
            usage: None,
        }
    }
    
}

impl Stream for AnthropicStreamProcessor {
    type Item = Result<ChatStreamItem, String>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut std::task::Context<'_>) -> std::task::Poll<Option<Self::Item>> {
        loop {
            // Return any pending results first
            if let Some(result) = self.pending_results.pop_front() {
                return std::task::Poll::Ready(Some(result));
            }

            // Poll the inner stream
            match self.inner.as_mut().poll_next(cx) {
                std::task::Poll::Ready(Some(chunk_result)) => {
                    match chunk_result {
                        Ok(chunk) => {
                            let lines = chunk.split(|&b| b == b'\n');

                            for line in lines {
                                if line.is_empty() {
                                    continue;
                                }

                                // Skip "data: " prefix from SSE
                                let line_str = String::from_utf8_lossy(line);
                                if line_str.starts_with("data: ") {
                                    let json_str = &line_str[6..];
                                    if json_str.trim() == "[DONE]" {
                                        self.pending_results.push_back(Ok(ChatStreamItem {
                                            content: String::new(),
                                            tool_calls: None,
                                            done: true,
                                            usage: None,
                                        }));
                                        continue;
                                    }

                                    if let Ok(event) = serde_json::from_str::<StreamingEvent>(json_str) {
                                        match event {
                                            StreamingEvent::ContentBlockDelta { delta, .. } => {
                                                match delta {
                                                    Delta::TextDelta { text } => {
                                                        self.pending_results.push_back(Ok(ChatStreamItem {
                                                            content: text,
                                                            tool_calls: None,
                                                            done: false,
                                                            usage: None,
                                                        }));
                                                    }
                                                    Delta::InputJsonDelta { partial_json } => {
                                                        // Find the most recently added tool (last in iteration order)
                                                        if let Some((_, accumulated_json)) = self.accumulating_tools.values_mut().last() {
                                                            accumulated_json.push_str(&partial_json);
                                                        }
                                                    }
                                                }
                                            }
                                            StreamingEvent::ContentBlockStart { content_block, .. } => {
                                                if let ContentBlock::ToolUse { id, name, input: _ } = content_block {
                                                    // Start accumulating a new tool call
                                                    self.accumulating_tools.insert(id, (name, String::new()));
                                                }
                                            }
                                            StreamingEvent::ContentBlockStop { .. } => {
                                                // Finish all accumulated tool calls
                                                let mut completed_tools = Vec::new();
                                                for (tool_id, (tool_name, accumulated_json)) in self.accumulating_tools.drain() {
                                                    if let Ok(arguments) = serde_json::from_str::<serde_json::Value>(&accumulated_json) {
                                                        // Create tool call with the ID properly stored
                                                        let tool_call = ToolCall {
                                                            id: Some(tool_id),
                                                            function: crate::core::Function {
                                                                name: tool_name,
                                                                arguments,
                                                            },
                                                        };
                                                        completed_tools.push(tool_call);
                                                    }
                                                }
                                                
                                                if !completed_tools.is_empty() {
                                                    self.pending_results.push_back(Ok(ChatStreamItem {
                                                        content: String::new(),
                                                        tool_calls: Some(completed_tools),
                                                        done: false,
                                                        usage: None,
                                                    }));
                                                }
                                            }
                                            StreamingEvent::MessageDelta { delta } => {
                                                if let Some(usage) = delta.usage {
                                                    self.usage = Some(TokenUsage {
                                                        prompt_tokens: Some(usage.input_tokens),
                                                        completion_tokens: Some(usage.output_tokens),
                                                        total_tokens: Some(usage.input_tokens + usage.output_tokens),
                                                    });
                                                }
                                            }
                                            StreamingEvent::MessageStop => {
                                                let usage = self.usage.clone();
                                                self.pending_results.push_back(Ok(ChatStreamItem {
                                                    content: String::new(),
                                                    tool_calls: None,
                                                    done: true,
                                                    usage,
                                                }));
                                            }
                                            StreamingEvent::Ping => {
                                                // Ignore ping events
                                            }
                                            _ => {
                                                // Handle other event types as needed
                                            }
                                        }
                                    }
                                }
                            }
                            // Continue the loop to check for pending results
                        }
                        Err(e) => return std::task::Poll::Ready(Some(Err(e.to_string())))
                    }
                }
                std::task::Poll::Ready(None) => return std::task::Poll::Ready(None),
                std::task::Poll::Pending => return std::task::Poll::Pending,
            }
        }
    }
}