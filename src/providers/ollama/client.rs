use base64::{Engine as _, engine::general_purpose};
use futures_util::{Stream, StreamExt};
use reqwest::Client;
use serde_json::json;
use std::error::Error;
use std::pin::Pin;

use crate::core::{Message, ToolCall, ChatStreamItem, PullProgress, ModelInfo, Tool, FallbackToolHandler, TokenUsage};
use super::{OllamaOptions, ChatResponse, Model, ListModelsResponse};
use super::utilities::StreamingXmlFilter;


impl Tool {
    fn to_json(&self) -> serde_json::Value {
        json!({
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            }
        })
    }
}

pub struct OllamaClient {
    client: Client,
    pub endpoint: String,
    pub model: String,
    tools: Vec<Tool>,
    debug_mode: bool,
}

impl OllamaClient {
    pub fn new(endpoint: String, model: String) -> Self {
        Self {
            client: Client::new(),
            endpoint,
            model,
            tools: Vec::new(),
            debug_mode: false,
        }
    }

    pub fn set_debug_mode(&mut self, debug: bool) {
        self.debug_mode = debug;
    }

    pub fn debug_mode(&self) -> bool {
        self.debug_mode
    }

    pub async fn add_tool(&mut self, tool: Tool) -> Result<(), Box<dyn Error>> {
        self.tools.push(tool);
        
        // Tool support is now determined dynamically when needed
        
        Ok(())
    }

    pub async fn is_fallback_mode(&self) -> bool {
        if self.tools.is_empty() {
            false // No tools, no fallback needed
        } else {
            // Dynamically check if model supports native tools
            !self.supports_tool_calls().await.unwrap_or(false)
        }
    }


    pub async fn supports_tool_calls(&self) -> Result<bool, Box<dyn Error>> {
        let model_info = self.show_model_info(&self.model).await?;

        // The definitive way to check tool support is the presence of .Tools in the template
        // All models that support tools use the .Tools variable in their prompt template
        let template = &model_info.template;
        let supports_tools = template.contains(".Tools") || template.contains(".tools");

        Ok(supports_tools)
    }

    pub async fn capabilities(&self) -> Result<crate::core::ProviderCapabilities, Box<dyn Error>> {
        Ok(crate::core::ProviderCapabilities {
            supports_tools: self.supports_tool_calls().await.unwrap_or(false),
            supports_vision: true,
        })
    }

    pub async fn list_local_models(&self) -> Result<Vec<Model>, Box<dyn Error>> {
        let response = self
            .client
            .get(&format!("{}/api/tags", self.endpoint))
            .send()
            .await?
            .json::<ListModelsResponse>()
            .await?;
        Ok(response.models)
    }

    pub async fn get_available_models(&self) -> Result<Vec<Model>, Box<dyn Error>> {
        self.list_local_models().await
    }

    pub async fn show_model_info(&self, model_name: &str) -> Result<ModelInfo, Box<dyn Error>> {
        let response = self
            .client
            .post(&format!("{}/api/show", self.endpoint))
            .json(&json!({ "name": model_name }))
            .send()
            .await?
            .json::<ModelInfo>()
            .await?;
        Ok(response)
    }

    pub async fn pull_model(&self, model_name: &str) -> Result<(), Box<dyn Error>> {
        println!("Pulling model: {}", model_name);
        let mut stream = self.pull_model_stream(model_name).await?;

        while let Some(progress) = stream.next().await {
            let progress = progress.map_err(|e| format!("Stream error: {}", e))?;
            println!("{}", progress.status);
        }
        Ok(())
    }

    pub async fn pull_model_stream(
        &self,
        model_name: &str,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<PullProgress, String>> + Send>>, Box<dyn Error>>
    {
        let stream = self
            .client
            .post(&format!("{}/api/pull", self.endpoint))
            .json(&json!({ "name": model_name, "stream": true }))
            .send()
            .await?
            .bytes_stream();

        let stream = stream.map(
            |item| -> Result<Vec<Result<PullProgress, String>>, Box<dyn Error>> {
                let chunk = item?;
                let lines = chunk.split(|&b| b == b'\n');
                let mut results = Vec::new();

                for line in lines {
                    if line.is_empty() {
                        continue;
                    }

                    let line_str = String::from_utf8_lossy(line);
                    match serde_json::from_str::<serde_json::Value>(&line_str) {
                        Ok(json) => {
                            results.push(Ok(PullProgress {
                                status: json
                                    .get("status")
                                    .and_then(|s| s.as_str())
                                    .unwrap_or("")
                                    .to_string(),
                                digest: json
                                    .get("digest")
                                    .and_then(|s| s.as_str())
                                    .map(|s| s.to_string()),
                                total: json.get("total").and_then(|n| n.as_u64()),
                                completed: json.get("completed").and_then(|n| n.as_u64()),
                            }));
                        }
                        Err(_) => {
                            results.push(Ok(PullProgress {
                                status: line_str.to_string(),
                                digest: None,
                                total: None,
                                completed: None,
                            }));
                        }
                    }
                }

                Ok(results)
            },
        );

        let flattened_stream = stream
            .map(
                |result: Result<Vec<Result<PullProgress, String>>, Box<dyn Error>>| match result {
                    Ok(items) => futures_util::stream::iter(items),
                    Err(e) => futures_util::stream::iter(vec![Err(e.to_string())]),
                },
            )
            .flatten();

        Ok(Box::pin(flattened_stream))
    }

    pub async fn send_chat_request_with_images(
        &self,
        messages: &[Message],
        image_paths: Vec<String>,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<ChatStreamItem, String>> + Send>>, Box<dyn Error>> {
        self.send_chat_request_with_images_stream_and_options(messages, image_paths, None).await
    }

    pub async fn send_chat_request_with_images_no_stream(
        &self,
        messages: &[Message],
        image_paths: Vec<String>,
    ) -> Result<(String, Option<Vec<ToolCall>>), Box<dyn Error>> {
        self.send_chat_request_with_images_no_stream_and_options(messages, image_paths, None).await
    }

    pub async fn send_chat_request_with_images_stream_and_options(
        &self,
        messages: &[Message],
        image_paths: Vec<String>,
        options: Option<OllamaOptions>,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<ChatStreamItem, String>> + Send>>, Box<dyn Error>> {
        let mut encoded_images = Vec::new();
        for image_path in image_paths {
            let image_bytes = std::fs::read(image_path)?;
            encoded_images.push(general_purpose::STANDARD.encode(image_bytes));
        }

        let mut messages_with_images = messages.to_vec();
        if let Some(last_message) = messages_with_images.last_mut() {
            last_message.images = Some(encoded_images);
        }

        self.send_chat_request_stream_with_options(&messages_with_images, options).await
    }

    pub async fn send_chat_request_with_images_no_stream_and_options(
        &self,
        messages: &[Message],
        image_paths: Vec<String>,
        options: Option<OllamaOptions>,
    ) -> Result<(String, Option<Vec<ToolCall>>), Box<dyn Error>> {
        let mut encoded_images = Vec::new();
        for image_path in image_paths {
            let image_bytes = std::fs::read(image_path)?;
            encoded_images.push(general_purpose::STANDARD.encode(image_bytes));
        }

        let mut messages_with_images = messages.to_vec();
        if let Some(last_message) = messages_with_images.last_mut() {
            last_message.images = Some(encoded_images);
        }

        self.send_chat_request_no_stream_with_options(&messages_with_images, options).await
    }

    pub async fn send_chat_request_with_images_data(
        &self,
        messages: &[Message],
        images_data: Vec<Vec<u8>>,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<ChatStreamItem, String>> + Send>>, Box<dyn Error>> {
        self.send_chat_request_with_images_data_stream_and_options(messages, images_data, None).await
    }

    pub async fn send_chat_request_with_images_data_no_stream(
        &self,
        messages: &[Message],
        images_data: Vec<Vec<u8>>,
    ) -> Result<(String, Option<Vec<ToolCall>>), Box<dyn Error>> {
        self.send_chat_request_with_images_data_no_stream_and_options(messages, images_data, None).await
    }

    pub async fn send_chat_request_with_images_data_stream_and_options(
        &self,
        messages: &[Message],
        images_data: Vec<Vec<u8>>,
        options: Option<OllamaOptions>,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<ChatStreamItem, String>> + Send>>, Box<dyn Error>> {
        let mut encoded_images = Vec::new();
        for image_bytes in images_data {
            encoded_images.push(general_purpose::STANDARD.encode(image_bytes));
        }

        let mut messages_with_images = messages.to_vec();
        if let Some(last_message) = messages_with_images.last_mut() {
            last_message.images = Some(encoded_images);
        }

        self.send_chat_request_stream_with_options(&messages_with_images, options).await
    }

    pub async fn send_chat_request_with_images_data_no_stream_and_options(
        &self,
        messages: &[Message],
        images_data: Vec<Vec<u8>>,
        options: Option<OllamaOptions>,
    ) -> Result<(String, Option<Vec<ToolCall>>), Box<dyn Error>> {
        let mut encoded_images = Vec::new();
        for image_bytes in images_data {
            encoded_images.push(general_purpose::STANDARD.encode(image_bytes));
        }

        let mut messages_with_images = messages.to_vec();
        if let Some(last_message) = messages_with_images.last_mut() {
            last_message.images = Some(encoded_images);
        }

        self.send_chat_request_no_stream_with_options(&messages_with_images, options).await
    }

    pub async fn send_chat_request(
        &self,
        messages: &[Message],
    ) -> Result<Pin<Box<dyn Stream<Item = Result<ChatStreamItem, String>> + Send>>, Box<dyn Error>> {
        self.send_chat_request_stream_with_options(messages, None).await
    }

    pub async fn send_chat_request_no_stream(
        &self,
        messages: &[Message],
    ) -> Result<(String, Option<Vec<ToolCall>>), Box<dyn Error>> {
        self.send_chat_request_no_stream_with_options(messages, None).await
    }

    pub async fn send_chat_request_no_stream_with_options(
        &self,
        messages: &[Message],
        options: Option<OllamaOptions>,
    ) -> Result<(String, Option<Vec<ToolCall>>), Box<dyn Error>> {
        let mut full_response = String::new();
        let mut tool_calls: Option<Vec<ToolCall>> = None;
        let mut stream = self.send_chat_request_stream_with_options(messages, options).await?;

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

    pub async fn send_chat_request_stream(
        &self,
        messages: &[Message],
    ) -> Result<Pin<Box<dyn Stream<Item = Result<ChatStreamItem, String>> + Send>>, Box<dyn Error>>
    {
        self.send_chat_request_stream_with_options(messages, None).await
    }

    pub async fn send_chat_request_stream_with_options(
        &self,
        messages: &[Message],
        options: Option<OllamaOptions>,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<ChatStreamItem, String>> + Send>>, Box<dyn Error>>
    {
        let mut messages_to_send = messages.to_vec();
        
        // In fallback mode, inject tool context into the system message
        let is_fallback = self.is_fallback_mode().await;
        if is_fallback && !self.tools.is_empty() {
            let tool_context = FallbackToolHandler::generate_tool_context(&self.tools);
            
            // Find existing system message or create one
            if let Some(system_msg) = messages_to_send.iter_mut().find(|msg| msg.role == "system") {
                system_msg.content.push_str(&tool_context);
            } else {
                // Insert system message at the beginning
                messages_to_send.insert(0, Message {
                    role: "system".to_string(),
                    content: format!("You are a helpful assistant.{}", tool_context),
                    images: None,
                    tool_calls: None,
                });
            }
        }

        let mut request_body = json!({
            "model": self.model,
            "messages": messages_to_send,
            "stream": true,
        });

        // Only add tools if not in fallback mode
        if !is_fallback && !self.tools.is_empty() {
            let tools_json: Vec<serde_json::Value> =
                self.tools.iter().map(|t| t.to_json()).collect();
            request_body["tools"] = serde_json::Value::Array(tools_json);
        }

        if let Some(opts) = options {
            request_body["options"] = serde_json::to_value(opts)?;
        }

        let stream = self
            .client
            .post(&format!("{}/api/chat", self.endpoint))
            .json(&request_body)
            .send()
            .await?
            .bytes_stream();

        let fallback_mode = self.is_fallback_mode().await;
        let debug_mode = self.debug_mode;
        
        // Create a stateful stream that handles tool calling internally
        let stream = futures_util::stream::unfold(
            (stream, StreamingXmlFilter::new(), String::new(), false),
            move |(mut stream, mut xml_filter, mut accumulated_raw, mut stream_done)| async move {
                match stream.next().await {
                    Some(chunk_result) => {
                        match chunk_result {
                            Ok(chunk) => {
                                let lines = chunk.split(|&b| b == b'\n');
                                let mut results = Vec::new();

                                for line in lines {
                                    if line.is_empty() {
                                        continue;
                                    }
                                    match serde_json::from_slice::<ChatResponse>(&line) {
                                        Ok(chat_response) => {
                                            let mut tool_calls = chat_response.message.tool_calls.clone();
                                            let raw_content = chat_response.message.content.clone();
                                            
                                            // Accumulate raw content for fallback tool detection
                                            accumulated_raw.push_str(&raw_content);
                                            
                                            // Apply XML filtering when debug is disabled
                                            let content = if !debug_mode {
                                                xml_filter.process_chunk(&raw_content)
                                            } else {
                                                raw_content.clone()
                                            };
                                            
                                            // On stream completion, check for fallback tool calls
                                            if chat_response.done && fallback_mode && tool_calls.is_none() {
                                                if let Some(fallback_tools) = FallbackToolHandler::parse_fallback_tool_calls(&accumulated_raw) {
                                                    tool_calls = Some(fallback_tools);
                                                }
                                                stream_done = true;
                                            }
                                            
                                            // Extract token usage if available (usually only on done=true)
                                            let usage = if chat_response.done {
                                                if let (Some(prompt_tokens), Some(completion_tokens)) = 
                                                    (chat_response.prompt_eval_count, chat_response.eval_count) {
                                                    Some(TokenUsage {
                                                        prompt_tokens: Some(prompt_tokens),
                                                        completion_tokens: Some(completion_tokens),
                                                        total_tokens: Some(prompt_tokens + completion_tokens),
                                                    })
                                                } else {
                                                    None
                                                }
                                            } else {
                                                None
                                            };
                                            
                                            results.push(Ok(ChatStreamItem {
                                                content,
                                                tool_calls,
                                                done: chat_response.done,
                                                usage,
                                            }));
                                        }
                                        Err(e) => {
                                            eprintln!("\nError parsing response: {}", e);
                                            eprintln!("Problematic line: {:?}", String::from_utf8_lossy(&line));
                                        }
                                    }
                                }
                                
                                Some((Ok(results), (stream, xml_filter, accumulated_raw, stream_done)))
                            }
                            Err(e) => Some((Err(Box::new(e) as Box<dyn Error>), (stream, xml_filter, accumulated_raw, stream_done)))
                        }
                    }
                    None => None
                }
            }
        );

        let flattened_stream = stream
            .map(
                |result| match result {
                    Ok(items) => futures_util::stream::iter(items),
                    Err(e) => futures_util::stream::iter(vec![Err(e.to_string())]),
                },
            )
            .flatten();

        Ok(Box::pin(flattened_stream))
    }

    pub async fn generate(
        &self,
        prompt: &str,
    ) -> Result<String, Box<dyn Error>> {
        self.generate_with_options(prompt, None).await
    }

    pub async fn generate_with_options(
        &self,
        prompt: &str,
        options: Option<OllamaOptions>,
    ) -> Result<String, Box<dyn Error>> {
        let mut request_body = json!({
            "model": self.model,
            "prompt": prompt,
            "stream": false,
        });

        if let Some(opts) = options {
            request_body["options"] = serde_json::to_value(opts)?;
        }

        let response = self
            .client
            .post(&format!("{}/api/generate", self.endpoint))
            .json(&request_body)
            .send()
            .await?;

        let response_json: serde_json::Value = response.json().await?;
        Ok(response_json["response"]
            .as_str()
            .unwrap_or("")
            .to_string())
    }

    pub async fn generate_stream(
        &self,
        prompt: &str,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<String, String>> + Send>>, Box<dyn Error>> {
        self.generate_stream_with_options(prompt, None).await
    }

    pub async fn generate_stream_with_options(
        &self,
        prompt: &str,
        options: Option<OllamaOptions>,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<String, String>> + Send>>, Box<dyn Error>> {
        let mut request_body = json!({
            "model": self.model,
            "prompt": prompt,
            "stream": true,
        });

        if let Some(opts) = options {
            request_body["options"] = serde_json::to_value(opts)?;
        }

        let stream = self
            .client
            .post(&format!("{}/api/generate", self.endpoint))
            .json(&request_body)
            .send()
            .await?
            .bytes_stream();

        let stream = stream.map(
            |item| -> Result<Vec<Result<String, String>>, Box<dyn Error>> {
                let chunk = item?;
                let lines = chunk.split(|&b| b == b'\n');
                let mut results = Vec::new();

                for line in lines {
                    if line.is_empty() {
                        continue;
                    }

                    match serde_json::from_slice::<serde_json::Value>(&line) {
                        Ok(json) => {
                            if let Some(response) = json["response"].as_str() {
                                results.push(Ok(response.to_string()));
                            }
                        }
                        Err(e) => {
                            results.push(Err(format!("Parse error: {}", e)));
                        }
                    }
                }

                Ok(results)
            },
        );

        let flattened_stream = stream
            .map(
                |result: Result<Vec<Result<String, String>>, Box<dyn Error>>| match result {
                    Ok(items) => futures_util::stream::iter(items),
                    Err(e) => futures_util::stream::iter(vec![Err(e.to_string())]),
                },
            )
            .flatten();

        Ok(Box::pin(flattened_stream))
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
                
                // In fallback mode, format tool response as user message with tool context
                let is_fallback = self.is_fallback_mode().await;
                let (role, content) = if is_fallback {
                    ("user".to_string(), format!("Tool response from {}: {}", tool_call.function.name, result))
                } else {
                    ("tool".to_string(), result)
                };
                
                tool_responses.push(Message {
                    role,
                    content,
                    images: None,
                    tool_calls: None,
                });
            }
        }
        tool_responses
    }

    pub async fn process_fallback_response(&self, content: &str) -> (String, Option<Vec<ToolCall>>) {
        let is_fallback = self.is_fallback_mode().await;
        if !is_fallback {
            return (content.to_string(), None);
        }

        FallbackToolHandler::process_fallback_response(content)
    }
}