use std::error::Error;
use std::pin::Pin;
use futures_util::{Stream, StreamExt};
use base64::{Engine as _, engine::general_purpose};

use crate::core::{Message, ToolCall, ChatStreamItem, PullProgress, ModelInfo, Tool, MonoModel};
use crate::providers::ollama::{OllamaClient, Model};
use crate::providers::anthropic::AnthropicClient;
use crate::providers::openai::OpenAIClient;

pub enum Provider {
    Ollama(OllamaClient),
    Anthropic(AnthropicClient),
    OpenAI(OpenAIClient),
}

pub struct NaoriAI {
    provider: Provider,
}

impl NaoriAI {
    /// Create Ollama client with endpoint URL and model name
    pub fn ollama(endpoint: String, model: String) -> Self {
        Self {
            provider: Provider::Ollama(OllamaClient::new(endpoint, model)),
        }
    }

    /// Create Anthropic client with API key and model name
    pub fn anthropic(api_key: String, model: String) -> Self {
        Self {
            provider: Provider::Anthropic(AnthropicClient::new(api_key, model)),
        }
    }

    /// Create OpenAI client with API key and model name
    pub fn openai(api_key: String, model: String) -> Self {
        Self {
            provider: Provider::OpenAI(OpenAIClient::new(api_key, model)),
        }
    }

    /// Create OpenRouter client with API key and model name (wraps OpenAI with OpenRouter base URL)
    pub fn openrouter(api_key: String, model: String) -> Self {
        Self {
            provider: Provider::OpenAI(OpenAIClient::with_base_url(
                api_key,
                model,
                "https://openrouter.ai/api/v1".to_string(),
            )),
        }
    }

    /// Create OpenAI client with custom base URL (for vLLM, local deployments, etc.)
    pub fn openai_custom(api_key: String, model: String, base_url: String) -> Self {
        Self {
            provider: Provider::OpenAI(OpenAIClient::with_base_url(api_key, model, base_url)),
        }
    }

    /// Add function tool to client. Automatically enables fallback mode for non-supporting models
    pub async fn add_tool(&mut self, tool: Tool) -> Result<(), Box<dyn Error>> {
        match &mut self.provider {
            Provider::Ollama(client) => client.add_tool(tool).await,
            Provider::Anthropic(client) => client.add_tool(tool).await,
            Provider::OpenAI(client) => client.add_tool(tool).await,
        }
    }

    /// Check if client is using fallback tool calling (XML prompting vs native tools)
    pub async fn is_fallback_mode(&self) -> bool {
        match &self.provider {
            Provider::Ollama(client) => client.is_fallback_mode().await,
            Provider::Anthropic(client) => client.is_fallback_mode().await,
            Provider::OpenAI(client) => client.is_fallback_mode().await,
        }
    }

    /// Enable/disable debug mode to show raw tool call XML in fallback mode
    pub fn set_debug_mode(&mut self, debug: bool) {
        match &mut self.provider {
            Provider::Ollama(client) => client.set_debug_mode(debug),
            Provider::Anthropic(client) => client.set_debug_mode(debug),
            Provider::OpenAI(client) => client.set_debug_mode(debug),
        }
    }

    /// Check if debug mode is enabled
    pub fn debug_mode(&self) -> bool {
        match &self.provider {
            Provider::Ollama(client) => client.debug_mode(),
            Provider::Anthropic(client) => client.debug_mode(),
            Provider::OpenAI(client) => client.debug_mode(),
        }
    }

    /// Check if model supports native tool calling by examining template
    pub async fn supports_tool_calls(&self) -> Result<bool, Box<dyn Error>> {
        match &self.provider {
            Provider::Ollama(client) => client.supports_tool_calls().await,
            Provider::Anthropic(client) => client.supports_tool_calls().await,
            Provider::OpenAI(client) => client.supports_tool_calls().await,
        }
    }

    /// Get provider capabilities (tool support, vision support)
    pub async fn capabilities(&self) -> Result<crate::core::ProviderCapabilities, Box<dyn Error>> {
        match &self.provider {
            Provider::Ollama(client) => client.capabilities().await,
            Provider::Anthropic(client) => client.capabilities().await,
            Provider::OpenAI(client) => client.capabilities().await,
        }
    }

    /// Send chat request with real-time streaming response
    pub async fn send_chat_request(
        &self,
        messages: &[Message],
    ) -> Result<Pin<Box<dyn Stream<Item = Result<ChatStreamItem, String>> + Send>>, Box<dyn Error>> {
        match &self.provider {
            Provider::Ollama(client) => client.send_chat_request(messages).await,
            Provider::Anthropic(client) => client.send_chat_request(messages).await,
            Provider::OpenAI(client) => client.send_chat_request(messages).await,
        }
    }

    /// Send chat request without streaming, returns complete response and tool calls
    pub async fn send_chat_request_no_stream(
        &self,
        messages: &[Message],
    ) -> Result<(String, Option<Vec<ToolCall>>), Box<dyn Error>> {
        match &self.provider {
            Provider::Ollama(client) => client.send_chat_request_no_stream(messages).await,
            Provider::Anthropic(client) => client.send_chat_request_no_stream(messages).await,
            Provider::OpenAI(client) => client.send_chat_request_no_stream(messages).await,
        }
    }

    /// Send chat request with images from file paths, returns real-time streaming response
    pub async fn send_chat_request_with_images(
        &self,
        messages: &[Message],
        image_paths: Vec<String>,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<ChatStreamItem, String>> + Send>>, Box<dyn Error>> {
        match &self.provider {
            Provider::Ollama(client) => client.send_chat_request_with_images(messages, image_paths).await,
            Provider::Anthropic(_) => {
                // For Anthropic, images should be encoded in the messages directly
                // This method is provided for backward compatibility with Ollama-style usage
                let mut messages_with_images = messages.to_vec();
                if let Some(last_message) = messages_with_images.last_mut() {
                    let mut encoded_images = Vec::new();
                    for image_path in image_paths {
                        let encoded = self.encode_image_file(&image_path).await?;
                        encoded_images.push(encoded);
                    }
                    last_message.images = Some(encoded_images);
                }
                self.send_chat_request(&messages_with_images).await
            }
            Provider::OpenAI(_) => {
                // For OpenAI, images should be encoded in the messages directly
                let mut messages_with_images = messages.to_vec();
                if let Some(last_message) = messages_with_images.last_mut() {
                    let mut encoded_images = Vec::new();
                    for image_path in image_paths {
                        let encoded = self.encode_image_file(&image_path).await?;
                        encoded_images.push(encoded);
                    }
                    last_message.images = Some(encoded_images);
                }
                self.send_chat_request(&messages_with_images).await
            }
        }
    }

    /// Send chat request with images from file paths, returns complete response and tool calls
    pub async fn send_chat_request_with_images_no_stream(
        &self,
        messages: &[Message],
        image_paths: Vec<String>,
    ) -> Result<(String, Option<Vec<ToolCall>>), Box<dyn Error>> {
        match &self.provider {
            Provider::Ollama(client) => client.send_chat_request_with_images_no_stream(messages, image_paths).await,
            Provider::Anthropic(_) => {
                // For Anthropic, images should be encoded in the messages directly
                let mut messages_with_images = messages.to_vec();
                if let Some(last_message) = messages_with_images.last_mut() {
                    let mut encoded_images = Vec::new();
                    for image_path in image_paths {
                        let encoded = self.encode_image_file(&image_path).await?;
                        encoded_images.push(encoded);
                    }
                    last_message.images = Some(encoded_images);
                }
                self.send_chat_request_no_stream(&messages_with_images).await
            }
            Provider::OpenAI(_) => {
                // For OpenAI, images should be encoded in the messages directly
                let mut messages_with_images = messages.to_vec();
                if let Some(last_message) = messages_with_images.last_mut() {
                    let mut encoded_images = Vec::new();
                    for image_path in image_paths {
                        let encoded = self.encode_image_file(&image_path).await?;
                        encoded_images.push(encoded);
                    }
                    last_message.images = Some(encoded_images);
                }
                self.send_chat_request_no_stream(&messages_with_images).await
            }
        }
    }

    /// Send chat request with image data from memory, returns real-time streaming response (single image: vec![data], multiple: vec![data1, data2])
    pub async fn send_chat_request_with_image_data(
        &self,
        messages: &[Message],
        images_data: Vec<Vec<u8>>,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<ChatStreamItem, String>> + Send>>, Box<dyn Error>> {
        match &self.provider {
            Provider::Ollama(client) => client.send_chat_request_with_images_data(messages, images_data).await,
            Provider::Anthropic(_) => {
                // For Anthropic, images should be encoded in the messages directly
                let mut messages_with_images = messages.to_vec();
                if let Some(last_message) = messages_with_images.last_mut() {
                    let mut encoded_images = Vec::new();
                    for image_data in images_data.clone() {
                        let encoded = self.encode_image_data(image_data).await?;
                        encoded_images.push(encoded);
                    }
                    last_message.images = Some(encoded_images);
                }
                self.send_chat_request(&messages_with_images).await
            }
            Provider::OpenAI(_) => {
                // For OpenAI, images should be encoded in the messages directly
                let mut messages_with_images = messages.to_vec();
                if let Some(last_message) = messages_with_images.last_mut() {
                    let mut encoded_images = Vec::new();
                    for image_data in images_data {
                        let encoded = self.encode_image_data(image_data).await?;
                        encoded_images.push(encoded);
                    }
                    last_message.images = Some(encoded_images);
                }
                self.send_chat_request(&messages_with_images).await
            }
        }
    }

    /// Send chat request with image data from memory, returns complete response and tool calls (single image: vec![data], multiple: vec![data1, data2])
    pub async fn send_chat_request_with_image_data_no_stream(
        &self,
        messages: &[Message],
        images_data: Vec<Vec<u8>>,
    ) -> Result<(String, Option<Vec<ToolCall>>), Box<dyn Error>> {
        match &self.provider {
            Provider::Ollama(client) => client.send_chat_request_with_images_data_no_stream(messages, images_data).await,
            Provider::Anthropic(_) => {
                // For Anthropic, images should be encoded in the messages directly
                let mut messages_with_images = messages.to_vec();
                if let Some(last_message) = messages_with_images.last_mut() {
                    let mut encoded_images = Vec::new();
                    for image_data in images_data.clone() {
                        let encoded = self.encode_image_data(image_data).await?;
                        encoded_images.push(encoded);
                    }
                    last_message.images = Some(encoded_images);
                }
                self.send_chat_request_no_stream(&messages_with_images).await
            }
            Provider::OpenAI(_) => {
                // For OpenAI, images should be encoded in the messages directly
                let mut messages_with_images = messages.to_vec();
                if let Some(last_message) = messages_with_images.last_mut() {
                    let mut encoded_images = Vec::new();
                    for image_data in images_data {
                        let encoded = self.encode_image_data(image_data).await?;
                        encoded_images.push(encoded);
                    }
                    last_message.images = Some(encoded_images);
                }
                self.send_chat_request_no_stream(&messages_with_images).await
            }
        }
    }

    /// Generate single completion from prompt without conversation context
    pub async fn generate(&self, prompt: &str) -> Result<String, Box<dyn Error>> {
        match &self.provider {
            Provider::Ollama(client) => client.generate(prompt).await,
            Provider::Anthropic(client) => {
                // Convert prompt to messages format for Anthropic
                let messages = vec![Message {
                    role: "user".to_string(),
                    content: prompt.to_string(),
                    images: None,
                    tool_calls: None,
                }];
                let (response, _) = client.send_chat_request_no_stream(&messages).await?;
                Ok(response)
            }
            Provider::OpenAI(client) => {
                // Convert prompt to messages format for OpenAI
                let messages = vec![Message {
                    role: "user".to_string(),
                    content: prompt.to_string(),
                    images: None,
                    tool_calls: None,
                }];
                let (response, _) = client.send_chat_request_no_stream(&messages).await?;
                Ok(response)
            }
        }
    }

    /// Generate streaming completion from prompt without conversation context
    pub async fn generate_stream(
        &self,
        prompt: &str,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<String, String>> + Send>>, Box<dyn Error>> {
        match &self.provider {
            Provider::Ollama(client) => client.generate_stream(prompt).await,
            Provider::Anthropic(client) => {
                // Convert prompt to messages format for Anthropic and convert stream
                let messages = vec![Message {
                    role: "user".to_string(),
                    content: prompt.to_string(),
                    images: None,
                    tool_calls: None,
                }];
                let stream = client.send_chat_request(&messages).await?;
                let mapped_stream = stream.map(|item| {
                    match item {
                        Ok(chat_item) => Ok(chat_item.content),
                        Err(e) => Err(e),
                    }
                });
                Ok(Box::pin(mapped_stream))
            }
            Provider::OpenAI(client) => {
                // Convert prompt to messages format for OpenAI and convert stream
                let messages = vec![Message {
                    role: "user".to_string(),
                    content: prompt.to_string(),
                    images: None,
                    tool_calls: None,
                }];
                let stream = client.send_chat_request(&messages).await?;
                let mapped_stream = stream.map(|item| {
                    match item {
                        Ok(chat_item) => Ok(chat_item.content),
                        Err(e) => Err(e),
                    }
                });
                Ok(Box::pin(mapped_stream))
            }
        }
    }

    /// Get available models from any provider
    pub async fn get_available_models(&self) -> Result<Vec<MonoModel>, Box<dyn Error>> {
        match &self.provider {
            Provider::Ollama(client) => {
                let models = client.list_local_models().await?;
                Ok(models.into_iter().map(|m| MonoModel {
                    id: m.name.clone(),
                    name: m.name,
                    provider: "Ollama".to_string(),
                    size: Some(m.size),
                    created: None,
                }).collect())
            }
            Provider::Anthropic(client) => {
                let models = client.get_available_models().await?;
                Ok(models.into_iter().map(|m| MonoModel {
                    id: m.id.clone(),
                    name: m.display_name,
                    provider: "Anthropic".to_string(),
                    size: None,
                    created: Some(m.created_at.parse().unwrap_or(0)),
                }).collect())
            }
            Provider::OpenAI(client) => {
                let models = client.get_available_models().await?;
                Ok(models.into_iter().map(|m| MonoModel {
                    id: m.id.clone(),
                    name: m.id,
                    provider: "OpenAI".to_string(),
                    size: None,
                    created: Some(m.created),
                }).collect())
            }
        }
    }

    /// List locally installed models (legacy method, use get_available_models instead)
    pub async fn list_local_models(&self) -> Result<Vec<Model>, Box<dyn Error>> {
        match &self.provider {
            Provider::Ollama(client) => client.list_local_models().await,
            _ => Err("list_local_models is only supported for Ollama provider".into()),
        }
    }

    /// Get detailed model information including template and parameters
    pub async fn show_model_info(&self, model_name: &str) -> Result<ModelInfo, Box<dyn Error>> {
        match &self.provider {
            Provider::Ollama(client) => client.show_model_info(model_name).await,
            Provider::Anthropic(_) => Err("show_model_info is not supported for Anthropic provider".into()),
            Provider::OpenAI(_) => Err("show_model_info is not supported for OpenAI provider".into()),
        }
    }

    /// Download model from provider registry (provider-specific operation)
    pub async fn pull_model(&self, model_name: &str) -> Result<(), Box<dyn Error>> {
        match &self.provider {
            Provider::Ollama(client) => client.pull_model(model_name).await,
            Provider::Anthropic(_) => Err("pull_model is not supported for Anthropic provider".into()),
            Provider::OpenAI(_) => Err("pull_model is not supported for OpenAI provider".into()),
        }
    }

    /// Download model with streaming progress updates (provider-specific operation)
    pub async fn pull_model_stream(
        &self,
        model_name: &str,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<PullProgress, String>> + Send>>, Box<dyn Error>> {
        match &self.provider {
            Provider::Ollama(client) => client.pull_model_stream(model_name).await,
            Provider::Anthropic(_) => Err("pull_model_stream is not supported for Anthropic provider".into()),
            Provider::OpenAI(_) => Err("pull_model_stream is not supported for OpenAI provider".into()),
        }
    }

    /// Execute tool calls and return formatted messages for conversation continuation
    pub async fn handle_tool_calls(&self, tool_calls: Vec<ToolCall>) -> Vec<Message> {
        match &self.provider {
            Provider::Ollama(client) => client.handle_tool_calls(tool_calls).await,
            Provider::Anthropic(client) => client.handle_tool_calls(tool_calls).await,
            Provider::OpenAI(client) => client.handle_tool_calls(tool_calls).await,
        }
    }

    /// Parse fallback tool calls from response content and clean XML artifacts
    pub async fn process_fallback_response(&self, content: &str) -> (String, Option<Vec<ToolCall>>) {
        match &self.provider {
            Provider::Ollama(client) => client.process_fallback_response(content).await,
            Provider::Anthropic(client) => client.process_fallback_response(content).await,
            Provider::OpenAI(client) => client.process_fallback_response(content).await,
        }
    }

    /// Get current model name for display purposes
    pub fn model(&self) -> &str {
        match &self.provider {
            Provider::Ollama(client) => &client.model,
            Provider::Anthropic(client) => &client.model,
            Provider::OpenAI(client) => &client.model,
        }
    }

    /// Access underlying Ollama client for provider-specific operations
    pub fn as_ollama(&self) -> Option<&OllamaClient> {
        match &self.provider {
            Provider::Ollama(client) => Some(client),
            Provider::Anthropic(_) => None,
            Provider::OpenAI(_) => None,
        }
    }

    /// Access underlying Ollama client mutably for provider-specific operations
    pub fn as_ollama_mut(&mut self) -> Option<&mut OllamaClient> {
        match &mut self.provider {
            Provider::Ollama(client) => Some(client),
            Provider::Anthropic(_) => None,
            Provider::OpenAI(_) => None,
        }
    }

    /// Access underlying Anthropic client for provider-specific operations
    pub fn as_anthropic(&self) -> Option<&AnthropicClient> {
        match &self.provider {
            Provider::Ollama(_) => None,
            Provider::Anthropic(client) => Some(client),
            Provider::OpenAI(_) => None,
        }
    }

    /// Access underlying Anthropic client mutably for provider-specific operations
    pub fn as_anthropic_mut(&mut self) -> Option<&mut AnthropicClient> {
        match &mut self.provider {
            Provider::Ollama(_) => None,
            Provider::Anthropic(client) => Some(client),
            Provider::OpenAI(_) => None,
        }
    }

    /// Encode image file to base64 string for use in Message.images
    pub async fn encode_image_file(&self, path: &str) -> Result<String, Box<dyn std::error::Error>> {
        let image_bytes = std::fs::read(path)?;
        Ok(general_purpose::STANDARD.encode(image_bytes))
    }

    /// Encode image bytes to base64 string for use in Message.images
    pub async fn encode_image_data(&self, bytes: Vec<u8>) -> Result<String, Box<dyn std::error::Error>> {
        Ok(general_purpose::STANDARD.encode(bytes))
    }
}